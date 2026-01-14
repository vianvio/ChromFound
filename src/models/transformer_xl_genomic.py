import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field
from functools import partial


@dataclass
class TransformerXLConfig:
    d_model: int = 256
    n_head: int = 8
    n_layer: int = 12
    seq_len: int = 512
    mem_len: int = 512
    ext_len: int = 0
    max_pos_len: int = 10000
    dropout: float = 0.1
    dropatt: float = 0.0
    d_inner: int = 1024
    activation: str = 'gelu'
    layer_norm_epsilon: float = 1e-5
    tie_projections: bool = True
    tie_word_embeddings: bool = True


class PositionalEmbedding(nn.Module):
    """Learnable genomic position embeddings for chromatin accessibility data"""
    def __init__(self, demb, max_pos_len=10000):
        super(PositionalEmbedding, self).__init__()

        self.pos_emb = nn.Parameter(torch.randn(1, max_pos_len, demb) * 0.02)
        
    def forward(self, pos_seq, bsz=None):
        # pos_seq: [seq_len] or [bsz, seq_len] - contains genomic positions
        if pos_seq.dim() == 1:
            # Expand to batch dimension if needed
            pos_seq = pos_seq.unsqueeze(0).expand(bsz, -1) if bsz is not None else pos_seq.unsqueeze(0)
            
        # Clamp positions to valid range
        pos_seq = torch.clamp(pos_seq, 0, self.pos_emb.size(1) - 1)
        
        # Gather embeddings for each position
        pos_emb = self.pos_emb.expand(pos_seq.size(0), -1, -1)
        emb = torch.gather(pos_emb, 1, pos_seq.unsqueeze(-1).expand(-1, -1, self.pos_emb.size(-1)))
        
        return emb


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout, activation='gelu', pre_lnorm=False):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU() if activation == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = self.CoreNet(self.layer_norm(inp))
            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = self.CoreNet(inp)
            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out)

        return output


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0,
                 tgt_len=None, ext_len=None, mem_len=None, pre_lnorm=False,
                 layer_norm_epsilon=1e-5):
        super(RelPartialLearnableMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head
        self.dropout = dropout

        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)

        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_epsilon)
        self.scale = 1 / (d_head ** 0.5)

        self.pre_lnorm = pre_lnorm

        # Relative learnable parameters
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_net = nn.Linear(self.d_model, self.n_head * self.d_head, bias=False)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])

        x = x_padded[1:].view_as(x)

        return x

    def forward(self, w, r, attn_mask=None, mems=None):
        # w : [bsz x len_q x d_model]
        # r : [bsz x len_k x d_model]
        # attn_mask : [bsz x len_q x len_k]

        if mems is not None:
            c = torch.cat([mems, w], 1)
        else:
            c = w

        if self.pre_lnorm:
            # layer normalization
            w = self.layer_norm(w)
            r = self.layer_norm(r)
            c = self.layer_norm(c)

        head_q, head_k, head_v = torch.chunk(self.qkv_net(w), 3, dim=-1)
        head_q = head_q.view(w.size(0), w.size(1), self.n_head, self.d_head)
        head_k = head_k.view(c.size(0), c.size(1), self.n_head, self.d_head)
        head_v = head_v.view(c.size(0), c.size(1), self.n_head, self.d_head)

        # [bsz x len_k x n_head x d_head]
        r_head_k = self.r_net(r)
        r_head_k = r_head_k.view(r.size(0), r.size(1), self.n_head, self.d_head)

        # compute attention score
        rw_head_q = head_q + self.r_w_bias                                    # [bsz x len_q x n_head x d_head]
        AC = torch.einsum('bind,bjnd->bnij', (rw_head_q, head_k))             # [bsz x n_head x len_q x len_k]

        rr_head_q = head_q + self.r_r_bias
        BD = torch.einsum('bind,bjnd->bnij', (rr_head_q, r_head_k))           # [bsz x n_head x len_q x len_k]
        BD = self._rel_shift(BD)

        # [bsz x n_head x len_q x len_k]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            attn_score = attn_score.float().masked_fill(
                attn_mask[..., None], -float('inf')).type_as(attn_score)

        # [bsz x n_head x len_q x len_k]
        attn_prob = F.softmax(attn_score, dim=-1).type_as(attn_score)
        attn_prob = self.dropatt(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('bnij,bjnd->bind', (attn_prob, head_v))

        # [bsz x len_q x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = w + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(w + attn_out)

        return output


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout,
                 **kwargs):
        super(RelPartialLearnableDecoderLayer, self).__init__()

        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, **kwargs)
        self.pos_ff = PositionwiseFF(
            d_model, d_inner, dropout, pre_lnorm=kwargs.get('pre_lnorm'))

    def forward(self, dec_inp, r, dec_attn_mask=None, mems=None):
        output = self.dec_attn(dec_inp, r, attn_mask=dec_attn_mask,
                               mems=mems)
        output = self.pos_ff(output)

        return output


class GenomicTransformerXL(nn.Module):
    def __init__(self, config: TransformerXLConfig):
        super(GenomicTransformerXL, self).__init__()

        self.config = config
        self.n_head = config.n_head
        self.d_model = config.d_model
        self.d_head = config.d_model // config.n_head
        self.d_inner = config.d_inner
        self.dropout = config.dropout

        self.word_emb = nn.Linear(config.feature_num, config.d_model)  # Changed from embedding to linear for continuous values

        self.drop = nn.Dropout(config.dropout)

        self.n_layer = config.n_layer

        self.tgt_len = config.seq_len
        self.mem_len = config.mem_len
        self.ext_len = config.ext_len
        self.max_pos_len = config.max_pos_len

        self.layers = nn.ModuleList()
        for i in range(config.n_layer):
            self.layers.append(
                RelPartialLearnableDecoderLayer(
                    config.n_head, config.d_model, self.d_head, config.d_inner,
                    config.dropout, dropatt=config.dropatt, pre_lnorm=False,
                    layer_norm_epsilon=config.layer_norm_epsilon)
            )

        self.pos_emb = PositionalEmbedding(config.d_model, config.max_pos_len)
        self.r_w_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(self.n_head, self.d_head))

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def init_mems(self, bsz):
        if self.mem_len > 0:
            mems = []
            param = next(self.parameters())
            for i in range(self.n_layer + 1):
                empty = torch.empty(0, dtype=param.dtype, device=param.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None: return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.mem_len - self.ext_len` to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):

                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, value, chromosome, hg38_start, hg38_end, mems=None):
        # value: [batch_size, seq_len, 1] - normalized accessibility values
        # chromosome: [batch_size, seq_len] - chromosome indices
        # hg38_start: [batch_size, seq_len] - genomic start positions
        # hg38_end: [batch_size, seq_len] - genomic end positions
        
        # Get sequence length
        qlen, bsz = value.size(1), value.size(0)
        
        # Prepare input embeddings
        word_emb = self.word_emb(value)  # Convert continuous values to embeddings
        
        # Create combined positional information
        # Using start positions as the primary position indicator
        pos_seq = hg38_start  # Use genomic start positions as positional indicators
        
        # Generate positional embeddings
        pos_emb = self.pos_emb(pos_seq, bsz)
        
        # Apply dropout to embeddings
        word_emb = self.drop(word_emb)
        pos_emb = self.drop(pos_emb)
        
        # Prepare memory
        if mems is None:
            mems = self.init_mems(bsz)
        
        # Prepare masks
        hidden_states = [word_emb]
        attentions = []
        
        if mems is None:
            mems = [None] * len(self.layers)
        
        # Process through transformer layers
        for i, layer in enumerate(self.layers):
            # Get memory for this layer
            mem_i = mems[i] if i < len(mems) else None
            
            # Forward through layer
            output = layer(hidden_states[-1], pos_emb, mems=mem_i)
            hidden_states.append(output)
        
        # Return the final hidden states
        output = hidden_states[-1]
        
        # Update memories for next iteration
        new_mems = self._update_mems(hidden_states, mems, mlen=0, qlen=qlen)
        
        return output, new_mems


class GenomicTransformerXLLM(nn.Module):
    def __init__(self, **kwargs):
        super(GenomicTransformerXLLM, self).__init__()
        self.model_args = kwargs

        # Create config from model arguments
        config = TransformerXLConfig(
            d_model=kwargs.get("embedding_dim", 256),
            n_head=kwargs.get("n_head", 8),
            n_layer=kwargs.get("encoder_layers", 12),
            seq_len=kwargs.get("max_length", 512),
            mem_len=kwargs.get("mem_len", 512),
            feature_num=kwargs.get("feature_num", 1),
            max_pos_len=kwargs.get("max_pos_len", 100000)
        )

        self.backbone = GenomicTransformerXL(config)

        # Mask token prediction head
        self.mask_token_prediction = nn.Linear(
            config.d_model,
            kwargs.get("value_size", 1)
        )

    def forward(self, value, chromosome, hg38_start, hg38_end, key_padding_mask=None):
        x, _ = self.backbone(value, chromosome, hg38_start, hg38_end)
        logits = self.mask_token_prediction(x)
        return logits


class FinetuneGenomicTransformerXLCellType(GenomicTransformerXLLM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        max_length = self.model_args["max_length"]
        self.post_backbone_dropout = nn.Dropout(p=0.3)
        self.feature_projection = nn.Sequential(
            nn.Linear(self.model_args["embedding_dim"], 256),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1),
            nn.GELU()
        )
        self._init_weights(self.feature_projection)
        in_feature = max_length
        self.ft_cell_type_projection = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, self.model_args["cell_type_num"])
        )
        self._init_weights(self.ft_cell_type_projection)

        # Freeze the mask token prediction head during fine-tuning
        for name, param in self.mask_token_prediction.named_parameters():
            param.requires_grad = False

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, value, chromosome, hg38_start, hg38_end, **kwargs):
        x, _ = self.backbone(value, chromosome, hg38_start, hg38_end)
        x = self.feature_projection(x)
        x = torch.squeeze(x, dim=-1)
        x = self.post_backbone_dropout(x)
        x_cell_type_prediction = self.ft_cell_type_projection(x)
        return x_cell_type_prediction