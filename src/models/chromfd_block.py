import torch
from mamba_ssm.ops.triton.layer_norm import RMSNorm
from mamba_ssm.ops.triton.layer_norm import layer_norm_fn
from torch import nn

from chromfd_flashatt import ChromFoundTransformerBlock
from genomic_dynamic_conv import GenomicDynamicConvBlock


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, mlp_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,
        wpsa_window_size=0, shift_size=0, wpsa_heads=0, seq_length=0
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm, residual connection, and wpsa heads.

        Args:
            wpsa_window_size: Window size for wpsa.
            shift_size: Shift size for wpsa.
            wpsa_heads: Number of attention heads for wpsa.
            seq_length: Input sequence length (for wpsa initialization).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(32)
        self.reduction_layer = nn.Linear(dim, 32)
        self.expansion_layer = nn.Linear(32, dim)
        if mlp_cls is not nn.Identity:
            self.norm2 = norm_cls(dim)
            self.mlp = mlp_cls(dim)
        else:
            self.mlp = None
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

        if wpsa_window_size > 0:
            self.chromfound_block = ChromFoundTransformerBlock(
                dim=dim,
                input_resolution=seq_length,
                num_heads=wpsa_heads,
                window_size=wpsa_window_size,
                shift_size=shift_size
            )
        else:
            self.chromfound_block = None

        # Add genomic dynamic convolution module
        self.genomic_conv_block = GenomicDynamicConvBlock(
            embedding_dim=dim,
            num_layers=1,
            dropout=0.1
        )

        # Gating mechanism to control flow between different modules in reduced space
        self.gate = nn.Sequential(
            nn.Linear(32 + 32, 32),  # genomic_reduced (32) + transformer_output_reduced (32)
            nn.Sigmoid()
        )

    def forward(
        self, hidden_states, residual=None, inference_params=None, pos_start=None, pos_end=None, **mixer_kwargs
    ):
        """
        Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual)).
            inference_params: Inference parameters for mamba(required).
            pos_start: start positions for genomic features
            pos_end: end positions for genomic features
        """
        # Step 1: Residual connection and normalization
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            hidden_states, residual = layer_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
                is_rms_norm=isinstance(self.norm, RMSNorm)
            )

        # Step 2: Pass through TransformerBlock if present
        transformer_output = None
        if self.chromfound_block is not None:
            transformer_output = self.chromfound_block(hidden_states, padding_mask=None)
            # Apply reduction layer to transformer output to match mixer expectations
            transformer_output_reduced = self.reduction_layer(transformer_output)

        # Step 3: Apply genomic dynamic convolution
        genomic_conv_output = None
        if pos_start is not None and pos_end is not None:
            # Use the original hidden states (before transformer) for genomic convolution
            # and pass transformer output as other_module_output for combination
            genomic_conv_output = self.genomic_conv_block(
                hidden_states, pos_start, pos_end,
                other_module_output=transformer_output if transformer_output is not None else hidden_states
            )

        # Step 4: Combine outputs using gating mechanism if both are available
        if genomic_conv_output is not None and transformer_output is not None:
            # Apply reduction to genomic_conv_output to match transformer dimension for gating
            genomic_reduced = self.reduction_layer(genomic_conv_output)  # 256 -> 32
            # Combine reduced representations for gating
            combined_input = torch.cat([genomic_reduced, transformer_output_reduced], dim=-1)  # 32 + 32 = 64
            # Apply the gate to get weights for combination
            gate_weights = self.gate(combined_input)  # (B, L, 32)
            # Combine in reduced space
            combined_reduced = gate_weights * genomic_reduced + (1 - gate_weights) * transformer_output_reduced
            # Use the combined reduced representation for mixer
            hidden_states = combined_reduced  # 32-dim
        elif genomic_conv_output is not None:
            # Apply reduction to genomic conv output to match mixer expectation
            hidden_states = self.reduction_layer(genomic_conv_output)  # 256 -> 32
        elif transformer_output is not None:
            hidden_states = transformer_output_reduced  # Already 32-dim
        else:
            # If neither is available, use the original hidden_states after norm
            # Apply reduction to match mixer expectation
            hidden_states = self.reduction_layer(hidden_states)  # 256 -> 32

        # Step 5: Pass through mamba mixer (operates on 32-dim representation)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params, **mixer_kwargs)

        # Step 6: Expand back to original dimension after mixing
        hidden_states = self.expansion_layer(hidden_states)  # 32 -> 256

        # Step 6: Optional MLP block
        if self.mlp is not None:
            if not self.fused_add_norm:
                residual = hidden_states + residual
                residual = self.norm2(residual.to(dtype=self.norm2.weight.dtype))
                if self.residual_in_fp32:
                    residual = residual.to(torch.float32)
            else:
                hidden_states, residual = layer_norm_fn(
                    hidden_states,
                    self.norm2.weight,
                    self.norm2.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm2.eps,
                    is_rms_norm=isinstance(self.norm2, RMSNorm)
                )
            hidden_states = self.mlp(hidden_states)

        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
