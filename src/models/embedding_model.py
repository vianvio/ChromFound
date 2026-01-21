import torch
from torch import nn


class ValueEmbedding(nn.Module):
    def __init__(self, hidden_dim, do_fft):
        super(ValueEmbedding, self).__init__()
        self.embedding = nn.Linear(1, hidden_dim)
        self.do_fft = do_fft

    def forward(self, x):
        if self.do_fft:
            x = torch.fft.fft(x, dim=-1).real
        # x has shape [batch, seq_len, 1], so we don't need to unsqueeze again
        return self.embedding(x)


class PositionalEmbeddingWithDnaPosition(nn.Module):
    def __init__(self, d_model, batch_size, seq_length, positional_temp, device):
        super(PositionalEmbeddingWithDnaPosition, self).__init__()
        self.d_model = d_model
        self.positional_temp = positional_temp
        self.device = device

    def forward(self, x):
        # x shape: (batch_size, seq_length)
        batch_size, seq_length = x.shape
        # Create positional encoding dynamically based on input size
        encoding = torch.zeros(batch_size, seq_length, self.d_model, device=x.device)

        pos = x.float().unsqueeze(-1)  # (batch_size, seq_length, 1)
        _2i = torch.arange(0, self.d_model, 2, device=x.device)  # (d_model//2,)

        # Calculate sin/cos values
        div_term = 10000 ** (_2i.float() / self.d_model)
        div_term = div_term.unsqueeze(0).unsqueeze(0)  # (1, 1, d_model//2)

        pos_expanded = pos.expand(-1, -1, len(_2i))  # (batch_size, seq_length, d_model//2)

        encoding[:, :, 0::2] = torch.sin(pos_expanded / div_term / self.positional_temp)
        encoding[:, :, 1::2] = torch.cos(pos_expanded / div_term / self.positional_temp)

        return encoding


class PretrainEmbeddingSimple(nn.Module):
    def __init__(
        self,
        embedding_dim,
        chromosome_size,
        embedding_dropout,
        positional_embedding_type,
        positional_temp,
        batch_size,
        seq_length,
        device,
        chromatin_embedding
    ):
        super(PretrainEmbeddingSimple, self).__init__()
        self.value_embedding = ValueEmbedding(embedding_dim, False)
        self.chromatin_embedding = chromatin_embedding
        self.positional_embedding_type = positional_embedding_type
        if self.chromatin_embedding:
            self.chromosome_embedding = nn.Embedding(chromosome_size, embedding_dim)
            self.position_embedding = PositionalEmbeddingWithDnaPosition(
                embedding_dim,
                batch_size,
                seq_length,
                positional_temp,
                device
            )
        self.embedding_dropout = embedding_dropout
        self.dropout = nn.Dropout(p=self.embedding_dropout)

    def forward(self, value, chromosome, hg38_start, hg38_end):
        if self.chromatin_embedding:
            embedding = self.value_embedding(value) + self.chromosome_embedding(chromosome) + self.position_embedding(
                hg38_start) + self.position_embedding(hg38_end)
        else:
            embedding = self.value_embedding(value)
        if self.embedding_dropout > 0:
            return self.dropout(embedding)
        else:
            return embedding
