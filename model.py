import math
import torch
from torch import nn as nn
from torch import Tensor


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, *, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoetryNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        *,
        embed_size: int = 512,
        n_head=8,
        n_layer=4,
        hidden_size=512
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.embed_size = embed_size
        self.transformer = nn.Transformer(
            embed_size,
            nhead=n_head,
            num_decoder_layers=n_layer,
            num_encoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
        )
        self.liner = nn.Linear(embed_size, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_size)

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor):
        src = self.embed(src) * math.sqrt(self.embed_size)
        src = self.positional_encoding(src)

        self.transformer
        pass
