import math
import torch
from torch import nn as nn
from torch import Tensor, device


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, *, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class PoetryNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        device: device,
        *,
        embed_size: int = 512,
        n_head=8,
        n_layer=4,
        hidden_size=512
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, device=device)
        self.embed_size = embed_size
        self.sq = math.sqrt(self.embed_size)
        self.transformer = nn.Transformer(
            embed_size,
            nhead=n_head,
            num_decoder_layers=n_layer,
            num_encoder_layers=n_layer,
            batch_first=True,
            dim_feedforward=hidden_size,
            device=device,
        )
        self.liner = nn.Linear(embed_size, vocab_size)
        self.positional_encoding = PositionalEncoding(embed_size)

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
    ):
        src = self.embed(src) * self.sq
        src = self.positional_encoding(src)

        tgt = self.embed(tgt) * self.sq
        tgt = self.positional_encoding(tgt)

        out = self.transformer.forward(
            src,
            tgt,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=src_padding_mask,
            src_key_padding_mask=src_padding_mask,
            tgt_key_padding_mask=tgt_padding_mask,
        )
        out = self.liner.forward(out)
        return out

    def encode(
        self, src: Tensor, noise: bool = True, noise_intensity: float = 1
    ) -> Tensor:
        embeded = self.embed.forward(src)
        if noise:
            embeded += torch.rand_like(embeded) * noise_intensity
        return self.transformer.encoder.forward(
            self.positional_encoding.forward(embeded * self.sq)
        )

    def decode(self, tgt: Tensor, memory: Tensor) -> Tensor:
        return self.liner.forward(
            self.transformer.decoder.forward(
                self.positional_encoding.forward(self.embed(tgt) * self.sq),
                memory,
            )
        )
