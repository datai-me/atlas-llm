"""
Causal Transformer Language Model
---------------------------------
Autoregressive decoder-only architecture.
"""

import torch
import torch.nn as nn
from core.transformer.embeddings import TokenEmbedding, PositionalEncoding
from core.transformer.transformer_block import TransformerBlock
from core.attention.causal_mask import generate_causal_mask


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        max_len=512,
    ):
        super().__init__()

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        self.blocks = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        device = x.device
        B, T = x.size()

        causal_mask = generate_causal_mask(T, device)

        x = self.token_emb(x)
        x = self.pos_emb(x)

        for block in self.blocks:
            x = block(x, causal_mask)

        x = self.norm(x)
        return self.lm_head(x)
