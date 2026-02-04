"""
Transformer Language Model
--------------------------
Full causal language model.
"""

import torch.nn as nn
from core.transformer.embeddings import TokenEmbedding, PositionalEncoding
from core.transformer.transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        max_len: int = 512,
    ):
        super().__init__()

        self.token_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff)
                for _ in range(num_layers)
            ]
        )

        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        x = self.token_emb(x)
        x = self.pos_emb(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.lm_head(x)
