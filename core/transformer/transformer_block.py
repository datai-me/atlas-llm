"""
Causal Transformer Block
------------------------
Self-attention + feed-forward with pre-norm architecture.
"""

import torch.nn as nn
from core.attention.multi_head_attention import MultiHeadAttention


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask):
        # Pre-Norm attention (more stable for deep models)
        attn_out = self.attn(self.norm1(x), causal_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward
        ff_out = self.ff(self.norm2(x))
        x = x + self.dropout(ff_out)

        return x
