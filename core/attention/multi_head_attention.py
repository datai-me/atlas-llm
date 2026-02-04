"""
Multi-Head Causal Self-Attention
--------------------------------
Implements autoregressive attention with memory-efficient layout.
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Single projection for Q, K, V (better memory locality)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)

        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, causal_mask=None):
        """
        x: (batch_size, seq_len, d_model)
        causal_mask: (1, 1, seq_len, seq_len)
        """
        B, T, C = x.size()

        # Project once, then split
        qkv = self.qkv(x)
        qkv = qkv.view(B, T, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        # Apply causal mask
        if causal_mask is not None:
            scores = scores.masked_fill(causal_mask == 0, float("-inf"))

        # Softmax + dropout
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        output = torch.matmul(attn, v)

        # Recombine heads
        output = output.transpose(1, 2).contiguous()
        output = output.view(B, T, C)

        return self.out(output)
