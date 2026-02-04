"""
Causal Mask Utilities
---------------------
Generates autoregressive masks to prevent attention to future tokens.
"""

import torch


def generate_causal_mask(seq_len: int, device):
    """
    Create a lower-triangular causal mask.

    mask[i, j] = 1 if j <= i
                 0 otherwise

    Shape: (1, 1, seq_len, seq_len)
    Compatible with multi-head attention broadcasting.
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)
