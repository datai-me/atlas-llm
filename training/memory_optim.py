"""
Memory Optimization Utilities
-----------------------------
Gradient checkpointing to reduce activation memory.
"""

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_block(block, x, causal_mask):
    """
    Recomputes forward pass during backward to save memory.
    """
    def custom_forward(*inputs):
        return block(*inputs)

    return checkpoint(custom_forward, x, causal_mask)
