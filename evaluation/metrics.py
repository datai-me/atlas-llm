"""
Evaluation Metrics
------------------
Basic perplexity computation.
"""

import torch
import math


def perplexity(loss: float):
    """
    Compute perplexity from cross-entropy loss.
    """
    return math.exp(loss)
