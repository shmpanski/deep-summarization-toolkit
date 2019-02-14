from typing import Tuple

import torch


def autoregressive_mask(batch_size: int, size: Tuple[int], device: torch.device = 'cpu', delay=1) -> torch.ByteTensor:
    x = torch.ones(*size, device=device).triu(delay)

    return x.repeat(batch_size, 1, 1).byte()


def autoregressive_mask_like(tensor: torch.Tensor, delay=0) -> torch.ByteTensor:
    """Generate auto-regressive mask for tensor. It's used to preserving the auto-regressive property.

    Args:
        tensor (torch.Tensor): of shape ``(batch, seq_len)``.
        delay (int): delay value.

    Returns:
        torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
        illegal attention connections between decoder sequence tokens.

    """
    batch_size, seq_len = tensor.shape

    return autoregressive_mask(batch_size, (seq_len, seq_len), device=tensor.device, delay=delay)
