import torch


def autoregressive_mask(tensor):
    """Generate auto-regressive mask for tensor. It's used to preserving the auto-regressive property.

    Args:
        tensor (torch.Tensor): of shape ``(batch, seq_len)``.

    Returns:
        torch.Tensor: a byte mask tensor of shape ``(batch, seq_len, seq_len)`` containing mask for
        illegal attention connections between decoder sequence tokens.

    """
    batch_size, seq_len = tensor.shape
    x = torch.ones(
        seq_len, seq_len, device=tensor.device).tril(-1).transpose(0, 1)

    return x.repeat(batch_size, 1, 1).byte()
