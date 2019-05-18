import scipy.special
import torch


def ive(v: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """Modified Bessel function.

    Args:
        v (torch.Tensor): value.
        z (torch.Tensor): value.

    Returns:
        torch.Tensor: Modified Bessel function value.
    """
    np_z = z.detach().cpu().numpy()
    np_v = v.detach().cpu().numpy()
    result = scipy.special.ive(np_v, np_z, dtype=np_z.dtype)
    return torch.Tensor(result).to(z.device)


class logC(torch.autograd.Function):
    """Log of normalization term for NLLvMF loss.

    Args:
        m (torch.Tensor): m value.
        k (torch.Tensor): k value.

    Returns:
        torch.Tensor: log normalization term.
    """

    @staticmethod
    def forward(ctx, m, k):
        Im2_1 = ive(m / 2 - 1, k)
        Cmk = (m / 2 - 1) * torch.log(k) - m * 0.3990899 - torch.log(1e-12 + Im2_1)
        ctx.save_for_backward(Im2_1, m, k)
        return Cmk

    @staticmethod
    def backward(ctx, grad_output):
        Im2_1, m, k, = ctx.saved_tensors
        Im2 = ive(m / 2, k)
        return None, -grad_output * Im2 / (Im2_1 + 1e-12)


logC = logC.apply


def approx_logC(m: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Approximated log of normalization term for NLLvMF loss.

    Args:
        m (torch.Tensor): m value.
        k (torch.Tensor): k value.

    Returns:
        torch.Tensor: log normalization term.
    """
    t = torch.sqrt(torch.pow(m + 1, 2) + torch.pow(k, 2))
    return t - (m - 1) * torch.log(m - 1 + t)
