import torch
from torch import nn

from dst.nn.functional import logC, approx_logC


class NLLvMF(nn.Module):
    """Negative log-likelihood Von Mises-Fisher loss.

    Args:
        p (int, optional): normalization power. Defaults to 2.
        l1 (float, optional): penalisation term. Defaults to 0.02.
        l2 (float, optional): penalisation term. Defaults to 0.1.
        use_approximation (bool, optional): whether to use Bessel function approximation. Defaults to False.

    Input:
        - **predicted** of shape `(batch, emb_size)`: a :class:`torch.FloatTensor` with predicted vectors.
        - **target** of shape `(batch, emb_size)`: a :class:`torch.FloatTensor` with target vectors.

    Output:
        - **loss**: a :class:`torch.FloatTensor` with loss value.
    """

    def __init__(self, p=2, l1=0.02, l2=0.1, use_approximation=False):
        super(NLLvMF, self).__init__()
        self.p = p
        self.l1 = l1
        self.l2 = l2
        self.logCfunc = approx_logC if use_approximation else logC

    def forward(self, predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        batch_size, emb_size = target.shape

        m = torch.full((batch_size,), emb_size, device=predicted.device)
        norm = torch.norm(predicted, p=self.p, dim=-1)
        # predicted = nn.functional.normalize(predicted, dim=-1)
        dot = (predicted * target).sum(-1)
        # loss = -self.logCfunc(m, norm) + torch.log(1 + norm) * (self.l1 - dot)
        loss = -self.logCfunc(m, norm) + self.l1 * norm - self.l2 * dot
        return loss.sum() / batch_size
