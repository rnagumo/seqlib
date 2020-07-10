
"""Base class for sequential class."""

from typing import Optional, Tuple, Dict

import math

from torch import Tensor, nn


class BaseSequentialVAE(nn.Module):
    """Base class for sequential model."""

    def forward(self, x: Tensor) -> Tensor:
        """Forwards to reconstruct given data.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.

        Returns:
            recon (torch.Tensor): Reconstructed observations.
        """

        recon, *_ = self.sample(x)

        return recon

    def loss_func(self, x: Tensor, mask: Optional[Tensor] = None,
                  beta: float = 1.0) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            mask (torch.Tensor, optional): Sequence mask for valid data.
            beta (float, optional): Beta coefficient of KL term.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, x: Optional[Tensor] = None, time_step: int = 0
               ) -> Tuple[Tensor, ...]:
        """Reconstructs and samples observations.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            time_step (int, optional): Time step for prediction.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or sampled
                data. The first element should be reconstructed observations.
        """

        raise NotImplementedError


def kl_divergence_normal(mu0: Tensor, var0: Tensor, mu1: Tensor, var1: Tensor,
                         reduce: bool = True) -> Tensor:
    """Kullback Leibler divergence for 1-D Normal distributions.

    p = N(mu0, var0)
    q = N(mu1, var1)
    KL(p||q) = 1/2 * (var0/var1 + (mu1-mu0)^2/var1 - 1 + log(var1/var0))

    Args:
        mu0 (torch.Tensor): Mean vector of p, size.
        var0 (torch.Tensor): Diagonal variance of p.
        mu1 (torch.Tensor): Mean vector of q.
        var1 (torch.Tensor): Diagonal variance of q.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        kl (torch.Tensor): Calculated kl divergence for each data.
    """

    diff = mu1 - mu0
    kl = (var0 / var1 + diff ** 2 / var1 - 1 + (var1 / var0).log()) * 0.5

    if reduce:
        return kl.sum(-1)
    return kl


def nll_normal(x: Tensor, mu: Tensor, var: Tensor, reduce: bool = True
               ) -> Tensor:
    """Negative log likelihood for 1-D Normal distribution.

    Args:
        x (torch.Tensor): Inputs tensor, size `(*, dim)`.
        mu (torch.Tensor): Mean vector, size `(*, dim)`.
        var (torch.Tensor): Variance vector, size `(*, dim)`.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data, size `(*,)` if
            `reduce` is `True`, `(*, dim)` otherwise.
    """

    nll = 0.5 * ((2 * math.pi * var).log() + (x - mu) ** 2 / var)

    if reduce:
        return nll.sum(-1)
    return nll
