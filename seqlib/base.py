
"""Base class for sequential class."""

from typing import Optional, Tuple, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BaseSequentialVAE(nn.Module):
    """Base class for sequential model."""

    def forward(self, x: Tensor, time_steps: int = 0) -> Tensor:
        """Forwards to reconstruct given data.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            time_steps (int, optional): Time step for prediction.

        Returns:
            recon (torch.Tensor): Reconstructed observations.
        """

        recon, *_ = self.sample(x, time_steps)

        return recon

    def loss_func(self, x: Tensor, mask: Optional[Tensor] = None,
                  beta: float = 1.0) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            mask (torch.Tensor, optional): Sequence mask for valid data with
                binary values, size `(b, l)`.
            beta (float, optional): Beta coefficient of KL term.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, x: Optional[Tensor] = None, time_steps: int = 0,
               batch_size: int = 1) -> Tuple[Tensor, ...]:
        """Reconstructs and samples observations.

        Args:
            x (torch.Tensor, optional): Observation tensor, size
                `(b, l, c, h, w)`.
            time_steps (int, optional): Time step for prediction.
            batch_size (int, optional): Batch size for samping, used if `x` is
                `None`.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or sampled
                data. The first element should be reconstructed observations.

        Raises:
            ValueError: If `x` is `None` and `time_steps` is non positive.
        """

        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        """Device property of this model.

        Returns:
            device (torch.device): Device information.

        Raises:
            RuntimeError: If device cannot be determined.
        """

        devices = (set(param.device for param in self.parameters())
                   | set(buf.device for buf in self.buffers()))

        if len(devices) != 1:
            raise RuntimeError(
                    f"Device cannot be determined: {len(devices)} "
                    "different devices found.")

        return next(iter(devices))


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


def nll_bernoulli(x: Tensor, probs: Tensor, reduce: bool = True) -> Tensor:
    """Negative log likelihood for Bernoulli distribution.

    Ref)
    https://pytorch.org/docs/stable/_modules/torch/distributions/bernoulli.html#Bernoulli
    https://github.com/pytorch/pytorch/blob/master/torch/distributions/utils.py#L75

    Args:
        x (torch.Tensor): Inputs tensor, size `(*, dim)`.
        probs (torch.Tensor): Probability parameter, size `(*, dim)`.
        reduce (bool, optional): If `True`, sum calculated loss for each
            data point.

    Returns:
        nll (torch.Tensor): Calculated nll for each data, size `(*,)` if
            `reduce` is `True`, `(*, dim)` otherwise.
    """

    probs = probs.clamp(min=1e-6, max=1 - 1e-6)
    logits = torch.log(probs) - torch.log1p(-probs)
    nll = -F.binary_cross_entropy_with_logits(logits, x, reduction="none")

    if reduce:
        return nll.sum(-1)
    return nll
