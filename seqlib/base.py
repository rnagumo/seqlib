
"""Base class for sequential class."""

from typing import Optional, Tuple, Dict

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
