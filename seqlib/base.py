
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

        (recon, *_), _ = self.inference(x)

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

        _, loss_dict = self.inference(x, mask, beta)

        return loss_dict

    def inference(self, x: Tensor, mask: Optional[Tensor] = None,
                  beta: float = 1.0
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inferences with given observations.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            mask (torch.Tensor, optional): Sequence mask for valid data.
            beta (float, optional): Beta coefficient of KL term.

        Returns:
            data (tuple of torch.Tensor): Tuple of reconstructed observations
                and latent variables. The first element of the tuple should be
                recontstructed observations.
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, x: Optional[Tensor] = None, time_step: int = 0) -> Tensor:
        """Reconstructs and samples observations.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            time_step (int, optional): Time step for prediction.

        Returns:
            sample (torch.Tensor): Reconstructed and sampled observations, size
                `(b, l + t, c, h, w)`.
        """

        raise NotImplementedError
