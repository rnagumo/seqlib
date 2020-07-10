
"""Base class for sequential class."""

from typing import Optional, Tuple, Dict

from torch import Tensor, nn


class BaseSequentialModel(nn.Module):
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

    def loss_func(self, x: Tensor) -> Dict[str, Tensor]:
        """Loss function.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.

        Returns:
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        _, loss_dict = self.inference(x)

        return loss_dict

    def inference(self, x: Tensor
                  ) -> Tuple[Tuple[Tensor, ...], Dict[str, Tensor]]:
        """Inferences with given observations.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.

        Returns:
            data (tuple of torch.Tensor): Tuple of reconstructed observations
                and latent variables. The first element of the tuple will be
                returned in `forward` method.
            loss_dict (dict of [str, torch.Tensor]): Dict of lossses.
        """

        raise NotImplementedError

    def sample(self, z: Optional[Tensor] = None) -> Tensor:
        """Samples observations.

        Args:
            z (torch.Tensor, optional): Latent variable, size `(b, l, d)`.
            time_step (int, optional): Time step for prediction.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, l, c, h, w)`.
        """

        raise NotImplementedError
