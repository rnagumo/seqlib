
"""Deep Markov Model (DMM)."""

from typing import Tuple, Optional, Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseSequentialVAE, kl_divergence_normal, nll_bernoulli


class StochasticPrior(nn.Module):
    """Stochastic prior: p(z|h).

    Args:
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, z_dim: int):
        super().__init__()

        self.fc1 = nn.Linear(z_dim, z_dim)
        self.fc2 = nn.Linear(z_dim, z_dim)

    def forward(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode hidden state to the parameters of latent distribution.

        Args:
            z (torch.Tensor): Latent states, size `(b, z)`.

        Returns:
            loc (torch.Tensor): Mean of Gausssian, size `(b, z)`.
            var (torch.Tensor): Variance of Gausssian, size `(b, z)`.
        """

        loc = self.fc1(z)
        var = F.softplus(self.fc2(z))

        return loc, var


class Generator(nn.Module):
    """Generator of observations: p(x|h, z).

    Args:
        x_channels (int): Channel number of observations.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, x_channels: int, z_dim: int):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1024),
            nn.ReLU(),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, x_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z: Tensor) -> Tensor:
        """Generates observations given latents and hiddens.

        Args:
            z (torch.Tensor): Latent states, size `(b, z)`.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        h = self.fc(z)
        h = h.view(-1, 64, 4, 4)
        x = self.deconv(h)

        return x


class Inference(nn.Module):
    """Inference of latents: q(z|x, h).

    Args:
        x_channels (int): Channel number of observations.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, x_channels: int, z_dim: int):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(x_channels, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, z_dim),
        )

        self.fc1 = nn.Linear(z_dim * 2, z_dim)
        self.fc2 = nn.Linear(z_dim * 2, z_dim)

    def forward(self, x: Tensor, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Variational inference of latents.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            z (torch.Tensor): Latent states, size `(b, z)`.

        Returns:
            loc (torch.Tensor): Mean of Gausssian, size `(b, z)`.
            var (torch.Tensor): Variance of Gausssian, size `(b, z)`.
        """

        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        h = torch.cat([x, z], dim=1)
        loc = self.fc1(h)
        var = F.softplus(self.fc2(h))

        return loc, var


class DeepMarkovModel(BaseSequentialVAE):
    """Deep Markov Model (DMM).

    Args:
        x_channels (int): Channel number of observations.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, x_channels: int, z_dim: int):
        super().__init__()

        self.x_channels = x_channels
        self.z_dim = z_dim

        self.prior = StochasticPrior(z_dim)
        self.decoder = Generator(x_channels, z_dim)
        self.encoder = Inference(x_channels, z_dim)

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

        # Data size
        batch, seq_len, *_ = x.size()

        # Initial parameter
        z_t = x.new_zeros((batch, self.z_dim))

        # Losses
        nll_loss = x.new_zeros((batch,))
        kl_loss = x.new_zeros((batch,))

        for t in range(seq_len):
            # 1. Sample stochastic latents
            p_z_t_mu, p_z_t_var = self.prior(z_t)
            q_z_t_mu, q_z_t_var = self.encoder(x[:, t], z_t)

            z_t = q_z_t_mu + q_z_t_var ** 0.5 * torch.randn_like(q_z_t_var)

            # 2. Decode observations
            recon = self.decoder(z_t)

            # Calculate loss
            _nll_loss_t = nll_bernoulli(x[:, t], recon, reduce=False)
            _nll_loss_t = _nll_loss_t.sum(dim=[1, 2, 3])

            _kl_loss_t = kl_divergence_normal(
                q_z_t_mu, q_z_t_var, p_z_t_mu, p_z_t_var, reduce=True)

            # Set mask
            if mask is not None:
                _nll_loss_t = _nll_loss_t * mask[:, t]
                _kl_loss_t = _kl_loss_t * mask[:, t]

            # Accumulate loss
            nll_loss += _nll_loss_t
            kl_loss += _kl_loss_t

        # Multiply beta coefficient
        kl_loss *= beta

        # Returned loss dict
        loss_dict = {"loss": (nll_loss + kl_loss).mean(),
                     "nll_loss": nll_loss.mean(), "kl_loss": kl_loss.mean()}

        return loss_dict

    def sample(self, x: Optional[Tensor] = None, time_steps: int = 0,
               batch_size: int = 1) -> Tuple[Tensor, ...]:
        """Reconstructs and samples observations.

        Args:
            x (torch.Tensor): Observation tensor, size `(b, l, c, h, w)`.
            time_steps (int, optional): Time step for prediction.
            batch_size (int, optional): Batch size for samping, used when `x`
                is `None`.

        Returns:
            samples (tuple of torch.Tensor): Tuple of reconstructed or sampled
                data. The first element should be reconstructed observations.
        """

        # Data size
        if x is not None:
            batch, recon_len, *_ = x.size()
        else:
            batch = batch_size
            recon_len = 0

            # Dummy input
            x = torch.rand(batch, 1, self.x_channels, 64, 64,
                           device=self.device)

        # Total sequence length (reconstruction and sample)
        seq_len = recon_len + time_steps

        if seq_len <= 0:
            raise ValueError(
                f"Sequence length must be positive, but given {seq_len}")

        # Initial parameter
        z_t = x.new_zeros((batch, self.z_dim))

        # Sampled data
        recon_list = []
        z_list = []

        for t in range(seq_len):
            # 1. Sample stochastic latents
            if t < recon_len:
                z_t_mu, z_t_var = self.encoder(x[:, t], z_t)
            else:
                z_t_mu, z_t_var = self.prior(z_t)

            z_t = z_t_mu + z_t_var ** 0.5 * torch.randn_like(z_t_var)

            # 2. Decode observations
            recon_t = self.decoder(z_t)

            # Add sampled data to list
            recon_list.append(recon_t)
            z_list.append(z_t)

        # Convert list to tensor, size (l, b, *)
        recon = torch.stack(recon_list)
        z = torch.stack(z_list)

        # Reshape: (l, b, *) -> (b, l, *)
        recon = recon.transpose(0, 1)
        z = z.transpose(0, 1)

        return recon, z