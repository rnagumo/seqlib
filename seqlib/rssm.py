"""Recurrent State-Space Model (RSSM).

RSSM is also known as Variational RNN.

ref)
* Chung+ 2015, "A Recurrent Latent Variable Model for Sequential Data"
"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .base import BaseSequentialVAE, kl_divergence_normal, nll_bernoulli


class Transition(nn.Module):
    """Time step transition.

    Args:
        h_dim (int): Dimension size of hidden states.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.fz = nn.Linear(z_dim, h_dim)
        self.rnn_cell = nn.GRUCell(h_dim, h_dim)

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Transition one time step.

        Args:
            h (torch.Tensor): Hidden states, size `(b, h)`.
            z (torch.Tensor): Latent states, size `(b, z)`.

        Returns:
            h (torch.Tensor): Hidden state at next time step, size `(b, h)`.
        """

        h = self.rnn_cell(self.fz(z), h)

        return h


class StochasticPrior(nn.Module):
    """Stochastic prior: p(z|h).

    Args:
        h_dim (int): Dimension size of hidden states.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)

    def forward(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode hidden state to the parameters of latent distribution.

        Args:
            h (torch.Tensor): RNN hidden state.

        Returns:
            loc (torch.Tensor): Mean of Gausssian, size `(b, z)`.
            var (torch.Tensor): Variance of Gausssian, size `(b, z)`.
        """

        loc = self.fc1(h)
        var = F.softplus(self.fc2(h))

        return loc, var


class Generator(nn.Module):
    """Generator of observations: p(x|h, z).

    Args:
        x_channels (int): Channel number of observations.
        h_dim (int): Dimension size of hidden states.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, x_channels: int, h_dim: int, z_dim: int) -> None:
        super().__init__()

        self.fz = nn.Linear(z_dim, h_dim)

        self.fc = nn.Sequential(
            nn.Linear(h_dim * 2, 256),
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

    def forward(self, h: Tensor, z: Tensor) -> Tensor:
        """Generates observations given latents and hiddens.

        Args:
            h (torch.Tensor): Hidden states, size `(b, h)`.
            z (torch.Tensor): Latent states, size `(b, z)`.

        Returns:
            x (torch.Tensor): Sampled observations, size `(b, c, h, w)`.
        """

        h = torch.cat([h, self.fz(z)], dim=1)
        h = self.fc(h)
        h = h.view(-1, 64, 4, 4)
        x = self.deconv(h)

        return x


class Inference(nn.Module):
    """Inference of latents: q(z|x, h).

    Args:
        x_channels (int): Channel number of observations.
        h_dim (int): Dimension size of hidden states.
        z_dim (int): Dimension size of latent states.
    """

    def __init__(self, x_channels: int, h_dim: int, z_dim: int) -> None:
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
            nn.Linear(256, h_dim),
        )

        self.fc1 = nn.Linear(h_dim * 2, z_dim)
        self.fc2 = nn.Linear(h_dim * 2, z_dim)

    def forward(self, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        """Variational inference of latents.

        Args:
            x (torch.Tensor): Observations, size `(b, c, h, w)`.
            h (torch.Tensor): Hidden states, size `(b, h)`.

        Returns:
            loc (torch.Tensor): Mean of Gausssian, size `(b, z)`.
            var (torch.Tensor): Variance of Gausssian, size `(b, z)`.
        """

        x = self.conv(x)
        x = x.view(-1, 1024)
        x = self.fc(x)

        h = torch.cat([x, h], dim=1)
        loc = self.fc1(h)
        var = F.softplus(self.fc2(h))

        return loc, var


class RecurrentSSM(BaseSequentialVAE):
    """Recurrent State-Space model (RSSM).

    Args:
        x_channels (int, optional): Channel number of observations.
        h_dim (int, optional): Dimension size of hidden states.
        z_dim (int, optional): Dimension size of latent states.
        beta (float, optional): Beta coefficient of KL term.
        do_anneal (bool, optional): If `True`, beta is given from kwargs.
    """

    def __init__(
        self,
        x_channels: int = 3,
        h_dim: int = 10,
        z_dim: int = 10,
        beta: float = 10.0,
        do_anneal: bool = False,
    ) -> None:
        super().__init__()

        self.beta = beta
        self.do_anneal = do_anneal

        self.transition = Transition(h_dim, z_dim)
        self.prior = StochasticPrior(h_dim, z_dim)
        self.decoder = Generator(x_channels, h_dim, z_dim)
        self.encoder = Inference(x_channels, h_dim, z_dim)

        self.h_0: Tensor
        self.z_0: Tensor
        self.register_buffer("h_0", torch.zeros(1, h_dim))
        self.register_buffer("z_0", torch.zeros(1, z_dim))

    def loss_func(
        self, x: Tensor, mask: Optional[Tensor] = None, beta: float = 1.0
    ) -> Dict[str, Tensor]:

        batch, seq_len, *_ = x.size()

        h_t = self.h_0.repeat(batch, 1)
        z_t = self.z_0.repeat(batch, 1)

        nll_loss = x.new_zeros((batch,))
        kl_loss = x.new_zeros((batch,))

        for t in range(seq_len):
            # 1. Update rnn
            h_t = self.transition(h_t, z_t)

            # 2. Sample stochastic latents
            p_z_t_mu, p_z_t_var = self.prior(h_t)
            q_z_t_mu, q_z_t_var = self.encoder(x[:, t], h_t)
            z_t = q_z_t_mu + q_z_t_var ** 0.5 * torch.randn_like(q_z_t_var)

            # 3. Decode observations
            recon = self.decoder(h_t, z_t)

            _nll_loss_t = nll_bernoulli(x[:, t], recon, reduce=False)
            _nll_loss_t = _nll_loss_t.sum(dim=[1, 2, 3])
            _kl_loss_t = kl_divergence_normal(
                q_z_t_mu, q_z_t_var, p_z_t_mu, p_z_t_var, reduce=True
            )

            if mask is not None:
                _nll_loss_t = _nll_loss_t * mask[:, t]
                _kl_loss_t = _kl_loss_t * mask[:, t]

            nll_loss += _nll_loss_t
            kl_loss += _kl_loss_t

        kl_loss *= beta if self.do_anneal else self.beta
        loss_dict = {
            "loss": (nll_loss + kl_loss).mean(),
            "nll_loss": nll_loss.mean(),
            "kl_loss": kl_loss.mean(),
        }

        return loss_dict

    def sample(
        self, x: Optional[Tensor] = None, time_steps: int = 0, batch_size: int = 1
    ) -> Tuple[Tensor, ...]:

        if x is not None:
            batch, recon_len, *_ = x.size()
        else:
            batch = batch_size
            recon_len = 0

        seq_len = recon_len + time_steps
        if seq_len <= 0:
            raise ValueError(f"Sequence length must be positive, but given {seq_len}")

        h_t = self.h_0.repeat(batch, 1)
        z_t = self.z_0.repeat(batch, 1)

        recon_list = []
        h_list = []
        z_list = []

        for t in range(seq_len):
            # 1. Update rnn
            h_t = self.transition(h_t, z_t)

            # 2. Sample stochastic latents
            if t < recon_len:
                assert x is not None
                z_t_mu, z_t_var = self.encoder(x[:, t], h_t)
            else:
                z_t_mu, z_t_var = self.prior(h_t)

            z_t = z_t_mu + z_t_var ** 0.5 * torch.randn_like(z_t_var)

            # 3. Decode observations
            recon_t = self.decoder(h_t, z_t)

            recon_list.append(recon_t)
            h_list.append(h_t)
            z_list.append(z_t)

        recon = torch.stack(recon_list)
        h = torch.stack(h_list)
        z = torch.stack(z_list)

        # Reshape: (l, b, *) -> (b, l, *)
        recon = recon.transpose(0, 1)
        h = h.transpose(0, 1)
        z = z.transpose(0, 1)

        return recon, h, z
