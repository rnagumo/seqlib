
import pytest

import torch
import seqlib


def test_loss_func() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    x = torch.rand(8, 6, 3, 64, 64)
    loss_dict = model.loss_func(x)

    assert isinstance(loss_dict, dict)
    assert set(["loss", "kl_loss", "nll_loss"]) <= set(loss_dict.keys())
    assert loss_dict["kl_loss"] > 0


def test_loss_func_with_mask() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    x = torch.rand(8, 6, 3, 64, 64)
    mask = torch.ones(8, 6)
    mask[:, 3:] = 0
    loss_dict = model.loss_func(x, mask)

    assert isinstance(loss_dict, dict)
    assert set(["loss", "kl_loss", "nll_loss"]) <= set(loss_dict.keys())
    assert loss_dict["kl_loss"] > 0


def test_sample_with_reconstruct() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    x = torch.rand(8, 6, 3, 64, 64)
    recon, z = model.sample(x)

    assert recon.size() == x.size()
    assert z.size(), (8, 6, z_dim)


def test_sample_without_reconstruct() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    recon, z = model.sample(batch_size=2, time_steps=4)

    assert recon.size() == (2, 4, x_channels, 64, 64)
    assert z.size() == (2, 4, z_dim)

    with pytest.raises(ValueError):
        _ = model.sample()


def test_sample_with_reconstruct_and_predict() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    x = torch.rand(8, 6, 3, 64, 64)
    recon, z = model.sample(x, time_steps=4)

    assert recon.size() == (8, 10, 3, 64, 64)
    assert z.size() == (8, 10, z_dim)


def test_forward() -> None:
    x_channels = 3
    z_dim = 4
    model = seqlib.DeepMarkovModel(x_channels, z_dim)

    x = torch.rand(8, 6, 3, 64, 64)
    recon = model(x)

    assert recon.size() == x.size()
