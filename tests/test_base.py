import torch

import seqlib


def test_kl_batch() -> None:
    batch = 10
    x_dim = 5

    mu0 = torch.randn(batch, x_dim)
    var0 = torch.rand(batch, x_dim) + 0.01
    mu1 = torch.randn(batch, x_dim)
    var1 = torch.rand(batch, x_dim) + 0.01

    kl = seqlib.kl_divergence_normal(mu0, var0, mu1, var1)
    assert kl.size() == (batch,)
    assert (kl >= 0).all()


def test_kl_batch_num() -> None:
    batch = 10
    num_points = 8
    x_dim = 5

    mu0 = torch.randn(batch, num_points, x_dim)
    var0 = torch.rand(batch, num_points, x_dim) + 0.01
    mu1 = torch.randn(batch, num_points, x_dim)
    var1 = torch.rand(batch, num_points, x_dim) + 0.01

    kl = seqlib.kl_divergence_normal(mu0, var0, mu1, var1)
    assert kl.size() == (batch, num_points)
    assert (kl >= 0).all()


def test_kl_same() -> None:
    batch = 10
    x_dim = 5

    mu0 = torch.randn(batch, x_dim)
    var0 = torch.rand(batch, x_dim) + 0.01

    kl = seqlib.kl_divergence_normal(mu0, var0, mu0, var0)
    assert kl.size() == (batch,)
    assert (kl >= 0).all()


def test_nll_bernoulli() -> None:
    batch = 10
    x_dim = 5

    x = torch.rand(batch, x_dim)
    probs = torch.rand(batch, x_dim)

    nll = seqlib.nll_bernoulli(x, probs)

    assert nll.size() == (batch,)
    assert (nll >= 0).all()
