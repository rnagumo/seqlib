
import unittest

import torch

import seqlib


class TestRecurrentSSM(unittest.TestCase):

    def setUp(self):
        self.x_channels = 3
        self.h_dim = 5
        self.z_dim = 4
        self.model = seqlib.RecurrentSSM(
            self.x_channels, self.h_dim, self.z_dim)

    def test_loss_func(self):
        x = torch.rand(8, 6, 3, 64, 64)
        loss_dict = self.model.loss_func(x)

        self.assertIsInstance(loss_dict, dict)
        self.assertTrue(
            set(["loss", "kl_loss", "nll_loss"]) <= set(loss_dict.keys()))
        self.assertGreater(loss_dict["kl_loss"], 0)

    def test_loss_func_with_mask(self):
        x = torch.rand(8, 6, 3, 64, 64)
        mask = torch.ones(8, 6)
        mask[:, 3:] = 0
        loss_dict = self.model.loss_func(x, mask)

        self.assertIsInstance(loss_dict, dict)
        self.assertTrue(
            set(["loss", "kl_loss", "nll_loss"]) <= set(loss_dict.keys()))
        self.assertGreater(loss_dict["kl_loss"], 0)

    def test_sample_with_reconstruct(self):
        x = torch.rand(8, 6, 3, 64, 64)
        recon, h, z = self.model.sample(x)

        self.assertTupleEqual(recon.size(), x.size())
        self.assertTupleEqual(h.size(), (8, 6, self.h_dim))
        self.assertTupleEqual(z.size(), (8, 6, self.z_dim))

    def test_sample_without_reconstruct(self):
        recon, h, z = self.model.sample(batch_size=2, time_steps=4)

        self.assertTupleEqual(recon.size(), (2, 4, self.x_channels, 64, 64))
        self.assertTupleEqual(h.size(), (2, 4, self.h_dim))
        self.assertTupleEqual(z.size(), (2, 4, self.z_dim))

        with self.assertRaises(ValueError):
            _ = self.model.sample()

    def test_sample_with_reconstruct_and_predict(self):
        x = torch.rand(8, 6, 3, 64, 64)
        recon, h, z = self.model.sample(x, time_steps=4)

        self.assertTupleEqual(recon.size(), (8, 10, 3, 64, 64))
        self.assertTupleEqual(h.size(), (8, 10, self.h_dim))
        self.assertTupleEqual(z.size(), (8, 10, self.z_dim))

    def test_forward(self):
        x = torch.rand(8, 6, 3, 64, 64)
        recon, h, z = self.model.sample(x)

        self.assertTupleEqual(recon.size(), x.size())


if __name__ == "__main__":
    unittest.main()
