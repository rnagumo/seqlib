
import unittest

import seqlib


class TestSequentialMNIST(unittest.TestCase):

    def test_dataset(self):
        data_num = 4
        seq_len = 10
        dataset = seqlib.SequentialMNIST(
            data_num=data_num, seq_len=seq_len,
            root="../data/mnist", train=True, download=False)

        # Getitem
        data, target = dataset[[0, 1]]
        self.assertTupleEqual(data.size(), (2, 10, 3, 64, 64))
        self.assertTrue((0 <= data).all() and (data <= 1).all())

        self.assertTupleEqual(target.size(), (2, 10))

        # Length
        self.assertEqual(len(dataset), 4)

    def test_dataset_colored(self):
        data_num = 4
        seq_len = 10
        dataset = seqlib.SequentialMNIST(
            data_num=data_num, seq_len=seq_len, color=True,
            root="../data/mnist", train=True, download=False)

        # Getitem
        data, target = dataset[[0, 1]]
        self.assertTupleEqual(data.size(), (2, 10, 3, 64, 64))
        self.assertTrue((0 <= data).all() and (data <= 1).all())

        self.assertTupleEqual(target.size(), (2, 10))

        # Length
        self.assertEqual(len(dataset), 4)


if __name__ == "__main__":
    unittest.main()
