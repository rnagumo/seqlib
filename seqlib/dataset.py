
"""Dataset class for sequential MNIST.

Ref)
https://github.com/davidtellez/contrastive-predictive-coding/blob/master/data_utils.py
"""

from typing import List, Tuple

from PIL import Image

import torch
from torch import Tensor
from torchvision import datasets, transforms


class SequentialMNIST(datasets.MNIST):
    """Sequential MNIST dataset.

    This dataset contains 3-channels MNIST images.

    Args:
        data_num (int): Number of sequences.
        seq_len (int): Length of each sequence.
        color (bool, optional): If `True`, coloring pixels.
        path_lena (str, optional): Path to Lena image used for coloring.

    Attributes:
        indices (torch.Tensor): Indices for sequences.
    """

    def __init__(self, data_num: int, seq_len: int, color: bool = False,
                 path_lena: str = "", **kwargs):
        super().__init__(**kwargs)

        self.data_num = data_num
        self.seq_len = seq_len
        self.color = color

        # Preprocess all MNIST data
        self._preprocess_data(path_lena)

        # Generate indices for sequences
        self.indices = torch.tensor(
            [self._generate_indices() for _ in range(data_num)])

    def _preprocess_data(self, path_lena: str) -> None:
        """Initialize data.

        * Convert images to tensor.
        * Modify color distribution.

        Args:
            path_lena (str): Path to lena image used for coloring.
        """

        # Transform for MNIST image
        _transform = transforms.Compose(
            [transforms.Resize(64), transforms.ToTensor()])

        # Transform for Lena image
        _transform_lena = transforms.Compose(
            [transforms.RandomCrop(64), transforms.ToTensor()])

        # Load lena image if necessary
        if self.color:
            lena_image = Image.open(path_lena).convert("RGB")

        # Convert images to tensor
        data_list = []
        for img in self.data:
            # Image to tensor
            img = Image.fromarray(img.numpy(), mode="L")
            img = _transform(img)

            # Convert channel dim to RGB: (3, h, w)
            img = img.repeat(3, 1, 1)

            # Modify color distribution of images
            if self.color:
                # Binarize image
                img[img >= 0.5] = 1.0
                img[img < 0.5] = 0.0

                # Image to tensor
                color_img = _transform_lena(lena_image)

                # Randomly alter color distribution
                color_img = (color_img + torch.rand(3, 1, 1)) / 2

                # Invert color of pixels at number
                color_img[img == 1] = 1 - color_img[img == 1]
                img = color_img

            # Add to data list
            data_list.append(img)

        # Conver list to tensor: (b, h, w)
        self.data = torch.stack(data_list)

    def _generate_indices(self) -> List[int]:
        """Generates indices for 1 sequence.

        Returns:
            indices (list of int): Indices of images.
        """

        # Current number
        n = torch.randint(0, 10, (1,)).item()

        indices = []
        for _ in range(self.seq_len):
            # Sample random index of specified number
            t, *_ = torch.where(self.targets == n)
            idx = t[torch.multinomial(t.float(), 1)].item()
            indices.append(idx)

            # Go to next number
            n = (n + 1) % 10

        return indices

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        """Get sequence.

        Args:
            index (int): Index of data from batch.

        Returns:
            img (torch.Tensor): Sequence of images, size `(l, c, h, w)`.
            target (torch.Tensor): Sequence of targets, size `(l,)`.
        """

        img = self.data[self.indices[index]]
        target = self.targets[self.indices[index]]

        return img, target

    def __len__(self) -> int:
        """Number of batch.

        Returns:
            data_num (int): Number of sequences.
        """

        return self.data_num
