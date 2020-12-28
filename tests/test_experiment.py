import tempfile
from copy import deepcopy
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import seqlib
from seqlib.base import BaseSequentialVAE


def test_trainer_run() -> None:

    model = TempModel()
    train_data = TempDataset()
    test_data = TempDataset()

    org_params = deepcopy(model.state_dict())

    with tempfile.TemporaryDirectory() as logdir:
        trainer = seqlib.Trainer(logdir=logdir)
        trainer.run(model, train_data, test_data)

        root = trainer._logdir
        assert (root / "training.log").exists()
        assert (root / "config.json").exists()

    updated_params = model.state_dict()
    for key in updated_params:
        assert not (updated_params[key] == org_params[key]).all()


class TempModel(BaseSequentialVAE):
    def __init__(self) -> None:
        super().__init__()

        self.encoder = nn.Linear(64 * 64 * 3, 10)
        self.decoder = nn.Linear(10, 64 * 64 * 3)

    def loss_func(
        self, x: Tensor, mask: Optional[Tensor] = None, beta: float = 1.0
    ) -> Dict[str, Tensor]:

        batch, seq_len, *_ = x.size()
        loss = x.new_zeros((batch,))

        for t in range(seq_len):
            x_t = x[:, t]
            x_t = x_t.view(-1, 64 * 64 * 3)
            z_t = self.encoder(x_t)
            recon = self.decoder(z_t)

            loss += F.mse_loss(recon, x_t)

        return {"loss": loss}

    def sample(
        self, x: Optional[Tensor] = None, time_steps: int = 0, batch_size: int = 1
    ) -> Tuple[Tensor, ...]:

        seq_len = 10 + time_steps
        x = torch.rand(seq_len, batch_size, 10)

        recon_list = []
        z_list = []
        for t in range(seq_len):
            x_t = x[:, t]
            x_t = x_t.view(-1, 64 * 64 * 3)
            z_t = self.encoder(x_t)
            recon = self.decoder(z_t)

            recon_list.append(recon.view(-1, 3, 64, 64))
            z_list.append(z_t)

        recon = torch.stack(recon_list)
        z = torch.stack(z_list)

        recon = recon.transpose(0, 1)
        z = z.transpose(0, 1)

        return recon, z


class TempDataset(torch.utils.data.Dataset):
    def __init__(self) -> None:
        super().__init__()

        self._data = torch.rand(10, 8, 3, 64, 64)
        self._label = torch.randint(0, 100, (10, 8))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        return self._data[index], self._label[index]

    def __len__(self) -> int:
        return self._data.size(0)
