import argparse
import json
import os
import pathlib
import random

import torch

import seqlib


def main() -> None:

    args = init_args()

    config_path = pathlib.Path(os.getenv("CONFIG_PATH", "./examples/config.json"))
    with config_path.open() as f:
        config = json.load(f)

    logdir = str(pathlib.Path(os.getenv("LOGDIR", "./logs/"), os.getenv("EXPERIMENT_NAME", "tmp")))
    dataset_name = os.getenv("DATASET_NAME", "mnist")
    data_dir = pathlib.Path(os.getenv("DATASET_DIR", "./data/"), dataset_name)

    params = vars(args)
    args_seed = params.pop("seed")
    args_cuda = params.pop("cuda")
    args_model = params.pop("model")

    torch.manual_seed(args_seed)
    random.seed(args_seed)

    use_cuda = torch.cuda.is_available() and args_cuda != "null"
    gpus = args_cuda if use_cuda else ""

    params.update(
        {
            "logdir": str(logdir),
            "gpus": gpus,
        }
    )

    model_dict = {
        "rssm": seqlib.RecurrentSSM,
        "dmm": seqlib.DeepMarkovModel,
    }
    model = model_dict[args_model](**config[f"{args_model}_params"])

    train_data = seqlib.SequentialMNIST(
        root=data_dir, train=True, download=True, **config["dataset_params"]
    )
    test_data = seqlib.SequentialMNIST(
        root=data_dir, train=False, download=True, **config["dataset_params"]
    )

    trainer = seqlib.Trainer(**params)
    trainer.run(model, train_data, test_data)


def init_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ML training")
    parser.add_argument("--cuda", type=str, default="0", help="CUDA devices by comma separation.")
    parser.add_argument("--model", type=str, default="rssm", help="Model name.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size.")
    parser.add_argument("--max-steps", type=int, default=2, help="Number of gradient steps.")
    parser.add_argument("--max-grad-value", type=float, default=5.0, help="Clipping value.")
    parser.add_argument("--max-grad-norm", type=float, default=100.0, help="Clipping norm.")
    parser.add_argument("--test-interval", type=int, default=2, help="Interval steps for testing.")
    parser.add_argument("--save-interval", type=int, default=2, help="Interval steps for saving.")

    return parser.parse_args()


if __name__ == "__main__":
    main()
