import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from dataset import IndonesiaAddressDataset
from model import get_default_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.seed)

    tokenizer, model = get_default_model()
    dataset = IndonesiaAddressDataset.from_json(args.dataset_dir / "train_dataset.json")
    dataset.set_tokenizer(tokenizer)

    print("> Dataset created")

    dataset.prepare_all_and_dump(args.dataset_dir, "train_prepared.json")

    print("> Dataset prepared")


def parse_args():
    parser = ArgumentParser()

    # Configs

    # Trainers
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default="dataset/")

    # Actions
    parser.add_argument("--do_all", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Misc
    parser.add_argument("--seed", type=int, default=0x06902029)

    args = parser.parse_args()
    if args.do_all:
        for key in args.__dict__:
            if key.startswith("do_"):
                setattr(args, key, True)
    return args


if __name__ == "__main__":
    main(parse_args())
