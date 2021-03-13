import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch

from dataset import IndonesiaAddressDataset
from preprocess_dataset import preprocess_dataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.seed)

    if args.do_preprocess:
        print("> Preprocessing dataset")
        preprocess_dataset(args)

    train_dataset = IndonesiaAddressDataset.from_json(
        args.dataset_dir / f"train_{args.bert_name.replace('/', '-')}.json",
    )

    print("> Dataset prepared")


def parse_args():
    parser = ArgumentParser()

    # Configs
    parser.add_argument("--bert_name", default="cahya/bert-base-indonesian-522M")

    # Trainers
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default="dataset/")

    # Actions
    parser.add_argument("--do_preprocess", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=0x06902024 ^ 0x06902029 ^ 0x06902066 ^ 0x06902074
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
