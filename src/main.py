import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel

from dataset import IndonesiaAddressDataset
from model import HamsBert

from preprocess_dataset import preprocess_dataset
from train import train
from predict import predict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    set_seed(args.seed)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset, batch_size=args.batch_size, num_workers=args.num_workers, **kwargs
        )

    if args.do_preprocess:
        print("> Preprocessing dataset")
        preprocess_dataset(args)

    if args.do_train:
        train_loader = to_dataloader(
            IndonesiaAddressDataset.from_json(
                args.dataset_dir / f"train_{args.bert_name.replace('/', '-')}.json",
            ),
            shuffle=True
        )

        val_loader = to_dataloader(
            IndonesiaAddressDataset.from_json(
                args.dataset_dir / f"val_{args.bert_name.replace('/', '-')}.json",
            )
        )

        model = HamsBert(backbone=BertModel.from_pretrained(args.bert_name))

        train(
            model,
            train_loader,
            val_loader,
            lr=args.learning_rate,
            epochs=args.epochs,
            early_stopping=args.early_stopping,
            model_path=args.checkpoint_dir / "model_best.pt",
        )

    if args.do_predict:
        model = HamsBert.from_checkpoint()

        test_loader = to_dataloader(
            IndonesiaAddressDataset.from_json(
                args.dataset_dir / f"test_{args.bert_name.replace('/', '-')}.json",
            )
        )

        predict(model, test_loader, output_csv=args.output_csv)


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
    parser.add_argument("--output_csv", type=Path, default="output.csv")

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
