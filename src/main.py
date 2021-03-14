import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import BertModel

from dataset import IndonesiaAddressDataset
from dataset_utils import create_batch
from model import HamsBert

from preprocess_dataset import preprocess_dataset
from train import train
from predict import predict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    print(args)
    set_seed(args.seed)

    def to_dataloader(dataset, **kwargs):
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            collate_fn=create_batch,
            **kwargs,
        )

    if args.do_preprocess:
        print("> Preprocessing dataset")
        preprocess_dataset(args)

    if args.do_train:
        train_loader = to_dataloader(
            IndonesiaAddressDataset.from_json(
                args.dataset_dir / f"train_{args.bert_name.replace('/', '-')}.json",
            ),
            shuffle=True,
        )

        val_loader = to_dataloader(
            IndonesiaAddressDataset.from_json(
                args.dataset_dir / f"val_{args.bert_name.replace('/', '-')}.json",
            )
        )

        model = HamsBert(backbone=BertModel.from_pretrained(args.bert_name))

        if not args.checkpoint_dir.is_dir():
            args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        train(
            model,
            train_loader,
            val_loader,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            epochs=args.epochs,
            early_stopping=args.early_stopping,
            model_path=args.checkpoint_dir / "model_best.pt",
            device=args.device,
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--early_stopping", type=int, default=5)
    parser.add_argument("--weight_decay", type=float, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default="dataset/")
    parser.add_argument("--checkpoint_dir", type=Path, default="checkpoints/default/")
    parser.add_argument("--output_csv", type=Path, default="output.csv")

    # Actions
    parser.add_argument("--do_preprocess", action="store_true")
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=0x06902024 ^ 0x06902029 ^ 0x06902066 ^ 0x06902074
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu", default='0')

    args = parser.parse_args()

    args.device = torch.device("cpu" if args.cpu else "cuda")
    if not args.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    return args


if __name__ == "__main__":
    main(parse_args())
