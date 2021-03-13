import json
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from dataset_utils import align_to_address


def check_dataset(args):
    train_df = pd.read_csv(args.dataset_dir / "train.csv")

    total_length = len(train_df)

    assert_split_length = 0
    for extract in train_df["POI/street"]:
        if len(extract.split("/")) != 2:
            assert_split_length += 1

    print(f"! Extract length after split is not 2 - {assert_split_length} / {total_length}")

    assert_continuous = 0
    for ID, address, extract in zip(
        train_df["id"], train_df["raw_address"], train_df["POI/street"]
    ):
        address = address.replace("sd neg ", "sd negeri ")
        address = address.replace("yaya ", "yayasan ")
        poi, street = extract.split("/")
        if not (poi in address and street in address):
            assert_continuous += 1
            # print(f"{ID} - {address} - {extract}")

    print(f"! Not continuous - {assert_continuous} / {total_length}")


def preprocess_dataset(args):
    train_df = pd.read_csv(args.dataset_dir / "train.csv")

    data = map(lambda p: p[1], train_df.iterrows())

    data = map(
        lambda row: {
            "id": row["id"],
            "address": row["raw_address"],
            "poi": row["POI/street"].split("/")[0],
            "street": row["POI/street"].split("/")[1],
        },
        data
    )

    data = map(align_to_address, data)

    with open(args.dataset_dir / "train_dataset.json", "w") as f:
        json.dump(list(data), f, indent=2)
        print(f"> data saved to {args.dataset_dir / 'train_dataset.json'}")


def main(args):
    if args.do_check:
        check_dataset(args)

    if args.do_preprocess:
        preprocess_dataset(args)


def parse_args():
    parser = ArgumentParser()

    # Filesystem
    parser.add_argument("--dataset_dir", type=Path, default="dataset/")

    # Actions
    parser.add_argument("--do_check", action="store_true")
    parser.add_argument("--do_preprocess", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
