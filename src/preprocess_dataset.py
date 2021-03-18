import sys
import json
from tqdm import tqdm

import pandas as pd
from transformers.models.bert.tokenization_bert import BertTokenizer
from sklearn.model_selection import train_test_split

from dataset_utils import align_tokenized, generate_fake_dataset


def prepare(data, tokenizer, train=True, debug=False):
    tokenized_address = ["[CLS]"] + tokenizer.tokenize(data["address"]) + ["[SEP]"]
    data |= {"input_ids": tokenizer.convert_tokens_to_ids(tokenized_address)}

    if train:
        tokenized_poi = tokenizer.tokenize(data["poi"])
        tokenized_street = tokenizer.tokenize(data["street"])

        poi_begin_index, poi_end_index = align_tokenized(tokenized_address, tokenized_poi)
        street_begin_index, street_end_index = align_tokenized(tokenized_address, tokenized_street)

        if debug:
            print(tokenized_address, tokenized_poi, tokenized_street, file=sys.stderr)
            print(poi_begin_index, poi_end_index)
            print(street_begin_index, street_end_index)

        data |= {
            "scores_poi": [
                1 if poi_begin_index <= i <= poi_end_index else 0
                for i in range(len(tokenized_address))
            ],
            "scores_street": [
                1 if street_begin_index <= i <= street_end_index else 0
                for i in range(len(tokenized_address))
            ],
        }

    if debug:
        print(data)

    return data


def preprocess_dataset(args):
    tokenizer = BertTokenizer.from_pretrained(args.bert_name)

    # Handle training data
    train_df = pd.read_csv(args.dataset_dir / f"{args.file_prefix}train.csv")

    data = [
        {
            "id": row["id"],
            "address": row["raw_address"],
            "poi": row["POI/street"].split("/")[0],
            "street": row["POI/street"].split("/")[1],
        }
        for _, row in tqdm(
            train_df.iterrows(), desc="Iterating train csv", total=len(train_df), ncols=80
        )
    ]

    dump_json = args.dataset_dir / f"{args.file_prefix}train.json"
    with open(dump_json, "w") as f:
        json.dump(data, f, indent=2)
        print(f"> Raw Dataset saved to {dump_json}")

    tokenized_data = [
        prepare(d, tokenizer) for d in tqdm(data, desc="Preparing train dataset", ncols=80)
    ]

    train_data, val_data, train_df, valid_df = train_test_split(
        tokenized_data, train_df, test_size=0.2, random_state=args.seed
    )
    train_df.sort_values(by="id").to_csv(
        args.dataset_dir / f"{args.file_prefix}train_split.csv", index=False
    )
    valid_df.sort_values(by="id").to_csv(
        args.dataset_dir / f"{args.file_prefix}valid_split.csv", index=False
    )

    train_dataset_json = (
        args.dataset_dir / f"{args.file_prefix}train_{args.bert_name.replace('/', '-')}.json"
    )
    with open(train_dataset_json, "w") as f:
        print(f"> Tokenized train dataset saved to {train_dataset_json}")
        json.dump(train_data, f, indent=2)

    val_dataset_json = (
        args.dataset_dir / f"{args.file_prefix}val_{args.bert_name.replace('/', '-')}.json"
    )
    with open(val_dataset_json, "w") as f:
        print(f"> Tokenized validation dataset saved to {val_dataset_json}")
        json.dump(val_data, f, indent=2)

    # Handle testing data
    test_df = pd.read_csv(args.dataset_dir / f"{args.file_prefix}test.csv")

    data = [
        {
            "id": row["id"],
            "address": row["raw_address"],
        }
        for _, row in tqdm(
            test_df.iterrows(), desc="Iterating test csv", total=len(test_df), ncols=80
        )
    ]

    dump_json = args.dataset_dir / f"{args.file_prefix}test.json"
    with open(dump_json, "w") as f:
        json.dump(data, f, indent=2)
        print(f"> Raw Dataset saved to {dump_json}")

    tokenized_data = [
        prepare(d, tokenizer, train=False)
        for d in tqdm(data, desc="Preparing test dataset", ncols=80)
    ]

    test_dataset_json = (
        args.dataset_dir / f"{args.file_prefix}test_{args.bert_name.replace('/', '-')}.json"
    )
    with open(test_dataset_json, "w") as f:
        json.dump(tokenized_data, f, indent=2)


if __name__ == "__main__":
    fake_data = generate_fake_dataset()
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")

    fake_data = [prepare(d, tokenizer, debug=True) for d in fake_data]
