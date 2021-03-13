import sys
import json
from tqdm import trange

from torch.utils.data import Dataset

from dataset_utils import align_tokenized, generate_fake_dataset
from model import get_default_model
from tokenizer_utils import tokenize_addr


class IndonesiaAddressDataset(Dataset):
    train_keys = ("id", "address", "poi", "street", "poi_index", "street_index")
    test_keys = ("id", "address")

    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            data = json.load(f)
        return cls(data=data)

    def __init__(self, data):
        self.data = data
        self.tokenizer = None

    def __len__(self):
        return self.data

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer

    def tokenize(self, text):
        return tokenize_addr(self.tokenizer, text)

    def prepare_all_and_dump(self, dump_json_path=None):
        for i in trange(len(self.data), desc="prepare dataset", leave=False, ncols=80):
            self.prepare(i)

        if dump_json_path:
            with open(dump_json_path, "w") as f:
                json.dump(self.data, indent=2)

    def prepare(self, index, debug=False):
        data = self.data[index]
        tokenized_address = ["[CLS]"] + self.tokenizer.tokenize(data["address"]) + ["[SEP]"]
        tokenized_poi = self.tokenizer.tokenize(data["poi"])
        tokenized_street = self.tokenizer.tokenize(data["street"])

        poi_begin_index, poi_end_index = align_tokenized(tokenized_address, tokenized_poi)
        street_begin_index, street_end_index = align_tokenized(tokenized_address, tokenized_street)

        if debug:
            print(tokenized_address, tokenized_poi, tokenized_street, file=sys.stderr)
            print(poi_begin_index, poi_end_index)
            print(street_begin_index, street_end_index)

        self.data[index] = data | {
            "prepared": True,
            "input_ids": self.tokenizer.convert_tokens_to_ids(tokenized_address),
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
            print(self.data[index])

    def __getitem__(self, index):
        if not self.data[index].get("prepared", False):
            self.prepare(index)
        return self.data[index]


if __name__ == "__main__":
    fake_data = generate_fake_dataset()
    tokenizer, model = get_default_model()

    dataset = IndonesiaAddressDataset(fake_data)
    dataset.set_tokenizer(tokenizer)

    dataset.prepare(0, debug=True)
    dataset.prepare(1, debug=True)
