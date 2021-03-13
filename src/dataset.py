import sys
import json
from tqdm import trange

from torch.utils.data import Dataset

from dataset_utils import generate_fake_dataset
from model import get_default_model


class IndonesiaAddressDataset(Dataset):
    @classmethod
    def from_json(cls, json_path, **kwargs):
        with open(json_path) as f:
            data = json.load(f)
        return cls(data=data, **kwargs)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data

    def __getitem__(self, index):
        return self.data[index]


if __name__ == "__main__":
    fake_data = generate_fake_dataset()
    tokenizer, model = get_default_model()

    dataset = IndonesiaAddressDataset(fake_data, tokenizer=tokenizer)

    dataset.prepare(0, debug=True)
    dataset.prepare(1, debug=True)
