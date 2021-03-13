import json

import torch
from torch.utils.data import Dataset

from dataset_utils import generate_fake_dataset


class IndonesiaAddressDataset(Dataset):
    @classmethod
    def from_json(cls, json_path, **kwargs):
        with open(json_path) as f:
            data = json.load(f)
        return cls(data=data, **kwargs)

    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def get_train_item(self, index):
        data = self.data[index]
        return {
            "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.long),
            "scores_poi": torch.as_tensor(data["scores_poi"], dtype=torch.float32),
            "scores_street": torch.as_tensor(data["scores_street"], dtype=torch.float32),
        }

    def get_test_item(self, index):
        data = self.data[index]
        return {
            "id": data["id"],
            "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.long),
        }

    def __getitem__(self, index):
        return self.get_train_item(index) if self.train else self.get_test_item(index)


if __name__ == "__main__":
    fake_data = generate_fake_dataset()
