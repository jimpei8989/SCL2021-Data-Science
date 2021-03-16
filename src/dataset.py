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

    def __init__(self, data, train=True, for_pretraining=False):
        self.data = data
        self.train = train
        self.for_pretraining = for_pretraining

    def __len__(self):
        return len(self.data)

    def get_item_for_pretraining(self, index):
        data = self.data[index]
        return {
            "address": data["address"],
            "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.long),
        }

    def get_train_item(self, index):
        data = self.data[index]
        return {
            "address": data["address"],
            "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.long),
            "scores_poi": torch.as_tensor(data["scores_poi"], dtype=torch.float32),
            "scores_street": torch.as_tensor(data["scores_street"], dtype=torch.float32),
        }

    def get_test_item(self, index):
        data = self.data[index]
        return {
            "id": data["id"],
            "address": data["address"],
            "input_ids": torch.as_tensor(data["input_ids"], dtype=torch.long),
        }

    def __getitem__(self, index):
        if self.for_pretraining:
            return self.get_item_for_pretraining(index)
        elif self.train:
            return self.get_train_item(index)
        else:
            return self.get_test_item(index)


if __name__ == "__main__":
    fake_data = generate_fake_dataset()
    dataset = IndonesiaAddressDataset(
        fake_data,
        for_pretraining=True,
    )

    print(dataset[0])
