import json

from torch.utils.data import Dataset


class IndonesiaAddressDataset(Dataset):
    @classmethod
    def from_json(cls, json_path):
        with open(json_path) as f:
            data = json.load(f)
        return cls(data=data)

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return self.data
