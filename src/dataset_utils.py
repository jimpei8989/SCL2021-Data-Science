from typing import List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence


def generate_fake_dataset():
    return [
        {
            "id": 10,
            "address": "cikahuripan sd neg boj 02 klap boj, no 5 16877",
            "poi": "sd negeri bojong 02",
            "street": "klap boj",
            "poi_index": 1,
            "street_index": 5,
        },
        {
            "id": 299979,
            "address": "graha tirta,tirta dahlia no.5,waru,sidoarjo",
            "poi": "graha tirta",
            "street": "",
            "poi_index": False,
            "street_index": -1,
        },
    ]


def _find_matching(address, target):
    if target == "":
        return -1

    address = address.replace(",", "").split()
    target = target.split()

    for i in range(len(address) - len(target) + 1):
        if all(b.startswith(a) for a, b in zip(address[i : i + len(target)], target)):
            return i
    return False


def align_to_address(data):
    return data | {
        "poi_index": _find_matching(data["address"], data["poi"]),
        "street_index": _find_matching(data["address"], data["street"]),
    }


def align_tokenized(address: List[str], target: List[str]) -> Tuple[int, int]:
    """
    Align two tokenized text (List[str]) by returning the begin index (inclusive) and end index
    (inclusive). If target is an empty string, the return values are two -1, if not found, both
    are -2.
    """
    if len(target) == 0:
        return -1, -1
    else:
        for i in range(len(address) - len(target) + 1):
            if address[i].startswith("##"):
                continue

            index_a, index_t = i, 0
            while index_a < len(address) and index_t < len(target):
                if address[index_a].startswith("##"):
                    index_a += 1
                elif target[index_t].startswith("##"):
                    index_t += 1
                elif target[index_t].startswith(address[index_a]):
                    index_a += 1
                    index_t += 1
                else:
                    break
            else:
                if index_t == len(target):
                    return i, index_a - 1
        return -2, -2


def create_batch(samples):
    ret = {
        k: [s[k] for s in samples]
        if not isinstance(samples[0][k], torch.Tensor)
        else pad_sequence([s[k] for s in samples], batch_first=True)
        for k in samples[0]
    }

    # Add mask for training
    if "scores_poi" in ret:
        ret |= {
            "mask": pad_sequence(
                [torch.ones(s["scores_poi"].shape[0]) for s in samples], batch_first=True
            )
        }

    return ret
