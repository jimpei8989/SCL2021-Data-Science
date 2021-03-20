import json
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from reconstruct_tokenized import reconstruct
from predict import to_continuous


def reduce(probs: List[List[float]], method="average"):
    if method == "average":
        return np.mean(probs, axis=0)
    elif method == "vote":
        return np.mean(np.round(probs), axis=0)
    else:
        raise ValueError(f"Invalid method {method}")


def main(args):
    raw_outputs = []
    for ff in args.output_prob_jsons:
        with open(ff) as f:
            raw_outputs.append(json.load(f))

    print(raw_outputs[0][0].keys())

    assert all(len(raw_outputs[0]) == len(ro) for ro in raw_outputs[1:])

    assert all(
        all(
            (
                raw_outputs[0][i]["tokenized"] == ro[i]["tokenized"]
                and raw_outputs[0][i]["address"] == ro[i]["address"]
            )
            for ro in raw_outputs[1:]
        )
        for i in range(len(raw_outputs[0]))
    )

    outputs = []
    for i, sample in enumerate(tqdm(raw_outputs[0])):
        scores_poi = to_continuous(reduce([ro[i]["scores_poi"] for ro in raw_outputs]))
        scores_street = to_continuous(reduce([ro[i]["scores_street"] for ro in raw_outputs]))

        poi = reconstruct(sample["address"], sample["tokenized"], scores_poi)
        street = reconstruct(sample["address"], sample["tokenized"], scores_street)

        outputs.append([sample["id"], f"{poi}/{street}"])

    pd.DataFrame(outputs, columns=["id", "POI/street"]).sort_values(by="id").to_csv(
        args.output_csv, index=False
    )


def parse_arguments():
    parser = ArgumentParser()

    parser.add_argument("output_prob_jsons", nargs="+")

    parser.add_argument("--method", choices=["average", "vote"], default="average")
    parser.add_argument("--output_csv", type=Path, default="ensemble.csv")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_arguments())
