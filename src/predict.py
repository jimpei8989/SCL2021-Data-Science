import json

import torch
import pandas as pd
from tqdm import tqdm
import numpy as np

from reconstruct_tokenized import reconstruct


def to_continuous(mask, single):
    """
    mask: [List] value represents probability
    """

    def find_max_sub_arr(prob):
        accumulate = [0]
        for v in prob:
            accumulate.append(accumulate[-1] + v)

        max_ans, max_rec = 0, [0, 0]
        left_min, left_min_idx = 100000, -1
        for i, v in enumerate(accumulate):
            if v <= left_min:
                left_min = v
                left_min_idx = i
            if v - left_min > max_ans:
                max_ans = v - left_min
                max_rec = [left_min_idx, i]

        return [0] * max_rec[0] + [1] * (max_rec[1] - max_rec[0]) + [0] * (len(prob) - max_rec[1])

    mask = np.array(mask) - 0.5

    if single:
        return find_max_sub_arr(mask)
    else:
        return [find_max_sub_arr(m) for m in mask]


def predict(model, tokenizer, test_dataloader, output_csv, device, output_probs_json=None):
    model.to(device)
    model.eval()

    outputs = []
    raw_outputs = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, ncols=80, desc="Predicting"):
            pred = model(batch["input_ids"].to(device)).to("cpu")

            poi_masks = to_continuous(pred[..., 0].tolist())
            street_masks = to_continuous(pred[..., 1].tolist())

            for ID, addr, poi_probs, poi_mask, street_probs, street_mask in zip(
                batch["id"],
                batch["address"],
                pred[..., 0].tolist(),
                poi_masks,
                pred[..., 1].tolist(),
                street_masks,
            ):
                tokenized = tokenizer.tokenize(addr)
                poi = reconstruct(addr, tokenized, poi_mask[1 : len(tokenized) + 1])
                street = reconstruct(addr, tokenized, street_mask[1 : len(tokenized) + 1])
                outputs.append([ID, f"{poi}/{street}"])
                raw_outputs.append(
                    {
                        "id": ID,
                        "address": addr,
                        "tokenized": tokenized,
                        "poi_probs": poi_probs[1 : len(tokenized) + 1],
                        "street_probs": street_probs[1 : len(tokenized) + 1],
                    }
                )

    pd.DataFrame(outputs, columns=["id", "POI/street"]).sort_values(by="id").to_csv(
        output_csv, index=False
    )

    if output_probs_json:
        with open(output_probs_json, "w") as f:
            json.dump(sorted(raw_outputs, key=lambda d: d["id"]), f, indent=2)
