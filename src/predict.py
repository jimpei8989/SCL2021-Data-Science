import torch
import pandas as pd
from tqdm import tqdm

from reconstruct_tokenized import reconstruct


def ids_to_str(tokenizer, ids, must_continue=False):
    ret = ""
    for c in ids:
        if c != 0:
            token = tokenizer.convert_ids_to_tokens(int(c.item()))
            if token[0] not in [".", ","]:
                ret += " "
            ret = ret + token[2:] if token[:2] == "##" else ret + token

    return ret.strip()


def get_res(tokenizer, poi, street, must_continue=False):
    ret = [
        f"{ids_to_str(tokenizer, p, must_continue)}/{ids_to_str(tokenizer, s, must_continue)}"
        for (p, s) in zip(poi, street)
    ]
    return ret


def predict(model, tokenizer, test_dataloader, output_csv, device):
    model.to(device)
    model.eval()

    outputs = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, ncols=80, desc="Predicting"):
            pred = model(batch["input_ids"].to(device)).to("cpu")

            poi_masks = pred[..., 0].gt(0.5).tolist()
            street_masks = pred[..., 1].gt(0.5).tolist()

            for ID, addr, poi_mask, street_mask in zip(
                batch["id"], batch["address"], poi_masks, street_masks
            ):
                tokenized = tokenizer.tokenize(addr)
                poi = reconstruct(addr, tokenized, poi_mask[1: len(tokenized) + 1])
                street = reconstruct(addr, tokenized, street_mask[1: len(tokenized) + 1])
                outputs.append([ID, f"{poi}/{street}"])

    pd.DataFrame(outputs, columns=["id", "POI/street"]).sort_values(by="id").to_csv(
        output_csv, index=False
    )
