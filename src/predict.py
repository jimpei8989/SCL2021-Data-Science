import torch
import pandas as pd


def ids_to_str(tokenizer, ids, must_continue=False):
    ret = ''
    for c in ids:
        if c != 0:
            token = tokenizer.convert_ids_to_tokens(int(c.item()))
            if token[0] not in ['.', ',']:
                ret += ' '
            ret = ret + token[2:] if token[:2] == '##' else ret + token

    return ret.strip()


def get_res(tokenizer, poi, street, must_continue=False):
    ret = [f"{ids_to_str(tokenizer, p, must_continue)}/{ids_to_str(tokenizer, s, must_continue)}" for (p, s) in zip(poi, street)]
    return ret


def predict(model, tokenizer, test_dataloader, output_csv, device):
    model.to(device)
    model.eval()

    ids, opt = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_dataloader):
            pred = model(batch["input_ids"].to(device)).to('cpu')

            ids += batch['id']
            poi = batch["input_ids"] * torch.round(pred[:, :, 0])
            street = batch["input_ids"] * torch.round(pred[:, :, 1])
            opt += get_res(tokenizer, poi, street)

    pd.DataFrame(data=[[i, j] for i, j in zip(ids, opt)], columns=['id', 'POI/street']).to_csv(output_csv, index=False)
