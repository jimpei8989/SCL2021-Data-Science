import json
from typing import List
from tqdm import tqdm

from transformers.models.bert.tokenization_bert import BertTokenizer


def reconstruct(source: str, tokenized: List[str], mask=None):
    # assert mask is None or len(tokenized) == len(mask)

    cursor = 0
    begining_indices = []
    for token in tokenized:
        if token.startswith("##"):
            begining_indices.append(None)
        else:
            cursor = source.find(token, cursor)
            begining_indices.append(cursor)

    begining_indices.append(len(source))
    ending_indices = []  # non-inclusive index

    cursor = 0
    for i in range(len(tokenized)):
        if begining_indices[i] is None:
            ending_indices.append(None)
        else:
            cursor += 1
            while begining_indices[cursor] is None:
                cursor += 1
            ending_indices.append(begining_indices[cursor])

    if mask is None:
        mask = [True for _ in tokenized]

    char_mask = [False for _ in source]
    for i, m in enumerate(mask):
        if m and not tokenized[i].startswith("##"):
            for j in range(begining_indices[i], ending_indices[i]):
                char_mask[j] = True

    return "".join(c for c, m in zip(source, char_mask) if m).strip()


def test():
    tokenizer = BertTokenizer.from_pretrained("cahya/bert-base-indonesian-522M")
    with open("dataset/train.json") as f:
        data = json.load(f)

    for sample in tqdm(data, desc="Testing training data"):
        tokenized = tokenizer.tokenize(sample["address"])
        reconstructed = reconstruct(sample["address"], tokenized)

        if not reconstructed == sample["address"]:
            print(f"!!! {sample['address']} - {tokenized} - {reconstructed}")
            exit()


if __name__ == "__main__":
    test()