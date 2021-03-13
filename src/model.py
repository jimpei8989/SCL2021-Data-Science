from transformers import BertTokenizer, BertModel
import torch.nn as nn

from tokenizer_utils import tokenize_addr


class HamsBert(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(nn.Linear(768, 2), nn.Sigmoid())

    def forward(self, x):
        x = self.backbone(**x)
        x = self.fc(x["last_hidden_state"])
        return x


def get_default_model(model_name="cahya/bert-base-indonesian-522M"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    backbone = BertModel.from_pretrained(model_name)
    model = HamsBert(backbone)
    return tokenizer, model


if __name__ == "__main__":
    tokenizer, model = get_default_model()

    text = "graha tirta,tirta dahlia no.5,waru,sidoarjo"
    encoded_input, tokened_res = tokenize_addr(tokenizer, text)
    print(encoded_input, "\n", tokened_res)

    opt = model(encoded_input)
