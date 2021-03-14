from transformers import BertTokenizer, BertModel
from torch import nn
import torch

from tokenizer_utils import tokenize_addr


class HamsBert(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.fc = nn.Sequential(nn.Linear(768, 2), nn.Sigmoid())

    def from_checkpoint(path):
        model = HamsBert(backbone=BertModel.from_pretrained('cahya/bert-base-indonesian-522M'))
        model.load_state_dict(torch.load(path / "model_best.pt"))

        return model

    def forward(self, x):
        """
        Arguments
            x: torch.LongTensor, of shape (BS, sequence length)

        Returns
            y: torch.FloatTensor, of shape (BS, sequence length, 2)
        """
        x = self.backbone(input_ids=x)
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
