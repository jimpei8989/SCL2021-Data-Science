from itertools import chain
from pathlib import Path
from typing import Iterable, Optional


def further_pretrain(
    model,
    dataloaders: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 0,
    epochs: int = 1,
    model_path: Optional[Path] = None,
    device=None
):
    model.train()

    for epoch in range(1, epochs + 1):
        for batch in chain.from_iterable(dataloaders):
            pass

    # save model to checkpoint_dir, save it as "pretrained_bert.pt"
