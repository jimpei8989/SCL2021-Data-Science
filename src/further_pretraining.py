from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
from torch.optim import Adam

from utils.timer import Timer


def mask_tokens(inputs, mask=None, mask_id=4, mlm_probability=0.15):
    labels = inputs.clone()

    probability_matrix = torch.full(labels.shape, mlm_probability)
    probability_matrix.masked_fill_(labels.eq(0), 0)

    if mask is not None:
        probability_matrix.masked_fill_(mask.eq(0), 0)

    masked_indices = torch.bernoulli(probability_matrix).bool()

    inputs[masked_indices] = mask_id
    # change the unmasked label to -100 to get rid of loss calculation
    labels[~masked_indices] = -100

    return inputs, labels


def further_pretrain(
    bert,
    dataloader: Iterable,
    lr: float = 1e-3,
    weight_decay: float = 0,
    epochs: int = 1,
    bert_save_dir: Optional[Path] = None,
    bert_checkpoint_dir: Optional[Path] = None,
    device=None,
):
    bert.train()
    bert.to(device)

    optimizer = Adam(bert.parameters(), lr=lr, weight_decay=weight_decay)

    min_loss = 10000
    for epoch in range(1, epochs + 1):
        with Timer(verbose=False) as et:
            print(f"Pretraining epoch {epoch}/{epochs}")
            epoch_losses = []
            for i, batch in enumerate(dataloader):
                optimizer.zero_grad()

                input_ids, labels = mask_tokens(batch["input_ids"], mask=batch["mask"])

                bert_output = bert(input_ids.to(device), labels=labels.to(device))

                loss = bert_output.loss
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.item())
                print(f"Batch {i}/{len(dataloader)} - loss = {loss.item():.4f}", end="\r")

            print(f"\\ [Train] time: {et.get_time():7.3f}s - loss = {np.mean(epoch_losses):.4f}")

        # save model to checkpoint_dir, save it as "pretrained_bert.pt"
        if np.mean(epoch_losses) < min_loss:
            print(f"Improve from {min_loss}, Save checkpoint to {bert_save_dir}")
            min_loss = np.mean(epoch_losses)
            bert.bert.save_pretrained(bert_save_dir)

        if epoch % 5 == 0:
            bert.bert.save_pretrained(bert_checkpoint_dir / f"epoch_{epoch:02d}")
