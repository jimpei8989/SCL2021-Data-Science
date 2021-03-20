import os
import torch
from torch.optim import Adam
import torch.nn as nn

from predict import predict
from utils.timer import timer


def evaluate(model, tokenizer, checkpoint_dir, train_loader=None, val_loader=None, device=None):
    for loader, data_type in zip([train_loader, val_loader], ["train", "val"]):
        if loader is not None:
            # predict and output
            opt_path = checkpoint_dir / f"{data_type}_opt.csv"
            predict(model, tokenizer, loader, output_csv=opt_path, device=device)

            # mapping
            opt_map_path = checkpoint_dir / f"{data_type}_map_opt.csv"
            os.system(
                f"python src/mapping.py -i {opt_path} -m dataset/train_split.csv -o {opt_map_path}"
            )

            print(f"========== Result of {data_type} set ==========")
            print("----------    Without mapping    ----------")
            os.system(
                f"python src/score.py -i dataset/train.csv -m dataset/train_split.csv -o {opt_path}"
            )
            print("----------     After mapping    ----------")
            os.system(
                f"python src/score.py -i dataset/train.csv -m dataset/train_split.csv -o {opt_map_path}"
            )


def train(
    model,
    train_loader,
    val_loader,
    lr=1e-4,
    weight_decay=0,
    epochs=100,
    early_stopping=5,
    freeze_backbone=False,
    model_path=None,
    device=None,
    magic_method=False,
):
    """
    Training function
        model: (nn.Module) BERT-based model
        train_loader: (DataLoader) Dataloader of training set
        val_loader: (DataLoader) Dataloader of validation set
        lr: (float) learning rate
        epochs: (int) maximum epochs
        early_stopping: (int) patience of not improved in validation set,
                              if set to -1, means not adapt early stopping
        model_path: (str) if not None, save the best model to the path
    """
    criterion = nn.BCELoss()
    if freeze_backbone:
        optimizer = Adam(model.fc.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.to(device)

    @timer
    def iterate_dataloader(dataloader, train=False):
        model.train() if train else model.eval()

        total_loss, total_token_acc, total_sentence_acc = 0, 0, 0
        with torch.set_grad_enabled(train):
            for i, batch in enumerate(dataloader):
                if train:
                    optimizer.zero_grad()

                pred = model(batch["input_ids"].to(device))

                # Apply mask
                pred = pred * batch["mask"].to(device).unsqueeze(-1)

                y = torch.stack([batch["scores_poi"], batch["scores_street"]], dim=-1).to(device)
                loss = criterion(pred, y.to(device))

                # loss_poi = criterion(pred[..., 0], batch["scores_poi"].to(device))
                # loss_street = criterion(pred[..., 1], batch["scores_street"].to(device))
                # loss = loss_poi + loss_street

                if train:
                    loss.backward()
                    optimizer.step()

                correct = (y == torch.round(pred)).to(torch.float)
                batch_token_acc = correct.mean()
                batch_sentence_acc = correct.reshape(correct.shape[0], -1).min(dim=1)[0].mean(0)

                total_loss += loss.item()
                total_token_acc += batch_token_acc
                total_sentence_acc += batch_sentence_acc

                if train:
                    print(
                        f"Batch {i}/{len(train_loader)}: loss = {loss.item():.4f}, "
                        f"acc_per_token = {batch_token_acc:.4f}, "
                        f"acc_per_sentence = {batch_sentence_acc:.4f}",
                        end="\r",
                    )

        return tuple(
            map(lambda v: v / len(dataloader), [total_loss, total_token_acc, total_sentence_acc])
        )

    min_val_loss, not_improved = 100000, 0
    for e in range(epochs):
        if magic_method:
            model.freeze(freeze_nums=4 + e)
        print(f"Epoch {e+1}/{epochs}")
        train_time, (train_loss, train_token_acc, train_sentence_acc) = iterate_dataloader(
            train_loader, train=True
        )
        print(
            f"| [Train] time: {train_time:7.3f}s - ",
            f"loss = {train_loss:.4f}, "
            f"acc_per_token = {train_token_acc:.4f}, "
            f"acc_per_sentence = {train_sentence_acc:.4f}"
        )

        val_time, (val_loss, val_token_acc, val_sentence_acc) = iterate_dataloader(val_loader)
        print(
            f"\\ [Val]   time: {val_time:7.3f}s - "
            f"loss = {val_loss:.4f}, "
            f"acc_per_token = {val_token_acc:.4f}, "
            f"acc_per_sentence = {val_sentence_acc:.4f}"
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            not_improved = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
                print(f"Improved! Save to {model_path} ... ")
        else:
            not_improved += 1
            if early_stopping > 0 and not_improved == early_stopping:
                break
