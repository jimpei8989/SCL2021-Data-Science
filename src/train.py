import torch
from torch.optim import Adam
import torch.nn as nn


def evaluate(model, loader):
    criterion = nn.BCELoss()

    model.eval()
    val_loss, val_acc, val_acc_sen = 0.0, 0.0, 0.0
    val_num = 0
    with torch.no_grad():
        for (x, y) in loader:
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            val_loss += loss.item() * y.shape[0]

            correct = (y == torch.round(pred)).to(torch.float)
            val_acc += correct.mean() * y.shape[0]
            val_acc_sen += correct.reshape(correct.shape[0], -1).min(dim=1)[0].mean() * y.shape[0]
            val_num += y.shape[0]

    val_loss, val_acc, val_acc_sen = val_loss / val_num, val_acc / val_num, val_acc_sen / val_num
    print(
        f"[Val] loss = {val_loss:.4f}, acc_per_token = {val_acc:.4f}, "
        f"acc_per_sentence = {val_acc_sen:.4f}"
    )
    return val_loss, val_acc, val_acc_sen


def train(model, train_loader, val_loader, lr=1e-4, epochs=100, early_stopping=5, model_path=None):
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
    optimizer = Adam(model.parameters(), lr=lr)

    model = model.cuda()

    def iterate_dataloader(dataloader, train=False):
        model.train() if train else model.eval()

        total_loss, total_token_acc, total_sentence_acc = 0, 0, 0
        with torch.set_grad_enabled(train):
            for i, batch in enumerate(dataloader):
                if train:
                    optimizer.zero_grad()

                pred = model(batch["input_ids"].cuda())

                y = torch.stack([batch["scores_poi"], batch["scores_street"]], dim=-1)
                loss = criterion(pred, y)

                # loss_poi = criterion(pred[..., 0], batch["scores_poi"].cuda())
                # loss_street = criterion(pred[..., 1], batch["scores_street"].cuda())
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
            map(lambda v: v / len(dataloader)), [total_loss, total_token_acc, total_sentence_acc]
        )

    min_val_loss, not_improved = 100000, 0
    for e in range(epochs):
        train_loss, train_token_acc, train_sentence_acc = iterate_dataloader(
            train_loader, train=True
        )
        print(
            f"Epoch {e+1}/{epochs}: [Train] loss = {train_loss:.4f}, "
            f"acc_per_token = {train_token_acc:.4f}, "
            f"acc_per_sentence = {train_sentence_acc:.4f}"
        )

        val_loss, val_token_acc, val_sentence_acc = iterate_dataloader(val_loader)
        print(
            f"[Val] loss = {val_loss:.4f}, "
            f"acc_per_token = {val_token_acc:.4f}, "
            f"acc_per_sentence = {val_sentence_acc:.4f}"
        )

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            not_improved = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            not_improved += 1
            if early_stopping > 0 and not_improved == early_stopping:
                break
