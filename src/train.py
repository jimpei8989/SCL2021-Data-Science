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

            correct = (y == (pred > 0.5)).to(torch.float)
            val_acc += correct.mean() * y.shape[0]
            val_acc_sen += correct.reshape(correct.shape[0], -1).min(dim=1)[0].mean() * y.shape[0]
            val_num += y.shape[0]

    val_loss, val_acc, val_acc_sen = val_loss / val_num, val_acc / val_num, val_acc_sen / val_num
    print(f'[Val] loss = {val_loss:.4f}, acc_per_token = {val_acc:.4f}, acc_per_sentence = {val_acc_sen:.4f}')

    model.train()
    return val_loss, val_acc, val_acc_sen


def train(model, train_loader, val_loader, lr=1e-4, epochs=100, early_stopping=5, model_path=None):
    '''
    Training function 
        model: (nn.Module) BERT-based model
        train_loader: (DataLoader) Dataloader of training set
        val_loader: (DataLoader) Dataloader of validation set
        lr: (float) learning rate
        epochs: (int) maximum epochs
        early_stopping: (int) patience of not improved in validation set, 
                              if set to -1, means not adapt early stopping
        model_path: (str) if not None, save the best model to the path
    '''
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=lr)

    model = model.cuda()
    min_val_loss, not_improved = 100000, 0
    for e in range(epochs):
        train_loss, train_acc, train_acc_sen = 0.0, 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            correct = (y == (pred > 0.5)).to(torch.float)
            batch_acc = correct.mean()
            batch_acc_sen = correct.reshape(correct.shape[0], -1).min(dim=1)[0].mean()

            print(f'Batch {i}/{len(train_loader)}: loss = {loss.item():.4f}, acc_per_token = {batch_acc:.4f}, acc_per_sentence = {batch_acc_sen:.4f}     ', end='\r')

            train_loss += loss.item()
            train_acc += batch_acc
            train_acc_sen += batch_acc_sen

        train_loss, train_acc, train_acc_sen = train_loader / len(train_loader), train_acc / len(train_loader), train_acc_sen / len(train_loader)
        print(f'Epoch {e+1}/{epochs}: [Train] loss = {train_loss:.4f}, acc_per_token = {train_acc:.4f}, acc_per_sentence = {train_acc_sen:.4f}')
        val_loss, _, _ = evaluate(model, val_loader)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            not_improved = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            not_improved += 1
            if early_stopping > 0 and not_improved == early_stopping:
                break
