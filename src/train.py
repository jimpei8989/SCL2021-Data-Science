import torch
from torch.optim import Adam
import torch.nn as nn


def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        for (x, y) in loader:
            x, y = x.cuda(), y.cuda()
            y_pred = model(x)

    model.train()


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
        train_loss, train_acc = 0.0, 0.0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        print(f'...TODO...')
        val_loss, val_acc = evaluate(model, val_loader)

        if val_loss < min_val_loss:
            min_val_loss = val_loss
            not_improved = 0
            if model_path is not None:
                torch.save(model.state_dict(), model_path)
        else:
            not_improved += 1
            if early_stopping > 0 and not_improved == early_stopping:
                break
