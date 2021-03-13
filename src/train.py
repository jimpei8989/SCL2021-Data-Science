import torch
import torch.nn as nn


def evaluate(model, loader):
    pass


def train(model, train_loader, val_loader, lr=1e-3, epochs=100, early_stopping=5, model_path=None):
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
    pass
