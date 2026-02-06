
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch.optim import Adam
from sklearn.metrics import accuracy_score, roc_auc_score

from ..utils.visualize import plot_results
from ..utils.transfer_learning import freeze_backbone, unfreeze_last_backbone_layer, print_nb_trainable_params

#-------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, criterion, loader, device):
    model.eval()

    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
      for x, _, y in loader:
          x = x.to(device)
          y = y.float().to(device)

          logits = model(x)
          loss = criterion(logits, y)

          losses.append(loss.item())
          preds_all.append(torch.sigmoid(logits))
          targets_all.append(y)

      preds_all = torch.cat(preds_all).cpu()
      targets_all = torch.cat(targets_all).cpu()

      acc = accuracy_score(targets_all, preds_all > 0.5)
      auc = roc_auc_score(targets_all, preds_all)

    return np.mean(losses), acc, auc

#-------------------------------------------------------------------------
def train_model(model, train_loader, val_loader, device, 
                    epochs=30, pos_weight=None, model_name="", 
                    apply_freeze=False, frozen_epochs = 10, res_path=""):
    """
    Training loop.

    Parameters:
    - model: your PyTorch model
    - train_loader, val_loader: data loaders
    - epochs: total epochs
    - model_name: name of the model
    """
    model.to(device)

    if pos_weight is not None:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    if "transformer" in model_name.split("_"):
        pass
    else:
        if apply_freeze:
            freeze_backbone(model)
            print_nb_trainable_params(model)
            # unfreeze_last_backbone_layer(model)
            # print_nb_trainable_params(model)

    optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

    loss_list, val_loss_list = [], []
    acc_list, auc_list = [], []
    for epoch in range(epochs):

        if epoch == frozen_epochs and apply_freeze:
            if "transformer" in model_name.split("_"):
                model.unfreeze_backbone_transformer()
            else:
                unfreeze_last_backbone_layer(model)
                print_nb_trainable_params(model)

            optimizer = Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4
            )

        model.train()
        total_loss = 0

        for x, _, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            preds = model(x).squeeze()
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        train_loss = total_loss/len(train_loader)
        val_loss, acc, auc = evaluate(model, criterion, val_loader, device)
        print(f"Epoch {epoch+1}/{epochs} | Val Acc: {acc:.4f} | Val AUC: {auc:.4f}")

        loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        acc_list.append(acc)
        auc_list.append(auc)

    
    plot_results(loss_list, val_loss_list, acc_list, auc_list, 
                    model_name=model_name, res_path=res_path, save=True)