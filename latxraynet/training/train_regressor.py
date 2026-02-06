import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

from ..utils.visualize import plot_metrics
from ..utils.metrics import classification_metrics
from ..utils.geometrics import extract_geometric_features
from ..utils.mask_utils import crop_to_mask

#-------------------------------------------------------------------------
def train_reg_with_roi(cls_model, seg_model, train_loader, val_loader, pos_weight, 
                            device, num_epochs=40, res_path=""):

    # ---- OPTIMIZER (LOW LR) ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, cls_model.parameters()),
        lr=1e-5,
        weight_decay=1e-4
    )

    criterion = nn.MSELoss()  # target: adenoid-to-nasopharynx ratio

    ema_score = None
    ema_decay = 0.9
    best_ema_score = float('inf')  # lower MSE is better

    for epoch in range(num_epochs):
        cls_model.train()
        total_loss = 0.
        for img, mask, label in tqdm(train_loader):
            img, ratio = img.to(device), label.to(device)

            with torch.no_grad():
                pred_mask = torch.sigmoid(seg_model(img)) > 0.5

            # ---- ROI CROP ----
            roi_imgs = torch.stack([
                crop_to_mask(i, m) for i, m in zip(img, pred_mask)
            ]).to(device)

            geom_feats = extract_geometric_features(pred_mask).to(device)

            pred_ratio = cls_model(roi_imgs, geom_feats).squeeze()
            loss = criterion(pred_ratio, ratio)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---- VALIDATION ----
        cls_model.eval()
        val_losses = []
        y_true, y_pred = [], []

        with torch.no_grad():
            for img, mask, label in tqdm(val_loader):
                img, ratio = img.to(device), label.to(device)

                pred_mask = torch.sigmoid(seg_model(img)) > 0.5
                roi_imgs = torch.stack([
                    crop_to_mask(i, m) for i, m in zip(img, pred_mask)
                ]).to(device)

                geom_feats = extract_geometric_features(pred_mask).to(device)

                pred_ratio = cls_model(roi_imgs, geom_feats).squeeze()
                val_losses.append(criterion(pred_ratio, ratio).item())

                # ---- store for metrics ----
                y_true.extend(label.numpy())
                y_pred.extend(pred_ratio.cpu().numpy())

        # ---- compute AUC ----
        auc = roc_auc_score(y_true, y_pred)

        threshold = 0.5
        pred_label = (np.array(y_pred) > threshold).astype(int)
        accuracy = (pred_label == np.array(y_true)).mean()

        val_loss = sum(val_losses) / len(val_losses)

        # EMA smoothing
        if ema_score is None:
            ema_score = val_loss
        else:
            ema_score = ema_decay * ema_score + (1 - ema_decay) * val_loss

        print(f"Epoch {epoch:02d} | MSE: {train_loss:.4f} | Val MSE: {val_loss:.4f} | EMA Val: {ema_score:.4f}")
        print(f"           Val AUC: {auc:.4f} | Val Acc: {accuracy:.3f}")

        # Save best EMA model
        if ema_score < best_ema_score:
            best_ema_score = ema_score
            torch.save(cls_model.state_dict(), "best_late_fusion_regression.pth")