import torch
import numpy as np
from tqdm import tqdm

from ..utils.visualize import plot_metrics
from ..utils.metrics import classification_metrics
from ..utils.geometrics import extract_geometric_features
from ..utils.mask_utils import crop_to_mask


#-------------------------------------------------------------------------
def train_late_fusion_cls(cls_model, seg_model, train_loader, val_loader, pos_weight, 
                        device, model_name = "", num_epochs=40, res_path = ""):

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, cls_model.parameters()),
        lr=1e-5,
        weight_decay=1e-4
    )

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    acc_list, auc_list, sp_list, st_list = [], [], [], []

    for epoch in range(num_epochs):
        # ---------------- TRAIN ----------------
        cls_model.train()
        total_loss = 0

        for img, mask, label in tqdm(train_loader):
            img, label = img.to(device), label.to(device)

            with torch.no_grad():
                seg_logits = seg_model(img)
                pred_mask = torch.sigmoid(seg_logits) > 0.5

            roi_imgs = torch.stack([
                crop_to_mask(i, m) for i, m in zip(img, pred_mask)
            ]).to(device)

            geom = extract_geometric_features(pred_mask).to(device)

            logits = cls_model(img, roi_imgs, geom).squeeze()
            loss = criterion(logits, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)

        # ---------------- VALIDATION ----------------
        cls_model.eval()
        y_true, y_pred, y_prob = [], [], []
        total_loss = 0.
        with torch.no_grad():
            for img, mask, label in val_loader:
                img = img.to(device)

                pred_mask = torch.sigmoid(seg_model(img)) > 0.5
                roi_imgs = torch.stack([
                    crop_to_mask(i, m) for i, m in zip(img, pred_mask)
                ]).to(device)

                geom = extract_geometric_features(pred_mask).to(device)

                logits = cls_model(img, roi_imgs, geom)
                loss = criterion(logits, label.to(device))

                prob = torch.sigmoid(logits).cpu().numpy()
                pred = (prob > 0.5).astype(int)

                y_true.extend(label.numpy())
                y_pred.extend(pred)
                y_prob.extend(prob)

                total_loss += loss.item()

        val_loss = total_loss / len(val_loader)

        acc, f1, auc, sens, spec = classification_metrics(y_true, y_pred, y_prob)

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss:.4f}")
        print(f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"Sensitivity: {sens:.3f} | Specificity: {spec:.3f}")

        acc_list.append(acc)
        auc_list.append(auc)
        sp_list.append(float(spec))
        st_list.append(float(sens))
    
    plot_metrics(acc_list, auc_list, sp_list, st_list, 
                    model_name = model_name, res_path = res_path, save=True)