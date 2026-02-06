
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

from ..utils.visualize import plot_metrics
from ..utils.metrics import classification_metrics
from ..utils.geometrics import extract_geometric_features


#-------------------------------------------------------------------------
def train_cls(seg_model, clf_model, train_loader, val_loader, pos_weight, 
                    device, model_name, num_epochs=40, res_path=""):
    
    optimizer = torch.optim.Adam(clf_model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_auc = 0
    acc_list, auc_list, sp_list, st_list = [], [], [], []
    for epoch in range(num_epochs):
        # ----- TRAIN -----
        clf_model.train()
        train_loss = 0.
        for img, mask, label in tqdm(train_loader):
            img, mask, label = img.to(device), mask.to(device), label.to(device)

            with torch.no_grad():
                pred_mask = torch.sigmoid(seg_model(img)) > 0.5

            geom = extract_geometric_features(pred_mask).to(device)
            logits = clf_model(img, geom).squeeze()

            loss = criterion(logits.view(-1), label.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # ----- VALIDATE -----
        clf_model.eval()
        y_true, y_pred, y_prob = [], [], []
        
        with torch.no_grad():
            for img, mask, label in val_loader:
                img, mask = img.to(device), mask.to(device)
                label = label.cpu().numpy()

                pred_mask = torch.sigmoid(seg_model(img)) > 0.5
                geom = extract_geometric_features(pred_mask).to(device)

                prob = torch.sigmoid(clf_model(img, geom)).cpu().numpy()
                # pred = (prob > 0.5).astype(int)

                y_true.extend(label)
                # y_pred.extend(pred)
                y_prob.extend(prob)

        # ---- THRESHOLD SELECTION (OUTSIDE LOOP) ----
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        youden_index = tpr - fpr
        best_threshold = thresholds[youden_index.argmax()]

        # ---- FINAL METRICS ----
        y_pred = (np.array(y_prob) > best_threshold).astype(int)

        acc, f1, auc, sens, spec = classification_metrics(
            y_true, y_pred, y_prob
        )

        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {train_loss/len(train_loader):.4f}")
        print(f"thresh: {best_threshold}")
        print(f"Acc: {acc:.3f} | F1: {f1:.3f} | AUC: {auc:.3f}")
        print(f"Sensitivity: {sens:.3f} | Specificity: {spec:.3f}")

        acc_list.append(acc)
        auc_list.append(auc)
        sp_list.append(float(spec))
        st_list.append(float(sens))

        if auc > best_auc:
            best_auc = auc
            torch.save({
                "model": clf_model.state_dict(),
                "threshold": best_threshold
            }, "best_classifier_with_threshold.pth")

    plot_metrics(acc_list, auc_list, sp_list, st_list, 
                    model_name = model_name, res_path = res_path, save=True)
