
import torch

from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    confusion_matrix, balanced_accuracy_score,
    precision_score, recall_score
)


#-------------------------------------------------------------------------
def classification_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn + 1e-5)
    specificity = tn / (tn + fp + 1e-5)

    return acc, f1, auc, sensitivity, specificity

#-------------------------------------------------------------------------
def check_metrics(loader, model, device, num_classes):
    # Prepare model for evaluation
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_scores = []
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x).softmax(-1)
            predictions = torch.argmax(scores, dim=1)
            
            # Collect all predictions and targets
            all_predictions.append(predictions)
            all_targets.append(y)
            all_scores.append(scores)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions).cpu().numpy()
    all_targets = torch.cat(all_targets).cpu().numpy()
    all_scores = torch.cat(all_scores).cpu().numpy()
    
    # Compute the final metrics using torcheval's functional metrics
    test_precision_score = precision_score(all_targets, all_predictions, average='macro')
    test_recall_score = recall_score(all_targets, all_predictions, average='macro')
    test_f1_score = f1_score(all_targets, all_predictions, average='macro')
    test_roc_auc_score = roc_auc_score(all_targets, all_scores, average='macro', multi_class='ovo')
    test_balanced_accuracy_score = balanced_accuracy_score(all_targets, all_predictions)
    test_accuracy_score = accuracy_score(all_targets, all_predictions)
    
    return (test_precision_score, test_recall_score, 
                test_f1_score, test_roc_auc_score, 
                    test_balanced_accuracy_score, test_accuracy_score)
