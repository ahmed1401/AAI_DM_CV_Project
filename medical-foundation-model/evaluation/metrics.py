"""
Métriques de classification et segmentation.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def compute_classification_metrics(all_logits, all_labels, multi_label: bool = True):
    logits = np.concatenate(all_logits, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    if multi_label:
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)
        acc = (preds == labels).mean()
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(labels, probs, average='macro')
        except Exception:
            auc = float('nan')
        return {'Accuracy': float(acc), 'F1 (macro)': float(f1), 'AUC (macro)': float(auc)}
    else:
        preds = logits.argmax(axis=1)
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average='macro', zero_division=0)
        try:
            auc = roc_auc_score(labels, logits, multi_class='ovr', average='macro')
        except Exception:
            auc = float('nan')
        return {'Accuracy': float(acc), 'F1 (macro)': float(f1), 'AUC (macro)': float(auc)}


def compute_segmentation_metrics(all_preds, all_masks):
    preds = np.concatenate(all_preds, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    # Dice global simplifié
    dice_scores = []
    for cls in [0,1,2,3]:
        p = (preds == cls).astype(int)
        m = (masks == cls).astype(int)
        inter = (p*m).sum()
        denom = p.sum() + m.sum()
        dice = (2*inter / denom) if denom>0 else 0.0
        dice_scores.append(dice)
    return {'Dice Score': float(np.mean(dice_scores))}
