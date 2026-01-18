"""
Fonctions utilitaires d'entraînement et d'évaluation communes (boucles, métriques).
"""
import torch
import torch.nn as nn
from typing import Dict, Tuple

from evaluation.metrics import compute_classification_metrics, compute_segmentation_metrics


def make_loss_functions(MONAI_AVAILABLE: bool = False):
    chest_criterion = nn.BCEWithLogitsLoss()
    ham_criterion = nn.CrossEntropyLoss()
    if MONAI_AVAILABLE:
        from monai.losses import DiceCELoss
        brats_criterion = DiceCELoss(to_onehot_y=True, softmax=True, include_background=True)
    else:
        brats_criterion = nn.CrossEntropyLoss()
    return chest_criterion, ham_criterion, brats_criterion


def train_epoch(model, train_loaders, optimizer, device) -> Dict[str, float]:
    model.train()
    chest_loader, brats_loader, ham_loader = train_loaders
    n_chest = len(chest_loader) if chest_loader else 0
    n_brats = len(brats_loader) if brats_loader else 0
    n_ham = len(ham_loader) if ham_loader else 0
    max_batches = max(n_chest, n_brats, n_ham)

    chest_criterion, ham_criterion, brats_criterion = make_loss_functions()
    running_loss = {'chest': 0.0, 'brats': 0.0, 'ham': 0.0}
    count = {'chest': 0, 'brats': 0, 'ham': 0}

    chest_iter = iter(chest_loader) if chest_loader else None
    brats_iter = iter(brats_loader) if brats_loader else None
    ham_iter = iter(ham_loader) if ham_loader else None

    for b in range(max_batches):
        if chest_iter and b < n_chest:
            batch = next(chest_iter, None)
            if batch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(images, task_id=0)
                loss = chest_criterion(outputs, labels)
                loss.backward(); optimizer.step()
                running_loss['chest'] += loss.item(); count['chest'] += 1
        if brats_iter and b < n_brats:
            batch = next(brats_iter, None)
            if batch:
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                optimizer.zero_grad()
                outputs = model(images, task_id=1)
                loss = brats_criterion(outputs, masks) if masks.dim()==3 else brats_criterion(outputs, masks)
                loss.backward(); optimizer.step()
                running_loss['brats'] += loss.item(); count['brats'] += 1
        if ham_iter and b < n_ham:
            batch = next(ham_iter, None)
            if batch:
                images = batch['image'].to(device)
                labels = batch['label'].to(device)
                optimizer.zero_grad()
                outputs = model(images, task_id=2)
                loss = ham_criterion(outputs, labels)
                loss.backward(); optimizer.step()
                running_loss['ham'] += loss.item(); count['ham'] += 1

    for k in running_loss:
        running_loss[k] = running_loss[k] / max(1, count[k])
    return running_loss


def evaluate_model(model, test_loaders, device) -> Dict[str, Dict[str, float]]:
    model.eval()
    results = {}
    chest_loader, brats_loader, ham_loader = test_loaders
    # Chest
    if chest_loader:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in chest_loader:
                images = batch['image'].to(device)
                labels = batch['label'].cpu().numpy()
                logits = model(images, task_id=0).cpu().numpy()
                all_logits.append(logits); all_labels.append(labels)
        results['ChestX-ray14'] = compute_classification_metrics(all_logits, all_labels, multi_label=True)
    # BraTS
    if brats_loader:
        all_preds, all_masks = [], []
        with torch.no_grad():
            for batch in brats_loader:
                images = batch['image'].to(device)
                masks = batch['mask'].cpu().numpy()
                logits = model(images, task_id=1).cpu()
                preds = logits.argmax(dim=1).numpy()
                all_preds.append(preds); all_masks.append(masks)
        results['BraTS2020'] = compute_segmentation_metrics(all_preds, all_masks)
    # HAM
    if ham_loader:
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in ham_loader:
                images = batch['image'].to(device)
                labels = batch['label'].cpu().numpy()
                logits = model(images, task_id=2).cpu().numpy()
                all_logits.append(logits); all_labels.append(labels)
        results['HAM10000'] = compute_classification_metrics(all_logits, all_labels, multi_label=False)
    return results


def train_model(model, train_loaders, test_loaders, model_name: str, epochs: int = 15, lr: float = 1e-4, device: str = 'cuda') -> Tuple[Dict, Dict]:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    history = {'chest_loss': [], 'brats_loss': [], 'ham_loss': []}
    for epoch in range(1, epochs+1):
        losses = train_epoch(model, train_loaders, optimizer, device)
        history['chest_loss'].append(losses['chest'])
        history['brats_loss'].append(losses['brats'])
        history['ham_loss'].append(losses['ham'])
        print(f"[{model_name}] Epoch {epoch}/{epochs} - Losses: {losses}")
    results = evaluate_model(model, test_loaders, device)
    return history, results
