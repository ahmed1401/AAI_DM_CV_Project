"""
Comparaison des méthodes: Full FT, Linear Probe, Adapters, Visual Prompts.
Produit un CSV et affiche un résumé.
"""
import copy
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vit_backbone import create_vit
from models.multi_task_head import MultiTaskViT
from data.chestxray_loader import ChestXrayDataset, prepare_chest_data
from data.brats_loader import BraTSSegmentationDataset, prepare_brats_data
from data.ham10000_loader import HAMDataset, prepare_ham_data
from training.train_utils import train_model

IMG_SIZE = 224
BS = 24
EPOCHS = 15


def load_data(CHEST_DIR, BRATS_DIR, HAM_DIR):
    normalize = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        normalize
    ])
    test_tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(), normalize])
    chest_train, chest_test = prepare_chest_data(CHEST_DIR)
    brats_train, brats_test = prepare_brats_data(BRATS_DIR)
    ham_train, ham_test = prepare_ham_data(HAM_DIR)
    loaders = (
        DataLoader(ChestXrayDataset(chest_train, train_tf), batch_size=BS, shuffle=True, num_workers=2) if chest_train else None,
        DataLoader(BraTSSegmentationDataset(brats_train, IMG_SIZE, augment=True), batch_size=BS, shuffle=True, num_workers=2) if brats_train else None,
        DataLoader(HAMDataset(ham_train, train_tf), batch_size=BS, shuffle=True, num_workers=2) if ham_train else None,
    )
    test_loaders = (
        DataLoader(ChestXrayDataset(chest_test, test_tf), batch_size=BS, shuffle=False, num_workers=2) if chest_test else None,
        DataLoader(BraTSSegmentationDataset(brats_test, IMG_SIZE, augment=False), batch_size=BS, shuffle=False, num_workers=2) if brats_test else None,
        DataLoader(HAMDataset(ham_test, test_tf), batch_size=BS, shuffle=False, num_workers=2) if ham_test else None,
    )
    return loaders, test_loaders


def main(CHEST_DIR: str, BRATS_DIR: str, HAM_DIR: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    train_loaders, test_loaders = load_data(CHEST_DIR, BRATS_DIR, HAM_DIR)
    vit_base = create_vit('vit_base_patch16_224', pretrained=True)

    models = {
        'Full Fine-Tuning': MultiTaskViT(copy.deepcopy(vit_base), freeze_backbone=False, use_adapters=False, use_prompts=False, img_size=IMG_SIZE).to(device),
        'Linear Probing': MultiTaskViT(copy.deepcopy(vit_base), freeze_backbone=True, use_adapters=False, use_prompts=False, img_size=IMG_SIZE).to(device),
        'Adapter Layers': MultiTaskViT(copy.deepcopy(vit_base), freeze_backbone=True, use_adapters=True, use_prompts=False, img_size=IMG_SIZE).to(device),
        'Visual Prompt Tuning': MultiTaskViT(copy.deepcopy(vit_base), freeze_backbone=True, use_adapters=False, use_prompts=True, img_size=IMG_SIZE).to(device),
    }

    comparison_data = []
    histories = {}

    for name, model in models.items():
        print(f"\n=== Entraînement: {name} ===")
        hist, res = train_model(model, train_loaders, test_loaders, name, epochs=EPOCHS, lr=1e-4, device=device)
        histories[name] = hist
        for task, metrics in res.items():
            row = {'Method': name, 'Task': task, 'Trainable Params': model.get_trainable_params()}
            row.update(metrics)
            comparison_data.append(row)

    df = pd.DataFrame(comparison_data)
    df.to_csv('comparison_results.csv', index=False)
    print("\n📊 Résultats complets:\n", df)


if __name__ == '__main__':
    CHEST_DIR = '/kaggle/input/data'
    BRATS_DIR = '/kaggle/input/brats20-dataset-training-validation'
    HAM_DIR = '/kaggle/input/skin-cancer-mnist-ham10000'
    main(CHEST_DIR, BRATS_DIR, HAM_DIR)
