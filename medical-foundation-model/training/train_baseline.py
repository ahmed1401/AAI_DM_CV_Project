"""
Entraînement Baseline: Full Fine-Tuning (15 epochs)
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from models.vit_backbone import create_vit, freeze_vit
from models.multi_task_head import MultiTaskViT
from data.chestxray_loader import ChestXrayDataset, prepare_chest_data
from data.brats_loader import BraTSSegmentationDataset, prepare_brats_data
from data.ham10000_loader import HAMDataset, prepare_ham_data
from training.train_utils import train_model

SEED = 42
IMG_SIZE = 224
BS = 24
EPOCHS = 15


def main(CHEST_DIR: str, BRATS_DIR: str, HAM_DIR: str, device: str = None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Device: {device}")

    # Transforms ImageNet
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

    # Données
    chest_train, chest_test = prepare_chest_data(CHEST_DIR)
    brats_train, brats_test = prepare_brats_data(BRATS_DIR)
    ham_train, ham_test = prepare_ham_data(HAM_DIR)

    chest_train_ds = ChestXrayDataset(chest_train, train_tf) if chest_train else None
    chest_test_ds = ChestXrayDataset(chest_test, test_tf) if chest_test else None
    brats_train_ds = BraTSSegmentationDataset(brats_train, IMG_SIZE, augment=True) if brats_train else None
    brats_test_ds = BraTSSegmentationDataset(brats_test, IMG_SIZE, augment=False) if brats_test else None
    ham_train_ds = HAMDataset(ham_train, train_tf) if ham_train else None
    ham_test_ds = HAMDataset(ham_test, test_tf) if ham_test else None

    chest_train_loader = DataLoader(chest_train_ds, batch_size=BS, shuffle=True, num_workers=2) if chest_train_ds else None
    chest_test_loader = DataLoader(chest_test_ds, batch_size=BS, shuffle=False, num_workers=2) if chest_test_ds else None
    brats_train_loader = DataLoader(brats_train_ds, batch_size=BS, shuffle=True, num_workers=2) if brats_train_ds else None
    brats_test_loader = DataLoader(brats_test_ds, batch_size=BS, shuffle=False, num_workers=2) if brats_test_ds else None
    ham_train_loader = DataLoader(ham_train_ds, batch_size=BS, shuffle=True, num_workers=2) if ham_train_ds else None
    ham_test_loader = DataLoader(ham_test_ds, batch_size=BS, shuffle=False, num_workers=2) if ham_test_ds else None

    vit = create_vit('vit_base_patch16_224', pretrained=True)
    model = MultiTaskViT(vit, freeze_backbone=False, use_adapters=False, use_prompts=False, img_size=IMG_SIZE).to(device)

    history, results = train_model(model, (chest_train_loader, brats_train_loader, ham_train_loader), (chest_test_loader, brats_test_loader, ham_test_loader), 'Full Fine-Tuning', epochs=EPOCHS, lr=1e-4, device=device)
    print("\n📊 Résultats Baseline:", results)


if __name__ == '__main__':
    # Chemins par défaut Kaggle (adapter selon votre environnement)
    CHEST_DIR = '/kaggle/input/data'
    BRATS_DIR = '/kaggle/input/brats20-dataset-training-validation'
    HAM_DIR = '/kaggle/input/skin-cancer-mnist-ham10000'
    main(CHEST_DIR, BRATS_DIR, HAM_DIR)
