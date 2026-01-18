"""
Chargement du dataset ChestX-ray14 (classification multi-label)
Adapté du notebook 15-epochs.
"""
import os
import random
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

# Définition des classes de maladies thoraciques
CHEST_DISEASES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
    'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax'
]
NUM_CHEST_CLASSES = len(CHEST_DISEASES)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def find_chest_csv(chest_dir: str) -> str:
    """Recherche du fichier CSV des annotations dans les sous-dossiers."""
    for root, _, files in os.walk(chest_dir):
        for f in files:
            if 'Data_Entry' in f and f.endswith('.csv'):
                return os.path.join(root, f)
    return ''


def find_image_dirs(chest_dir: str) -> List[str]:
    """Recherche des répertoires contenant des images (.png)."""
    img_dirs = []
    for root, _, files in os.walk(chest_dir):
        if any(f.endswith('.png') for f in files):
            img_dirs.append(root)
    # Cas Kaggle: images_/images
    if not img_dirs:
        for d in os.listdir(chest_dir):
            if d.startswith('images_'):
                subpath = os.path.join(chest_dir, d, 'images')
                img_dirs.append(subpath if os.path.exists(subpath) else os.path.join(chest_dir, d))
    return img_dirs


class ChestXrayDataset(Dataset):
    """
    Dataset PyTorch pour ChestX-ray14.
    - Tâche: Classification multi-label (14 classes)
    - Sortie: dict {image: Tensor[3,H,W], label: Tensor[14], task_id: 0}
    """
    def __init__(self, data_list: List[Tuple[str, np.ndarray]], transform=None):
        self.data_list = data_list
        self.transform = transform
        self.task_id = 0
        self.task_type = 'classification'
        self.num_classes = NUM_CHEST_CLASSES

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label_vec = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {
            'image': img,
            'label': torch.from_numpy(label_vec),
            'task_id': self.task_id
        }


def prepare_chest_data(chest_dir: str, max_samples: int = 15000) -> Tuple[List, List]:
    """
    Prépare les paires (chemin_image, vecteur_labels) et effectue le split train/test.
    - max_samples: limite le nombre d'exemples pour accélérer l'entraînement.
    """
    print("\n📁 Préparation ChestX-ray14...")
    csv_path = find_chest_csv(chest_dir)
    if not csv_path:
        print("   ❌ CSV introuvable dans", chest_dir)
        return [], []
    print(f"   ✅ CSV trouvé: {csv_path}")

    df = pd.read_csv(csv_path)
    img_dirs = find_image_dirs(chest_dir)
    if not img_dirs:
        print("   ❌ Aucun répertoire d'images trouvé.")
        return [], []
    print(f"   ✅ Répertoires d'images: {len(img_dirs)}")

    label_to_idx = {d: i for i, d in enumerate(CHEST_DISEASES)}
    data = []

    for _, row in df.iterrows():
        if len(data) >= max_samples:
            break
        fname = row['Image Index']
        labels_str = str(row['Finding Labels'])

        img_path = None
        for img_dir in img_dirs:
            candidate = os.path.join(img_dir, fname)
            if os.path.exists(candidate):
                img_path = candidate
                break
        if not img_path:
            continue

        label_vec = np.zeros(NUM_CHEST_CLASSES, dtype=np.float32)
        for label in labels_str.split('|'):
            label = label.strip()
            if label in label_to_idx:
                label_vec[label_to_idx[label]] = 1.0
        data.append((img_path, label_vec))

    print(f"   ✅ Images collectées: {len(data)}")
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED)
    print(f"   ▶ Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data
