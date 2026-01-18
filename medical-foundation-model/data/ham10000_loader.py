"""
Chargement du dataset HAM10000 (classification multi-classe)
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

HAM_CLASS_NAMES = {
    'akiec': 'Kératose actinique',
    'bcc': 'Carcinome basocellulaire',
    'bkl': 'Kératose bénigne',
    'df': 'Dermatofibrome',
    'mel': 'Mélanome',
    'nv': 'Nævus mélanocytaire',
    'vasc': 'Lésions vasculaires'
}


class HAMDataset(Dataset):
    """
    Dataset PyTorch pour HAM10000.
    - Tâche: Classification multi-classe (7 classes)
    - Sortie: dict {image: Tensor[3,H,W], label: LongTensor[], task_id: 2}
    """
    def __init__(self, data_list: List[Tuple[str, int]], transform=None):
        self.data_list = data_list
        self.transform = transform
        self.task_id = 2
        self.task_type = 'classification'
        self.num_classes = 7

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path, label = self.data_list[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return {'image': img, 'label': torch.tensor(label, dtype=torch.long), 'task_id': self.task_id}


def prepare_ham_data(ham_dir: str) -> Tuple[List, List]:
    """
    Prépare les paires (chemin_image, label) et effectue le split train/test avec stratification.
    """
    print("\n📁 Préparation HAM10000...")
    csv_path = None
    for f in os.listdir(ham_dir):
        if 'metadata' in f.lower() and f.endswith('.csv'):
            csv_path = os.path.join(ham_dir, f)
            break
    if not csv_path:
        print("   ❌ Fichier metadata introuvable.")
        return [], []
    print(f"   ✅ CSV metadata: {csv_path}")

    df = pd.read_csv(csv_path)
    label_to_idx = {dx: i for i, dx in enumerate(sorted(df['dx'].unique()))}
    print(f"   ▶ Classes: {label_to_idx}")

    img_dirs = [os.path.join(ham_dir, d) for d in os.listdir(ham_dir) if os.path.isdir(os.path.join(ham_dir, d))]
    if not img_dirs:
        img_dirs = [ham_dir]

    data = []
    for _, row in df.iterrows():
        img_id = row['image_id']
        dx = row['dx']
        img_path = None
        for img_dir in img_dirs:
            for ext in ['.jpg', '.png', '.jpeg']:
                candidate = os.path.join(img_dir, f"{img_id}{ext}")
                if os.path.exists(candidate):
                    img_path = candidate
                    break
            if img_path:
                break
        if img_path:
            data.append((img_path, label_to_idx[dx]))

    print(f"   ✅ Images collectées: {len(data)}")
    labels = [d[1] for d in data]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=SEED, stratify=labels)
    print(f"   ▶ Train: {len(train_data)} | Test: {len(test_data)}")
    return train_data, test_data
