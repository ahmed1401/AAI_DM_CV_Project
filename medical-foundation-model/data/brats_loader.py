"""
Chargement du dataset BraTS2020 (segmentation 2D des IRM cérébrales)
Adapté du notebook 15-epochs.
"""
import os
import random
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

# Détection optionnelle de nibabel pour lire les NIfTI
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BRATS_CLASSES = ['Background', 'NCR/NET', 'Edema', 'Enhancing']
NUM_BRATS_CLASSES = 4


class BraTSSegmentationDataset(Dataset):
    """
    Dataset 2D pour BraTS2020.
    - Entrée: tranche FLAIR mise à l'échelle en 3 canaux
    - Sortie: dict {image: Tensor[3,H,W], mask: LongTensor[H,W], task_id: 1}
    """
    def __init__(self, slice_data: List[Tuple[np.ndarray, np.ndarray, str, int]], img_size: int = 224, augment: bool = False):
        self.slice_data = slice_data
        self.img_size = img_size
        self.augment = augment
        self.task_id = 1
        self.task_type = 'segmentation'
        self.num_classes = NUM_BRATS_CLASSES
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.slice_data)

    def __getitem__(self, idx):
        flair_slice, seg_slice, patient_id, slice_idx = self.slice_data[idx]
        flair_resized = np.array(Image.fromarray(flair_slice).resize((self.img_size, self.img_size), Image.BILINEAR))
        seg_resized = np.array(Image.fromarray(seg_slice.astype(np.uint8)).resize((self.img_size, self.img_size), Image.NEAREST))

        flair_3ch = np.stack([flair_resized] * 3, axis=0).astype(np.float32)
        flair_3ch = (flair_3ch - flair_3ch.min()) / (flair_3ch.max() - flair_3ch.min() + 1e-8)
        image = torch.from_numpy(flair_3ch)
        image = self.normalize(image)

        seg_remapped = seg_resized.copy()
        seg_remapped[seg_resized == 4] = 3
        mask = torch.from_numpy(seg_remapped).long()

        if self.augment and random.random() > 0.5:
            image = torch.flip(image, dims=[2])
            mask = torch.flip(mask, dims=[1])
        return {'image': image, 'mask': mask, 'task_id': self.task_id}


def _candidate_training_paths(brats_dir: str) -> List[str]:
    return [
        os.path.join(brats_dir, 'BraTS2020_TrainingData', 'MICCAI_BraTS2020_TrainingData'),
        os.path.join(brats_dir, 'MICCAI_BraTS2020_TrainingData'),
        os.path.join(brats_dir, 'BraTS2020_TrainingData'),
        brats_dir,
    ]


def prepare_brats_data(brats_dir: str, max_patients: int = 100, img_size: int = 224) -> Tuple[List, List]:
    """
    Extrait des tranches 2D (FLAIR + segmentation) pour BraTS.
    Sélectionne des tranches médianes avec contenu tumoral pour un entraînement efficace.
    """
    print("\n📁 Préparation BraTS2020 (segmentation)...")
    if not NIBABEL_AVAILABLE:
        print("   ❌ nibabel non disponible. Installez 'nibabel'.")
        return [], []

    training_dir = None
    for path in _candidate_training_paths(brats_dir):
        if os.path.exists(path):
            patient_folders = [d for d in os.listdir(path) if d.startswith('BraTS20_Training_') and os.path.isdir(os.path.join(path, d))]
            if len(patient_folders) > 10:
                training_dir = path
                print(f"   ✅ Dossier d'entraînement détecté: {training_dir}")
                print(f"   ▶ Patients: {len(patient_folders)}")
                break
    if not training_dir:
        print("   ❌ Dossier des patients introuvable.")
        return [], []

    patient_dirs = sorted([d for d in os.listdir(training_dir) if os.path.isdir(os.path.join(training_dir, d)) and d.startswith('BraTS20_Training_')])
    print(f"   ▶ Traitement de {min(max_patients, len(patient_dirs))} patients...")

    all_slices = []
    patients_used = 0
    patients_skipped = 0

    for patient in patient_dirs[:max_patients]:
        patient_path = os.path.join(training_dir, patient)
        flair_file = None
        seg_file = None
        try:
            files = os.listdir(patient_path)
        except Exception:
            patients_skipped += 1
            continue
        for f in files:
            f_lower = f.lower()
            if 'flair' in f_lower and (f.endswith('.nii.gz') or f.endswith('.nii')):
                flair_file = os.path.join(patient_path, f)
            elif 'seg' in f_lower and (f.endswith('.nii.gz') or f.endswith('.nii')):
                seg_file = os.path.join(patient_path, f)
        if not flair_file or not seg_file:
            patients_skipped += 1
            continue

        try:
            flair_vol = nib.load(flair_file).get_fdata()
            seg_vol = nib.load(seg_file).get_fdata()
            depth = flair_vol.shape[2]
            slices_extracted = 0
            for slice_idx in range(depth // 4, 3 * depth // 4, 2):
                seg_slice = seg_vol[:, :, slice_idx]
                if np.sum(seg_slice > 0) < 100:
                    continue
                flair_slice = flair_vol[:, :, slice_idx]
                flair_min, flair_max = flair_slice.min(), flair_slice.max()
                if flair_max - flair_min > 0:
                    flair_norm = ((flair_slice - flair_min) / (flair_max - flair_min) * 255).astype(np.uint8)
                else:
                    flair_norm = np.zeros_like(flair_slice, dtype=np.uint8)
                all_slices.append((flair_norm, seg_slice, patient, slice_idx))
                slices_extracted += 1
            if slices_extracted > 0:
                patients_used += 1
        except Exception as e:
            print(f"   ⚠️ Erreur chargement {patient}: {e}")
            patients_skipped += 1
            continue

    print(f"   ✅ Tranches extraites: {len(all_slices)} (patients utilisés: {patients_used}, ignorés: {patients_skipped})")
    if len(all_slices) == 0:
        return [], []

    random.shuffle(all_slices)
    split_idx = int(0.8 * len(all_slices))
    train_slices = all_slices[:split_idx]
    test_slices = all_slices[split_idx:]
    print(f"   ▶ Train: {len(train_slices)} | Test: {len(test_slices)}")
    return train_slices, test_slices
