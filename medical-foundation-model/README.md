# Medical Foundation Model (Multi-Task ViT)

Projet modulaire issu du notebook multi-tâches (15 époques) pour l'imagerie médicale. Compare quatre stratégies d'adaptation autour d'un Vision Transformer (ViT-Base):
- Full Fine-Tuning
- Linear Probing
- Adapter Layers (PEFT)
- Visual Prompt Tuning (PEFT)

## Architecture du Répertoire

```
medical-foundation-model/
├── data/                # Chargement et préparation des datasets
│   ├── chestxray_loader.py
│   ├── brats_loader.py
│   └── ham10000_loader.py
├── models/              # Définition des modules de modèle
│   ├── vit_backbone.py
│   ├── visual_prompt.py
│   ├── multi_task_head.py
│   └── adapter_layers.py
├── training/            # Boucles et stratégies d'entraînement
│   ├── train_baseline.py
│   ├── train_prompt.py
│   └── train_adapter.py
├── evaluation/          # Métriques et interprétabilité
│   ├── metrics.py
│   └── interpretability.py
├── experiments/         # Comparaison des méthodes
│   └── compare_methods.py
├── notebooks/           # Notebooks d'exploration
│   ├── Data_exploration.ipynb
│   ├── diagnostic-m-dical-multi-t-ches-5epochs.ipynb
│   └── diagnostic-m-dical-multi-t-ches-10epochs.ipynb
│   └── diagnostic-m-dical-multi-t-ches-15epochs.ipynb
│
└── requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Chemins des Datasets

Par défaut (Kaggle):
- ChestX-ray14: `/kaggle/input/data`
- BraTS2020: `/kaggle/input/brats20-dataset-training-validation`
- HAM10000: `/kaggle/input/skin-cancer-mnist-ham10000`

Pour un environnement local, remplacez ces chemins dans les scripts d'entraînement.

## Lancement Entraînement (15 Époques)

- Full Fine-Tuning:
```bash
python training/train_baseline.py
```
- Visual Prompt Tuning:
```bash
python training/train_prompt.py
```
- Adapter Layers:
```bash
python training/train_adapter.py
```

## Comparaison des Méthodes

```bash
python experiments/compare_methods.py
```
Génère `comparison_results.csv` et un résumé console. Vous pouvez prolonger pour produire des figures.

## Évaluation et Interprétabilité

- Métriques: `evaluation/metrics.py`
- Grad-CAM++: `evaluation/interpretability.py`

Exemple d'utilisation Grad-CAM++ (classification):
```python
from evaluation.interpretability import gradcam_visualization
viz = gradcam_visualization(model, img_tensor, target_class=3, task_id=0, prompt_length=10)
```



## Auteurs / Crédits
- Ahmed HAJJEJ -- Skander HAKOUNA -- Samia THAMEUR -- Mohamed Ali CHIBANI -- Ayhem BOUKARI
