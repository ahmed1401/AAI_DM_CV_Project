# Medical Foundation Model (Multi‑Task ViT)

**Auteurs**: Ahmed HAJJEJ · Skander HAKOUNA · Samia THAMEUR · Mohamed Ali CHIBANI · Ayhem BOUKARI

Projet modulaire issu du notebook multi‑tâches pour l'imagerie médicale. Il compare quatre stratégies d'adaptation autour d'un Vision Transformer (ViT‑Base/16) partagé et des têtes spécifiques aux tâches:
- Full Fine‑Tuning
- Linear Probing
- Adapter Layers (PEFT)
- Visual Prompt Tuning (PEFT)

## Aperçu

- Tâches couvertes: Classification multi‑label (ChestX‑ray14), Segmentation (BraTS2020), Classification multi‑classe (HAM10000)
- Backbone: ViT‑Base/16 pré‑entraîné (partagé)
- Segmentation: Décodeur de type UNETR
- Adaptation efficace: Adapters et Visual Prompts
- Exécutions: Notebooks et scripts avec 5/10/15 époques

## Architecture du Répertoire

```
medical-foundation-model/
├── data/                # Chargement et préparation des datasets
│   ├── chestxray_loader.py
│   ├── brats_loader.py
│   └── ham10000_loader.py
├── models/              # Modules de modèle
│   ├── vit_backbone.py
│   ├── visual_prompt.py
│   ├── multi_task_head.py
│   └── adapter_layers.py
├── training/            # Stratégies d'entraînement
│   ├── train_baseline.py
│   ├── train_prompt.py
│   └── train_adapter.py
├── evaluation/          # Métriques et interprétabilité
│   ├── metrics.py
│   └── interpretability.py
├── experiments/         # Comparaison des méthodes
│   └── compare_methods.py
├── notebooks/           # Notebooks d'exploration et d'expérimentation
│   ├── Data_exploration.ipynb
│   ├── diagnostic-m-dical-multi-t-ches-5epochs.ipynb
│   ├── diagnostic-m-dical-multi-t-ches-10epochs.ipynb
│   └── diagnostic-m-dical-multi-t-ches-15epochs.ipynb
└── requirements.txt
```

## Prérequis

- Python 3.9+ recommandé
- GPU CUDA (optionnel mais fortement conseillé pour la segmentation)
- Pip ou Conda pour la gestion d'environnement

## Installation

```bash
pip install -r requirements.txt
```

## Datasets

Chemins par défaut (Kaggle):
- ChestX‑ray14: /kaggle/input/data
- BraTS2020: /kaggle/input/brats20-dataset-training-validation
- HAM10000: /kaggle/input/skin-cancer-mnist-ham10000

Configuration locale (exemple, Windows):
```bash
set CHESTXRAY_PATH="D:\\datasets\\ChestX-ray14"
set BRATS2020_PATH="D:\\datasets\\BraTS2020"
set HAM10000_PATH="D:\\datasets\\HAM10000"
```
Adaptez ces chemins dans les notebooks et les scripts d'entraînement si nécessaire.

## Lancer l'entraînement

Méthodes (epochs configurables dans les scripts):
- Full Fine‑Tuning:
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

## Notebooks

- Exploration et vérifications: notebooks/Data_exploration.ipynb
- Entraînements comparatifs:
	- notebooks/diagnostic-m-dical-multi-t-ches-5epochs.ipynb
	- notebooks/diagnostic-m-dical-multi-t-ches-10epochs.ipynb
	- notebooks/diagnostic-m-dical-multi-t-ches-15epochs.ipynb

## Comparaison des méthodes

```bash
python experiments/compare_methods.py
```
Produit un fichier de résultats (comparison_results.csv) et un résumé console. Vous pouvez étendre pour générer des figures (courbes de pertes, métriques par tâche, efficacité paramétrique).

## Évaluation & Interprétabilité

- Métriques: evaluation/metrics.py
- Grad‑CAM++: evaluation/interpretability.py

Exemple (classification):
```python
from evaluation.interpretability import gradcam_visualization
viz = gradcam_visualization(model, img_tensor, target_class=3, task_id=0, prompt_length=10)
```

## Crédit & Contact

Projet académique. Pour questions ou suggestions, merci de créer une issue ou de nous contacter.
