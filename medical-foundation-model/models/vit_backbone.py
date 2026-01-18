"""
Backbone Vision Transformer (ViT) pré-entraîné via timm.
Remplace la tête par une identité et gère le gel des paramètres.
"""
import timm
import torch
import torch.nn as nn


def create_vit(model_name: str = 'vit_base_patch16_224', pretrained: bool = True):
    """Crée le modèle ViT pré-entraîné et supprime la tête de classification."""
    vit = timm.create_model(model_name, pretrained=pretrained)
    # Remplacer la tête par une identité (on utilise nos têtes spécifiques ensuite)
    if hasattr(vit, 'head'):
        vit.head = nn.Identity()
    return vit


def freeze_vit(vit: torch.nn.Module, freeze: bool = True):
    """Active/désactive la mise à jour des poids du backbone."""
    for p in vit.parameters():
        p.requires_grad = not freeze
    return vit
