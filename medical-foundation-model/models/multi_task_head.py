"""
Têtes multi-tâches et modèle principal MultiTaskViT.
Inclut le décodeur UNETR pour la segmentation et l'intégration des Adapters/Prompts.
"""
import math
import torch
import torch.nn as nn
from models.adapter_layers import Adapter
from models.visual_prompt import VisualPromptManager


class ConvBlock(nn.Module):
    """Bloc conv 3x3 + BN + ReLU (doublé)."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class UNETRDecoder(nn.Module):
    """
    Décodeur UNETR-like pour transformer les tokens ViT en carte de segmentation.
    """
    def __init__(self, embed_dim: int = 768, num_classes: int = 4, img_size: int = 224, patch_size: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.feature_size = img_size // patch_size
        channels = [embed_dim, 512, 256, 128, 64]
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU())
        self.decoder_blocks = nn.ModuleList([
            nn.Sequential(nn.ConvTranspose2d(channels[0], channels[1], 2, 2), ConvBlock(channels[1], channels[1])),
            nn.Sequential(nn.ConvTranspose2d(channels[1], channels[2], 2, 2), ConvBlock(channels[2], channels[2])),
            nn.Sequential(nn.ConvTranspose2d(channels[2], channels[3], 2, 2), ConvBlock(channels[3], channels[3])),
            nn.Sequential(nn.ConvTranspose2d(channels[3], channels[4], 2, 2), ConvBlock(channels[4], channels[4])),
        ])
        self.seg_head = nn.Conv2d(channels[4], num_classes, kernel_size=1)

    def forward(self, vit_features: torch.Tensor) -> torch.Tensor:
        B = vit_features.shape[0]
        patch_tokens = vit_features[:, 1:, :]  # enlever CLS
        patch_tokens = self.proj(patch_tokens)
        x = patch_tokens.transpose(1, 2).reshape(B, self.embed_dim, self.feature_size, self.feature_size)
        for block in self.decoder_blocks:
            x = block(x)
        return self.seg_head(x)


class MultiTaskViT(nn.Module):
    """
    Modèle multi-tâches basé sur ViT.
    - Tâche 0: ChestX-ray14 (classification multi-label, 14 classes)
    - Tâche 1: BraTS2020 (segmentation, 4 classes)
    - Tâche 2: HAM10000 (classification multi-classe, 7 classes)
    """
    def __init__(self, vit_backbone: nn.Module, num_chest: int = 14, num_brats: int = 4, num_ham: int = 7,
                 freeze_backbone: bool = False, use_adapters: bool = False, use_prompts: bool = False,
                 adapter_hidden: int = 64, prompt_length: int = 10, img_size: int = 224):
        super().__init__()
        self.vit = vit_backbone
        self.embed_dim = vit_backbone.embed_dim
        self.freeze_backbone = freeze_backbone
        self.use_adapters = use_adapters
        self.use_prompts = use_prompts
        self.prompt_length = prompt_length
        self.img_size = img_size

        # Têtes par tâche
        self.chest_head = nn.Linear(self.embed_dim, num_chest)
        self.brats_decoder = UNETRDecoder(embed_dim=self.embed_dim, num_classes=num_brats, img_size=img_size, patch_size=16)
        self.ham_head = nn.Linear(self.embed_dim, num_ham)

        # Adapters
        if use_adapters:
            num_blocks = len(self.vit.blocks)
            self.adapters = nn.ModuleDict({
                'chest': nn.ModuleList([Adapter(self.embed_dim, adapter_hidden) for _ in range(num_blocks)]),
                'brats': nn.ModuleList([Adapter(self.embed_dim, adapter_hidden) for _ in range(num_blocks)]),
                'ham': nn.ModuleList([Adapter(self.embed_dim, adapter_hidden) for _ in range(num_blocks)])
            })

        # Prompts
        self.prompt_mgr = VisualPromptManager(self.embed_dim, prompt_length, task_names=['chest', 'brats', 'ham']) if use_prompts else None

    def forward_features(self, x: torch.Tensor, task_name: str) -> torch.Tensor:
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Injecter prompts si requis
        if self.use_prompts:
            x = self.prompt_mgr(x, task_name)
        # Position embeddings (gérer longueur augmentée)
        num_tokens = x.size(1)
        if self.vit.pos_embed.size(1) >= num_tokens:
            pos_embed = self.vit.pos_embed[:, :num_tokens, :]
        else:
            cls_pos = self.vit.pos_embed[:, :1, :]
            patch_pos = self.vit.pos_embed[:, 1:, :]
            prompt_pos = torch.zeros(1, num_tokens - patch_pos.size(1) - 1, self.embed_dim, device=x.device)
            pos_embed = torch.cat([cls_pos, prompt_pos, patch_pos], dim=1)
        x = x + pos_embed
        x = self.vit.pos_drop(x)
        # Passer dans les blocs Transformer
        for i, block in enumerate(self.vit.blocks):
            if self.freeze_backbone:
                with torch.no_grad():
                    x = block(x)
            else:
                x = block(x)
            if self.use_adapters:
                adapter_out = self.adapters[task_name][i](x)
                x = x + adapter_out
        # Norme finale
        if self.freeze_backbone:
            with torch.no_grad():
                x = self.vit.norm(x)
        else:
            x = self.vit.norm(x)
        return x

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        task_names = ['chest', 'brats', 'ham']
        task_name = task_names[task_id]
        features = self.forward_features(x, task_name)
        if task_id == 0:
            return self.chest_head(features[:, 0])  # token CLS
        elif task_id == 1:
            # Retirer les prompts avant le décodeur
            if self.use_prompts:
                features = torch.cat([features[:, :1], features[:, 1 + self.prompt_length:]], dim=1)
            return self.brats_decoder(features)
        elif task_id == 2:
            return self.ham_head(features[:, 0])
        else:
            raise ValueError(f"task_id inconnu: {task_id}")

    def get_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
