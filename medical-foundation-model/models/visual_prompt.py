"""
Implémentation des Visual Prompts (prompts apprenables insérés dans la séquence ViT).
"""
import torch
import torch.nn as nn


class VisualPromptManager(nn.Module):
    """
    Gère des prompts par tâche (chest, brats, ham) et leur injection dans la séquence.
    """
    def __init__(self, embed_dim: int, prompt_length: int = 10, task_names=None):
        super().__init__()
        if task_names is None:
            task_names = ['chest', 'brats', 'ham']
        self.embed_dim = embed_dim
        self.prompt_length = prompt_length
        # Prompts par tâche
        self.prompts = nn.ParameterDict({
            name: nn.Parameter(torch.randn(1, prompt_length, embed_dim) * 0.02)
            for name in task_names
        })

    def forward(self, tokens: torch.Tensor, task_name: str) -> torch.Tensor:
        """
        Insère les prompts après le token CLS: [CLS][PROMPTS][PATCHES].
        - tokens: [B, 1+N, D] (incluant CLS et patchs)
        - task_name: nom de la tâche
        """
        B = tokens.size(0)
        prompts = self.prompts[task_name].expand(B, -1, -1)
        # Concat: CLS | PROMPTS | PATCHES
        return torch.cat([tokens[:, :1], prompts, tokens[:, 1:]], dim=1)
