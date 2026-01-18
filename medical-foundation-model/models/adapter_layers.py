"""
Couches Adapter pour un fine-tuning efficace (Houlsby et al.).
"""
import torch
import torch.nn as nn


class Adapter(nn.Module):
    """Bottleneck linéaire: down-projection → GELU → up-projection."""
    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(embed_dim, hidden_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(hidden_dim, embed_dim)
        # Initialisation proche de l'identité pour la stabilité
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(x)))
