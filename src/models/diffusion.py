"""
Diffusion model placeholder for music generation.
Not used in current pipeline (Tasks 1-4 use AE, VAE, Transformer, RLHF).
"""
import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    """Score-based / DDPM-style diffusion model stub."""

    def __init__(self, input_dim: int = 88, hidden_dim: int = 256, num_steps: int = 1000):
        super().__init__()
        self.num_steps = num_steps
        self.net = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = t.float().unsqueeze(-1) / self.num_steps
        xt = torch.cat([x, t_emb.expand(x.size(0), 1)], dim=-1)
        return self.net(xt)
