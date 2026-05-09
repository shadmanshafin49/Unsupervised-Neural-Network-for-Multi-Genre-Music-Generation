import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, WINDOW_SIZE


class RewardModel(nn.Module):
    """Predicts human preference score (1-5) from a piano roll segment."""
    input_dim = NUM_PITCHES * WINDOW_SIZE  # 88 * 64 = 5632

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, roll):
        # roll: (batch, WINDOW_SIZE, NUM_PITCHES)
        flat = roll.view(roll.size(0), -1)
        return self.net(flat) * 4 + 1  # scale to [1, 5]
