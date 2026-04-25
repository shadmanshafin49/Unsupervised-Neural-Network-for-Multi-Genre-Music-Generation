import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE


class Encoder(nn.Module):
    def __init__(self, input_dim=NUM_PITCHES, hidden_dim=HIDDEN_DIM,
                 latent_dim=LATENT_DIM, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        # x: (batch, seq_len, input_dim)
        _, (h_n, _) = self.lstm(x)
        z = self.fc(h_n[-1])  # (batch, latent_dim)
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=NUM_PITCHES, seq_len=WINDOW_SIZE, num_layers=2):
        super().__init__()
        self.seq_len = seq_len
        self.fc = nn.Linear(latent_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, z):
        # z: (batch, latent_dim)
        h = self.fc(z)
        h = h.unsqueeze(1).repeat(1, self.seq_len, 1)  # (batch, seq_len, hidden_dim)
        out, _ = self.lstm(h)
        return self.sigmoid(self.output_fc(out))


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat
