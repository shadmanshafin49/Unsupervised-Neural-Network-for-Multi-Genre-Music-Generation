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
        _, (h_n, _) = self.lstm(x)
        return self.fc(h_n[-1])


class Decoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=NUM_PITCHES, seq_len=WINDOW_SIZE, num_layers=2):
        super().__init__()
        self.seq_len   = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        # z initialises the LSTM hidden state
        self.fc_z  = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm  = nn.LSTM(
            input_size=output_dim,   # previous piano-roll step as input
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.output_fc = nn.Linear(hidden_dim, output_dim)

    def _init_hidden(self, z):
        batch = z.size(0)
        h = self.fc_z(z).view(batch, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, x_teacher=None):
        """
        z:         (batch, latent_dim)
        x_teacher: (batch, seq_len, output_dim) during training (teacher forcing)
                   None during inference (autoregressive)
        Returns:   (batch, seq_len, output_dim) raw logits
        """
        h0, c0 = self._init_hidden(z)
        batch = z.size(0)

        if x_teacher is not None:
            # Teacher forcing: SOS zeros prepended, drop last target step
            sos   = torch.zeros(batch, 1, self.output_dim, device=z.device)
            x_in  = torch.cat([sos, x_teacher[:, :-1, :]], dim=1)
            out, _ = self.lstm(x_in, (h0, c0))
            return self.output_fc(out)
        else:
            # Autoregressive inference (generates self.seq_len steps)
            outputs = []
            x_t    = torch.zeros(batch, 1, self.output_dim, device=z.device)
            hidden = (h0, c0)
            for _ in range(self.seq_len):
                out, hidden = self.lstm(x_t, hidden)
                logit = self.output_fc(out)
                outputs.append(logit)
                x_t = torch.sigmoid(logit).detach()
            return torch.cat(outputs, dim=1)


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z, x_teacher=x)  # teacher forcing during training
        return z, x_hat
