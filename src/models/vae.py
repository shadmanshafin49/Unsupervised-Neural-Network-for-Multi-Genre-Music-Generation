import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, HIDDEN_DIM, LATENT_DIM, WINDOW_SIZE, NUM_GENRES, GENRE_EMBED_DIM


class VAEEncoder(nn.Module):
    def __init__(self, input_dim=NUM_PITCHES + GENRE_EMBED_DIM,
                 hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM,
                 output_dim=NUM_PITCHES, seq_len=WINDOW_SIZE, num_layers=2):
        super().__init__()
        self.seq_len    = seq_len
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.fc_z  = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.lstm  = nn.LSTM(
            input_size=output_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def _init_hidden(self, z):
        batch = z.size(0)
        h = self.fc_z(z).view(batch, self.num_layers, self.hidden_dim)
        h = h.permute(1, 0, 2).contiguous()
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, x_teacher=None):
        h0, c0 = self._init_hidden(z)
        batch  = z.size(0)

        if x_teacher is not None:
            sos   = torch.zeros(batch, 1, self.output_dim, device=z.device)
            x_in  = torch.cat([sos, x_teacher[:, :-1, :]], dim=1)
            out, _ = self.lstm(x_in, (h0, c0))
            return self.out_fc(out)
        else:
            outputs = []
            x_t    = torch.zeros(batch, 1, self.output_dim, device=z.device)
            hidden = (h0, c0)
            for _ in range(self.seq_len):
                out, hidden = self.lstm(x_t, hidden)
                logit = self.out_fc(out)
                outputs.append(logit)
                x_t = torch.sigmoid(logit).detach()
            return torch.cat(outputs, dim=1)


class VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.genre_emb = nn.Embedding(NUM_GENRES, GENRE_EMBED_DIM)
        self.encoder   = VAEEncoder()
        self.decoder   = VAEDecoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, genre_ids):
        batch, seq_len, _ = x.shape
        g = self.genre_emb(genre_ids).unsqueeze(1).expand(-1, seq_len, -1)
        x_with_genre = torch.cat([x, g], dim=-1)

        mu, log_var = self.encoder(x_with_genre)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decoder(z, x_teacher=x)  # teacher forcing
        return mu, log_var, z, x_hat

    @staticmethod
    def kl_loss(mu, log_var):
        return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / mu.size(0)
