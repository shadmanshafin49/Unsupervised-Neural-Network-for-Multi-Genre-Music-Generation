import torch
import torch.nn as nn
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, D_MODEL, N_HEADS, N_LAYERS, DROPOUT, NUM_GENRES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model=D_MODEL, max_len=512, dropout=DROPOUT):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(NUM_PITCHES, D_MODEL)
        self.genre_embed = nn.Embedding(NUM_GENRES, D_MODEL)
        self.pos_enc     = PositionalEncoding(D_MODEL)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=D_MODEL,
            nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4,
            dropout=DROPOUT,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=N_LAYERS)
        self.output_proj = nn.Linear(D_MODEL, NUM_PITCHES)
        self.sigmoid     = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def make_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return mask.bool()

    def forward(self, x, genre_ids):
        """
        x:         (batch, seq_len, 88)
        genre_ids: (batch,)
        Returns:   (batch, seq_len, 88)
        """
        x_proj = self.input_proj(x)
        g_tok  = self.genre_embed(genre_ids).unsqueeze(1)   # (batch, 1, D_MODEL)
        x_proj = torch.cat([g_tok, x_proj], dim=1)          # (batch, seq_len+1, D_MODEL)
        x_proj = self.pos_enc(x_proj)

        full_len    = x_proj.size(1)
        causal_mask = self.make_causal_mask(full_len, x.device)

        out = self.transformer(
            tgt=x_proj,
            memory=x_proj,
            tgt_mask=causal_mask,
            memory_mask=causal_mask,
        )
        out = out[:, 1:, :]  # remove genre token
        return self.sigmoid(self.output_proj(out))
