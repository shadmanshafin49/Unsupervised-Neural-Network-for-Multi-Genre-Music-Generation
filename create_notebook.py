"""
Run this script once locally:
    python create_notebook.py
It creates kaggle_train.ipynb which you upload to Kaggle.
"""
import json

def md(text):
    lines = text.split('\n')
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [l + '\n' for l in lines[:-1]] + ([lines[-1]] if lines[-1] else [])
    }

def code(text):
    lines = text.rstrip('\n').split('\n')
    source = [l + '\n' for l in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

cells = []

# ─────────────────────────────────────────────
# Title
# ─────────────────────────────────────────────
cells.append(md(
"""# CSE425 – Music Generation: Kaggle Training Notebook
**Tasks 1 · 2 · 3  |  LSTM Autoencoder → VAE → Transformer**

### Before running
1. **Preprocess locally** — run `python run_preprocessing.py` in your project root.
   This produces 4 files in `data/train_test_split/`:
   `train_data.npy`, `test_data.npy`, `train_labels.npy`, `test_labels.npy`
2. **Upload those 4 files** to Kaggle as a new Dataset (Datasets → New Dataset).
3. **Add that dataset** to this notebook (click **+ Add data** in Kaggle).
4. **Set `NPY_SLUG`** in the Configuration cell below to match your dataset slug.
5. Enable GPU: *Settings → Accelerator → GPU T4 x2* (or P100).
6. Click **Run All**.

Outputs (trained models + generated MIDIs + plots) are saved to `/kaggle/working/outputs/`.
A download ZIP is produced in the final cell."""
))

# ─────────────────────────────────────────────
# 1 · Install
# ─────────────────────────────────────────────
cells.append(code(
"""!pip install pretty_midi -q
print("pretty_midi installed.")"""
))

# ─────────────────────────────────────────────
# 2 · Imports & GPU
# ─────────────────────────────────────────────
cells.append(code(
"""import os, math, random, shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import pretty_midi

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))"""
))

# ─────────────────────────────────────────────
# 3 · Configuration
# ─────────────────────────────────────────────
cells.append(md("""## Configuration
**Edit `NPY_SLUG` to match the slug of your uploaded preprocessed-data dataset.**"""))

cells.append(code(
"""# ── USER SETTING ────────────────────────────────────────────────────────────
NPY_SLUG = 'your-npy-dataset-slug'   # e.g. 'cse425-preprocessed'
# ────────────────────────────────────────────────────────────────────────────

NPY_DIR  = f'/kaggle/input/{NPY_SLUG}'

OUT_DIR  = '/kaggle/working/outputs'
MIDI_OUT = os.path.join(OUT_DIR, 'generated_midis')
PLOTS    = os.path.join(OUT_DIR, 'plots')
for d in [OUT_DIR, MIDI_OUT, PLOTS,
          os.path.join(MIDI_OUT, 'task1'),
          os.path.join(MIDI_OUT, 'task2'),
          os.path.join(MIDI_OUT, 'task3'),
          os.path.join(MIDI_OUT, 'baselines')]:
    os.makedirs(d, exist_ok=True)

GENRES = {
    'Blues': 0, 'Classical': 1, 'Country': 2, 'Electronic': 3,
    'Folk': 4, 'Hip-Hop': 5, 'Jazz': 6, 'Pop': 7, 'Rock': 8,
}
NUM_GENRES   = len(GENRES)
NUM_PITCHES  = 88
MIDI_MIN     = 21
WINDOW_SIZE  = 64
STEP_DUR     = 0.125

LATENT_DIM      = 64
HIDDEN_DIM      = 256
GENRE_EMBED_DIM = 16
D_MODEL         = 256
N_HEADS         = 8
N_LAYERS        = 4
DROPOUT         = 0.1

BATCH_SIZE  = 32
LR_AE       = 1e-3
LR_VAE      = 1e-3
LR_TR       = 5e-4
EPOCHS_AE   = 50
EPOCHS_VAE  = 50
EPOCHS_TR   = 80
WARMUP_KL   = 10

GEN_THRESHOLD = 0.30

print('Config ready.')
print(f'NPY source: {NPY_DIR}')"""
))

# ─────────────────────────────────────────────
# 4 · Load preprocessed .npy files
# ─────────────────────────────────────────────
cells.append(md("## Load Preprocessed Data"))

cells.append(code(
"""X_train = np.load(os.path.join(NPY_DIR, 'train_data.npy'))
X_test  = np.load(os.path.join(NPY_DIR, 'test_data.npy'))
y_train = np.load(os.path.join(NPY_DIR, 'train_labels.npy'))
y_test  = np.load(os.path.join(NPY_DIR, 'test_labels.npy'))

print(f'Train : {X_train.shape}  labels: {y_train.shape}')
print(f'Test  : {X_test.shape}   labels: {y_test.shape}')
sparsity = 1 - float(X_train.mean())
print(f'Sparsity: {sparsity:.3f}  ({sparsity*100:.1f}% silent cells)')

kw = dict(batch_size=BATCH_SIZE, num_workers=0, pin_memory=(device.type == 'cuda'))

train_dl = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    shuffle=True, **kw)
test_dl = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    shuffle=False, **kw)

train_dl_ae = DataLoader(
    TensorDataset(torch.FloatTensor(X_train)), shuffle=True, **kw)
test_dl_ae = DataLoader(
    TensorDataset(torch.FloatTensor(X_test)), shuffle=False, **kw)

pos_rate       = float(X_train.mean())
pos_weight_val = min((1 - pos_rate) / pos_rate, 30.0)
POS_W = torch.tensor([pos_weight_val]).to(device)
print(f'pos_weight = {pos_weight_val:.1f}')"""
))

# ─────────────────────────────────────────────
# 5 · MIDI export helper
# ─────────────────────────────────────────────
cells.append(code(
"""def piano_roll_to_midi(roll, threshold=GEN_THRESHOLD, tempo=120, save_path=None):
    pm    = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)
    binary = roll > threshold
    for pitch_idx in range(binary.shape[1]):
        in_note = False; t_start = 0.0
        for t in range(binary.shape[0]):
            t_sec = t * STEP_DUR
            if binary[t, pitch_idx] and not in_note:
                t_start = t_sec; in_note = True
            elif not binary[t, pitch_idx] and in_note:
                if t_sec - t_start >= STEP_DUR:
                    piano.notes.append(pretty_midi.Note(
                        velocity=80, pitch=pitch_idx + MIDI_MIN,
                        start=t_start, end=t_sec))
                in_note = False
        if in_note:
            t_end = binary.shape[0] * STEP_DUR
            if t_end - t_start >= STEP_DUR:
                piano.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch_idx + MIDI_MIN,
                    start=t_start, end=t_end))
    pm.instruments.append(piano)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pm.write(save_path)
    return pm

print('MIDI export helper ready.')"""
))

# ─────────────────────────────────────────────
# TASK 1 – LSTM Autoencoder
# ─────────────────────────────────────────────
cells.append(md(
"""---
## Task 1 – LSTM Autoencoder
**Architecture:** 2-layer LSTM encoder → 64-dim latent → 2-layer LSTM decoder.
The decoder uses `z` to initialise its hidden state and generates autoregressively
(each step's sigmoid output feeds back as the next step's input).
**Loss:** BCEWithLogitsLoss with pos_weight for sparse piano rolls.
**Generation:** encode a real training sample → get `z` → decode autoregressively for 256 steps (32 s)."""
))

cells.append(code(
"""class AEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(NUM_PITCHES, HIDDEN_DIM, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


class AEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.hidden_dim = HIDDEN_DIM
        self.output_dim = NUM_PITCHES
        self.seq_len    = WINDOW_SIZE
        # z initialises the LSTM hidden state (num_layers * hidden_dim weights)
        self.fc_z       = nn.Linear(LATENT_DIM, HIDDEN_DIM * 2)
        self.lstm       = nn.LSTM(NUM_PITCHES, HIDDEN_DIM, 2, batch_first=True, dropout=0.2)
        self.output_fc  = nn.Linear(HIDDEN_DIM, NUM_PITCHES)

    def _init_hidden(self, z):
        batch = z.size(0)
        h = self.fc_z(z).view(batch, 2, HIDDEN_DIM).permute(1, 0, 2).contiguous()
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, x_teacher=None):
        h0, c0 = self._init_hidden(z)
        batch  = z.size(0)
        if x_teacher is not None:
            # Teacher forcing during training
            sos  = torch.zeros(batch, 1, NUM_PITCHES, device=z.device)
            x_in = torch.cat([sos, x_teacher[:, :-1]], dim=1)
            out, _ = self.lstm(x_in, (h0, c0))
            return self.output_fc(out)
        else:
            # Autoregressive inference (self.seq_len steps)
            outputs = []
            x_t    = torch.zeros(batch, 1, NUM_PITCHES, device=z.device)
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
        self.encoder = AEEncoder()
        self.decoder = AEDecoder()

    def forward(self, x):
        z     = self.encoder(x)
        x_hat = self.decoder(z, x_teacher=x)   # teacher forcing during training
        return z, x_hat

print('Autoencoder defined.')"""
))

cells.append(code(
"""ae_model = Autoencoder().to(device)
ae_opt   = torch.optim.Adam(ae_model.parameters(), lr=LR_AE)
ae_crit  = nn.BCEWithLogitsLoss(pos_weight=POS_W)

ae_train_h, ae_val_h = [], []

for epoch in range(1, EPOCHS_AE + 1):
    ae_model.train()
    epoch_loss = 0.0
    for (x,) in tqdm(train_dl_ae, desc=f'AE {epoch}/{EPOCHS_AE}', leave=False):
        x = x.to(device)
        ae_opt.zero_grad()
        _, x_hat = ae_model(x)
        loss = ae_crit(x_hat, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ae_model.parameters(), 1.0)
        ae_opt.step()
        epoch_loss += loss.item()
    ae_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x,) in test_dl_ae:
            x = x.to(device)
            _, x_hat = ae_model(x)
            val_loss += ae_crit(x_hat, x).item()
    ae_train_h.append(epoch_loss / len(train_dl_ae))
    ae_val_h.append(val_loss   / len(test_dl_ae))
    if epoch % 10 == 0 or epoch == 1:
        print(f'AE Epoch {epoch:3d}  Train={ae_train_h[-1]:.4f}  Val={ae_val_h[-1]:.4f}')

torch.save(ae_model.state_dict(), f'{OUT_DIR}/autoencoder.pth')

plt.figure(figsize=(9, 4))
plt.plot(ae_train_h, label='Train'); plt.plot(ae_val_h, label='Val', linestyle='--')
plt.title('Task 1 – Autoencoder BCE Loss'); plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend(); plt.tight_layout()
plt.savefig(f'{PLOTS}/task1_loss.png', dpi=150); plt.show()
print('Task 1 model + plot saved.')"""
))

cells.append(code(
"""# --- Task 1 Generation: encode real samples, decode autoregressively for 256 steps ---
def generate_long_ae(model, z, total_steps=256, threshold=GEN_THRESHOLD):
    h0, c0 = model.decoder._init_hidden(z)
    outputs = []
    x_t    = torch.zeros(1, 1, NUM_PITCHES, device=z.device)
    hidden = (h0, c0)
    model.eval()
    with torch.no_grad():
        for _ in range(total_steps):
            out, hidden = model.decoder.lstm(x_t, hidden)
            logit       = model.decoder.output_fc(out)
            step        = (torch.sigmoid(logit) > threshold).float()
            outputs.append(step.squeeze().cpu().numpy())  # (88,)
            x_t = step.detach()
    return np.stack(outputs, axis=0)   # (total_steps, 88)

note_counts  = (X_train > 0.5).sum(axis=(1, 2))
good_indices = np.where(note_counts > 20)[0]
step_size    = max(1, len(good_indices) // 5)
selected     = [good_indices[i * step_size] for i in range(5)]

print('Generating 5 Task 1 MIDI samples (256 steps = 32 s each) ...')
for i, idx in enumerate(selected, 1):
    x = torch.FloatTensor(X_train[idx:idx+1]).to(device)
    with torch.no_grad():
        z, _ = ae_model(x)
    roll = generate_long_ae(ae_model, z, total_steps=256)
    piano_roll_to_midi(roll, save_path=f'{MIDI_OUT}/task1/task1_sample_{i}.mid')
    print(f'  Saved task1_sample_{i}.mid  ({int(roll.sum())} active cells)')"""
))

# ─────────────────────────────────────────────
# TASK 2 – VAE
# ─────────────────────────────────────────────
cells.append(md(
"""---
## Task 2 – Variational Autoencoder (multi-genre)
**Architecture:** genre-conditioned LSTM-VAE. Decoder uses `z` as LSTM initial hidden state
and generates autoregressively (same design as Task 1 decoder).
**Loss:** BCEWithLogitsLoss + KL divergence with KL annealing (β: 0→1 over 10 epochs).
**Generation:** encode a genre-specific training sample → get `μ` → decode autoregressively 256 steps.
8-point latent interpolation Classical ↔ Jazz using `μ₁` → `μ₂`."""
))

cells.append(code(
"""class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm      = nn.LSTM(NUM_PITCHES + GENRE_EMBED_DIM, HIDDEN_DIM, 2,
                                 batch_first=True, dropout=0.2)
        self.fc_mu     = nn.Linear(HIDDEN_DIM, LATENT_DIM)
        self.fc_logvar = nn.Linear(HIDDEN_DIM, LATENT_DIM)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc_mu(h[-1]), self.fc_logvar(h[-1])


class VAEDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_layers = 2
        self.hidden_dim = HIDDEN_DIM
        self.output_dim = NUM_PITCHES
        self.seq_len    = WINDOW_SIZE
        self.fc_z       = nn.Linear(LATENT_DIM, HIDDEN_DIM * 2)
        self.lstm       = nn.LSTM(NUM_PITCHES, HIDDEN_DIM, 2, batch_first=True, dropout=0.2)
        self.out_fc     = nn.Linear(HIDDEN_DIM, NUM_PITCHES)

    def _init_hidden(self, z):
        batch = z.size(0)
        h = self.fc_z(z).view(batch, 2, HIDDEN_DIM).permute(1, 0, 2).contiguous()
        c = torch.zeros_like(h)
        return h, c

    def forward(self, z, x_teacher=None):
        h0, c0 = self._init_hidden(z)
        batch  = z.size(0)
        if x_teacher is not None:
            sos  = torch.zeros(batch, 1, NUM_PITCHES, device=z.device)
            x_in = torch.cat([sos, x_teacher[:, :-1]], dim=1)
            out, _ = self.lstm(x_in, (h0, c0))
            return self.out_fc(out)
        else:
            outputs = []
            x_t    = torch.zeros(batch, 1, NUM_PITCHES, device=z.device)
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

    def reparameterize(self, mu, lv):
        return mu + torch.exp(0.5 * lv) * torch.randn_like(mu)

    def forward(self, x, gids):
        B, T, _ = x.shape
        g = self.genre_emb(gids).unsqueeze(1).expand(-1, T, -1)
        mu, lv = self.encoder(torch.cat([x, g], dim=-1))
        z      = self.reparameterize(mu, lv)
        x_hat  = self.decoder(z, x_teacher=x)   # teacher forcing
        return mu, lv, z, x_hat

    @staticmethod
    def kl(mu, lv):
        return -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp()) / mu.size(0)

print('VAE defined.')"""
))

cells.append(code(
"""vae_model = VAE().to(device)
vae_opt   = torch.optim.Adam(vae_model.parameters(), lr=LR_VAE)
vae_recon = nn.BCEWithLogitsLoss(pos_weight=POS_W)

vae_train_h, vae_val_h = [], []
vae_recon_h, vae_kl_h  = [], []

for epoch in range(1, EPOCHS_VAE + 1):
    beta = min(1.0, (epoch - 1) / WARMUP_KL)

    vae_model.train()
    t_tot = t_rec = t_kl = 0.0
    for x, g in tqdm(train_dl, desc=f'VAE {epoch}/{EPOCHS_VAE}', leave=False):
        x, g = x.to(device), g.to(device)
        vae_opt.zero_grad()
        mu, lv, z, x_hat = vae_model(x, g)
        rec  = vae_recon(x_hat, x)
        kl   = VAE.kl(mu, lv)
        loss = rec + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 1.0)
        vae_opt.step()
        t_tot += loss.item(); t_rec += rec.item(); t_kl += kl.item()

    vae_model.eval()
    v_tot = 0.0
    with torch.no_grad():
        for x, g in test_dl:
            x, g = x.to(device), g.to(device)
            mu, lv, z, x_hat = vae_model(x, g)
            v_tot += (vae_recon(x_hat, x) + beta * VAE.kl(mu, lv)).item()

    n = len(train_dl)
    vae_train_h.append(t_tot / n); vae_val_h.append(v_tot / len(test_dl))
    vae_recon_h.append(t_rec / n); vae_kl_h.append(t_kl / n)
    if epoch % 10 == 0 or epoch == 1:
        print(f'VAE {epoch:3d}  beta={beta:.2f}  Train={vae_train_h[-1]:.4f}'
              f'  Recon={vae_recon_h[-1]:.4f}  KL={vae_kl_h[-1]:.4f}'
              f'  Val={vae_val_h[-1]:.4f}')

torch.save(vae_model.state_dict(), f'{OUT_DIR}/vae.pth')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(vae_train_h, label='Train'); a1.plot(vae_val_h, label='Val', linestyle='--')
a1.set_title('Task 2 – VAE Total Loss'); a1.legend()
a2.plot(vae_recon_h, label='Reconstruction', color='steelblue')
a2.plot(vae_kl_h,    label='KL Divergence',  color='coral')
a2.axvline(WARMUP_KL, color='gray', linestyle='--', label=f'Warmup end (ep {WARMUP_KL})')
a2.set_title('Task 2 – Recon vs KL'); a2.legend()
plt.tight_layout(); plt.savefig(f'{PLOTS}/task2_loss.png', dpi=150); plt.show()
print('Task 2 model + plots saved.')"""
))

cells.append(code(
"""# --- Task 2 Generation ---
def generate_long_vae(model, z, total_steps=256, threshold=GEN_THRESHOLD):
    h0, c0 = model.decoder._init_hidden(z)
    outputs = []
    x_t    = torch.zeros(1, 1, NUM_PITCHES, device=z.device)
    hidden = (h0, c0)
    model.eval()
    with torch.no_grad():
        for _ in range(total_steps):
            out, hidden = model.decoder.lstm(x_t, hidden)
            logit       = model.decoder.out_fc(out)
            step        = (torch.sigmoid(logit) > threshold).float()
            outputs.append(step.squeeze().cpu().numpy())  # (88,)
            x_t = step.detach()
    return np.stack(outputs, axis=0)

genre_list = list(GENRES.items())
samples    = genre_list + genre_list[:4]   # 13 total

print('Generating 13 VAE samples (256 steps = 32 s each) ...')
for si, (name, gid) in enumerate(samples, 1):
    candidates = np.where(y_train == gid)[0]
    data_idx   = candidates[(si - 1) % len(candidates)]
    x_s = torch.FloatTensor(X_train[data_idx:data_idx+1]).to(device)
    g_s = torch.LongTensor([gid]).to(device)
    with torch.no_grad():
        mu, _, _, _ = vae_model(x_s, g_s)
    roll = generate_long_vae(vae_model, mu, total_steps=256)
    piano_roll_to_midi(roll, save_path=f'{MIDI_OUT}/task2/task2_{name}_{si}.mid')
    print(f'  Saved task2_{name}_{si}.mid  ({int(roll.sum())} active cells)')

# 8-point latent interpolation: Classical mu1 -> Jazz mu2
print('\\nLatent interpolation: Classical <-> Jazz (8 steps) ...')
cid = np.where(y_train == GENRES['Classical'])[0][0]
jid = np.where(y_train == GENRES['Jazz'])[0][0]
x1  = torch.FloatTensor(X_train[cid:cid+1]).to(device)
x2  = torch.FloatTensor(X_train[jid:jid+1]).to(device)
g1  = torch.LongTensor([GENRES['Classical']]).to(device)
g2  = torch.LongTensor([GENRES['Jazz']]).to(device)
with torch.no_grad():
    mu1, _, _, _ = vae_model(x1, g1)
    mu2, _, _, _ = vae_model(x2, g2)

for i in range(8):
    alpha    = i / 7
    z_interp = (1 - alpha) * mu1 + alpha * mu2
    roll     = generate_long_vae(vae_model, z_interp, total_steps=256)
    fname    = f'interp_{i+1}of8_a{alpha:.2f}_Classical_Jazz.mid'
    piano_roll_to_midi(roll, save_path=f'{MIDI_OUT}/task2/{fname}')
    print(f'  Step {i+1}/8  alpha={alpha:.3f}')"""
))

# ─────────────────────────────────────────────
# TASK 3 – Transformer
# ─────────────────────────────────────────────
cells.append(md(
"""---
## Task 3 – Transformer-Based Music Generator
**Architecture:** GPT-style decoder-only `TransformerEncoder` with causal mask + genre token prepended.
**Loss:** BCEWithLogitsLoss + CosineAnnealing LR.  **Metric:** Perplexity = `exp(avg_val_BCE)`.
**Generation:** sliding window of `WINDOW_SIZE` steps keeps the model in-distribution;
256 autoregressive steps = ~32 seconds per composition."""
))

cells.append(code(
"""class PositionalEncoding(nn.Module):
    def __init__(self, d=D_MODEL, max_len=512, drop=DROPOUT):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(max_len).float().unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.drop(x + self.pe[:, :x.size(1)])


class MusicTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj  = nn.Linear(NUM_PITCHES, D_MODEL)
        self.genre_embed = nn.Embedding(NUM_GENRES, D_MODEL)
        self.pos_enc     = PositionalEncoding()
        enc_layer = nn.TransformerEncoderLayer(
            d_model=D_MODEL, nhead=N_HEADS,
            dim_feedforward=D_MODEL * 4, dropout=DROPOUT, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=N_LAYERS)
        self.output_proj = nn.Linear(D_MODEL, NUM_PITCHES)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _causal_mask(self, n, dev):
        return torch.triu(torch.ones(n, n, device=dev), diagonal=1).bool()

    def forward(self, x, gids):
        xp  = self.input_proj(x)
        g   = self.genre_embed(gids).unsqueeze(1)
        xp  = torch.cat([g, xp], dim=1)
        xp  = self.pos_enc(xp)
        out = self.transformer(xp, mask=self._causal_mask(xp.size(1), x.device))
        return self.output_proj(out[:, 1:])   # raw logits, drop genre token

print('MusicTransformer defined.')"""
))

cells.append(code(
"""tr_model = MusicTransformer().to(device)
tr_opt   = torch.optim.Adam(tr_model.parameters(), lr=LR_TR)
tr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(tr_opt, EPOCHS_TR)
tr_crit  = nn.BCEWithLogitsLoss(pos_weight=POS_W)

tr_train_h, tr_ppl_h = [], []

for epoch in range(1, EPOCHS_TR + 1):
    tr_model.train()
    epoch_loss = 0.0
    for x, g in tqdm(train_dl, desc=f'TR {epoch}/{EPOCHS_TR}', leave=False):
        x, g  = x.to(device), g.to(device)
        x_in  = x[:, :-1]
        x_tgt = x[:, 1:]
        tr_opt.zero_grad()
        loss = tr_crit(tr_model(x_in, g), x_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tr_model.parameters(), 1.0)
        tr_opt.step()
        epoch_loss += loss.item()
    tr_sched.step()

    tr_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, g in test_dl:
            x, g = x.to(device), g.to(device)
            val_loss += tr_crit(tr_model(x[:, :-1], g), x[:, 1:]).item()

    avg_t = epoch_loss / len(train_dl)
    avg_v = val_loss   / len(test_dl)
    ppl   = math.exp(min(avg_v, 20))
    tr_train_h.append(avg_t); tr_ppl_h.append(ppl)
    if epoch % 10 == 0 or epoch == 1:
        print(f'TR {epoch:3d}  Train={avg_t:.4f}  Val PPL={ppl:.2f}')

torch.save(tr_model.state_dict(), f'{OUT_DIR}/transformer.pth')
print(f'\\nFinal perplexity: {tr_ppl_h[-1]:.2f}')

fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.plot(tr_train_h); a1.set_title('Train Loss'); a1.set_xlabel('Epoch')
a2.plot(tr_ppl_h, color='coral')
a2.axhline(15, color='green', linestyle='--', label='Target PPL = 15')
a2.set_title('Validation Perplexity'); a2.set_xlabel('Epoch'); a2.legend()
plt.tight_layout(); plt.savefig(f'{PLOTS}/task3_perplexity.png', dpi=150); plt.show()
print('Task 3 model + plot saved.')"""
))

cells.append(code(
"""# --- Task 3 Generation: sliding window keeps model in-distribution ---
def generate_sequence(genre_id, steps=256, seed_steps=8, temperature=0.7):
    tr_model.eval()
    g    = torch.LongTensor([genre_id]).to(device)
    seed = np.zeros((seed_steps, NUM_PITCHES), dtype=np.float32)
    for t in range(seed_steps):
        ps = np.random.choice(NUM_PITCHES, np.random.randint(2, 6), replace=False)
        seed[t, ps] = 1.0
    gen = list(seed)
    with torch.no_grad():
        for _ in range(steps - seed_steps):
            # Use only the last WINDOW_SIZE steps so the model stays in-distribution
            ctx   = torch.FloatTensor(np.array(gen[-WINDOW_SIZE:])).unsqueeze(0).to(device)
            logit = tr_model(ctx, g)[0, -1]
            probs = torch.sigmoid(logit / temperature)
            step  = (probs > GEN_THRESHOLD).float().cpu().numpy()
            gen.append(step)
    return np.array(gen)

genre_list  = list(GENRES.items())
to_generate = (genre_list * 2)[:10]

print('Generating 10 Transformer compositions (256 steps = 32 s each) ...')
for i, (name, gid) in enumerate(to_generate, 1):
    roll = generate_sequence(gid)
    piano_roll_to_midi(roll, save_path=f'{MIDI_OUT}/task3/task3_{name}_{i}.mid')
    print(f'  [{i}/10] task3_{name}_{i}.mid  ({int(roll.sum())} active cells)')"""
))

# ─────────────────────────────────────────────
# Baselines & Evaluation
# ─────────────────────────────────────────────
cells.append(md(
"""---
## Baselines & Evaluation
Two required baselines: **Random Note Generator** and **Markov Chain**.
Metrics: Rhythm Diversity · Repetition Ratio · Pitch Histogram Similarity."""
))

cells.append(code(
"""# Random baseline
print('Generating random baseline ...')
for i in range(1, 6):
    roll = np.zeros((WINDOW_SIZE, NUM_PITCHES), dtype=np.float32)
    for t in range(WINDOW_SIZE):
        ps = np.random.choice(NUM_PITCHES, np.random.randint(1, 5), replace=False)
        roll[t, ps] = 1.0
    piano_roll_to_midi(roll, save_path=f'{MIDI_OUT}/baselines/random_{i}.mid')
print('  Done.')

# Markov baseline
def build_markov(X, top_k=32):
    T = {}
    for roll in X:
        for t in range(len(roll) - 1):
            k = tuple(np.where(roll[t] > 0.5)[0][:top_k])
            T.setdefault(k, []).append(roll[t + 1].copy())
    return T

print('Building Markov model ...')
transitions = build_markov(X_train[:1000])

print('Generating Markov baseline ...')
for i in range(1, 6):
    seed  = X_train[np.random.randint(len(X_train))][0]
    gen   = [seed]
    state = tuple(np.where(seed > 0.5)[0])
    for _ in range(WINDOW_SIZE - 1):
        if state in transitions:
            nxt = random.choice(transitions[state])
        else:
            nxt = random.choice(list(transitions.values()))[0]
        gen.append(nxt)
        state = tuple(np.where(nxt > 0.5)[0])
    piano_roll_to_midi(np.array(gen), save_path=f'{MIDI_OUT}/baselines/markov_{i}.mid')
print('  Done.')"""
))

cells.append(code(
"""# Evaluation metrics
def pitch_histogram(roll):
    h = np.zeros(12)
    for p in np.where(roll > 0.5)[1]:
        h[(p + MIDI_MIN) % 12] += 1
    return h / h.sum() if h.sum() > 0 else h

def rhythm_diversity(roll):
    durs = []
    b = (roll > 0.5).astype(np.uint8)
    for p in range(b.shape[1]):
        on = False; d = 0
        for t in range(b.shape[0]):
            if b[t, p]: on = True; d += 1
            elif on:    durs.append(d); d = 0; on = False
        if on: durs.append(d)
    return len(set(durs)) / len(durs) if durs else 0.0

def repetition_ratio(roll, pl=4):
    b    = (roll > 0.5).astype(np.uint8)
    pats = [tuple(b[s:s+pl].flatten()) for s in range(0, len(roll)-pl, pl)]
    return 1.0 - len(set(pats)) / len(pats) if pats else 0.0

def ph_similarity(roll_a, roll_b):
    return float(np.sum(np.abs(pitch_histogram(roll_a) - pitch_histogram(roll_b))))

def load_folder_rolls(folder, prefix=None, n=5):
    rolls = []
    files = sorted([f for f in os.listdir(folder) if f.endswith('.mid')])
    if prefix:
        files = [f for f in files if f.startswith(prefix)]
    for f in files[:n]:
        try:
            pm   = pretty_midi.PrettyMIDI(os.path.join(folder, f))
            roll = pm.get_piano_roll(fs=8)[MIDI_MIN:MIDI_MIN+NUM_PITCHES].T
            roll = (roll > 0).astype(np.float32)
            if len(roll) >= WINDOW_SIZE:
                rolls.append(roll[:WINDOW_SIZE])
        except Exception:
            pass
    return rolls

ref_rolls = load_folder_rolls(f'{MIDI_OUT}/task3')
ref_avg   = np.mean(ref_rolls, axis=0) if ref_rolls else None

tasks = [
    ('Random Generator',    f'{MIDI_OUT}/baselines', 'random'),
    ('Markov Chain',        f'{MIDI_OUT}/baselines', 'markov'),
    ('Task 1: LSTM AE',     f'{MIDI_OUT}/task1',     None),
    ('Task 2: VAE',         f'{MIDI_OUT}/task2',     None),
    ('Task 3: Transformer', f'{MIDI_OUT}/task3',     None),
]

print(f'\\n{"Model":<25} {"Rhythm Div":>12} {"Repetition":>12} {"Pitch Sim":>12}')
print('-' * 65)
results = {}
for label, folder, prefix in tasks:
    rolls = load_folder_rolls(folder, prefix)
    if not rolls:
        print(f'{label:<25}  no data'); continue
    rd  = np.mean([rhythm_diversity(r) for r in rolls])
    rep = np.mean([repetition_ratio(r) for r in rolls])
    ps  = np.mean([ph_similarity(r, ref_avg) for r in rolls]) if ref_avg is not None else None
    results[label] = dict(rhythm_diversity=rd, repetition=rep, pitch_sim=ps)
    ps_str = f'{ps:.3f}' if ps is not None else '  -'
    print(f'{label:<25} {rd:>12.3f} {rep:>12.3f} {ps_str:>12}')

with open(f'{OUT_DIR}/evaluation_table.txt', 'w') as fh:
    fh.write(f'{"Model":<25} {"Rhythm Div":>12} {"Repetition":>12} {"Pitch Sim":>12}\\n')
    fh.write('-' * 65 + '\\n')
    for label, m in results.items():
        ps_str = f'{m["pitch_sim"]:.3f}' if m.get('pitch_sim') is not None else '  -'
        fh.write(f'{label:<25} {m["rhythm_diversity"]:>12.3f} '
                 f'{m["repetition"]:>12.3f} {ps_str:>12}\\n')
print('\\nEvaluation table saved.')"""
))

# ─────────────────────────────────────────────
# Download
# ─────────────────────────────────────────────
cells.append(md("## Download Outputs\nRun the cell below, then click the file in the Kaggle **Output** panel to download."))

cells.append(code(
"""shutil.make_archive('/kaggle/working/music_generation_outputs', 'zip', OUT_DIR)
print('ZIP created: /kaggle/working/music_generation_outputs.zip')
print()
print('Contents:')
for root, dirs, files in os.walk(OUT_DIR):
    level = root.replace(OUT_DIR, '').count(os.sep)
    if level < 2:
        indent = '  ' * level
        print(f'{indent}{os.path.basename(root)}/')
        for f in files:
            print(f'{indent}  {f}')"""
))

# ─────────────────────────────────────────────
# Write .ipynb
# ─────────────────────────────────────────────
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.12"
        }
    },
    "cells": cells
}

out_path = "kaggle_train.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Created: {out_path}")
print(f"Cells  : {len(cells)}")
print()
print("Next steps:")
print("  1. Run:  python create_notebook.py")
print("  2. Upload kaggle_train.ipynb to Kaggle")
print("  3. Add your preprocessed .npy dataset, set NPY_SLUG in Config cell")
print("  4. Enable GPU, Run All")
print("  5. Download outputs/autoencoder.pth, vae.pth, transformer.pth")
print("  6. Copy them to local outputs/models/")
print("  7. Run generation scripts locally")
