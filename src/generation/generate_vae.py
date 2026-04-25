"""Run this after training the VAE (train_vae.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import OUTPUTS_DIR, SPLIT_DIR, LATENT_DIM, GENRES
from models.vae import VAE
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VAE().to(device)
model.load_state_dict(
    torch.load(os.path.join(OUTPUTS_DIR, 'vae.pth'), map_location=device))
model.eval()

genre_list = list(GENRES.items())  # [(name, id), ...]

# Generate at least one sample per genre (9 genres -> 9 samples + extras)
print("Generating VAE samples (one per genre + extras)...")
samples = genre_list + genre_list[:4]  # 13 samples total

for idx, (genre_name, genre_id) in enumerate(samples, 1):
    z = torch.randn(1, LATENT_DIM).to(device)
    with torch.no_grad():
        x_hat = model.decoder(z)
    roll = x_hat.squeeze(0).cpu().numpy()
    path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2',
                        f'task2_{genre_name}_{idx}.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  Saved: task2_{genre_name}_{idx}.mid")

# Latent interpolation: Classical (1) <-> Jazz (6)
print("\nRunning latent interpolation: Classical <-> Jazz...")
X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
y_train = np.load(os.path.join(SPLIT_DIR, 'train_labels.npy'))

classical_idx = np.where(y_train == GENRES["Classical"])[0][0]
jazz_idx      = np.where(y_train == GENRES["Jazz"])[0][0]

x_classical = torch.FloatTensor(X_train[classical_idx:classical_idx+1]).to(device)
x_jazz      = torch.FloatTensor(X_train[jazz_idx:jazz_idx+1]).to(device)
g_classical  = torch.LongTensor([GENRES["Classical"]]).to(device)
g_jazz       = torch.LongTensor([GENRES["Jazz"]]).to(device)

with torch.no_grad():
    mu1, lv1, z1, _ = model(x_classical, g_classical)
    mu2, lv2, z2, _ = model(x_jazz,      g_jazz)

for alpha_pct in [0, 25, 50, 75, 100]:
    alpha    = alpha_pct / 100.0
    z_interp = (1 - alpha) * z1 + alpha * z2
    with torch.no_grad():
        x_hat = model.decoder(z_interp)
    roll = x_hat.squeeze(0).cpu().numpy()
    path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2',
                        f'interp_{alpha_pct}pct_Classical_Jazz.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  Interpolation {alpha_pct}%: saved")

print("\nDone! Check outputs/generated_midis/task2/")
