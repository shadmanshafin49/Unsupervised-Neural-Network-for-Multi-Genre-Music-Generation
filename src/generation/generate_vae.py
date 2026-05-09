"""Run this after training the VAE (train_vae.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import OUTPUTS_DIR, MODELS_DIR, SPLIT_DIR, NUM_PITCHES, GENERATION_THRESHOLD, GENRES
from models.vae import VAE
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = VAE().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, 'vae.pth'), map_location=device))
model.eval()

X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
y_train = np.load(os.path.join(SPLIT_DIR, 'train_labels.npy'))

genre_list = list(GENRES.items())


def generate_long_vae(model, z, total_steps=256, threshold=GENERATION_THRESHOLD, device='cpu'):
    """Autoregressively decode z for total_steps steps."""
    h0, c0 = model.decoder._init_hidden(z)
    outputs = []
    x_t     = torch.zeros(1, 1, NUM_PITCHES, device=device)
    hidden  = (h0, c0)
    with torch.no_grad():
        for _ in range(total_steps):
            out, hidden = model.decoder.lstm(x_t, hidden)
            logit       = model.decoder.out_fc(out)
            probs       = torch.sigmoid(logit)
            step_out    = (probs > threshold).float()
            outputs.append(step_out.squeeze().cpu().numpy())  # (88,)
            x_t = step_out.detach()
    return np.stack(outputs, axis=0)


# One sample per genre + 4 extras (13 total), each 256 steps = 32s
print("Generating VAE samples (256 steps = 32s each)...")
samples = genre_list + genre_list[:4]

for sample_idx, (genre_name, genre_id) in enumerate(samples, 1):
    candidates = np.where(y_train == genre_id)[0]
    data_idx   = candidates[(sample_idx - 1) % len(candidates)]
    x_sample   = torch.FloatTensor(X_train[data_idx:data_idx+1]).to(device)
    g_tensor   = torch.LongTensor([genre_id]).to(device)
    with torch.no_grad():
        mu, _, _, _ = model(x_sample, g_tensor)

    roll = generate_long_vae(model, mu, total_steps=256, device=device)
    path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2',
                        f'task2_{genre_name}_{sample_idx}.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  Saved: task2_{genre_name}_{sample_idx}.mid  ({int(roll.sum())} active cells)")

# Latent interpolation: Classical <-> Jazz, 8 steps, 256 steps each
print("\nRunning latent interpolation: Classical <-> Jazz (8 steps)...")

classical_idx = np.where(y_train == GENRES["Classical"])[0][0]
jazz_idx      = np.where(y_train == GENRES["Jazz"])[0][0]

x_classical  = torch.FloatTensor(X_train[classical_idx:classical_idx+1]).to(device)
x_jazz       = torch.FloatTensor(X_train[jazz_idx:jazz_idx+1]).to(device)
g_classical  = torch.LongTensor([GENRES["Classical"]]).to(device)
g_jazz       = torch.LongTensor([GENRES["Jazz"]]).to(device)

with torch.no_grad():
    mu1, _, _, _ = model(x_classical, g_classical)
    mu2, _, _, _ = model(x_jazz,      g_jazz)

for i in range(8):
    alpha    = i / 7
    z_interp = (1 - alpha) * mu1 + alpha * mu2
    roll     = generate_long_vae(model, z_interp, total_steps=256, device=device)
    path     = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2',
                            f'interp_{i+1}of8_alpha{alpha:.2f}_Classical_Jazz.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  Interpolation {i+1}/8 (alpha={alpha:.3f}): saved")

print("\nDone! Check outputs/generated_midis/task2/")
