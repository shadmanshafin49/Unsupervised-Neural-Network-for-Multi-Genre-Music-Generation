"""Run this after training the autoencoder (train_ae.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import SPLIT_DIR, OUTPUTS_DIR, LATENT_DIM
from models.autoencoder import Autoencoder
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task1'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(
    torch.load(os.path.join(OUTPUTS_DIR, 'autoencoder.pth'), map_location=device))
model.eval()

X_train      = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
sample_batch = torch.FloatTensor(X_train[:200]).to(device)

with torch.no_grad():
    z_encoded, _ = model(sample_batch)

z_mean = z_encoded.mean(dim=0)
z_std  = z_encoded.std(dim=0)

print("Generating 5 MIDI samples from Task 1 (LSTM Autoencoder)...")
for i in range(1, 6):
    noise    = torch.randn(1, LATENT_DIM).to(device)
    z_sample = z_mean + z_std * noise

    with torch.no_grad():
        x_hat = model.decoder(z_sample)

    roll      = x_hat.squeeze(0).cpu().numpy()
    save_path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task1',
                             f'task1_sample_{i}.mid')
    piano_roll_to_midi(roll, save_path=save_path)
    print(f"  Saved: task1_sample_{i}.mid")

print("Done! Check outputs/generated_midis/task1/")
