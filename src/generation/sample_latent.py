"""Run this after training the autoencoder (train_ae.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import SPLIT_DIR, OUTPUTS_DIR, MODELS_DIR, NUM_PITCHES, GENERATION_THRESHOLD
from models.autoencoder import Autoencoder
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task1'), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Autoencoder().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, 'autoencoder.pth'), map_location=device))
model.eval()

X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))

# Pick 5 well-spread samples with enough active notes
note_counts  = (X_train > 0.5).sum(axis=(1, 2))
good_indices = np.where(note_counts > 20)[0]
step         = max(1, len(good_indices) // 5)
selected     = [good_indices[i * step] for i in range(5)]


def generate_long(model, z, total_steps=256, threshold=GENERATION_THRESHOLD, device='cpu'):
    """Autoregressively decode z for total_steps steps (any length)."""
    h0, c0 = model.decoder._init_hidden(z)
    outputs = []
    x_t     = torch.zeros(1, 1, NUM_PITCHES, device=device)
    hidden  = (h0, c0)
    with torch.no_grad():
        for _ in range(total_steps):
            out, hidden = model.decoder.lstm(x_t, hidden)
            logit       = model.decoder.output_fc(out)
            probs       = torch.sigmoid(logit)
            step_out    = (probs > threshold).float()
            outputs.append(step_out.squeeze().cpu().numpy())  # (88,)
            x_t = step_out.detach()
    return np.stack(outputs, axis=0)  # (total_steps, 88)


print("Generating 5 MIDI samples from Task 1 (LSTM Autoencoder, 256 steps = 32s)...")
for i, idx in enumerate(selected, 1):
    x = torch.FloatTensor(X_train[idx:idx+1]).to(device)
    with torch.no_grad():
        z, _ = model(x)

    roll      = generate_long(model, z, total_steps=256, device=device)
    save_path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task1',
                             f'task1_sample_{i}.mid')
    piano_roll_to_midi(roll, save_path=save_path)
    active    = int(roll.sum())
    print(f"  Saved: task1_sample_{i}.mid  ({active} active cells)")

print("Done! Check outputs/generated_midis/task1/")
