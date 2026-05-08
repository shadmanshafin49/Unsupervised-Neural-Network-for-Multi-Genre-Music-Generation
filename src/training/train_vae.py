import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SPLIT_DIR, OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR, BATCH_SIZE, LR_VAE, EPOCHS_VAE
from models.vae import VAE

os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
y_train = np.load(os.path.join(SPLIT_DIR, 'train_labels.npy'))
X_test  = np.load(os.path.join(SPLIT_DIR, 'test_data.npy'))
y_test  = np.load(os.path.join(SPLIT_DIR, 'test_labels.npy'))

train_dl = DataLoader(
    TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
    batch_size=BATCH_SIZE, shuffle=True)
test_dl = DataLoader(
    TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test)),
    batch_size=BATCH_SIZE, shuffle=False)

model     = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_VAE)

# BCEWithLogitsLoss with pos_weight to handle piano-roll sparsity (~98% zeros)
pos_rate       = float(X_train.mean())
pos_weight_val = min((1.0 - pos_rate) / pos_rate, 30.0)
recon_loss_fn  = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))

WARMUP_EPOCHS = 10  # KL annealing: beta ramps from 0 → 1 over first 10 epochs

train_losses, test_losses = [], []
train_kl_losses, train_recon_losses = [], []

for epoch in range(1, EPOCHS_VAE + 1):
    # KL annealing schedule
    beta = min(1.0, (epoch - 1) / WARMUP_EPOCHS)

    model.train()
    epoch_loss = 0.0
    epoch_recon = 0.0
    epoch_kl = 0.0
    for x, genres in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS_VAE}", leave=False):
        x, genres = x.to(device), genres.to(device)
        optimizer.zero_grad()
        mu, log_var, z, x_hat = model(x, genres)
        recon = recon_loss_fn(x_hat, x)
        kl    = VAE.kl_loss(mu, log_var)
        loss  = recon + beta * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss  += loss.item()
        epoch_recon += recon.item()
        epoch_kl    += kl.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, genres in test_dl:
            x, genres = x.to(device), genres.to(device)
            mu, log_var, z, x_hat = model(x, genres)
            recon = recon_loss_fn(x_hat, x)
            kl    = VAE.kl_loss(mu, log_var)
            val_loss += (recon + beta * kl).item()

    avg_train = epoch_loss  / len(train_dl)
    avg_val   = val_loss    / len(test_dl)
    avg_recon = epoch_recon / len(train_dl)
    avg_kl    = epoch_kl    / len(train_dl)
    train_losses.append(avg_train)
    test_losses.append(avg_val)
    train_recon_losses.append(avg_recon)
    train_kl_losses.append(avg_kl)
    print(f"Epoch {epoch:3d} | beta={beta:.2f} | Train: {avg_train:.4f} "
          f"(Recon={avg_recon:.4f} KL={avg_kl:.4f}) | Val: {avg_val:.4f}")

os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'vae.pth'))
print("Model saved to outputs/models/vae.pth")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses,  label='Val Loss')
ax1.set_title("Task 2 - VAE Total Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(train_recon_losses, label='Reconstruction Loss', color='steelblue')
ax2.plot(train_kl_losses,    label='KL Divergence',       color='coral')
ax2.axvline(x=WARMUP_EPOCHS, color='gray', linestyle='--', label=f'Warmup end (ep {WARMUP_EPOCHS})')
ax2.set_title("Task 2 - Recon vs KL Loss")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()

plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'task2_loss.png'), dpi=150)
plt.show()
print("Saved: outputs/plots/task2_loss.png")
