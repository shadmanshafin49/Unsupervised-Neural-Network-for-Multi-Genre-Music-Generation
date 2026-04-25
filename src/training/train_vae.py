import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SPLIT_DIR, OUTPUTS_DIR, PLOTS_DIR, BATCH_SIZE, LR_VAE, EPOCHS_VAE, BETA_KL
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

model         = VAE().to(device)
optimizer     = torch.optim.Adam(model.parameters(), lr=LR_VAE)
recon_loss_fn = nn.MSELoss(reduction='mean')

train_losses, test_losses = [], []

for epoch in range(1, EPOCHS_VAE + 1):
    model.train()
    epoch_loss = 0.0
    for x, genres in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS_VAE}", leave=False):
        x, genres = x.to(device), genres.to(device)
        optimizer.zero_grad()
        mu, log_var, z, x_hat = model(x, genres)
        recon = recon_loss_fn(x_hat, x)
        kl    = VAE.kl_loss(mu, log_var)
        loss  = recon + BETA_KL * kl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, genres in test_dl:
            x, genres = x.to(device), genres.to(device)
            mu, log_var, z, x_hat = model(x, genres)
            recon = recon_loss_fn(x_hat, x)
            kl    = VAE.kl_loss(mu, log_var)
            val_loss += (recon + BETA_KL * kl).item()

    avg_train = epoch_loss / len(train_dl)
    avg_val   = val_loss   / len(test_dl)
    train_losses.append(avg_train)
    test_losses.append(avg_val)
    print(f"Epoch {epoch:3d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, 'vae.pth'))
print("Model saved to outputs/vae.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses,  label='Val Loss')
plt.title("Task 2 - VAE Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'task2_loss.png'), dpi=150)
plt.show()
print("Saved: outputs/plots/task2_loss.png")
