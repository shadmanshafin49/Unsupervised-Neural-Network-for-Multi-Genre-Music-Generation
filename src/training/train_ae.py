import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import SPLIT_DIR, OUTPUTS_DIR, PLOTS_DIR, BATCH_SIZE, LR_AE, EPOCHS_AE
from models.autoencoder import Autoencoder

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
X_test  = np.load(os.path.join(SPLIT_DIR, 'test_data.npy'))

train_dl = DataLoader(TensorDataset(torch.FloatTensor(X_train)),
                      batch_size=BATCH_SIZE, shuffle=True)
test_dl  = DataLoader(TensorDataset(torch.FloatTensor(X_test)),
                      batch_size=BATCH_SIZE, shuffle=False)

model     = Autoencoder().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_AE)
criterion = nn.MSELoss()

train_losses, test_losses = [], []

for epoch in range(1, EPOCHS_AE + 1):
    model.train()
    epoch_loss = 0.0
    for (x,) in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS_AE}", leave=False):
        x = x.to(device)
        optimizer.zero_grad()
        _, x_hat = model(x)
        loss = criterion(x_hat, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for (x,) in test_dl:
            x = x.to(device)
            _, x_hat = model(x)
            val_loss += criterion(x_hat, x).item()

    avg_train = epoch_loss / len(train_dl)
    avg_val   = val_loss   / len(test_dl)
    train_losses.append(avg_train)
    test_losses.append(avg_val)
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")

torch.save(model.state_dict(), os.path.join(OUTPUTS_DIR, 'autoencoder.pth'))
print("Model saved to outputs/autoencoder.pth")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='steelblue')
plt.plot(test_losses,  label='Val Loss',   color='coral')
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Task 1 - Autoencoder Reconstruction Loss")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'task1_loss.png'), dpi=150)
plt.show()
print("Loss curve saved to outputs/plots/task1_loss.png")
