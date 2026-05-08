import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from config import (SPLIT_DIR, OUTPUTS_DIR, MODELS_DIR, PLOTS_DIR, BATCH_SIZE,
                    LR_TRANSFORMER, EPOCHS_TRANSFORMER)
from models.transformer import MusicTransformer

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

model     = MusicTransformer().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR_TRANSFORMER)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_TRANSFORMER)

# BCEWithLogitsLoss: numerically stable, expects raw logits (model no longer applies sigmoid)
pos_rate       = float(X_train.mean())
pos_weight_val = min((1.0 - pos_rate) / pos_rate, 30.0)
criterion      = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val]).to(device))


def compute_perplexity(loss_val):
    return math.exp(min(loss_val, 20))


train_losses, val_ppls = [], []

for epoch in range(1, EPOCHS_TRANSFORMER + 1):
    model.train()
    epoch_loss = 0.0
    for x, genres in tqdm(train_dl, desc=f"Epoch {epoch}/{EPOCHS_TRANSFORMER}", leave=False):
        x, genres = x.to(device), genres.to(device)
        x_in  = x[:, :-1, :]
        x_tgt = x[:, 1:,  :]
        optimizer.zero_grad()
        x_hat = model(x_in, genres)
        loss  = criterion(x_hat, x_tgt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x, genres in test_dl:
            x, genres = x.to(device), genres.to(device)
            x_in, x_tgt = x[:, :-1, :], x[:, 1:, :]
            x_hat = model(x_in, genres)
            val_loss += criterion(x_hat, x_tgt).item()

    avg_train = epoch_loss / len(train_dl)
    avg_val   = val_loss   / len(test_dl)
    ppl       = compute_perplexity(avg_val)
    train_losses.append(avg_train)
    val_ppls.append(ppl)
    print(f"Epoch {epoch:3d} | Train Loss: {avg_train:.4f} | Val PPL: {ppl:.2f}")

os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'transformer.pth'))
print("Model saved to outputs/models/transformer.pth")

import json
metrics_path = os.path.join(OUTPUTS_DIR, 'training_metrics.json')
metrics = {}
if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
final_ppl = val_ppls[-1]
final_bce = math.log(final_ppl)
metrics['transformer_val_ppl'] = round(final_ppl, 6)
metrics['transformer_val_bce'] = round(final_bce, 6)
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Saved transformer_val_ppl={final_ppl:.4f} -> {metrics_path}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses)
ax1.set_title("Train Loss")
ax1.set_xlabel("Epoch")
ax2.plot(val_ppls, color='coral')
ax2.set_title("Validation Perplexity")
ax2.set_xlabel("Epoch")
ax2.axhline(y=15, color='green', linestyle='--', label='Target PPL=15')
ax2.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'task3_perplexity.png'), dpi=150)
plt.show()
print(f"Final Perplexity: {val_ppls[-1]:.2f}")
print("Saved: outputs/plots/task3_perplexity.png")
