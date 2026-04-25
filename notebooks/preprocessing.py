"""
Preprocessing notebook / script.
Run from the project root: python notebooks/preprocessing.py
Or open as a Jupyter notebook by converting: jupyter nbconvert --to notebook preprocessing.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import RAW_MIDI_DIR, SPLIT_DIR, GENRES, PLOTS_DIR
from preprocessing.piano_roll import build_dataset, visualize_piano_roll

os.makedirs(SPLIT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Build dataset ───────────────────────────────────────────────────────────
genre_dirs = {genre: os.path.join(RAW_MIDI_DIR, genre) for genre in GENRES.keys()}

print("Building dataset...")
X, y = build_dataset(genre_dirs)
print(f"Total: {len(X)} segments of shape {X[0].shape}")

# ── 2. Train/test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape} | Test: {X_test.shape}")

np.save(os.path.join(SPLIT_DIR, 'train_data.npy'),   X_train)
np.save(os.path.join(SPLIT_DIR, 'test_data.npy'),    X_test)
np.save(os.path.join(SPLIT_DIR, 'train_labels.npy'), y_train)
np.save(os.path.join(SPLIT_DIR, 'test_labels.npy'),  y_test)
print("Saved to data/train_test_split/")

# ── 3. Genre distribution bar chart ───────────────────────────────────────────
genre_id_to_name = {v: k for k, v in GENRES.items()}
genre_counts     = {genre_id_to_name[i]: int((y_train == i).sum())
                    for i in sorted(set(y_train))}

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(genre_counts.keys(), genre_counts.values(), color='steelblue')
ax.set_xlabel("Genre")
ax.set_ylabel("Training Segments")
ax.set_title("Genre Distribution in Training Set")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'genre_distribution.png'), dpi=150)
plt.show()
print("Genre distribution plot saved.")

# ── 4. Visualize sample piano rolls from each genre ───────────────────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 9))
axes = axes.flatten()
for i, (genre_name, genre_id) in enumerate(GENRES.items()):
    idx  = np.where(y_train == genre_id)[0]
    if len(idx) == 0:
        continue
    roll = X_train[idx[0]]
    axes[i].imshow(roll.T, aspect='auto', origin='lower', cmap='Blues',
                   interpolation='nearest')
    axes[i].set_title(genre_name)
    axes[i].set_xlabel("Time Steps")
    axes[i].set_ylabel("Pitch")
plt.suptitle("Sample Piano Rolls by Genre")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'sample_piano_roll.png'), dpi=150)
plt.show()
print("Sample piano roll grid saved to outputs/plots/sample_piano_roll.png")

# ── 5. Statistics ─────────────────────────────────────────────────────────────
print("\nDataset Statistics:")
print(f"  Piano roll shape:  {X_train.shape[1]} steps x {X_train.shape[2]} pitches")
print(f"  Sparsity (train):  {1.0 - X_train.mean():.3f}")
print(f"  Unique genres:     {len(GENRES)}")
print(f"  Total segments:    {len(X)}")
