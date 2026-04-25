"""
Run this script once to build the train/test dataset from MIDI files.
Execute from the project root: python run_preprocessing.py
"""
import sys
import os
sys.path.append('src')

import numpy as np
from sklearn.model_selection import train_test_split
from config import RAW_MIDI_DIR, SPLIT_DIR, GENRES, PLOTS_DIR
from preprocessing.piano_roll import build_dataset, visualize_piano_roll

os.makedirs(SPLIT_DIR,  exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# Map genre names to their raw MIDI folders
genre_dirs = {
    genre: os.path.join(RAW_MIDI_DIR, genre)
    for genre in GENRES.keys()
}

# Verify folders exist
for genre, folder in genre_dirs.items():
    if not os.path.isdir(folder):
        print(f"WARNING: folder not found: {folder}")

print("Building dataset from all genres...")
X, y = build_dataset(genre_dirs)

if len(X) == 0:
    print("ERROR: No segments found. Check that MIDI files exist in data/raw_midi/")
    sys.exit(1)

# 80/20 train-test split, stratified by genre
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

np.save(os.path.join(SPLIT_DIR, 'train_data.npy'),   X_train)
np.save(os.path.join(SPLIT_DIR, 'test_data.npy'),    X_test)
np.save(os.path.join(SPLIT_DIR, 'train_labels.npy'), y_train)
np.save(os.path.join(SPLIT_DIR, 'test_labels.npy'),  y_test)

print(f"\nDataset saved:")
print(f"  train: {X_train.shape}  labels: {y_train.shape}")
print(f"  test:  {X_test.shape}   labels: {y_test.shape}")

# Genre distribution report
print("\nGenre distribution in training set:")
genre_id_to_name = {v: k for k, v in GENRES.items()}
for gid in sorted(set(y_train)):
    count = (y_train == gid).sum()
    print(f"  {genre_id_to_name[gid]:<15}: {count} segments")

# Save a sample piano roll plot for the report
visualize_piano_roll(
    X_train[0],
    title="Sample Piano Roll - Training Data",
    save_path=os.path.join(PLOTS_DIR, 'sample_piano_roll.png'),
)
print(f"\nSample piano roll saved to outputs/plots/sample_piano_roll.png")
