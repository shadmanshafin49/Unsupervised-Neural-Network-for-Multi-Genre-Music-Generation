import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import WINDOW_SIZE, NUM_PITCHES, GENRES
from preprocessing.midi_parser import load_midi_as_piano_roll


def segment_piano_roll(roll, window_size=WINDOW_SIZE, hop=None):
    """
    Chop a piano roll into fixed-length windows.
    Returns: list of np.ndarray, each shape (window_size, NUM_PITCHES)
    """
    if hop is None:
        hop = window_size

    segments = []
    T = roll.shape[0]
    for start in range(0, T - window_size + 1, hop):
        seg = roll[start:start + window_size]
        if seg.sum() > 0:
            segments.append(seg)
    return segments


def build_dataset(genre_dirs, window_size=WINDOW_SIZE):
    """
    Build full dataset from genre folders.
    genre_dirs: dict {genre_name: folder_path}
    Returns: (X, y) where X shape=(N, window_size, NUM_PITCHES), y shape=(N,)
    """
    all_segments = []
    all_labels   = []

    for genre_name, folder in genre_dirs.items():
        label = GENRES[genre_name]
        midi_files = [
            f for f in os.listdir(folder)
            if f.lower().endswith('.mid') or f.lower().endswith('.midi')
        ]
        print(f"  Processing {genre_name}: {len(midi_files)} files")

        for fname in midi_files:
            fpath = os.path.join(folder, fname)
            roll = load_midi_as_piano_roll(fpath)
            if roll is None or roll.shape[0] < window_size:
                continue
            segs = segment_piano_roll(roll, window_size)
            all_segments.extend(segs)
            all_labels.extend([label] * len(segs))

    X = np.array(all_segments, dtype=np.float32)
    y = np.array(all_labels,   dtype=np.int64)
    print(f"  Total segments: {len(X)}, shape: {X.shape}")
    return X, y


def normalize_roll(roll):
    return np.clip(roll, 0.0, 1.0)


def visualize_piano_roll(roll, title="Piano Roll", save_path=None):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.imshow(roll.T, aspect='auto', origin='lower', cmap='Blues',
              interpolation='nearest')
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Pitch (Piano Key Index)")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
    plt.show()
