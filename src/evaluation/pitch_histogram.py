"""
Standalone pitch-class histogram utilities (also available via metrics.py).
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIDI_MIN

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F',
                 'F#', 'G', 'G#', 'A', 'A#', 'B']


def compute_histogram(roll, threshold=0.5):
    """Return a 12-bin normalized pitch class histogram from a piano roll."""
    hist   = np.zeros(12)
    binary = (roll > threshold).astype(np.uint8)
    active = np.argwhere(binary)
    for _, pitch_idx in active:
        hist[(pitch_idx + MIDI_MIN) % 12] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def plot_histogram(roll, title="Pitch Class Histogram", save_path=None):
    import matplotlib.pyplot as plt
    hist = compute_histogram(roll)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(PITCH_CLASSES, hist, color='steelblue')
    ax.set_xlabel("Pitch Class")
    ax.set_ylabel("Relative Frequency")
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compare_histograms(rolls_a, rolls_b, label_a="Model A", label_b="Model B",
                       save_path=None):
    """Plot side-by-side pitch histograms for two sets of rolls."""
    import matplotlib.pyplot as plt
    hist_a = np.mean([compute_histogram(r) for r in rolls_a], axis=0)
    hist_b = np.mean([compute_histogram(r) for r in rolls_b], axis=0)

    x     = np.arange(12)
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x - width/2, hist_a, width, label=label_a, color='steelblue')
    ax.bar(x + width/2, hist_b, width, label=label_b, color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(PITCH_CLASSES)
    ax.set_ylabel("Relative Frequency")
    ax.set_title("Pitch Class Distribution Comparison")
    ax.legend()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
