"""
Standalone rhythm analysis utilities (also available via metrics.py).
"""
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import STEPS_PER_BAR


def note_density(roll, threshold=0.5):
    """Average number of simultaneously active pitches per time step."""
    binary = (roll > threshold).astype(np.uint8)
    return float(binary.sum(axis=1).mean())


def syncopation_score(roll, threshold=0.5):
    """
    Fraction of notes that start on weak beats (odd 16th-note positions).
    Higher = more syncopated.
    """
    binary = (roll > threshold).astype(np.uint8)
    weak_beats = [i for i in range(binary.shape[0]) if (i % STEPS_PER_BAR) % 2 != 0]
    if not weak_beats:
        return 0.0

    note_onsets = 0
    weak_onsets = 0
    for pitch_idx in range(binary.shape[1]):
        prev = 0
        for t in range(binary.shape[0]):
            cur = binary[t, pitch_idx]
            if cur == 1 and prev == 0:
                note_onsets += 1
                if t in weak_beats:
                    weak_onsets += 1
            prev = cur

    if note_onsets == 0:
        return 0.0
    return weak_onsets / note_onsets


def polyphony_rate(roll, threshold=0.5, min_voices=2):
    """Fraction of time steps with at least min_voices simultaneously active."""
    binary = (roll > threshold).astype(np.uint8)
    counts = binary.sum(axis=1)
    return float((counts >= min_voices).mean())


def rhythm_summary(roll, threshold=0.5):
    """Return dict of all rhythm metrics for a piano roll."""
    return {
        'note_density':     note_density(roll, threshold),
        'syncopation':      syncopation_score(roll, threshold),
        'polyphony_rate':   polyphony_rate(roll, threshold),
    }
