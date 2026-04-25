import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, MIDI_MIN


def pitch_histogram(roll):
    """
    Compute 12-bin pitch class histogram (C, C#, D, ..., B).
    roll: (steps, 88)
    Returns: np.ndarray of shape (12,), normalized to sum=1
    """
    hist   = np.zeros(12)
    active = np.where(roll > 0.5)
    for pitch_idx in active[1]:
        midi_pitch  = pitch_idx + MIDI_MIN
        pitch_class = midi_pitch % 12
        hist[pitch_class] += 1
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def pitch_histogram_similarity(roll_a, roll_b):
    """L1 distance between pitch class histograms. Lower = more similar."""
    return float(np.sum(np.abs(pitch_histogram(roll_a) - pitch_histogram(roll_b))))


def rhythm_diversity_score(roll, threshold=0.5):
    """
    Ratio of unique note durations to total notes.
    Higher = more rhythmically diverse.
    """
    durations = []
    binary    = (roll > threshold).astype(np.uint8)

    for pitch_idx in range(binary.shape[1]):
        in_note = False
        dur     = 0
        for t in range(binary.shape[0]):
            if binary[t, pitch_idx]:
                in_note = True
                dur    += 1
            else:
                if in_note:
                    durations.append(dur)
                    dur     = 0
                    in_note = False
        if in_note:
            durations.append(dur)

    if not durations:
        return 0.0
    return len(set(durations)) / len(durations)


def repetition_ratio(roll, threshold=0.5, pattern_len=4):
    """
    Ratio of repeated patterns to total patterns.
    Lower = less repetitive = better.
    """
    binary   = (roll > threshold).astype(np.uint8)
    patterns = []
    for start in range(0, len(roll) - pattern_len, pattern_len):
        pat = tuple(binary[start:start + pattern_len].flatten())
        patterns.append(pat)
    if not patterns:
        return 0.0
    return 1.0 - (len(set(patterns)) / len(patterns))


def evaluate_all(roll, reference_roll=None):
    results = {
        'rhythm_diversity': rhythm_diversity_score(roll),
        'repetition_ratio': repetition_ratio(roll),
    }
    if reference_roll is not None:
        results['pitch_histogram_similarity'] = pitch_histogram_similarity(
            roll, reference_roll)
    return results


def print_metrics_table(model_results):
    print(f"\n{'Model':<30} {'Rhythm Div':>12} {'Repetition':>12} {'Pitch Sim':>12}")
    print("-" * 68)
    for model_name, metrics in model_results.items():
        rd  = f"{metrics.get('rhythm_diversity', 0):.3f}"
        rep = f"{metrics.get('repetition_ratio', 0):.3f}"
        ps  = metrics.get('pitch_histogram_similarity')
        ps_str = f"{ps:.3f}" if ps is not None else "    -"
        print(f"{model_name:<30} {rd:>12} {rep:>12} {ps_str:>12}")
