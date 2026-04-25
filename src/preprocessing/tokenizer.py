"""
Minimal tokenizer stub — piano rolls are used directly as continuous tensors.
This module is reserved for future token-based (e.g. MIDI-like event) representations.
"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, GENERATION_THRESHOLD


def roll_to_tokens(roll, threshold=GENERATION_THRESHOLD):
    """Convert piano roll to a list of (time_step, pitch) active-note tokens."""
    binary = (roll > threshold).astype(np.uint8)
    tokens = []
    for t in range(binary.shape[0]):
        active = np.where(binary[t] > 0)[0]
        for pitch in active:
            tokens.append((t, int(pitch)))
    return tokens


def tokens_to_roll(tokens, total_steps, num_pitches=NUM_PITCHES):
    """Reconstruct a piano roll from (time_step, pitch) token list."""
    roll = np.zeros((total_steps, num_pitches), dtype=np.float32)
    for t, pitch in tokens:
        if 0 <= t < total_steps and 0 <= pitch < num_pitches:
            roll[t, pitch] = 1.0
    return roll
