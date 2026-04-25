import pretty_midi
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MIDI_MIN, NUM_PITCHES, STEP_DURATION


def parse_midi(filepath):
    """
    Load a MIDI file and return a list of note events.
    Each event: (pitch_index, velocity_norm, start_step, end_step)
    pitch_index: 0-87 (relative to piano range starting at MIDI 21)
    """
    try:
        midi = pretty_midi.PrettyMIDI(filepath)
    except Exception:
        return None

    notes = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        for note in instrument.notes:
            pitch_idx = note.pitch - MIDI_MIN
            if pitch_idx < 0 or pitch_idx >= NUM_PITCHES:
                continue
            velocity_norm = note.velocity / 127.0
            start_step = int(note.start / STEP_DURATION)
            end_step   = int(note.end   / STEP_DURATION)
            if end_step <= start_step:
                end_step = start_step + 1
            notes.append((pitch_idx, velocity_norm, start_step, end_step))

    return notes if notes else None


def notes_to_piano_roll(notes, total_steps=None):
    """
    Convert note list to binary piano roll matrix.
    Returns: np.ndarray of shape (total_steps, NUM_PITCHES), dtype float32
    """
    if not notes:
        return None

    max_step = max(end for _, _, _, end in notes)
    if total_steps is None:
        total_steps = max_step

    roll = np.zeros((total_steps, NUM_PITCHES), dtype=np.float32)
    for pitch_idx, velocity, start, end in notes:
        end = min(end, total_steps)
        roll[start:end, pitch_idx] = 1.0

    return roll


def load_midi_as_piano_roll(filepath):
    """Full pipeline: MIDI file -> piano roll matrix."""
    notes = parse_midi(filepath)
    if notes is None:
        return None
    return notes_to_piano_roll(notes)
