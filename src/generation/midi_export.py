import numpy as np
import pretty_midi
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import NUM_PITCHES, MIDI_MIN, STEP_DURATION, GENERATION_THRESHOLD


def piano_roll_to_midi(roll, threshold=GENERATION_THRESHOLD,
                       tempo=120.0, save_path=None):
    """
    Convert a piano roll array to a MIDI file.
    roll: np.ndarray of shape (time_steps, NUM_PITCHES), values in [0,1]
    Returns: pretty_midi.PrettyMIDI object
    """
    midi  = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    piano = pretty_midi.Instrument(program=0)

    binary_roll = (roll > threshold).astype(np.uint8)
    T, P        = binary_roll.shape

    for pitch_idx in range(P):
        pitch      = pitch_idx + MIDI_MIN
        in_note    = False
        note_start = 0.0

        for t in range(T):
            is_active = binary_roll[t, pitch_idx]
            if is_active and not in_note:
                note_start = t * STEP_DURATION
                in_note    = True
            elif not is_active and in_note:
                piano.notes.append(pretty_midi.Note(
                    velocity=80, pitch=pitch,
                    start=note_start, end=t * STEP_DURATION,
                ))
                in_note = False

        if in_note:
            piano.notes.append(pretty_midi.Note(
                velocity=80, pitch=pitch,
                start=note_start, end=T * STEP_DURATION,
            ))

    midi.instruments.append(piano)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        midi.write(save_path)
    return midi
