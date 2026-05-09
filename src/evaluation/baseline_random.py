import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OUTPUTS_DIR, NUM_PITCHES, WINDOW_SIZE
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'baselines'), exist_ok=True)


def generate_random(steps=WINDOW_SIZE, notes_per_step=4):
    """Generate a piano roll where each step has random notes."""
    roll = np.zeros((steps, NUM_PITCHES), dtype=np.float32)
    for t in range(steps):
        n_notes = np.random.randint(1, notes_per_step + 1)
        pitches = np.random.choice(NUM_PITCHES, size=n_notes, replace=False)
        roll[t, pitches] = 1.0
    return roll


if __name__ == "__main__":
    for i in range(1, 6):
        roll = generate_random()
        piano_roll_to_midi(roll, save_path=os.path.join(
            OUTPUTS_DIR, 'generated_midis', 'baselines', f'random_{i}.mid'))
        print(f"Random baseline {i} saved.")
