"""
Run full evaluation: load generated MIDI files from each task folder,
compute metrics, and print a comparison table.
Execute from the project root: python run_evaluation.py
"""
import sys
import os
sys.path.append('src')

import numpy as np
from config import OUTPUTS_DIR
from evaluation.metrics import evaluate_all, print_metrics_table
from preprocessing.midi_parser import load_midi_as_piano_roll


def load_generated(task_folder, n=5):
    """Load first n generated MIDI files from a task subfolder as 64-step piano rolls."""
    folder = os.path.join(OUTPUTS_DIR, 'generated_midis', task_folder)
    if not os.path.isdir(folder):
        return []
    rolls = []
    for fname in sorted(os.listdir(folder))[:n]:
        if not fname.endswith('.mid'):
            continue
        roll = load_midi_as_piano_roll(os.path.join(folder, fname))
        if roll is not None and len(roll) >= 64:
            rolls.append(roll[:64])
    return rolls


# Use Task 3 (Transformer) output as the reference pitch distribution
reference_rolls = load_generated('task3', n=5)
reference_avg   = np.mean(reference_rolls, axis=0) if reference_rolls else None

model_results = {}
tasks = [
    ('baselines', 'Random Generator',    'random'),
    ('baselines', 'Markov Chain',        'markov'),
    ('task1',     'Task 1: LSTM AE',     None),
    ('task2',     'Task 2: VAE',         None),
    ('task3',     'Task 3: Transformer', None),
    ('task4',     'Task 4: RLHF',        None),
]

for folder, model_name, prefix in tasks:
    rolls = load_generated(folder, n=5)
    if prefix:
        # Filter to only the relevant baseline files
        folder_path = os.path.join(OUTPUTS_DIR, 'generated_midis', folder)
        if os.path.isdir(folder_path):
            all_files = [f for f in sorted(os.listdir(folder_path))
                         if f.startswith(prefix) and f.endswith('.mid')]
            rolls = []
            for fname in all_files[:5]:
                roll = load_midi_as_piano_roll(os.path.join(folder_path, fname))
                if roll is not None and len(roll) >= 64:
                    rolls.append(roll[:64])

    if not rolls:
        print(f"No rolls found for {model_name}, skipping.")
        continue

    metrics_list = [evaluate_all(r, reference_avg) for r in rolls]
    avg = {k: float(np.mean([m[k] for m in metrics_list if k in m]))
           for k in metrics_list[0]}
    model_results[model_name] = avg

print_metrics_table(model_results)
