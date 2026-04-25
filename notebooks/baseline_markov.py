"""
Markov baseline notebook / script.
Run from the project root: python notebooks/baseline_markov.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from config import SPLIT_DIR, OUTPUTS_DIR, PLOTS_DIR
from evaluation.baseline_markov import build_markov_model, markov_generate
from evaluation.metrics import evaluate_all, print_metrics_table
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'baselines'), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── 1. Load training data ──────────────────────────────────────────────────────
print("Loading training data...")
X_train = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
y_train = np.load(os.path.join(SPLIT_DIR, 'train_labels.npy'))
print(f"  Loaded {len(X_train)} training segments.")

# ── 2. Build Markov model ──────────────────────────────────────────────────────
print("Building Markov transition model...")
transitions = build_markov_model(X_train)
print(f"  Unique states: {len(transitions)}")

# ── 3. Generate 5 samples ─────────────────────────────────────────────────────
print("Generating 5 Markov samples...")
markov_rolls = []
for i in range(1, 6):
    seed = X_train[np.random.randint(len(X_train))][0]
    roll = markov_generate(transitions, seed)
    markov_rolls.append(roll)
    path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'baselines', f'markov_{i}.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  markov_{i}.mid saved")

# ── 4. Visualize ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 4))
for i, ax in enumerate(axes):
    ax.imshow(markov_rolls[i].T, aspect='auto', origin='lower',
              cmap='Blues', interpolation='nearest')
    ax.set_title(f"Markov Sample {i+1}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Pitch")
plt.suptitle("Markov Chain Baseline - Generated Piano Rolls")
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'markov_samples.png'), dpi=150)
plt.show()

# ── 5. Quick metrics ──────────────────────────────────────────────────────────
metrics_list = [evaluate_all(r) for r in markov_rolls]
avg_rhythm   = float(np.mean([m['rhythm_diversity'] for m in metrics_list]))
avg_rep      = float(np.mean([m['repetition_ratio'] for m in metrics_list]))
print(f"\nMarkov Baseline Metrics (avg over 5 samples):")
print(f"  Rhythm Diversity: {avg_rhythm:.3f}")
print(f"  Repetition Ratio: {avg_rep:.3f}")
