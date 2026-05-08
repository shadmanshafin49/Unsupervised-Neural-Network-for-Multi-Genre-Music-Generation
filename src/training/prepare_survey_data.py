"""
Parse human survey CSV and generate matching piano rolls for reward model training.

Survey samples 1-6 correspond to task3 generations in alphabetical genre order:
  1=Blues, 2=Classical, 3=Country, 4=Electronic, 5=Folk, 6=Hip-Hop
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from config import OUTPUTS_DIR, MODELS_DIR, WINDOW_SIZE, NUM_PITCHES, GENRES
from models.transformer import MusicTransformer
from generation.generate_transformer import generate_sequence

SURVEY_CSV = os.path.join(
    OUTPUTS_DIR, 'survey_results',
    "Music Quality Survey (Responses) - Form Responses 1.csv"
)
OUT_DIR = os.path.join(OUTPUTS_DIR, 'survey_results')
os.makedirs(OUT_DIR, exist_ok=True)

# ── 1. Parse CSV ────────────────────────────────────────────────────────────
raw = pd.read_csv(SURVEY_CSV)

# The 6 rating columns are everything after Timestamp
rating_cols = raw.columns[1:]  # columns 1-6

rows = []
for _, row in raw.iterrows():
    for sample_idx, col in enumerate(rating_cols, start=1):
        rows.append({'sample_id': sample_idx, 'musicality': int(row[col])})

scores_df = pd.DataFrame(rows)
scores_path = os.path.join(OUT_DIR, 'pre_rl_scores.csv')
scores_df.to_csv(scores_path, index=False)
print(f"Saved {len(scores_df)} rating rows -> {scores_path}")

avg = scores_df.groupby('sample_id')['musicality'].mean()
print("Average scores per sample:")
for sid, mean_score in avg.items():
    print(f"  Sample {sid}: {mean_score:.2f}")

# ── 2. Generate piano rolls for samples 1-6 ─────────────────────────────────
# Samples 1-6 map to the first 6 genres in alphabetical order
genre_list = sorted(GENRES.items(), key=lambda x: x[0])  # alphabetical
sample_genres = genre_list[:6]  # Blues, Classical, Country, Electronic, Folk, Hip-Hop

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MusicTransformer().to(device)
model.load_state_dict(
    torch.load(os.path.join(MODELS_DIR, 'transformer.pth'), map_location=device))
model.eval()
print(f"\nLoaded transformer")

rolls = []
for genre_name, genre_id in sample_genres:
    roll = generate_sequence(genre_id, model, device, seed_steps=8, total_steps=256)
    # Take the first WINDOW_SIZE steps for the reward model input
    window = roll[:WINDOW_SIZE]
    if window.shape[0] < WINDOW_SIZE:
        pad = np.zeros((WINDOW_SIZE - window.shape[0], NUM_PITCHES), dtype=np.float32)
        window = np.vstack([window, pad])
    rolls.append(window)
    print(f"  Generated roll for sample ({genre_name}): shape {window.shape}")

rolls_arr = np.array(rolls, dtype=np.float32)  # (6, WINDOW_SIZE, NUM_PITCHES)
rolls_path = os.path.join(OUT_DIR, 'pre_rl_rolls.npy')
np.save(rolls_path, rolls_arr)
print(f"\nSaved piano rolls shape={rolls_arr.shape}")
print("Done. Ready to run train_reward_model.py")
