"""
Generates the full evaluation table.
- Loss/Perplexity from training plots (authoritative values)
- Rhythm Div, Repetition, Pitch Sim computed via src/evaluation/metrics.py
  on piano-roll arrays derived from the generated MIDI files
"""
import os, sys, math
import numpy as np
import pretty_midi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'src'))

from config import MIDI_MIN, WINDOW_SIZE
from evaluation.metrics import rhythm_diversity_score, repetition_ratio, pitch_histogram_similarity, pitch_histogram

MGO_DIR   = os.path.join(BASE_DIR, 'music_generation_outputs')
MIDI_ROOT = os.path.join(MGO_DIR, 'generated_midis')

# ── Loss / Perplexity from training plots ─────────────────────────────────────
# Task 1: final val BCE loss ~ 1.013  (task1_loss.png, epoch 50)
# Task 2: final val total   ~ 1.020  (task2_loss.png; KL collapsed to ~0)
# Task 3: final val BCE     ~ 0.22   (task3_perplexity.png) -> PPL = exp(0.22)
AE_LOSS  = 1.013
VAE_LOSS = 1.020
TR_PPL   = math.exp(0.22)

FS = 16   # frames per second used during preprocessing

# ── MIDI -> piano-roll ────────────────────────────────────────────────────────
def midi_to_roll(path, fs=FS):
    """Returns (T, 88) binary piano-roll or None on error."""
    try:
        pm   = pretty_midi.PrettyMIDI(path)
        roll = pm.get_piano_roll(fs=fs)          # (128, T)
        roll = roll[MIDI_MIN:MIDI_MIN + 88, :]   # (88,  T)
        roll = (roll > 0).astype(np.float32).T   # (T,   88)
        return roll if roll.sum() > 0 else None
    except Exception:
        return None


def avg_metrics_folder(folder, name_filter=None, ref_roll=None):
    """Compute averaged metrics for all .mid files in folder."""
    rds, reps, pss = [], [], []
    for f in sorted(os.listdir(folder)):
        if not f.endswith('.mid'):
            continue
        if name_filter and name_filter not in f:
            continue
        roll = midi_to_roll(os.path.join(folder, f))
        if roll is None:
            continue
        rds.append(rhythm_diversity_score(roll))
        reps.append(repetition_ratio(roll))
        if ref_roll is not None:
            pss.append(pitch_histogram_similarity(roll, ref_roll))

    if not rds:
        return 0.0, 0.0, float('nan')
    rd  = float(np.mean(rds))
    rep = float(np.mean(reps))
    ps  = float(np.mean(pss)) if pss else float('nan')
    return rd, rep, ps


# ── build a reference roll from all Task-1 files ──────────────────────────────
def build_ref_roll(folder):
    rolls = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.mid'):
            r = midi_to_roll(os.path.join(folder, f))
            if r is not None:
                rolls.append(r)
    if not rolls:
        return None
    return np.concatenate(rolls, axis=0)   # (Ttotal, 88)


print("Computing MIDI metrics from piano-rolls...")
t1_dir = os.path.join(MIDI_ROOT, 'task1')
t2_dir = os.path.join(MIDI_ROOT, 'task2')
t3_dir = os.path.join(MIDI_ROOT, 'task3')
bl_dir = os.path.join(MIDI_ROOT, 'baselines')

ref_roll = build_ref_roll(t1_dir)   # Task 1 AE output as reference

rd_rand,   rep_rand,   ps_rand   = avg_metrics_folder(bl_dir, 'random', ref_roll)
rd_markov, rep_markov, ps_markov = avg_metrics_folder(bl_dir, 'markov', ref_roll)
rd_ae,     rep_ae,     _         = avg_metrics_folder(t1_dir, None,     None)
rd_vae,    rep_vae,    ps_vae    = avg_metrics_folder(t2_dir, None,     ref_roll)
rd_tr,     rep_tr,     ps_tr     = avg_metrics_folder(t3_dir, None,     ref_roll)

# Task 1 pitch-sim vs itself = 0 by definition
ps_ae = 0.0

# ── assemble rows ─────────────────────────────────────────────────────────────
DASH = '--'
rows = [
    # name                  loss      ppl     rd        rep       ps        human   genre
    ("Random Generator",    DASH,     DASH,   rd_rand,   rep_rand,   ps_rand,   "N/A", "None"),
    ("Markov Chain",        DASH,     DASH,   rd_markov, rep_markov, ps_markov, "N/A", "Weak"),
    ("Task 1: LSTM AE",     AE_LOSS,  DASH,   rd_ae,     rep_ae,     ps_ae,     "N/A", "Single Genre"),
    ("Task 2: VAE",         VAE_LOSS, DASH,   rd_vae,    rep_vae,    ps_vae,    "N/A", "Moderate (9 genres)"),
    ("Task 3: Transformer", DASH,     TR_PPL, rd_tr,     rep_tr,     ps_tr,     "N/A", "Strong (9 genres)"),
]

# ── format & print ────────────────────────────────────────────────────────────
HDR = ["Model", "BCE Loss", "Perplexity", "Rhythm Div", "Repetition", "Pitch Sim", "Human Score", "Genre Control"]
W   = [28, 10, 11, 11, 11, 10, 12, 22]

def fmt(v, w):
    if isinstance(v, float):
        return f"{v:.4f}".ljust(w)
    return str(v).ljust(w)

sep     = '-' * (sum(W) + 2 * (len(W) - 1))
hdr_str = "  ".join(h.ljust(W[i]) for i, h in enumerate(HDR))

lines = [hdr_str, sep]
for name, loss, ppl, rd, rep, ps, hs, gc in rows:
    lines.append("  ".join([
        name.ljust(W[0]),
        fmt(loss, W[1]),
        fmt(ppl,  W[2]),
        fmt(rd,   W[3]),
        fmt(rep,  W[4]),
        fmt(ps,   W[5]),
        hs.ljust(W[6]),
        gc.ljust(W[7]),
    ]))
lines.append(sep)
lines += [
    "",
    "Notes:",
    "  BCE Loss    : final val BCEWithLogitsLoss from training plots",
    "                Task 1 epoch-50 val ~1.013 | Task 2 epoch-50 val ~1.020 (KL~0)",
    "  Perplexity  : exp(avg val BCE/frame); Task 3 final val BCE~0.22 -> PPL~1.25",
    "  Rhythm Div  : unique quantised durations / total notes  (higher = more variety)",
    "  Repetition  : 1 - (unique 4-step piano-roll patterns / total patterns)  (lower = better)",
    "  Pitch Sim   : L1 distance vs Task 1 AE pitch-class histogram  (lower = more similar to AE)",
    "  Human Score : NOT YET COLLECTED -- requires listening survey (1-5, >=10 participants)",
    "  Task 2 note : KL divergence collapsed to ~0 (posterior collapse visible in task2_loss.png)",
]

output = "\n".join(lines)
print(output)

out_path = os.path.join(MGO_DIR, 'evaluation_table_full.txt')
with open(out_path, 'w', encoding='utf-8') as fh:
    fh.write(output + "\n")
print(f"\nSaved -> {out_path}")
