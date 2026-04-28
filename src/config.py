import os

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR     = os.path.join(BASE_DIR, "data")
RAW_MIDI_DIR = os.path.join(DATA_DIR, "raw_midi")
SPLIT_DIR    = os.path.join(DATA_DIR, "train_test_split")
OUTPUTS_DIR  = os.path.join(BASE_DIR, "outputs")
PLOTS_DIR    = os.path.join(OUTPUTS_DIR, "plots")
MIDI_OUT_DIR = os.path.join(OUTPUTS_DIR, "generated_midis")

GENRES = {
    "Blues":       0,
    "Classical":   1,
    "Country":     2,
    "Electronic":  3,
    "Folk":        4,
    "Hip-Hop":     5,
    "Jazz":        6,
    "Pop":         7,
    "Rock":        8,
}
NUM_GENRES = len(GENRES)  # 9

NUM_PITCHES   = 88
MIDI_MIN      = 21
STEPS_PER_BAR = 16
WINDOW_SIZE   = 64
STEP_DURATION = 0.125

LATENT_DIM      = 64
HIDDEN_DIM      = 256
GENRE_EMBED_DIM = 16
D_MODEL         = 256
N_HEADS         = 8
N_LAYERS        = 4
DROPOUT         = 0.1

BATCH_SIZE           = 32
LR_AE                = 1e-3
LR_VAE               = 1e-3
LR_TRANSFORMER       = 5e-4
LR_RLHF              = 1e-4
EPOCHS_AE            = 50
EPOCHS_VAE           = 50
EPOCHS_TRANSFORMER   = 80
RL_STEPS             = 200
BETA_KL              = 0.001

GENERATION_THRESHOLD = 0.5
TEMPERATURE          = 1.0
