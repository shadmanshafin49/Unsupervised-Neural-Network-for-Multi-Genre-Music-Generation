# Unsupervised Neural Network for Multi-Genre Music Generation

**Course:** CSE425
**Institution:** BRAC University

---

## Team Members

| Name | Student ID | Serial |
|------|-----------|--------|
| Taskia Maisha | 22201433 | 511 |
| Shadman Sakib | 21301566 | 402 |

---

## Project Overview

This project implements a pipeline for **unsupervised multi-genre music generation** using deep learning. Starting from raw MIDI files spanning 9 genres, we build progressively more sophisticated generative models — from a plain Autoencoder to a Transformer, with an optional RLHF-tuned variant — and compare them against random and Markov-chain baselines using quantitative musical metrics.

**Genres covered:** Blues, Classical, Country, Electronic, Folk, Hip-Hop, Jazz, Pop, Rock

---

## Dataset

**Lakh MIDI Dataset (LMD)**  
Link: [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/)

> Colin Raffel. *"Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching."* PhD Thesis, 2016.

If running on the full dataset is not feasible, a 10–15 GB subset is sufficient.

---

## Tasks

| Task | Model | Description |
|------|-------|-------------|
| Task 1 | Autoencoder (AE) | Learns a compressed latent representation of piano-roll sequences; generates music by sampling/interpolating in latent space |
| Task 2 | Variational Autoencoder (VAE) | Adds a KL-regularized probabilistic latent space for smoother, more controlled generation |
| Task 3 | Transformer | Autoregressive sequence model conditioned on genre embeddings for step-by-step generation |
| Task 4 *(Bonus)* | RLHF | Trains a reward model on human preference scores and fine-tunes the Transformer with RL |
| Baseline | Random + Markov | Random sampling and first-order Markov chain; used as lower-bound reference |

---

## Repository Structure

```
music-generation-unsupervised/
│
├── run_preprocessing.py          # Entry point: parses raw MIDI → piano-roll numpy arrays
├── run_evaluation.py             # Entry point: runs all metrics and prints the final comparison table
├── compute_eval_table.py         # Computes and formats per-model evaluation metrics
├── create_notebook.py            # Utility to scaffold Jupyter notebooks from source files
├── kaggle_train.ipynb            # Kaggle-compatible training notebook (all tasks in one place)
├── requirements.txt              # Python dependencies
│
├── src/
│   ├── config.py                 # All hyperparameters and constants (genres, model dims, etc.)
│   │
│   ├── preprocessing/
│   │   ├── midi_parser.py        # Loads .mid files, extracts note events per genre
│   │   ├── piano_roll.py         # Converts note events to fixed-size piano-roll matrices
│   │   └── tokenizer.py          # Discretises and encodes piano rolls for model input
│   │
│   ├── models/
│   │   ├── autoencoder.py        # Convolutional AE encoder/decoder (Task 1)
│   │   ├── vae.py                # VAE with reparameterisation trick (Task 2)
│   │   ├── transformer.py        # Genre-conditioned Transformer decoder (Task 3)
│   │   ├── reward_model.py       # MLP reward model trained on human preference data (Task 4)
│   │   └── diffusion.py          # Experimental diffusion model (unused in final evaluation)
│   │
│   ├── training/
│   │   ├── train_ae.py           # Training loop for the Autoencoder
│   │   ├── train_vae.py          # Training loop for the VAE (reconstruction + KL loss)
│   │   ├── train_transformer.py  # Training loop for the Transformer (cross-entropy)
│   │   ├── prepare_survey_data.py # Exports generated samples for the human preference survey
│   │   ├── train_reward_model.py # Trains reward model on collected survey scores
│   │   └── train_rlhf.py         # Fine-tunes Transformer with PPO-style RL reward signal
│   │
│   ├── generation/
│   │   ├── sample_latent.py      # Samples from AE latent space and decodes to piano roll
│   │   ├── generate_vae.py       # Samples from VAE prior and decodes to piano roll
│   │   ├── generate_transformer.py # Autoregressive generation from Transformer
│   │   ├── generate_music.py     # Shared utilities for generation across models
│   │   └── midi_export.py        # Converts piano-roll arrays back to .mid files
│   │
│   └── evaluation/
│       ├── metrics.py            # Aggregates all metric scores for a set of MIDI files
│       ├── pitch_histogram.py    # Computes pitch class histogram and distribution entropy
│       ├── rhythm_score.py       # Measures rhythmic regularity and note density
│       ├── baseline_random.py    # Random-note baseline generator + evaluator
│       └── baseline_markov.py    # First-order Markov chain baseline generator + evaluator
│
├── notebooks/
│   ├── preprocessing.ipynb       # Interactive walkthrough of the preprocessing pipeline
│   └── baseline_markov.ipynb     # Markov baseline exploration and visualisations
│
├── data/
│   └── train_test_split/         # Preprocessed .npy piano-roll arrays (generated by run_preprocessing.py)
│
├── outputs/
│   ├── models/                   # Saved model checkpoints (.pt files)
│   ├── plots/                    # Loss curves, pitch histograms, evaluation charts
│   └── generated_midi/           # MIDI files produced by each model
│
└── report/
    ├── final_report.tex          # Full LaTeX report (Abstract → Conclusion)
    ├── neurips_2024.sty          # NeurIPS 2024 LaTeX style file
    ├── references.bib            # BibTeX references
    └── architecture_diagrams/    # Figures used in the report
```

---

## Setup and Installation

```bash
pip install -r requirements.txt
```

**Key dependencies:**

| Package | Purpose |
|---------|---------|
| `torch >= 2.0` | All neural network models |
| `pretty_midi` | MIDI file reading and writing |
| `music21` | Music theory utilities |
| `numpy`, `scipy` | Numerical operations |
| `scikit-learn` | Data splitting and metrics |
| `matplotlib` | Plots and visualisations |
| `pandas` | Results tables |
| `tqdm` | Progress bars |
| `jupyter` | Notebook support |

---

## Run Order

Execute the scripts in the following order to reproduce the full pipeline:

```bash
# 1. Preprocessing — builds data/train_test_split/*.npy
python run_preprocessing.py

# 2. Task 1: Autoencoder
python src/training/train_ae.py
python src/generation/sample_latent.py

# 3. Task 2: Variational Autoencoder
python src/training/train_vae.py
python src/generation/generate_vae.py

# 4. Task 3: Transformer
python src/training/train_transformer.py
python src/generation/generate_transformer.py

# 5. Task 4 (Optional / Bonus): RLHF
#    → Collect human survey, save pre_rl_scores.csv + pre_rl_rolls.npy first
python src/training/train_reward_model.py
python src/training/train_rlhf.py

# 6. Baselines
python src/evaluation/baseline_random.py
python src/evaluation/baseline_markov.py

# 7. Final evaluation table
python run_evaluation.py
```

---

## Model Configuration

All hyperparameters are centralised in [src/config.py](src/config.py):

| Parameter | Value |
|-----------|-------|
| Genres | 9 (Blues, Classical, Country, Electronic, Folk, Hip-Hop, Jazz, Pop, Rock) |
| Piano keys (`NUM_PITCHES`) | 88 |
| Steps per bar | 16 |
| Window size | 64 |
| Latent dim | 64 |
| Hidden dim | 256 |
| Transformer layers | 4 |
| Attention heads | 8 |
| Dropout | 0.1 |
| Batch size | 32 |

---

## Outputs and Generated MIDI Files

Generated MIDI files for all models are available on Google Drive:  
**[MIDI Files — Google Drive](YOUR_DRIVE_LINK_HERE)**  
*(Replace this link with the publicly accessible Drive folder before submission.)*

---

## Report

The full project report (PDF) is available at:  
**[Project Report — Google Drive](https://drive.google.com/file/d/1Gs91QOcRcvph6dV38OODAM3vUzajRBAQ/view?usp=sharing)**  
*(Replace this link with the publicly accessible PDF link before submission.)*

Sections: Abstract · Introduction · Methodology · Result Analysis · Conclusion

---

## Presentation / Demo Video

**[YouTube (Unlisted)](YOUR_YOUTUBE_LINK_HERE)**  
*(Replace this link with the YouTube unlisted video link before submission.)*

The video walkthrough covers: EDA → Preprocessing → Task 1 → Task 2 → Task 3 → Task 4 → Baseline comparison → Overall analysis.

---

## Individual Contributions

**Taskia Maisha (ID: 22201433, Serial: 511)**
- Preprocessing pipeline (`run_preprocessing.py`, `src/preprocessing/`)
- Autoencoder model and training (`src/models/autoencoder.py`, `src/training/train_ae.py`)
- Latent space sampling and MIDI export (`src/generation/sample_latent.py`, `src/generation/midi_export.py`)
- Report writing: Methodology and Result Analysis sections

**Shadman Sakib (ID: 21301566, Serial: 402)**
- VAE and Transformer models (`src/models/vae.py`, `src/models/transformer.py`)
- VAE and Transformer training and generation scripts
- RLHF pipeline (`src/training/train_reward_model.py`, `src/training/train_rlhf.py`)
- Evaluation framework (`src/evaluation/`, `run_evaluation.py`)
- Report writing: Abstract, Introduction, and Conclusion sections

---

## References

- Colin Raffel. *"Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching."* PhD Thesis, 2016.  
  Dataset: [https://colinraffel.com/projects/lmd/](https://colinraffel.com/projects/lmd/)
