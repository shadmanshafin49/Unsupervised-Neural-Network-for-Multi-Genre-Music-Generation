Unsupervised Neural Network for Multi-Genre Music Generation

Team members:
---------------
Taskia Maisha
ID: 22201433
Serial: 511
---------------
Shadman Sakib
ID: 21301566
Serial: 402
---------------

Dataset:
LinkL: https://colinraffel.com/projects/lmd/
Reference:
Colin Raffel. "Learning-Based Methods for Comparing Sequences, with Applications to Audio-to-MIDI Alignment and Matching". PhD Thesis, 2016.


## Run Order

```

python run_preprocessing.py            # builds data/train_test_split/*.npy
python src/training/train_ae.py        # Task 1
python src/generation/sample_latent.py
python src/training/train_vae.py       # Task 2
python src/generation/generate_vae.py
python src/training/train_transformer.py  # Task 3
python src/generation/generate_transformer.py
# → collect human survey, save pre_rl_scores.csv + pre_rl_rolls.npy
python src/training/train_reward_model.py
python src/training/train_rlhf.py     # Task 4
python src/evaluation/baseline_random.py
python src/evaluation/baseline_markov.py
python run_evaluation.py              # final metrics table

```