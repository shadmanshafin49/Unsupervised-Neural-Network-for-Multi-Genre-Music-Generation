"""
High-level entry point: generate music from any trained model.

Usage (from project root):
    python src/generation/generate_music.py --model ae --genre Classical --count 3
    python src/generation/generate_music.py --model vae --genre Jazz --count 5
    python src/generation/generate_music.py --model transformer --genre Rock --count 3
    python src/generation/generate_music.py --model rlhf --genre Pop --count 3
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import numpy as np
from config import OUTPUTS_DIR, LATENT_DIM, GENRES
from generation.midi_export import piano_roll_to_midi


def generate_ae(genre_name, count, device):
    from models.autoencoder import Autoencoder
    from config import SPLIT_DIR
    model = Autoencoder().to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUTS_DIR, 'autoencoder.pth'), map_location=device))
    model.eval()

    X_train      = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
    sample_batch = torch.FloatTensor(X_train[:200]).to(device)
    with torch.no_grad():
        z_encoded, _ = model(sample_batch)
    z_mean, z_std = z_encoded.mean(dim=0), z_encoded.std(dim=0)

    out_dir = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task1')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, count + 1):
        z     = z_mean + z_std * torch.randn(1, LATENT_DIM).to(device)
        with torch.no_grad():
            x_hat = model.decoder(z)
        roll = x_hat.squeeze(0).cpu().numpy()
        path = os.path.join(out_dir, f'ae_{genre_name}_{i}.mid')
        piano_roll_to_midi(roll, save_path=path)
        print(f"  Saved: {path}")


def generate_vae(genre_name, count, device):
    from models.vae import VAE
    model = VAE().to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUTS_DIR, 'vae.pth'), map_location=device))
    model.eval()

    out_dir = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task2')
    os.makedirs(out_dir, exist_ok=True)
    for i in range(1, count + 1):
        z = torch.randn(1, LATENT_DIM).to(device)
        with torch.no_grad():
            x_hat = model.decoder(z)
        roll = x_hat.squeeze(0).cpu().numpy()
        path = os.path.join(out_dir, f'vae_{genre_name}_{i}.mid')
        piano_roll_to_midi(roll, save_path=path)
        print(f"  Saved: {path}")


def generate_transformer_music(genre_name, count, device, model_tag='transformer'):
    from generation.generate_transformer import generate_sequence
    from models.transformer import MusicTransformer

    model_file = 'rlhf_model.pth' if model_tag == 'rlhf' else 'transformer.pth'
    model      = MusicTransformer().to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUTS_DIR, model_file), map_location=device))
    model.eval()

    task    = 'task4' if model_tag == 'rlhf' else 'task3'
    out_dir = os.path.join(OUTPUTS_DIR, 'generated_midis', task)
    os.makedirs(out_dir, exist_ok=True)

    genre_id = GENRES[genre_name]
    for i in range(1, count + 1):
        roll = generate_sequence(genre_id, model, device, seed_steps=8, total_steps=256)
        path = os.path.join(out_dir, f'{model_tag}_{genre_name}_{i}.mid')
        piano_roll_to_midi(roll, save_path=path)
        print(f"  Saved: {path}")


def main():
    parser = argparse.ArgumentParser(description="Generate music from trained models")
    parser.add_argument('--model',  choices=['ae', 'vae', 'transformer', 'rlhf'],
                        default='transformer')
    parser.add_argument('--genre',  default='Classical', choices=list(GENRES.keys()))
    parser.add_argument('--count',  type=int, default=3)
    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Generating {args.count} samples | model={args.model} | genre={args.genre}")
    if args.model == 'ae':
        generate_ae(args.genre, args.count, device)
    elif args.model == 'vae':
        generate_vae(args.genre, args.count, device)
    elif args.model in ('transformer', 'rlhf'):
        generate_transformer_music(args.genre, args.count, device, args.model)


if __name__ == "__main__":
    main()
