"""Run this after training the Transformer (train_transformer.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import (OUTPUTS_DIR, NUM_PITCHES, WINDOW_SIZE,
                    GENERATION_THRESHOLD, TEMPERATURE, GENRES)
from models.transformer import MusicTransformer
from generation.midi_export import piano_roll_to_midi


def generate_sequence(genre_id, model, device,
                      seed_steps=8, total_steps=256,
                      temperature=TEMPERATURE):
    """
    Autoregressively generate a piano roll sequence.
    model must already be on device and in eval mode.
    """
    genre_tensor = torch.LongTensor([genre_id]).to(device)

    seed = np.zeros((seed_steps, NUM_PITCHES), dtype=np.float32)
    for t in range(seed_steps):
        active = np.random.choice(NUM_PITCHES,
                                  size=np.random.randint(1, 5),
                                  replace=False)
        seed[t, active] = 1.0

    generated = list(seed)

    with torch.no_grad():
        for _ in range(total_steps - seed_steps):
            context    = torch.FloatTensor(np.array(generated)).unsqueeze(0).to(device)
            logits     = model(context, genre_tensor)
            last_logit = logits[0, -1, :]
            probs      = torch.sigmoid(last_logit / temperature)
            next_step  = (probs > GENERATION_THRESHOLD).float().cpu().numpy()
            generated.append(next_step)

    return np.array(generated)  # (total_steps, 88)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MusicTransformer().to(device)
    model.load_state_dict(
        torch.load(os.path.join(OUTPUTS_DIR, 'transformer.pth'), map_location=device))
    model.eval()

    os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task3'), exist_ok=True)

    genre_list  = list(GENRES.items())
    to_generate = (genre_list * 2)[:10]

    print(f"Generating {len(to_generate)} long compositions (256 steps each)...")
    for idx, (genre_name, genre_id) in enumerate(to_generate, 1):
        roll = generate_sequence(genre_id, model, device, seed_steps=8, total_steps=256)
        path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task3',
                            f'task3_{genre_name}_{idx}.mid')
        piano_roll_to_midi(roll, save_path=path)
        print(f"  [{idx}/{len(to_generate)}] task3_{genre_name}_{idx}.mid "
              f"({roll.shape[0]} steps)")

    print("Done! Check outputs/generated_midis/task3/")
