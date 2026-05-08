"""Run this after training the Transformer (train_transformer.py)."""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from config import (OUTPUTS_DIR, MODELS_DIR, NUM_PITCHES, WINDOW_SIZE,
                    GENERATION_THRESHOLD, TEMPERATURE, GENRES)
from models.transformer import MusicTransformer
from generation.midi_export import piano_roll_to_midi


def generate_sequence(genre_id, model, device,
                      seed_steps=8, total_steps=256,
                      temperature=0.7):
    """
    Autoregressively generate a piano roll.
    Uses a sliding window of WINDOW_SIZE to stay in-distribution.
    """
    genre_tensor = torch.LongTensor([genre_id]).to(device)

    seed = np.zeros((seed_steps, NUM_PITCHES), dtype=np.float32)
    for t in range(seed_steps):
        active = np.random.choice(NUM_PITCHES,
                                  size=np.random.randint(2, 6),
                                  replace=False)
        seed[t, active] = 1.0

    generated = list(seed)

    with torch.no_grad():
        for _ in range(total_steps - seed_steps):
            # Slide a window of WINDOW_SIZE steps so the model stays in-distribution
            context_arr = np.array(generated[-WINDOW_SIZE:])
            context     = torch.FloatTensor(context_arr).unsqueeze(0).to(device)
            logits      = model(context, genre_tensor)
            last_logit  = logits[0, -1, :]
            probs       = torch.sigmoid(last_logit / temperature)
            next_step   = (probs > GENERATION_THRESHOLD).float().cpu().numpy()
            generated.append(next_step)

    return np.array(generated)  # (total_steps, 88)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MusicTransformer().to(device)
    model.load_state_dict(
        torch.load(os.path.join(MODELS_DIR, 'transformer.pth'), map_location=device))
    model.eval()

    os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task3'), exist_ok=True)

    genre_list  = list(GENRES.items())
    to_generate = (genre_list * 2)[:10]

    print(f"Generating {len(to_generate)} compositions (256 steps = 32s each)...")
    for idx, (genre_name, genre_id) in enumerate(to_generate, 1):
        roll = generate_sequence(genre_id, model, device,
                                 seed_steps=8, total_steps=256, temperature=0.7)
        path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task3',
                            f'task3_{genre_name}_{idx}.mid')
        piano_roll_to_midi(roll, save_path=path)
        active = int(roll.sum())
        print(f"  [{idx}/{len(to_generate)}] task3_{genre_name}_{idx}.mid "
              f"({roll.shape[0]} steps, {active} active cells)")

    print("Done! Check outputs/generated_midis/task3/")
