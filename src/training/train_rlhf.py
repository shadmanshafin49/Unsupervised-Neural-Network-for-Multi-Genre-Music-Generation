import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from config import (OUTPUTS_DIR, PLOTS_DIR, LR_RLHF, RL_STEPS,
                    WINDOW_SIZE, NUM_PITCHES, GENRES)
from models.transformer import MusicTransformer
from models.reward_model import RewardModel
from generation.generate_transformer import generate_sequence
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'task4'), exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator = MusicTransformer().to(device)
generator.load_state_dict(
    torch.load(os.path.join(OUTPUTS_DIR, 'transformer.pth'), map_location=device))
generator.train()

reward_model = RewardModel().to(device)
reward_model.load_state_dict(
    torch.load(os.path.join(OUTPUTS_DIR, 'reward_model.pth'), map_location=device))
reward_model.eval()

optimizer      = torch.optim.Adam(generator.parameters(), lr=LR_RLHF)
genre_ids      = list(range(len(GENRES)))
reward_history = []

print(f"Starting RLHF fine-tuning for {RL_STEPS} steps...")

for step in trange(RL_STEPS):
    genre_id     = np.random.choice(genre_ids)
    genre_tensor = torch.LongTensor([genre_id]).to(device)

    seed = np.zeros((8, NUM_PITCHES), dtype=np.float32)
    for t in range(8):
        active = np.random.choice(NUM_PITCHES, size=2, replace=False)
        seed[t, active] = 1.0

    context = torch.FloatTensor(seed).unsqueeze(0).to(device)

    optimizer.zero_grad()
    logits   = generator(context, genre_tensor)              # (1, 7, 88)
    dist     = torch.distributions.Bernoulli(probs=logits)
    sample   = dist.sample()
    log_prob = dist.log_prob(sample).sum()

    sample_np = sample.squeeze(0).detach().cpu().numpy()    # (7, 88)
    padded    = np.zeros((WINDOW_SIZE, NUM_PITCHES), dtype=np.float32)
    padded[:sample_np.shape[0], :] = sample_np
    padded_t  = torch.FloatTensor(padded).unsqueeze(0).to(device)

    with torch.no_grad():
        reward = reward_model(padded_t).item()

    reward_norm = (reward - 3.0) / 2.0
    rl_loss     = -reward_norm * log_prob
    rl_loss.backward()
    torch.nn.utils.clip_grad_norm_(generator.parameters(), 0.5)
    optimizer.step()

    reward_history.append(reward)

    if (step + 1) % 50 == 0:
        avg_r = np.mean(reward_history[-50:])
        print(f"Step {step+1:4d} | Avg Reward (last 50): {avg_r:.3f}")

torch.save(generator.state_dict(), os.path.join(OUTPUTS_DIR, 'rlhf_model.pth'))
print("Saved: outputs/rlhf_model.pth")

plt.figure(figsize=(10, 4))
plt.plot(reward_history, alpha=0.4, label='Raw Reward')
window   = 20
smoothed = np.convolve(reward_history, np.ones(window) / window, mode='valid')
plt.plot(range(window - 1, len(reward_history)), smoothed, color='red',
         label=f'{window}-step avg')
plt.axhline(y=3.0, color='gray', linestyle='--', label='Baseline (score=3)')
plt.title("RLHF Training - Reward Over Time")
plt.xlabel("RL Step")
plt.ylabel("Predicted Human Score")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOTS_DIR, 'task4_reward_curve.png'), dpi=150)
plt.show()
print("Saved: outputs/plots/task4_reward_curve.png")

# Generate 10 post-RLHF samples using the fine-tuned model
print("Generating 10 post-RLHF MIDI samples...")
generator.eval()
genre_items = list(GENRES.items())
for i in range(10):
    genre_name, genre_id = genre_items[i % len(genre_items)]
    roll = generate_sequence(genre_id, generator, device, seed_steps=8, total_steps=256)
    path = os.path.join(OUTPUTS_DIR, 'generated_midis', 'task4',
                        f'task4_rlhf_{genre_name}_{i+1}.mid')
    piano_roll_to_midi(roll, save_path=path)
    print(f"  [{i+1}/10] Saved: task4_rlhf_{genre_name}_{i+1}.mid")
