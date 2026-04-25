import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import SPLIT_DIR, OUTPUTS_DIR, NUM_PITCHES, WINDOW_SIZE
from generation.midi_export import piano_roll_to_midi

os.makedirs(os.path.join(OUTPUTS_DIR, 'generated_midis', 'baselines'), exist_ok=True)


def build_markov_model(X_train, top_k=32):
    """
    Bigram Markov model over quantized piano roll states.
    Uses top-K active pitches as state key to keep memory manageable.
    """
    transitions = {}
    for roll in X_train:
        for t in range(len(roll) - 1):
            state = tuple(np.where(roll[t] > 0.5)[0][:top_k])
            if state not in transitions:
                transitions[state] = []
            transitions[state].append(roll[t + 1].copy())
    return transitions


def markov_generate(transitions, seed_step, steps=WINDOW_SIZE):
    generated     = [seed_step]
    current_state = tuple(np.where(seed_step > 0.5)[0])

    for _ in range(steps - 1):
        if current_state in transitions and transitions[current_state]:
            next_step = transitions[current_state][
                np.random.randint(len(transitions[current_state]))
            ]
        else:
            fallback_state = np.random.choice(list(transitions.keys()))
            next_step      = transitions[fallback_state][0]
        generated.append(next_step)
        current_state = tuple(np.where(next_step > 0.5)[0])

    return np.array(generated)


if __name__ == "__main__":
    print("Building Markov model from training data...")
    X_train     = np.load(os.path.join(SPLIT_DIR, 'train_data.npy'))
    transitions = build_markov_model(X_train)

    for i in range(1, 6):
        seed = X_train[np.random.randint(len(X_train))][0]
        roll = markov_generate(transitions, seed)
        piano_roll_to_midi(roll, save_path=os.path.join(
            OUTPUTS_DIR, 'generated_midis', 'baselines', f'markov_{i}.mid'))
        print(f"Markov baseline {i} saved.")
