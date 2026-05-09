import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from config import OUTPUTS_DIR, MODELS_DIR
from models.reward_model import RewardModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

survey_path    = os.path.join(OUTPUTS_DIR, 'survey_results', 'pre_rl_scores.csv')
pre_rl_path    = os.path.join(OUTPUTS_DIR, 'survey_results', 'pre_rl_rolls.npy')

df = pd.read_csv(survey_path)
avg_scores = df.groupby('sample_id')['musicality'].mean().reset_index()
avg_scores.columns = ['sample_id', 'avg_score']
print(avg_scores)

# pre_rl_rolls.npy must be saved during generate_transformer.py (shape: (N, 64, 88))
rolls  = np.load(pre_rl_path)
scores = avg_scores['avg_score'].values.astype(np.float32)

X_tensor = torch.FloatTensor(rolls)
y_tensor = torch.FloatTensor(scores)

model     = RewardModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

for epoch in range(75):
    model.train()
    optimizer.zero_grad()
    preds = model(X_tensor.to(device)).squeeze()
    loss  = criterion(preds, y_tensor.to(device))
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | MSE Loss: {loss.item():.4f}")

os.makedirs(MODELS_DIR, exist_ok=True)
torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'reward_model.pth'))
print("Saved: outputs/models/reward_model.pth")
