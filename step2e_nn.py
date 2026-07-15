#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage E: Neural Network (reduced grid, GPU)
Requires: results/stage_prep.pkl
Output:   results/stage_nn.pkl

Optimizations vs original:
  - Grid: 12 combos -> 4 combos
  - Epochs: 300 -> 150 (early stopping still active, patience=15)
  - batch_size: 2048 (unchanged, GPU-friendly)
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

OUT = os.environ.get('OUT', 'results/')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")


class BetaNet(nn.Module):
    def __init__(self, input_dim, hidden_sizes):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_nn(Xtr, ytr, Xva, yva, hidden_sizes, lr, weight_decay,
             epochs=150, batch_size=2048, patience=15):
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=DEVICE)

    model = BetaNet(Xtr.shape[1], hidden_sizes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=8, factor=0.5)

    dl = DataLoader(TensorDataset(Xtr_t, ytr_t), batch_size=batch_size, shuffle=True)

    best_val, pat, best_state = np.inf, 0, None
    for epoch in range(epochs):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            nn.MSELoss()(model(xb), yb).backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(Xva_t), yva_t).item()
        scheduler.step(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_val


def predict_nn(model, X):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32, device=DEVICE)).cpu().numpy()


print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel        = d['panel']
feat_cols    = d['feat_cols']
test_periods = d['test_periods']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

preds_nn = []

# Reduced grid: 4 combos (down from 12)
NN_GRID = [
    {'h': (64, 32),      'lr': 1e-3, 'wd': 1e-3},
    {'h': (128, 64),     'lr': 1e-3, 'wd': 1e-2},
    {'h': (128, 64, 32), 'lr': 5e-4, 'wd': 1e-3},
    {'h': (128, 64, 32), 'lr': 1e-3, 'wd': 1e-2},
]

for pi, per in enumerate(test_periods):
    tr_mask = panel['date'].isin(per['train'])
    va_mask = panel['date'].isin(per['val'])
    te_mask = panel['date'].isin(per['test'])

    Xtr = np.nan_to_num(panel.loc[tr_mask, feat_cols].values.astype(float))
    ytr = panel.loc[tr_mask, 'realized_beta'].values.astype(float)
    Xva = np.nan_to_num(panel.loc[va_mask, feat_cols].values.astype(float))
    yva = panel.loc[va_mask, 'realized_beta'].values.astype(float)
    Xte = np.nan_to_num(panel.loc[te_mask, feat_cols].values.astype(float))

    if len(Xtr) < 100 or len(Xte) < 50:
        continue

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(Xte)

    print(f"\n  Period {pi+1}/{len(test_periods)}: "
          f"{per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')} "
          f"(tr={len(Xtr)}, te={len(Xte)})")

    t1 = time.time()
    best_val, best_model = np.inf, None
    for cfg in NN_GRID:
        model, val = train_nn(Xtr_s, ytr, Xva_s, yva,
                              cfg['h'], cfg['lr'], cfg['wd'])
        print(f"    NN {cfg['h']} lr={cfg['lr']}: val={val:.6f}")
        if val < best_val:
            best_val, best_model = val, model

    preds_nn.append(predict_nn(best_model, Xte_s))
    print(f"    NN best val={best_val:.6f}, {time.time()-t1:.1f}s")

    # Free GPU memory between periods
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

# ── Quick eval ───────────────────────────────────────────────
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
unique_dates = sorted(set(dates_arr))

pred_nn = np.concatenate(preds_nn)
mse_list = []
for dt in unique_dates:
    mask = dates_arr == dt
    a = actual[mask]; p = pred_nn[mask]; w = weight[mask]
    w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    mse_list.append(float(np.sum(w * (a - p)**2)))
print(f"\n  NN_1 MSE: {np.mean(mse_list)*100:.4f}%")

save = {'preds': {'NN_1': pred_nn}}
with open(f'{OUT}/stage_nn.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"  Saved to {OUT}/stage_nn.pkl")
print("[Stage E Complete]")
