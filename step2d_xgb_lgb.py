#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage D: XGBoost + LightGBM (reduced grid)
Requires: results/stage_prep.pkl
Output:   results/stage_xgb_lgb.pkl

Optimizations vs original:
  - XGBoost: grid 12 -> 4 combos
  - LightGBM: grid 8 -> 4 combos
  - Both use GPU if available
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

OUT = "results/"

import torch
USE_GPU = torch.cuda.is_available()
XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
LGB_DEVICE  = 'gpu'  if USE_GPU else 'cpu'
print(f"GPU available: {USE_GPU}")

print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel        = d['panel']
feat_cols    = d['feat_cols']
test_periods = d['test_periods']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

preds_xgb, preds_lgb = [], []
fi_xgb, fi_lgb = [], []

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

    # ── XGBoost: 4 combos (down from 12) ──
    t1 = time.time()
    best, bm = np.inf, None
    for ne, md, lr in [(300, 4, .05), (300, 6, .1), (400, 4, .1), (400, 6, .05)]:
        m = xgb.XGBRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=.8, colsample_bytree=.8, min_child_weight=5,
            reg_alpha=.1, reg_lambda=1.,
            device=XGB_DEVICE, random_state=42, verbosity=0)
        m.fit(Xtr_s, ytr)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
    fi_xgb.append(bm.feature_importances_)
    preds_xgb.append(bm.predict(Xte_s))
    print(f"    XGBoost: {time.time()-t1:.1f}s")

    # ── LightGBM: 4 combos (down from 8) ──
    t1 = time.time()
    best, bm = np.inf, None
    for ne, nl, lr in [(300, 31, .05), (300, 63, .1), (400, 31, .1), (400, 63, .05)]:
        try:
            m = lgb.LGBMRegressor(
                n_estimators=ne, num_leaves=nl, learning_rate=lr,
                subsample=.8, colsample_bytree=.8, min_child_samples=5,
                reg_alpha=.1, reg_lambda=1.,
                device=LGB_DEVICE, random_state=42, verbose=-1)
            m.fit(Xtr_s, ytr)
        except Exception:
            m = lgb.LGBMRegressor(
                n_estimators=ne, num_leaves=nl, learning_rate=lr,
                subsample=.8, colsample_bytree=.8, min_child_samples=5,
                reg_alpha=.1, reg_lambda=1.,
                device='cpu', random_state=42, verbose=-1)
            m.fit(Xtr_s, ytr)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
    fi_lgb.append(bm.feature_importances_)
    preds_lgb.append(bm.predict(Xte_s))
    print(f"    LightGBM: {time.time()-t1:.1f}s")

# ── Quick eval ───────────────────────────────────────────────
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
unique_dates = sorted(set(dates_arr))

preds = {
    'XGBoost':  np.concatenate(preds_xgb),
    'LightGBM': np.concatenate(preds_lgb),
}

print("\n  Quick MSE summary:")
for mn, pred in preds.items():
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(w * (a - p)**2)))
    print(f"    {mn:12s}: {np.mean(mse_list)*100:.4f}%")

save = {
    'preds': preds,
    'fi_xgb': fi_xgb,
    'fi_lgb': fi_lgb,
    'feat_cols': feat_cols,
}
with open(f'{OUT}/stage_xgb_lgb.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"\n  Saved to {OUT}/stage_xgb_lgb.pkl")
print("[Stage D Complete]")
