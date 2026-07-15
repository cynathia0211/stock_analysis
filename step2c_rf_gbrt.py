#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage C: Random Forest + GBRT (reduced grid, controlled resource usage)
Requires: results/stage_prep.pkl
Output:   results/stage_rf_gbrt.pkl

Optimizations vs original:
  - RF:   grid 4 -> 2 combos, n_jobs=4 (not -1, avoid system freeze)
  - GBRT: grid 8 -> 3 combos, n_estimators reduced for speed
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')

OUT = "results/"
N_JOBS = 4  # Adjust down if machine struggles

print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel        = d['panel']
feat_cols    = d['feat_cols']
test_periods = d['test_periods']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

preds_rf, preds_gbrt = [], []
fi_rf = []

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

    # ── Random Forest: 2 combos (reduced from 4) ──
    t1 = time.time()
    best, bm = np.inf, None
    for ne, md in [(300, 8), (300, None)]:
        m = RandomForestRegressor(
            n_estimators=ne, max_depth=md,
            max_features='sqrt', min_samples_leaf=5,
            n_jobs=N_JOBS, random_state=42)
        m.fit(Xtr_s, ytr)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
    fi_rf.append(bm.feature_importances_)
    preds_rf.append(bm.predict(Xte_s))
    print(f"    RF: {time.time()-t1:.1f}s")

    # ── GBRT: 3 combos (reduced from 8), fewer estimators ──
    t1 = time.time()
    best, bm = np.inf, None
    for ne, md, lr in [(150, 3, .1), (200, 4, .05), (150, 5, .1)]:
        m = GradientBoostingRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=.8, min_samples_leaf=5,
            random_state=42, warm_start=False)
        m.fit(Xtr_s, ytr)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
    preds_gbrt.append(bm.predict(Xte_s))
    print(f"    GBRT: {time.time()-t1:.1f}s")

# ── Concatenate & quick eval ─────────────────────────────────
# Load bench_lm for actual/weight/dates
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
unique_dates = sorted(set(dates_arr))

preds = {
    'RF':   np.concatenate(preds_rf),
    'GBRT': np.concatenate(preds_gbrt),
}

print("\n  Quick MSE summary:")
for mn, pred in preds.items():
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(w * (a - p)**2)))
    print(f"    {mn:10s}: {np.mean(mse_list)*100:.4f}%")

save = {
    'preds': preds,
    'fi_rf': fi_rf,
    'feat_cols': feat_cols,
}
with open(f'{OUT}/stage_rf_gbrt.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"\n  Saved to {OUT}/stage_rf_gbrt.pkl")
print("[Stage C Complete]")
