#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage D (v2): XGBoost + LightGBM with loss-curve tracking.

New vs step2d_xgb_lgb.py:
  - Use eval_set during fit and extract per-iteration validation MSE
  - Persist the last period's best XGB and LGB models + Xte_s for SHAP analysis

Output: {OUT}/stage_xgb_lgb.pkl  (OUT defaults to results_v2/)
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
if not hasattr(np, 'bool'):
    np.bool = bool
import shap

OUT = os.environ.get('OUT', 'results_v2/')

import torch
USE_GPU = torch.cuda.is_available()
XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
LGB_DEVICE  = 'gpu'  if USE_GPU else 'cpu'
print(f"OUT = {OUT}")
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
shap_xgb, shap_lgb = [], []           # per-period mean |SHAP|
loss_xgb, loss_lgb = [], []           # list of dict(ntrees, mse)
last_xgb_model = None
last_lgb_model = None
last_Xte_s = None
SHAP_MAX = 5000

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

    # ── XGBoost: grid search + eval_set loss tracking ──
    t1 = time.time()
    best, bm, best_hist = np.inf, None, None
    for ne, md, lr in [(300, 4, .05), (300, 6, .1), (400, 4, .1), (400, 6, .05)]:
        m = xgb.XGBRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=.8, colsample_bytree=.8, min_child_weight=5,
            reg_alpha=.1, reg_lambda=1.,
            device=XGB_DEVICE, random_state=42, verbosity=0,
            eval_metric='rmse')
        m.fit(Xtr_s, ytr, eval_set=[(Xva_s, yva)], verbose=False)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
            rmse_hist = m.evals_result()['validation_0']['rmse']
            best_hist = np.array(rmse_hist) ** 2  # -> MSE
    fi_xgb.append(bm.feature_importances_)
    preds_xgb.append(bm.predict(Xte_s))
    loss_xgb.append({
        'ntrees': np.arange(1, len(best_hist) + 1),
        'mse': best_hist,
    })
    last_xgb_model = bm
    print(f"    XGBoost: {time.time()-t1:.1f}s, final val mse={best_hist[-1]:.6f}")

    # Compute per-period mean |SHAP| for XGBoost
    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    sv_xgb = shap.TreeExplainer(bm).shap_values(Xs_shap)
    shap_xgb.append(np.abs(sv_xgb).mean(axis=0))
    print(f"    XGBoost SHAP: computed for {len(Xs_shap)} rows")

    # ── LightGBM: grid search + eval_set loss tracking ──
    t1 = time.time()
    best, bm, best_hist = np.inf, None, None
    for ne, nl, lr in [(300, 31, .05), (300, 63, .1), (400, 31, .1), (400, 63, .05)]:
        evals_result = {}
        try:
            m = lgb.LGBMRegressor(
                n_estimators=ne, num_leaves=nl, learning_rate=lr,
                subsample=.8, colsample_bytree=.8, min_child_samples=5,
                reg_alpha=.1, reg_lambda=1.,
                device=LGB_DEVICE, random_state=42, verbose=-1)
            m.fit(Xtr_s, ytr,
                  eval_set=[(Xva_s, yva)],
                  eval_metric='l2',
                  callbacks=[lgb.record_evaluation(evals_result)])
        except Exception:
            evals_result = {}
            m = lgb.LGBMRegressor(
                n_estimators=ne, num_leaves=nl, learning_rate=lr,
                subsample=.8, colsample_bytree=.8, min_child_samples=5,
                reg_alpha=.1, reg_lambda=1.,
                device='cpu', random_state=42, verbose=-1)
            m.fit(Xtr_s, ytr,
                  eval_set=[(Xva_s, yva)],
                  eval_metric='l2',
                  callbacks=[lgb.record_evaluation(evals_result)])
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
            # evals_result structure: {'valid_0': {'l2': [...]}}
            v0 = list(evals_result.values())[0]
            mse_key = 'l2' if 'l2' in v0 else list(v0.keys())[0]
            best_hist = np.array(v0[mse_key])
    fi_lgb.append(bm.feature_importances_)
    preds_lgb.append(bm.predict(Xte_s))
    loss_lgb.append({
        'ntrees': np.arange(1, len(best_hist) + 1),
        'mse': best_hist,
    })
    last_lgb_model = bm
    print(f"    LightGBM: {time.time()-t1:.1f}s, final val mse={best_hist[-1]:.6f}")

    # Compute per-period mean |SHAP| for LightGBM
    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    sv_lgb = shap.TreeExplainer(bm).shap_values(Xs_shap)
    shap_lgb.append(np.abs(sv_lgb).mean(axis=0))
    print(f"    LightGBM SHAP: computed for {len(Xs_shap)} rows")

    last_Xte_s = Xte_s

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
    'shap_xgb': shap_xgb,
    'shap_lgb': shap_lgb,
    'feat_cols': feat_cols,
    'loss_xgb': loss_xgb,
    'loss_lgb': loss_lgb,
    'last_xgb_model': last_xgb_model,
    'last_lgb_model': last_lgb_model,
    'last_Xte_s': last_Xte_s,
}
with open(f'{OUT}/stage_xgb_lgb.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"\n  Saved to {OUT}/stage_xgb_lgb.pkl")
print("[Stage D v2 Complete]")
