#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage C (v2): Random Forest + GBRT with loss-curve tracking.

New vs step2c_rf_gbrt.py:
  - RF: record validation MSE as trees are accumulated (1..n_estimators, step=10)
  - GBRT: use staged_predict to record validation MSE per boosting iteration
  - Persist the last period's best RF model + Xte_s for later SHAP analysis.
  - Per-period mean |SHAP| saved in shap_rf for feature importance plots.

Output: {OUT}/stage_rf_gbrt.pkl  (OUT defaults to results_v2/)
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
warnings.filterwarnings('ignore')
if not hasattr(np, 'bool'):
    np.bool = bool
import shap

OUT = os.environ.get('OUT', 'results_v2/')
N_JOBS = 4
RF_STEP = 10
SHAP_MAX = 500   # cap rows for SHAP; max_depth=8 makes each call ~5s

print(f"OUT = {OUT}")
print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel        = d['panel']
feat_cols    = d['feat_cols']
test_periods = d['test_periods']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

# ── Reuse existing GBRT results to avoid re-running slow sklearn GBRT ──
print("Loading existing stage_rf_gbrt.pkl for GBRT cache ...")
with open(f'{OUT}/stage_rf_gbrt.pkl', 'rb') as f:
    old = pickle.load(f)
preds_gbrt = old['preds']['GBRT']                # already concatenated numpy array
loss_gbrt  = old.get('loss_gbrt', [])
print(f"  GBRT preds cached: {len(preds_gbrt)} rows, loss curves: {len(loss_gbrt)}")

preds_rf = []
fi_rf    = []
shap_rf  = []
loss_rf  = []
last_rf_model = None
last_Xte_s    = None

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

    # ── Random Forest: grid search + loss curve ──
    t1 = time.time()
    best, bm = np.inf, None
    for ne, md in [(300, 8), (500, 8)]:
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

    # Per-period mean |SHAP|
    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    t_shap = time.time()
    explainer = shap.TreeExplainer(bm, feature_perturbation='tree_path_dependent')
    sv = explainer.shap_values(Xs_shap, check_additivity=False)
    shap_rf.append(np.abs(sv).mean(axis=0))
    print(f"    RF SHAP: {time.time()-t_shap:.1f}s for {len(Xs_shap)} rows")

    # Loss curve
    n_total = len(bm.estimators_)
    ks = list(range(RF_STEP, n_total + 1, RF_STEP))
    if ks and ks[-1] != n_total:
        ks.append(n_total)
    all_tree_preds = np.array([t.predict(Xva_s) for t in bm.estimators_])
    cum_mean = np.cumsum(all_tree_preds, axis=0) / np.arange(1, n_total + 1)[:, None]
    mses = [float(np.mean((yva - cum_mean[k-1])**2)) for k in ks]
    loss_rf.append({'ntrees': np.array(ks), 'mse': np.array(mses)})
    print(f"    RF total: {time.time()-t1:.1f}s, mse={mses[-1]:.6f}")

    last_rf_model = bm
    last_Xte_s    = Xte_s

# ── Concatenate & quick eval ─────────────────────────────────
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
unique_dates = sorted(set(dates_arr))

preds = {
    'RF':   np.concatenate(preds_rf),
    'GBRT': preds_gbrt,
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
    'preds':         preds,
    'fi_rf':         fi_rf,
    'shap_rf':       shap_rf,
    'feat_cols':     feat_cols,
    'loss_rf':       loss_rf,
    'loss_gbrt':     loss_gbrt,
    'last_rf_model': last_rf_model,
    'last_Xte_s':    last_Xte_s,
}
with open(f'{OUT}/stage_rf_gbrt.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"\n  Saved to {OUT}/stage_rf_gbrt.pkl")
print("[Stage C v2 Complete]")
