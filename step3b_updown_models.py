#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train all models for one regime (up or down).
Usage: OUT=results_up/ python step3b_updown_models.py

Models: Benchmarks (OLS_5Y_M, OLS_1Y_D, BSW, HYBRID, FAMA_FRENCH, LONG_MEMO),
        LM, ELANET, RF, GBRT, XGBoost, LightGBM, NN_1.
Computes per-period mean |SHAP| for RF, XGBoost, LightGBM.
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
warnings.filterwarnings('ignore')

if not hasattr(np, 'bool'):
    np.bool = bool
import shap

OUT = os.environ.get('OUT', 'results_up/')
N_JOBS = 4
SHAP_MAX = 500
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_GPU = torch.cuda.is_available()
XGB_DEVICE = 'cuda' if USE_GPU else 'cpu'
LGB_DEVICE = 'gpu' if USE_GPU else 'cpu'

print(f"OUT = {OUT}")
print(f"Device: {DEVICE}, GPU: {USE_GPU}")

# ── Load prep data ───────────────────────────────────────────
print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel        = d['panel']
feat_cols    = d['feat_cols']
test_periods = d['test_periods']
beta_1y      = d['beta_1y']
beta_3m      = d['beta_3m']
beta_5y      = d['beta_5y']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

# ══════════════════════════════════════════════════════════════
# Stage B: Benchmarks + LM + ELANET
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Stage B: Benchmarks + LM + ELANET")
print("="*60)

all_actual, all_weight, all_date, all_stock = [], [], [], []
preds_lm, preds_elanet = [], []
valid_periods = []

for pi, per in enumerate(test_periods):
    tr_mask = panel['date'].isin(per['train'])
    va_mask = panel['date'].isin(per['val'])
    te_mask = panel['date'].isin(per['test'])

    Xtr = np.nan_to_num(panel.loc[tr_mask, feat_cols].values.astype(float))
    ytr = panel.loc[tr_mask, 'realized_beta'].values.astype(float)
    Xva = np.nan_to_num(panel.loc[va_mask, feat_cols].values.astype(float))
    yva = panel.loc[va_mask, 'realized_beta'].values.astype(float)
    Xte = np.nan_to_num(panel.loc[te_mask, feat_cols].values.astype(float))
    yte = panel.loc[te_mask, 'realized_beta'].values.astype(float)
    wte = panel.loc[te_mask, 'weight'].values.astype(float)
    dte = panel.loc[te_mask, 'date'].values
    ste = panel.loc[te_mask, 'stock'].values

    if len(Xtr) < 100 or len(Xte) < 50:
        continue

    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)
    Xte_s = sc.transform(Xte)

    all_actual.append(yte)
    all_weight.append(wte)
    all_date.append(dte)
    all_stock.append(ste)
    valid_periods.append(pi)

    print(f"\n  Period {pi+1}/{len(test_periods)}: "
          f"test {per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')} "
          f"(tr={len(Xtr)}, te={len(Xte)})")

    t1 = time.time()
    m = LinearRegression().fit(Xtr_s, ytr)
    preds_lm.append(m.predict(Xte_s))
    print(f"    LM: {time.time()-t1:.1f}s")

    t1 = time.time()
    Xc = np.vstack([Xtr_s, Xva_s]); yc = np.concatenate([ytr, yva])
    m = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=3000, n_jobs=N_JOBS)
    m.fit(Xc, yc)
    preds_elanet.append(m.predict(Xte_s))
    print(f"    ELANET: {time.time()-t1:.1f}s")

actual    = np.concatenate(all_actual)
weight    = np.concatenate(all_weight)
dates_arr = np.concatenate(all_date)
stocks_arr = np.concatenate(all_stock)
dates_ts  = pd.to_datetime(dates_arr)

def lookup_beta_fast(beta_df, dates_ts, stocks_arr):
    mi = pd.MultiIndex.from_arrays([dates_ts, stocks_arr], names=['date', 'stock'])
    stacked = beta_df.stack(dropna=False)
    stacked.index.names = ['date', 'stock']
    return stacked.reindex(mi).values.astype(float)

bench = {}
bench['OLS_1Y_D'] = lookup_beta_fast(beta_1y, dates_ts, stocks_arr)
bench['OLS_5Y_M'] = lookup_beta_fast(beta_5y, dates_ts, stocks_arr)
bench['BSW'] = np.clip(bench['OLS_1Y_D'], 0.5, 1.5)

b1y_arr = bench['OLS_1Y_D'].copy()
hybrid_df = pd.DataFrame({'date': dates_ts, 'b': b1y_arr})
cs_mean = hybrid_df.groupby('date')['b'].transform('mean')
bench['HYBRID'] = (0.33 * cs_mean + 0.67 * b1y_arr).values

ff = np.full_like(b1y_arr, np.nan)
ff_df = pd.DataFrame({'date': dates_ts, 'b': b1y_arr, 'idx': np.arange(len(b1y_arr))})
for d, grp in ff_df.groupby('date'):
    valid = grp['b'].notna()
    if valid.sum() >= 10:
        b_valid = grp.loc[valid, 'b']
        decile = pd.qcut(b_valid, 10, labels=False, duplicates='drop')
        for q in decile.unique():
            qm = decile == q
            ff[grp.loc[valid].loc[qm, 'idx'].values] = b_valid[qm].mean()
        ff[grp.loc[~valid, 'idx'].values] = b_valid.mean()
    else:
        ff[grp['idx'].values] = grp['b'].values
bench['FAMA_FRENCH'] = ff

b3m_arr = lookup_beta_fast(beta_3m, dates_ts, stocks_arr)
bench['LONG_MEMO'] = 0.2 * b3m_arr + 0.3 * b1y_arr + 0.5 * bench['OLS_5Y_M']

for k in bench:
    arr = bench[k]
    for d in np.unique(dates_arr):
        mask = dates_arr == d
        cs = np.nanmean(arr[mask])
        arr[mask] = np.where(np.isnan(arr[mask]), cs, arr[mask])
    bench[k] = arr

bench['LM']     = np.concatenate(preds_lm)
bench['ELANET'] = np.concatenate(preds_elanet)

save_bl = {
    'actual': actual, 'weight': weight,
    'dates_arr': dates_arr, 'stocks_arr': stocks_arr,
    'preds': bench,
}
with open(f'{OUT}/stage_bench_lm.pkl', 'wb') as f:
    pickle.dump(save_bl, f, protocol=4)
print(f"\n  Saved stage_bench_lm.pkl")

# ══════════════════════════════════════════════════════════════
# Stage C: RF + GBRT
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Stage C: RF + GBRT")
print("="*60)

preds_rf, preds_gbrt = [], []
fi_rf, shap_rf = [], []

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
          f"{per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')}")

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

    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    explainer = shap.TreeExplainer(bm, feature_perturbation='tree_path_dependent')
    sv = explainer.shap_values(Xs_shap, check_additivity=False)
    shap_rf.append(np.abs(sv).mean(axis=0))
    print(f"    RF: {time.time()-t1:.1f}s")

    t1 = time.time()
    best, bm = np.inf, None
    for ne, md, lr in [(150, 3, .1), (200, 4, .05), (150, 5, .1)]:
        m = GradientBoostingRegressor(
            n_estimators=ne, max_depth=md, learning_rate=lr,
            subsample=.8, min_samples_leaf=5, random_state=42)
        m.fit(Xtr_s, ytr)
        s = mean_squared_error(yva, m.predict(Xva_s))
        if s < best:
            best, bm = s, m
    preds_gbrt.append(bm.predict(Xte_s))
    print(f"    GBRT: {time.time()-t1:.1f}s")

save_rg = {
    'preds': {'RF': np.concatenate(preds_rf), 'GBRT': np.concatenate(preds_gbrt)},
    'fi_rf': fi_rf,
    'shap_rf': shap_rf,
    'feat_cols': feat_cols,
}
with open(f'{OUT}/stage_rf_gbrt.pkl', 'wb') as f:
    pickle.dump(save_rg, f, protocol=4)
print(f"\n  Saved stage_rf_gbrt.pkl")

# ══════════════════════════════════════════════════════════════
# Stage D: XGBoost + LightGBM
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Stage D: XGBoost + LightGBM")
print("="*60)

import xgboost as xgb
import lightgbm as lgb

preds_xgb, preds_lgb = [], []
fi_xgb, fi_lgb = [], []
shap_xgb, shap_lgb = [], []

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
          f"{per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')}")

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

    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    sv_x = shap.TreeExplainer(bm).shap_values(Xs_shap)
    shap_xgb.append(np.abs(sv_x).mean(axis=0))
    print(f"    XGBoost: {time.time()-t1:.1f}s")

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

    Xs_shap = Xte_s
    if len(Xs_shap) > SHAP_MAX:
        rng = np.random.default_rng(42 + pi)
        Xs_shap = Xs_shap[rng.choice(len(Xs_shap), SHAP_MAX, replace=False)]
    sv_l = shap.TreeExplainer(bm).shap_values(Xs_shap)
    shap_lgb.append(np.abs(sv_l).mean(axis=0))
    print(f"    LightGBM: {time.time()-t1:.1f}s")

save_xl = {
    'preds': {'XGBoost': np.concatenate(preds_xgb), 'LightGBM': np.concatenate(preds_lgb)},
    'fi_xgb': fi_xgb, 'fi_lgb': fi_lgb,
    'shap_xgb': shap_xgb, 'shap_lgb': shap_lgb,
    'feat_cols': feat_cols,
}
with open(f'{OUT}/stage_xgb_lgb.pkl', 'wb') as f:
    pickle.dump(save_xl, f, protocol=4)
print(f"\n  Saved stage_xgb_lgb.pkl")

# ══════════════════════════════════════════════════════════════
# Stage E: Neural Network
# ══════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("Stage E: Neural Network")
print("="*60)


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


NN_GRID = [
    {'h': (64, 32),      'lr': 1e-3, 'wd': 1e-3},
    {'h': (128, 64),     'lr': 1e-3, 'wd': 1e-2},
    {'h': (128, 64, 32), 'lr': 5e-4, 'wd': 1e-3},
    {'h': (128, 64, 32), 'lr': 1e-3, 'wd': 1e-2},
]

preds_nn = []

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
          f"{per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')}")

    t1 = time.time()
    best_val, best_model = np.inf, None
    for cfg in NN_GRID:
        model, val = train_nn(Xtr_s, ytr, Xva_s, yva, cfg['h'], cfg['lr'], cfg['wd'])
        if val < best_val:
            best_val, best_model = val, model

    preds_nn.append(predict_nn(best_model, Xte_s))
    print(f"    NN best val={best_val:.6f}, {time.time()-t1:.1f}s")

    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()

save_nn = {'preds': {'NN_1': np.concatenate(preds_nn)}}
with open(f'{OUT}/stage_nn.pkl', 'wb') as f:
    pickle.dump(save_nn, f, protocol=4)
print(f"\n  Saved stage_nn.pkl")

# ── Quick MSE summary ──────────────────────────────────────
unique_dates = sorted(set(dates_arr))
all_preds = dict(bench)
all_preds.update(save_rg['preds'])
all_preds.update(save_xl['preds'])
all_preds.update(save_nn['preds'])

print(f"\n{'='*60}")
print("Quick MSE summary")
print("="*60)
for mn, pred in sorted(all_preds.items()):
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(w * (a - p)**2)))
    print(f"  {mn:15s}: {np.mean(mse_list)*100:.4f}%")

print(f"\n[All models complete for {OUT}]")
