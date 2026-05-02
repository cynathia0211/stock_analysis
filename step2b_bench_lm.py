#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage B: Benchmarks + LM + ELANET
Requires: results/stage_prep.pkl
Output:   results/stage_bench_lm.pkl
"""
import os, time, pickle, warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

OUT = os.environ.get('OUT', 'results/')

# ── Load prep data ───────────────────────────────────────────
print("Loading prep data ...")
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    d = pickle.load(f)
panel       = d['panel']
feat_cols   = d['feat_cols']
test_periods = d['test_periods']
beta_1y     = d['beta_1y']
beta_3m     = d['beta_3m']
beta_5y     = d['beta_5y']
print(f"  Panel: {panel.shape}, {len(test_periods)} periods")

# ── Build test index arrays ──────────────────────────────────
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

    # LM
    t1 = time.time()
    m = LinearRegression().fit(Xtr_s, ytr)
    preds_lm.append(m.predict(Xte_s))
    print(f"    LM: {time.time()-t1:.1f}s")

    # ELANET — combine train+val, limit n_jobs to 4
    t1 = time.time()
    Xc = np.vstack([Xtr_s, Xva_s]); yc = np.concatenate([ytr, yva])
    m = ElasticNetCV(l1_ratio=[.1, .5, .9], cv=5, max_iter=3000, n_jobs=4)
    m.fit(Xc, yc)
    preds_elanet.append(m.predict(Xte_s))
    print(f"    ELANET: {time.time()-t1:.1f}s")

# ── Concatenate ──────────────────────────────────────────────
actual    = np.concatenate(all_actual)
weight    = np.concatenate(all_weight)
dates_arr = np.concatenate(all_date)
stocks_arr = np.concatenate(all_stock)
dates_ts  = pd.to_datetime(dates_arr)

# ── Benchmark predictions ────────────────────────────────────
print("\nBuilding benchmark predictions ...")

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

# Fill NaN benchmarks
for k in bench:
    arr = bench[k]
    for d in np.unique(dates_arr):
        mask = dates_arr == d
        cs = np.nanmean(arr[mask])
        arr[mask] = np.where(np.isnan(arr[mask]), cs, arr[mask])
    bench[k] = arr

bench['LM']     = np.concatenate(preds_lm)
bench['ELANET'] = np.concatenate(preds_elanet)

# ── Quick evaluation ─────────────────────────────────────────
print("\n  Quick MSE summary:")
unique_dates = sorted(set(dates_arr))
for mn, pred in bench.items():
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(w * (a - p)**2)))
    print(f"    {mn:15s}: {np.mean(mse_list)*100:.4f}%")

# ── Save ─────────────────────────────────────────────────────
save = {
    'actual': actual, 'weight': weight,
    'dates_arr': dates_arr, 'stocks_arr': stocks_arr,
    'preds': bench,
}
with open(f'{OUT}/stage_bench_lm.pkl', 'wb') as f:
    pickle.dump(save, f, protocol=4)
print(f"\n  Saved to {OUT}/stage_bench_lm.pkl")
print("[Stage B Complete]")
