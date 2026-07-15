#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage F (v2): Separate-figure evaluation + SHAP + Excel exports.

Requires (in $OUT):
  stage_prep.pkl, stage_bench_lm.pkl, stage_rf_gbrt.pkl, stage_xgb_lgb.pkl, stage_nn.pkl (optional)

Outputs (in $OUT):
  loss_{rf,gbrt,xgboost,lightgbm}.png           -- separate loss curves (dpi=300)
  pred_vs_actual_{rf,gbrt,xgboost,lightgbm}.png -- separate prediction plots
  fi_{rf,xgboost,lightgbm}.png                  -- separate feature importance
  shap_{rf,xgboost,lightgbm}.png                -- SHAP beeswarm density (dpi=300)
  metrics.xlsx                                  -- avg_mse, monthly_mse, r2 sheets
  feature_stats.xlsx                            -- non-industry feature mean+var
  summary_mse.csv                               -- backward compat
"""
import os, pickle, warnings
import numpy as np
# Compat: shap 0.41 uses deprecated np.bool
if not hasattr(np, 'bool'):
    np.bool = bool
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

import shap

# Compat: shap 0.41's summary_plot calls plt.colorbar without ax=, which
# newer matplotlib rejects. Patch to default ax=current axes.
_orig_colorbar = plt.colorbar
def _patched_colorbar(mappable=None, *args, **kwargs):
    if 'ax' not in kwargs and 'cax' not in kwargs:
        kwargs['ax'] = plt.gca()
    return _orig_colorbar(mappable, *args, **kwargs)
plt.colorbar = _patched_colorbar

OUT = os.environ.get('OUT', 'results_v2/')
SHAP_MAX_ROWS = 5000
print(f"OUT = {OUT}")

# ── Load all stages ──────────────────────────────────────────
print("Loading results ...")
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
all_preds = dict(bl['preds'])

rf_gbrt = xgb_lgb = nn = None
for fname, label in [('stage_rf_gbrt.pkl', 'RF/GBRT'),
                     ('stage_xgb_lgb.pkl', 'XGB/LGB'),
                     ('stage_nn.pkl',       'NN')]:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            dd = pickle.load(f)
        all_preds.update(dd['preds'])
        print(f"  Loaded {label}: {list(dd['preds'].keys())}")
        if fname == 'stage_rf_gbrt.pkl':   rf_gbrt = dd
        if fname == 'stage_xgb_lgb.pkl':   xgb_lgb = dd
        if fname == 'stage_nn.pkl':        nn = dd

print(f"  Total models: {list(all_preds.keys())}")

unique_dates = sorted(set(dates_arr))
unique_ts = [pd.Timestamp(d) for d in unique_dates]

# ── Weighted metrics ─────────────────────────────────────────
def weighted_r2(y, p, w):
    wn = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
    ybar = np.sum(wn * y)
    ss_res = np.sum(wn * (y - p) ** 2)
    ss_tot = np.sum(wn * (y - ybar) ** 2)
    return 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

monthly_mse = {}
r2_vals = {}
for mn, pred in all_preds.items():
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        wn = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(wn * (a - p) ** 2)))
    monthly_mse[mn] = mse_list
    r2_vals[mn] = weighted_r2(actual, pred, weight)

avg_mse = {mn: float(np.mean(v)) for mn, v in monthly_mse.items()}
sorted_models = sorted(avg_mse.items(), key=lambda x: x[1])

print("\n  Average MSE (×100%):")
for rank, (mn, mse) in enumerate(sorted_models, 1):
    print(f"  {rank:2d}  {mn:15s}  {mse*100:7.4f}   R²={r2_vals[mn]:+.4f}")

# Backward-compat CSV
pd.DataFrame(sorted_models, columns=['model', 'avg_mse']).to_csv(
    f'{OUT}/summary_mse.csv', index=False)

# ── Excel 1: metrics.xlsx ────────────────────────────────────
summary_df = pd.DataFrame({
    'model': [m for m, _ in sorted_models],
    'avg_mse': [mse for _, mse in sorted_models],
    'r2': [r2_vals[m] for m, _ in sorted_models],
})

monthly_df = pd.DataFrame(
    {mn: monthly_mse[mn] for mn in summary_df['model']},
    index=pd.to_datetime(unique_dates))
monthly_df.index.name = 'date'

r2_df = pd.DataFrame(
    {'model': list(r2_vals.keys()), 'r2': list(r2_vals.values())}
).sort_values('r2', ascending=False)

with pd.ExcelWriter(f'{OUT}/metrics.xlsx', engine='openpyxl') as w:
    summary_df.to_excel(w, sheet_name='avg_mse', index=False)
    monthly_df.to_excel(w, sheet_name='monthly_mse')
    r2_df.to_excel(w, sheet_name='r2', index=False)
print(f"  {OUT}/metrics.xlsx")

# ── Excel 2: feature_stats.xlsx (exclude IND_*) ─────────────
with open(f'{OUT}/stage_prep.pkl', 'rb') as f:
    prep = pickle.load(f)
panel = prep['panel']
feat_cols_all = prep['feat_cols']
non_ind_cols = [c for c in feat_cols_all if not c.startswith('IND_')]
stats_df = panel[non_ind_cols].agg(['mean', 'var']).T
stats_df.columns = ['mean', 'variance']
stats_df.index.name = 'feature'
stats_df.to_excel(f'{OUT}/feature_stats.xlsx', sheet_name='feature_stats')
print(f"  {OUT}/feature_stats.xlsx  ({len(non_ind_cols)} features)")

# ── Per-model loss plots ────────────────────────────────────
def plot_loss(curves, title, out_path):
    """curves: list of {'ntrees': arr, 'mse': arr} (one per period)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, c in enumerate(curves):
        ax.plot(c['ntrees'], c['mse'], alpha=0.3, lw=0.8, color='#4472C4')
    # Overlay mean curve (only if curves share length and x)
    lens = [len(c['ntrees']) for c in curves]
    if len(set(lens)) == 1:
        mses = np.stack([c['mse'] for c in curves])
        mean_mse = mses.mean(axis=0)
        ax.plot(curves[0]['ntrees'], mean_mse, color='red', lw=2, label='mean across periods')
        ax.legend()
    ax.set_xlabel('Number of trees / iterations')
    ax.set_ylabel('Validation MSE')
    ax.set_title(title, fontweight='bold')
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")

if rf_gbrt is not None:
    plot_loss(rf_gbrt['loss_rf'],   'Random Forest Loss (x=n_trees)',       f'{OUT}/loss_rf.png')
    plot_loss(rf_gbrt['loss_gbrt'], 'GBRT Loss (x=boosting iterations)',     f'{OUT}/loss_gbrt.png')
if xgb_lgb is not None:
    plot_loss(xgb_lgb['loss_xgb'],  'XGBoost Loss (x=boosting iterations)',  f'{OUT}/loss_xgboost.png')
    plot_loss(xgb_lgb['loss_lgb'],  'LightGBM Loss (x=boosting iterations)', f'{OUT}/loss_lightgbm.png')

# ── Per-model prediction vs actual (monthly value-weighted) ─
def plot_pred_vs_actual(model_name, out_path):
    pred = all_preds[model_name]
    actual_m, pred_m = [], []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        wn = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        actual_m.append(float(np.sum(wn * a)))
        pred_m.append(float(np.sum(wn * p)))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(unique_ts, actual_m, label='Actual (value-weighted)', color='black', lw=1.5)
    ax.plot(unique_ts, pred_m,   label=f'{model_name} predicted',  color='#ED7D31', lw=1.5, alpha=.85)
    ax.set_title(f'{model_name}: Predicted vs Actual (value-weighted cross-sectional mean)',
                 fontweight='bold')
    ax.set_xlabel('Date'); ax.set_ylabel('Realized beta')
    ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")

for mn, fn in [('RF', 'rf'), ('GBRT', 'gbrt'),
               ('XGBoost', 'xgboost'), ('LightGBM', 'lightgbm')]:
    if mn in all_preds:
        plot_pred_vs_actual(mn, f'{OUT}/pred_vs_actual_{fn}.png')

# ── Per-model feature importance ────────────────────────────
def plot_fi(imp_series, title, out_path, xlabel='Mean |SHAP value|'):
    imp_no_ind = imp_series[[c for c in imp_series.index if not c.startswith('IND_')]]
    top = imp_no_ind.nlargest(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top.values[::-1], color='#4472C4')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1])
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight='bold')
    ax.grid(axis='x', alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")

fi_entries = []
if rf_gbrt is not None:
    if rf_gbrt.get('shap_rf'):
        fi_entries.append(('RF', rf_gbrt['shap_rf'], rf_gbrt['feat_cols'], 'rf', 'Mean |SHAP value|'))
    elif rf_gbrt.get('fi_rf'):
        fi_entries.append(('RF', rf_gbrt['fi_rf'], rf_gbrt['feat_cols'], 'rf', 'Importance (normalized)'))
if xgb_lgb is not None:
    if xgb_lgb.get('shap_xgb'):
        fi_entries.append(('XGBoost', xgb_lgb['shap_xgb'], xgb_lgb['feat_cols'], 'xgboost', 'Mean |SHAP value|'))
    elif xgb_lgb.get('fi_xgb'):
        fi_entries.append(('XGBoost', xgb_lgb['fi_xgb'], xgb_lgb['feat_cols'], 'xgboost', 'Importance (normalized)'))
    if xgb_lgb.get('shap_lgb'):
        fi_entries.append(('LightGBM', xgb_lgb['shap_lgb'], xgb_lgb['feat_cols'], 'lightgbm', 'Mean |SHAP value|'))
    elif xgb_lgb.get('fi_lgb'):
        fi_entries.append(('LightGBM', xgb_lgb['fi_lgb'], xgb_lgb['feat_cols'], 'lightgbm', 'Importance (normalized)'))

for label, fi_list, fc, fn, xlabel in fi_entries:
    raw = np.mean(fi_list, axis=0)
    # For SHAP-based: use raw mean |SHAP|; for model FI: normalize to sum=1
    if xlabel == 'Mean |SHAP value|':
        s = pd.Series(raw, index=fc)
    else:
        s = pd.Series(raw / raw.sum() if raw.sum() > 0 else raw, index=fc)
    plot_fi(s, f'{label} Feature Importance (Top 15, ex-industry)', f'{OUT}/fi_{fn}.png', xlabel=xlabel)

# ── SHAP density plots (RF / XGBoost / LightGBM) ────────────
def compute_and_plot_shap(model, Xte_s, feat_cols_, model_label, out_path):
    if os.path.exists(out_path):
        print(f"  SHAP {model_label}: {out_path} exists, skipping")
        return
    # Subsample for speed / memory
    if len(Xte_s) > SHAP_MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(Xte_s), SHAP_MAX_ROWS, replace=False)
        Xs = Xte_s[idx]
    else:
        Xs = Xte_s
    print(f"  SHAP {model_label}: explaining {len(Xs)} rows × {Xs.shape[1]} features ...")
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(Xs)
    # Exclude IND_* columns from plot
    mask = np.array([not c.startswith('IND_') for c in feat_cols_])
    keep_cols = [c for c, m in zip(feat_cols_, mask) if m]
    sv_keep = sv[:, mask]
    Xs_keep = Xs[:, mask]
    shap.summary_plot(sv_keep, Xs_keep, feature_names=keep_cols,
                      show=False, max_display=20, plot_size=(10, 10))
    fig = plt.gcf()
    fig.suptitle(f'{model_label} SHAP (last test period)', fontweight='bold', y=1.01)
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close('all')
    print(f"  {out_path}")

if rf_gbrt is not None and rf_gbrt.get('last_rf_model') is not None:
    compute_and_plot_shap(rf_gbrt['last_rf_model'], rf_gbrt['last_Xte_s'],
                          rf_gbrt['feat_cols'], 'Random Forest',
                          f'{OUT}/shap_rf.png')
if xgb_lgb is not None and xgb_lgb.get('last_xgb_model') is not None:
    compute_and_plot_shap(xgb_lgb['last_xgb_model'], xgb_lgb['last_Xte_s'],
                          xgb_lgb['feat_cols'], 'XGBoost',
                          f'{OUT}/shap_xgboost.png')
if xgb_lgb is not None and xgb_lgb.get('last_lgb_model') is not None:
    compute_and_plot_shap(xgb_lgb['last_lgb_model'], xgb_lgb['last_Xte_s'],
                          xgb_lgb['feat_cols'], 'LightGBM',
                          f'{OUT}/shap_lightgbm.png')

print("\n[Stage F v2 Complete] All outputs in", OUT)
