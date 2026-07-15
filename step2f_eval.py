#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage F: Full evaluation + plots
Requires: results/stage_bench_lm.pkl
          results/stage_rf_gbrt.pkl
          results/stage_xgb_lgb.pkl   (optional)
          results/stage_nn.pkl         (optional)
Output:   results/fig*.png, results/summary_mse.csv
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = "results/"

# ── Load all stages ──────────────────────────────────────────
print("Loading results ...")
with open(f'{OUT}/stage_bench_lm.pkl', 'rb') as f:
    bl = pickle.load(f)
actual    = bl['actual']
weight    = bl['weight']
dates_arr = bl['dates_arr']
all_preds = dict(bl['preds'])  # benchmarks + LM + ELANET

for fname, label in [('stage_rf_gbrt.pkl', 'RF/GBRT'),
                     ('stage_xgb_lgb.pkl', 'XGB/LGB'),
                     ('stage_nn.pkl',       'NN')]:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        all_preds.update(d['preds'])
        print(f"  Loaded {label}: {list(d['preds'].keys())}")
    else:
        print(f"  Skipping {fname} (not found)")

print(f"  Total models: {list(all_preds.keys())}")

# ── DM test ──────────────────────────────────────────────────
def dm_test(mse_a, mse_b):
    d = np.array(mse_a) - np.array(mse_b)
    if len(d) < 2 or np.std(d) == 0:
        return 0.0
    return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))

# ── Evaluation ───────────────────────────────────────────────
print("\n" + "="*60)
print("Evaluation")
print("="*60)

unique_dates = sorted(set(dates_arr))
monthly_mse = {}
for mn, pred in all_preds.items():
    mse_list = []
    for dt in unique_dates:
        mask = dates_arr == dt
        a = actual[mask]; p = pred[mask]; w = weight[mask]
        w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
        mse_list.append(float(np.sum(w * (a - p)**2)))
    monthly_mse[mn] = mse_list

avg_mse = {mn: np.mean(v) for mn, v in monthly_mse.items()}
sorted_models = sorted(avg_mse.items(), key=lambda x: x[1])

print("\n  Average MSE (×100%):")
print(f"  {'Rank':>4s}  {'Model':15s}  {'MSE%':>8s}")
print("  " + "-"*32)
for rank, (mn, mse) in enumerate(sorted_models, 1):
    star = " ★" if rank <= 3 else ""
    print(f"  {rank:4d}  {mn:15s}  {mse*100:8.4f}{star}")

pd.DataFrame(sorted_models, columns=['model', 'avg_mse']).to_csv(
    f'{OUT}/summary_mse.csv', index=False)

best = sorted_models[0][0]
print(f"\n  DM test vs {best}:")
for mn, mse in sorted_models[1:]:
    t = dm_test(monthly_mse[mn], monthly_mse[best])
    sig = "***" if abs(t)>2.576 else "**" if abs(t)>1.96 else "*" if abs(t)>1.645 else ""
    print(f"    {mn:15s}: t={t:6.2f} {sig}")

# ── Plots ────────────────────────────────────────────────────
bench_order = ['OLS_5Y_M','OLS_1Y_D','BSW','HYBRID','FAMA_FRENCH','LONG_MEMO']
ml_order    = ['LM','ELANET','RF','GBRT','XGBoost','LightGBM','NN_1']
order = [m for m in bench_order + ml_order if m in avg_mse]
unique_ts = [pd.Timestamp(d) for d in unique_dates]

# Fig 1: Bar chart
fig, ax = plt.subplots(figsize=(14, 6))
vals = [avg_mse[m]*100 for m in order]
colors = ['#4472C4' if m in bench_order else '#ED7D31' for m in order]
bars = ax.bar(range(len(order)), vals, color=colors, edgecolor='k', lw=.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.002, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_ylabel('Value-Weighted MSE (%)')
ax.set_title('Figure 1: Average Forecast Errors', fontweight='bold', fontsize=14)
ax.legend(handles=[Patch(fc='#4472C4', label='Benchmark'),
                   Patch(fc='#ED7D31', label='Machine Learning')], fontsize=11)
ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_avg_mse.png', dpi=150); plt.close()
print("\n  fig1_avg_mse.png")

# Fig 2: MSE over time
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
c_bench = {'OLS_1Y_D':'#4472C4','OLS_5Y_M':'#ED7D31','BSW':'grey',
           'HYBRID':'#FFC000','LONG_MEMO':'#70AD47'}
for mn, c in c_bench.items():
    if mn in monthly_mse:
        ax1.plot(unique_ts, [v*100 for v in monthly_mse[mn]], label=mn, color=c, alpha=.7, lw=1)
available_ml = [m for m in ml_order if m in avg_mse]
if available_ml:
    best_ml = min(available_ml, key=lambda m: avg_mse[m])
    ax1.plot(unique_ts, [v*100 for v in monthly_mse[best_ml]],
             label=f'{best_ml}(Best ML)', color='red', lw=2, ls='--')
ax1.set_title('Panel A: Benchmarks vs Best ML', fontweight='bold')
ax1.set_ylabel('MSE (%)'); ax1.legend(fontsize=9); ax1.grid(alpha=.3)

c_ml = {'LM':'#4472C4','ELANET':'#ED7D31','RF':'grey','GBRT':'#FFC000',
        'XGBoost':'#5B9BD5','LightGBM':'#70AD47','NN_1':'#FF6384'}
for mn in available_ml:
    s = pd.Series([v*100 for v in monthly_mse[mn]], index=unique_ts).rolling(12, min_periods=1).mean()
    ax2.plot(s.index, s.values, label=mn, color=c_ml.get(mn,'k'), lw=1.5)
ax2.set_title('Panel B: ML Models (12-month rolling avg)', fontweight='bold')
ax2.set_ylabel('MSE (%)'); ax2.set_xlabel('Year'); ax2.legend(fontsize=9); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_mse_time.png', dpi=150); plt.close()
print("  fig2_mse_time.png")

# Fig 3: Relative error
if available_ml:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
    for ax, comp, clr, plabel in [(ax1,'OLS_1Y_D','#4472C4','A'),(ax2,'LM','#ED7D31','B')]:
        if comp in monthly_mse and best_ml in monthly_mse:
            rel = [1 - monthly_mse[best_ml][i] / monthly_mse[comp][i]
                   if monthly_mse[comp][i] > 0 else 0
                   for i in range(len(unique_ts))]
            rel100 = [r * 100 for r in rel]
            ax.bar(unique_ts, rel100, width=25, alpha=.5, color=clr, label='Monthly')
            df_r = pd.DataFrame({'date': unique_ts, 'rel': rel100})
            df_r['year'] = df_r['date'].dt.year
            yr = df_r.groupby('year').agg({'date': 'first', 'rel': 'mean'})
            ax.plot(yr['date'], yr['rel'], 'ro-', ms=5, lw=2, label='Yearly Avg')
            ax.axhline(0, color='k', lw=.5)
            ax.set_title(f'Panel {plabel}: {best_ml} vs {comp}', fontweight='bold')
            ax.set_ylabel('Relative MSE Reduction (%)')
            ax.legend(); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig3_relative_error.png', dpi=150); plt.close()
    print("  fig3_relative_error.png")

# Fig 4: Feature importance
fi_data = {}
for fname, key, label in [('stage_rf_gbrt.pkl','fi_rf','RF'),
                           ('stage_xgb_lgb.pkl','fi_xgb','XGBoost'),
                           ('stage_xgb_lgb.pkl','fi_lgb','LightGBM')]:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        with open(path, 'rb') as f:
            dd = pickle.load(f)
        if key in dd and dd[key]:
            feat_cols = dd['feat_cols']
            raw = np.mean(dd[key], axis=0)
            fi_data[label] = pd.Series(raw / raw.sum(), index=feat_cols)

if fi_data:
    n_plots = len(fi_data)
    fig, axes = plt.subplots(1, n_plots, figsize=(8*n_plots, 8))
    if n_plots == 1: axes = [axes]
    for ax, (label, imp) in zip(axes, fi_data.items()):
        imp_no_ind = imp[[c for c in imp.index if not c.startswith('IND_')]]
        top = imp_no_ind.nlargest(15)
        ax.barh(range(len(top)), top.values[::-1], color='#4472C4')
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top.index[::-1])
        ax.set_title(f'{label} Feature Importance', fontweight='bold')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig4_feature_importance.png', dpi=150); plt.close()
    print("  fig4_feature_importance.png")

print("\n[Stage F Complete] All results in results/")
