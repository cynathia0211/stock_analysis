#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation for upside/downside beta analysis.

For both up and down regimes, produces:
  1. VW-MAE and VW-MSE tables (Excel)
  2. DM test heatmap (MSE-based)
  3. SHAP feature importance for RF, XGBoost, LightGBM
  4. Pre/post 2020 sub-sample VW-MAE

Input:  results_up/ and results_down/ (stage_*.pkl files)
Output: results_updown/
"""
import os, pickle, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
                                    'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 300

OUT = 'results_updown/'
os.makedirs(OUT, exist_ok=True)

MODEL_ORDER = ['OLS_5Y_M', 'OLS_1Y_D', 'BSW', 'HYBRID', 'FAMA_FRENCH', 'LONG_MEMO',
               'LM', 'ELANET', 'RF', 'GBRT', 'XGBoost', 'LightGBM', 'NN_1']

# ── Load results for both regimes ────────────────────────────
def load_regime(regime_dir):
    print(f"Loading {regime_dir} ...")
    with open(f'{regime_dir}/stage_bench_lm.pkl', 'rb') as f:
        bl = pickle.load(f)
    actual    = bl['actual']
    weight    = bl['weight']
    dates_arr = bl['dates_arr']
    all_preds = dict(bl['preds'])

    for fname, label in [('stage_rf_gbrt.pkl', 'RF/GBRT'),
                         ('stage_xgb_lgb.pkl', 'XGB/LGB'),
                         ('stage_nn.pkl',       'NN')]:
        path = os.path.join(regime_dir, fname)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                dd = pickle.load(f)
            all_preds.update(dd['preds'])
            print(f"  Loaded {label}: {list(dd['preds'].keys())}")

    shap_data = {}
    rg_path = os.path.join(regime_dir, 'stage_rf_gbrt.pkl')
    if os.path.exists(rg_path):
        with open(rg_path, 'rb') as f:
            rg = pickle.load(f)
        if rg.get('shap_rf'):
            shap_data['RF'] = (rg['shap_rf'], rg['feat_cols'])

    xl_path = os.path.join(regime_dir, 'stage_xgb_lgb.pkl')
    if os.path.exists(xl_path):
        with open(xl_path, 'rb') as f:
            xl = pickle.load(f)
        if xl.get('shap_xgb'):
            shap_data['XGBoost'] = (xl['shap_xgb'], xl['feat_cols'])
        if xl.get('shap_lgb'):
            shap_data['LightGBM'] = (xl['shap_lgb'], xl['feat_cols'])

    model_order = [m for m in MODEL_ORDER if m in all_preds]
    print(f"  Models: {model_order}")
    return actual, weight, dates_arr, all_preds, model_order, shap_data


# ── DM test ──────────────────────────────────────────────────
def dm_test(mse_a, mse_b):
    d = np.array(mse_a) - np.array(mse_b)
    if len(d) < 2 or np.std(d) == 0:
        return 0.0
    return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))


# ── Compute monthly metrics ─────────────────────────────────
def compute_monthly_metrics(actual, weight, dates_arr, all_preds, model_order):
    unique_dates = sorted(set(dates_arr))
    monthly_mse = {}
    monthly_mae = {}
    for mn in model_order:
        pred = all_preds[mn]
        mse_list, mae_list = [], []
        for dt in unique_dates:
            mask = dates_arr == dt
            a = actual[mask]; p = pred[mask]; w = weight[mask]
            wn = w / w.sum() if w.sum() > 0 else np.ones_like(w) / len(w)
            mse_list.append(float(np.sum(wn * (a - p)**2)))
            mae_list.append(float(np.sum(wn * np.abs(a - p))))
        monthly_mse[mn] = mse_list
        monthly_mae[mn] = mae_list

    avg_mse = {mn: float(np.mean(v)) for mn, v in monthly_mse.items()}
    avg_mae = {mn: float(np.mean(v)) for mn, v in monthly_mae.items()}

    return unique_dates, monthly_mse, monthly_mae, avg_mse, avg_mae


# ── Pre/post 2020 MAE ───────────────────────────────────────
def compute_subsample_mae(unique_dates, monthly_mae, model_order):
    dates_ts = [pd.Timestamp(d) for d in unique_dates]
    cutoff = pd.Timestamp('2020-01-01')
    pre_idx  = [i for i, d in enumerate(dates_ts) if d < cutoff]
    post_idx = [i for i, d in enumerate(dates_ts) if d >= cutoff]

    pre_mae, post_mae = {}, {}
    for mn in model_order:
        vals = monthly_mae[mn]
        pre_mae[mn]  = float(np.mean([vals[i] for i in pre_idx])) if pre_idx else np.nan
        post_mae[mn] = float(np.mean([vals[i] for i in post_idx])) if post_idx else np.nan

    return pre_mae, post_mae


# ── Plot: DM heatmap ────────────────────────────────────────
def plot_dm_heatmap(monthly_mse, model_order, title, out_path):
    n = len(model_order)
    dm_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dm_matrix[i, j] = dm_test(monthly_mse[model_order[i]],
                                           monthly_mse[model_order[j]])

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(dm_matrix, cmap='RdYlGn_r', aspect='auto', vmin=-5, vmax=5)

    ax.set_xticks(range(n))
    ax.set_xticklabels(model_order, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(range(n))
    ax.set_yticklabels(model_order, fontsize=10)

    for i in range(n):
        for j in range(n):
            if i != j:
                val = dm_matrix[i, j]
                color = 'white' if abs(val) > 3 else 'black'
                ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                        fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label='DM Statistic', shrink=0.8)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")


# ── Plot: SHAP feature importance ────────────────────────────
def plot_shap_fi(shap_list, feat_cols, model_label, out_path, top_n=15):
    raw = np.mean(shap_list, axis=0)
    s = pd.Series(raw, index=feat_cols)
    s_no_ind = s[[c for c in s.index if not c.startswith('IND_')]]
    top = s_no_ind.nlargest(top_n)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(top)), top.values[::-1], color='#4472C4')
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(top.index[::-1])
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'{model_label} Feature Importance (Top {top_n}, ex-industry)',
                 fontweight='bold')
    ax.grid(axis='x', alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")


# ── Plot: VW-MAE bar chart (pre/post 2020) ──────────────────
def plot_subsample_mae(pre_mae, post_mae, model_order, title, out_path):
    models = [m for m in model_order if m in pre_mae]
    pre_vals  = [pre_mae[m] * 100 for m in models]
    post_vals = [post_mae[m] * 100 for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(14, 6))
    bars1 = ax.bar(x - width/2, pre_vals, width, label='Pre-2020', color='#4472C4')
    bars2 = ax.bar(x + width/2, post_vals, width, label='Post-2020', color='#ED7D31')

    for bars in [bars1, bars2]:
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                    f'{b.get_height():.3f}',
                    ha='center', va='bottom', fontsize=7, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('Value-Weighted MAE (%)')
    ax.set_title(title, fontweight='bold', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"  {out_path}")


# ══════════════════════════════════════════════════════════════
# Process each regime
# ══════════════════════════════════════════════════════════════
regimes = {}
for regime, regime_dir in [('up', 'results_up'), ('down', 'results_down')]:
    if not os.path.exists(f'{regime_dir}/stage_bench_lm.pkl'):
        print(f"\nSkipping {regime} regime (no results found)")
        continue

    print(f"\n{'='*60}")
    print(f"Processing {regime.upper()} regime")
    print(f"{'='*60}")

    actual, weight, dates_arr, all_preds, model_order, shap_data = load_regime(regime_dir)
    unique_dates, monthly_mse, monthly_mae, avg_mse, avg_mae = \
        compute_monthly_metrics(actual, weight, dates_arr, all_preds, model_order)
    pre_mae, post_mae = compute_subsample_mae(unique_dates, monthly_mae, model_order)

    regimes[regime] = {
        'model_order': model_order,
        'avg_mse': avg_mse,
        'avg_mae': avg_mae,
        'monthly_mse': monthly_mse,
        'monthly_mae': monthly_mae,
        'pre_mae': pre_mae,
        'post_mae': post_mae,
        'unique_dates': unique_dates,
        'shap_data': shap_data,
    }

    # Print tables
    sorted_by_mse = sorted(avg_mse.items(), key=lambda x: x[1])
    sorted_by_mae = sorted(avg_mae.items(), key=lambda x: x[1])

    print(f"\n  VW-MSE ranking ({regime}):")
    print(f"  {'Rank':>4s}  {'Model':15s}  {'MSE%':>8s}")
    print("  " + "-"*32)
    for rank, (mn, mse) in enumerate(sorted_by_mse, 1):
        print(f"  {rank:4d}  {mn:15s}  {mse*100:8.4f}")

    print(f"\n  VW-MAE ranking ({regime}):")
    print(f"  {'Rank':>4s}  {'Model':15s}  {'MAE%':>8s}")
    print("  " + "-"*32)
    for rank, (mn, mae) in enumerate(sorted_by_mae, 1):
        print(f"  {rank:4d}  {mn:15s}  {mae*100:8.4f}")

    print(f"\n  Pre/Post 2020 VW-MAE ({regime}):")
    print(f"  {'Model':15s}  {'Pre-2020%':>10s}  {'Post-2020%':>10s}")
    print("  " + "-"*40)
    for mn in model_order:
        print(f"  {mn:15s}  {pre_mae[mn]*100:10.4f}  {post_mae[mn]*100:10.4f}")

    # ── Generate plots ──
    label_zh = '上行' if regime == 'up' else '下行'

    # DM heatmap
    plot_dm_heatmap(monthly_mse, model_order,
                    f'DM Test Heatmap ({label_zh} Beta)\n(+: row has higher MSE)',
                    f'{OUT}/dm_heatmap_{regime}.png')

    # SHAP feature importance
    for model_name, (shap_list, feat_cols) in shap_data.items():
        fn = model_name.lower().replace(' ', '_')
        plot_shap_fi(shap_list, feat_cols, f'{model_name} ({label_zh} Beta)',
                     f'{OUT}/shap_{fn}_{regime}.png')

    # Pre/post 2020 MAE bar chart
    plot_subsample_mae(pre_mae, post_mae, model_order,
                       f'VW-MAE by Sub-period ({label_zh} Beta)',
                       f'{OUT}/mae_subsample_{regime}.png')


# ── Save combined Excel ─────────────────────────────────────
print(f"\n{'='*60}")
print("Saving Excel summary")
print("="*60)

with pd.ExcelWriter(f'{OUT}/metrics_updown.xlsx', engine='openpyxl') as writer:
    for regime in ['up', 'down']:
        if regime not in regimes:
            continue
        r = regimes[regime]
        label = 'Upside' if regime == 'up' else 'Downside'

        # VW-MSE summary
        sorted_mse = sorted(r['avg_mse'].items(), key=lambda x: x[1])
        mse_df = pd.DataFrame({
            'model': [m for m, _ in sorted_mse],
            'avg_vw_mse': [v for _, v in sorted_mse],
        })
        mse_df.to_excel(writer, sheet_name=f'{label}_MSE', index=False)

        # VW-MAE summary
        sorted_mae = sorted(r['avg_mae'].items(), key=lambda x: x[1])
        mae_df = pd.DataFrame({
            'model': [m for m, _ in sorted_mae],
            'avg_vw_mae': [v for _, v in sorted_mae],
        })
        mae_df.to_excel(writer, sheet_name=f'{label}_MAE', index=False)

        # Monthly MSE
        monthly_mse_df = pd.DataFrame(r['monthly_mse'],
                                       index=pd.to_datetime(r['unique_dates']))
        monthly_mse_df.index.name = 'date'
        monthly_mse_df.to_excel(writer, sheet_name=f'{label}_Monthly_MSE')

        # Monthly MAE
        monthly_mae_df = pd.DataFrame(r['monthly_mae'],
                                       index=pd.to_datetime(r['unique_dates']))
        monthly_mae_df.index.name = 'date'
        monthly_mae_df.to_excel(writer, sheet_name=f'{label}_Monthly_MAE')

        # Pre/post 2020 MAE
        subsample_df = pd.DataFrame({
            'model': r['model_order'],
            'pre_2020_mae': [r['pre_mae'][m] for m in r['model_order']],
            'post_2020_mae': [r['post_mae'][m] for m in r['model_order']],
        })
        subsample_df.to_excel(writer, sheet_name=f'{label}_MAE_2020', index=False)

print(f"  {OUT}/metrics_updown.xlsx")

print(f"\n[Evaluation Complete] All outputs in {OUT}")
