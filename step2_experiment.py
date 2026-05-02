#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estimating Stock Market Betas via Machine Learning (Chinese A-shares)
Based on Drobetz, Hollstein, Otto, and Prokopczuk (2025)

Removed: DIVPAY, RELPRC, DFY (no domestic data)
Removed benchmarks: EWMA_S, EWMA_L, VASICEK, KAROLYI
Added: XGBoost, LightGBM
"""
import os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.linear_model import LinearRegression, ElasticNetCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore')

PKL = "data/pkl/"
OUT = "results/"
os.makedirs(OUT, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────
def load(name):
    return pd.read_pickle(os.path.join(PKL, name + ".pkl"))

def vw_mse(actual, pred, weight):
    """Value-weighted MSE."""
    w = weight / weight.sum()
    return float(np.sum(w * (actual - pred)**2))

def dm_test(mse_a, mse_b):
    """DM test: positive t => a has higher MSE than b (b is better)."""
    d = np.array(mse_a) - np.array(mse_b)
    if len(d) < 2 or np.std(d) == 0:
        return 0.0
    return float(np.mean(d) / (np.std(d, ddof=1) / np.sqrt(len(d))))

# ── 1. Load data (from pkl, fast) ───────────────────────────
print("Loading pkl data ...")
t0 = time.time()
daily_ret   = load("daily_ret") / 100        # pct -> decimal
monthly_ret = load("monthly_ret") / 100
mkt_d       = load("market_daily_ret").iloc[:, 0] / 100
mkt_m       = load("market_monthly_ret").iloc[:, 0] / 100
mcap        = load("market_cap")
vol         = load("volume")
pe          = load("pe_ratio")
mkt_pe      = load("market_pe").iloc[:, 0]
rf          = load("rf_monthly")['rf']

listing     = load("listing_dates").set_index('stock_code')['listing_date']
ind_df      = load("industry").set_index('stock_code')['industry']

fund = {}
for name in ['fund_total_assets','fund_equity','fund_net_sales','fund_net_income',
             'fund_roe','fund_roa','fund_fixed_cost','fund_fixed_assets',
             'fund_gross_margin','fund_noa','fund_nwcap']:
    fund[name] = load(name)

# Remove market index from stock daily returns
if '000001.SH' in daily_ret.columns:
    daily_ret.drop(columns=['000001.SH'], inplace=True, errors='ignore')

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Daily: {daily_ret.shape}, Monthly: {monthly_ret.shape}")
print(f"  Period: {monthly_ret.index[0].strftime('%Y-%m')} ~ {monthly_ret.index[-1].strftime('%Y-%m')}")

# Common stocks
all_stocks = sorted(set(daily_ret.columns) & set(monthly_ret.columns))
print(f"  Common stocks: {len(all_stocks)}")

# ── 2. Vectorised beta computation ──────────────────────────
print("\nComputing rolling betas (vectorised) ...")
t0 = time.time()


def rolling_beta_matrix(stock_ret, market_ret, window):
    """Compute rolling beta for ALL stocks at once.
    stock_ret : DataFrame (dates × stocks)
    market_ret: Series (dates)
    window    : int (number of periods)
    Returns   : DataFrame (dates × stocks) of betas
    """
    aligned_mkt = market_ret.reindex(stock_ret.index)
    # Expanding/rolling cov and var
    betas = pd.DataFrame(index=stock_ret.index, columns=stock_ret.columns, dtype=float)

    # Use rolling windows
    for i in range(window, len(stock_ret)):
        s = stock_ret.iloc[i-window:i].values   # (window, n_stocks)
        m = aligned_mkt.iloc[i-window:i].values  # (window,)

        # Remove NaN: per-stock mask
        valid_m = np.isfinite(m)
        m_clean = m.copy()
        m_clean[~valid_m] = 0

        # Vectorised: cov(s, m) / var(m) for each stock
        s_vals = s.copy()
        s_vals[~np.isfinite(s_vals)] = 0

        # Count valid obs per stock
        valid_s = np.isfinite(stock_ret.iloc[i-window:i].values)
        valid_both = valid_s & valid_m[:, None]
        n_valid = valid_both.sum(axis=0)

        # Means
        m_sum = (m_clean[:, None] * valid_both).sum(axis=0)
        s_sum = (s_vals * valid_both).sum(axis=0)
        m_mean = np.where(n_valid > 0, m_sum / n_valid, 0)
        s_mean = np.where(n_valid > 0, s_sum / n_valid, 0)

        # Cov and var
        dm = m_clean[:, None] - m_mean[None, :]  # (window, n_stocks)
        ds = s_vals - s_mean[None, :]
        dm = dm * valid_both
        ds = ds * valid_both

        cov = (dm * ds).sum(axis=0) / np.maximum(n_valid - 1, 1)
        var = (dm * dm).sum(axis=0) / np.maximum(n_valid - 1, 1)

        beta_row = np.where((var > 1e-12) & (n_valid >= window * 0.5), cov / var, np.nan)
        betas.iloc[i] = beta_row

    return betas


# We compute betas at month-end dates only for efficiency
month_ends = monthly_ret.index.tolist()

# Precompute: for each month-end, which daily dates fall in 3m/1y windows
def _beta_vec(sv, mv, min_obs):
    """Vectorised beta from (T, N) stock returns and (T,) market returns."""
    valid_m = np.isfinite(mv)
    valid_s = np.isfinite(sv)
    valid = valid_s & valid_m[:, None]
    n = valid.sum(axis=0)
    sv_c = np.where(valid, sv, 0.0)
    mv_c = np.where(valid_m, mv, 0.0)
    m_sum = (mv_c[:, None] * valid).sum(axis=0)
    s_sum = (sv_c * valid).sum(axis=0)
    m_mean = np.where(n > 0, m_sum / n, 0)
    s_mean = np.where(n > 0, s_sum / n, 0)
    dm = (mv_c[:, None] - m_mean[None, :]) * valid
    ds = (sv_c - s_mean[None, :]) * valid
    cov = (dm * ds).sum(axis=0) / np.maximum(n - 1, 1)
    var_m = (dm * dm).sum(axis=0) / np.maximum(n - 1, 1)
    return np.where((var_m > 1e-12) & (n >= min_obs), cov / var_m, np.nan)


def compute_betas_at_months(daily_ret_df, mkt_daily, month_ends, window_months, min_obs):
    """Compute beta for each stock at each month-end using daily returns.
    Uses numpy array for results to avoid slow DataFrame.iloc assignment."""
    cols = daily_ret_df.columns
    n_months = len(month_ends)
    n_stocks = len(cols)
    result_arr = np.full((n_months, n_stocks), np.nan)
    daily_vals = daily_ret_df.values  # (D, N) preconvert once
    daily_idx = daily_ret_df.index
    mkt_vals = mkt_daily.reindex(daily_idx).values  # (D,)

    for i, mdate in enumerate(month_ends):
        start = mdate - pd.DateOffset(months=window_months)
        idx_mask = (daily_idx > start) & (daily_idx <= mdate)
        rows_idx = np.where(idx_mask)[0]
        if len(rows_idx) < min_obs:
            continue
        sv = daily_vals[rows_idx]  # (T, N)
        mv = mkt_vals[rows_idx]    # (T,)
        result_arr[i] = _beta_vec(sv, mv, min_obs)
        if (i + 1) % 48 == 0:
            print(f"    {mdate.strftime('%Y-%m')} ({i+1}/{n_months})", flush=True)

    return pd.DataFrame(result_arr, index=month_ends, columns=cols)


print("  OLS_3M_D (3-month daily) ...", flush=True)
beta_3m = compute_betas_at_months(daily_ret, mkt_d, month_ends, 3, 30)
print("  OLS_1Y_D (1-year daily) ...", flush=True)
beta_1y = compute_betas_at_months(daily_ret, mkt_d, month_ends, 12, 120)

# OLS_5Y_M: monthly returns, 5-year window
print("  OLS_5Y_M (5-year monthly) ...", flush=True)
monthly_vals = monthly_ret.values
monthly_idx = monthly_ret.index
mkt_m_vals = mkt_m.reindex(monthly_idx).values
n_mstocks = len(monthly_ret.columns)
beta_5y_arr = np.full((len(month_ends), n_mstocks), np.nan)
for i, mdate in enumerate(month_ends):
    start = mdate - pd.DateOffset(months=60)
    idx_mask = (monthly_idx > start) & (monthly_idx <= mdate)
    rows_idx = np.where(idx_mask)[0]
    if len(rows_idx) < 24:
        continue
    beta_5y_arr[i] = _beta_vec(monthly_vals[rows_idx], mkt_m_vals[rows_idx], 24)
beta_5y = pd.DataFrame(beta_5y_arr, index=month_ends, columns=monthly_ret.columns)

# Realized beta (1-year forward daily) for evaluation
print("  Realized betas (1-year forward) ...", flush=True)
daily_vals_r = daily_ret.values
daily_idx_r = daily_ret.index
mkt_d_vals = mkt_d.reindex(daily_idx_r).values
n_dstocks = len(daily_ret.columns)
realized_arr = np.full((len(month_ends), n_dstocks), np.nan)
for i, mdate in enumerate(month_ends):
    end = mdate + pd.DateOffset(months=12)
    idx_mask = (daily_idx_r > mdate) & (daily_idx_r <= end)
    rows_idx = np.where(idx_mask)[0]
    if len(rows_idx) < 120:
        continue
    realized_arr[i] = _beta_vec(daily_vals_r[rows_idx], mkt_d_vals[rows_idx], 120)
    if (i + 1) % 48 == 0:
        print(f"    {mdate.strftime('%Y-%m')} ({i+1}/{len(month_ends)})", flush=True)
realized_beta = pd.DataFrame(realized_arr, index=month_ends, columns=daily_ret.columns)

print(f"  Betas done in {time.time()-t0:.1f}s", flush=True)

# ── 3. Build features (vectorised) ──────────────────────────
print("\nBuilding features ...")
t0 = time.time()

# We'll build features month-by-month but vectorised across stocks
rows = []  # list of dicts for the panel

for idx, mdate in enumerate(month_ends):
    # Need realized beta (1y forward)
    rb = realized_beta.loc[mdate].dropna()
    if len(rb) < 50:
        continue

    stocks = rb.index.tolist()
    n = len(stocks)

    feat = pd.DataFrame(index=stocks)

    # ── Beta estimates ──
    feat['OLS_3M_D'] = beta_3m.loc[mdate, stocks] if mdate in beta_3m.index else np.nan
    feat['OLS_1Y_D'] = beta_1y.loc[mdate, stocks] if mdate in beta_1y.index else np.nan
    feat['OLS_5Y_M'] = beta_5y.loc[mdate, stocks] if mdate in beta_5y.index else np.nan

    # ── Accounting features (use fiscal year with 4-month lag) ──
    fy = mdate.year - 1 if mdate.month >= 5 else mdate.year - 2

    def get_fund(name, stks, year):
        df = fund[name]
        if year not in df.columns:
            return pd.Series(np.nan, index=stks)
        common = stks.intersection(df.index)
        s = pd.Series(np.nan, index=stks)
        if len(common) > 0:
            s.loc[common] = df.loc[common, year]
        return s.astype(float)

    stk_idx = pd.Index(stocks)
    ta = get_fund('fund_total_assets', stk_idx, fy)
    eq = get_fund('fund_equity', stk_idx, fy)
    ns = get_fund('fund_net_sales', stk_idx, fy)
    ni = get_fund('fund_net_income', stk_idx, fy)
    fc = get_fund('fund_fixed_cost', stk_idx, fy)
    fa = get_fund('fund_fixed_assets', stk_idx, fy)
    gm = get_fund('fund_gross_margin', stk_idx, fy)
    roa = get_fund('fund_roa', stk_idx, fy)
    roe = get_fund('fund_roe', stk_idx, fy)
    noa = get_fund('fund_noa', stk_idx, fy)
    nwcap = get_fund('fund_nwcap', stk_idx, fy)

    # Market cap at month-end
    mc = mcap.loc[mdate, stocks].astype(float) if mdate in mcap.index else pd.Series(np.nan, index=stocks)

    # AGE
    if listing is not None:
        common_l = stk_idx.intersection(listing.index)
        age_s = pd.Series(np.nan, index=stocks)
        if len(common_l) > 0:
            years_since = (mdate - listing.reindex(common_l)).dt.days / 365.25
            age_s.loc[common_l] = np.log(years_since.clip(lower=0.01))
        feat['AGE'] = age_s.values

    feat['AT'] = np.log(ta.clip(lower=1)).values
    feat['BM'] = np.log((eq / mc).clip(lower=1e-6)).values
    feat['CAPTURN'] = np.log((ns / ta).clip(lower=1e-6)).values
    feat['FINLEV'] = np.log((ta / mc).clip(lower=1e-6)).values
    feat['FXDCOS'] = np.log((fc / ta).clip(lower=1e-6)).values
    feat['GM'] = (gm / 100).values  # percentage -> ratio
    feat['LEV'] = np.log((ta / eq).clip(lower=1e-6)).values
    feat['LOGME'] = np.log(mc.clip(lower=1)).values
    feat['NOA'] = (noa / ta).values
    feat['NI'] = (ni / ta).values
    feat['OPLEV'] = (fa / ta).values
    feat['ROA'] = (roa / 100).values
    feat['ROE'] = (roe / 100).values
    feat['NWCAP'] = (nwcap / ta).values

    # ── Technical indicators ──
    # IVOL, COSKEW, KURT from daily returns (1y window)
    d1y_start = mdate - pd.DateOffset(months=12)
    d1y = daily_ret.loc[(daily_ret.index > d1y_start) & (daily_ret.index <= mdate), stocks]
    m1y = mkt_d.reindex(d1y.index)

    if len(d1y) >= 120:
        sv = d1y.values   # (T, N)
        mv = m1y.values   # (T,)
        valid_m = np.isfinite(mv)
        valid_s = np.isfinite(sv)
        valid = valid_s & valid_m[:, None]
        n_v = valid.sum(axis=0)

        sv_c = np.where(valid, sv, 0.0)
        mv_c = np.where(valid_m, mv, 0.0)

        m_sum = (mv_c[:, None] * valid).sum(axis=0)
        s_sum = (sv_c * valid).sum(axis=0)
        m_mean = np.where(n_v > 0, m_sum / n_v, 0)
        s_mean = np.where(n_v > 0, s_sum / n_v, 0)

        dm = (mv_c[:, None] - m_mean) * valid
        ds = (sv_c - s_mean) * valid

        cov_sm = (dm * ds).sum(axis=0) / np.maximum(n_v - 1, 1)
        var_mk = (dm * dm).sum(axis=0) / np.maximum(n_v - 1, 1)
        b1y = np.where((var_mk > 1e-12) & (n_v >= 60), cov_sm / var_mk, 0)
        alpha = s_mean - b1y * m_mean

        resid = sv_c - alpha - b1y * mv_c[:, None]
        resid = resid * valid
        resid_sq = (resid ** 2).sum(axis=0) / np.maximum(n_v - 1, 1)
        ivol = np.where(n_v >= 60, np.log(np.sqrt(resid_sq).clip(1e-10)), np.nan)
        feat['IVOL'] = ivol

        # COSKEW = E[e_i * e_m^2] / (std(e_i) * E[e_m^2])
        se = ds  # demeaned stock returns (already valid-masked)
        me = dm  # demeaned market returns
        coskew_num = (se * me**2).sum(axis=0) / np.maximum(n_v, 1)
        coskew_den = np.sqrt((se**2).sum(axis=0) / np.maximum(n_v, 1)) * \
                     ((me**2).sum(axis=0) / np.maximum(n_v, 1))
        feat['COSKEW'] = np.where((coskew_den > 1e-12) & (n_v >= 60),
                                   coskew_num / coskew_den, np.nan)

        # KURT
        s2 = (ds**2).sum(axis=0) / np.maximum(n_v, 1)
        s4 = (ds**4).sum(axis=0) / np.maximum(n_v, 1)
        feat['KURT'] = np.where((s2 > 1e-12) & (n_v >= 60), s4 / s2**2 - 3, np.nan)

    # MOM: cumulative return past 12 months (skip last month)
    mom_start = mdate - pd.DateOffset(months=12)
    mom_end = mdate - pd.DateOffset(months=1)
    mom_w = monthly_ret.loc[(monthly_ret.index >= mom_start) & (monthly_ret.index <= mom_end), stocks]
    if len(mom_w) >= 6:
        cum = (1 + mom_w).prod() - 1
        feat['MOM'] = cum.values

    # TURNOVER: log average monthly volume (12m)
    vol_start = mdate - pd.DateOffset(months=12)
    vol_w = vol.loc[(vol.index >= vol_start) & (vol.index <= mdate)]
    common_v = stk_idx.intersection(vol_w.columns)
    if len(vol_w) > 0 and len(common_v) > 0:
        avg_vol = vol_w[common_v].mean()
        turnover = pd.Series(np.nan, index=stocks)
        turnover.loc[common_v] = np.log(avg_vol.clip(lower=1))
        feat['TURNOVER'] = turnover.values

    # EP_COVAR / EP_VAR
    ep_start = mdate - pd.DateOffset(years=3)
    pe_w = pe.loc[(pe.index >= ep_start) & (pe.index <= mdate)]
    if len(pe_w) >= 12:
        common_pe = stk_idx.intersection(pe_w.columns)
        if len(common_pe) > 0:
            ep_stocks = (1.0 / pe_w[common_pe].replace(0, np.nan))
            mpe_w = mkt_pe.reindex(pe_w.index)
            ep_mkt = 1.0 / mpe_w.replace(0, np.nan)

            ep_var_s = pd.Series(np.nan, index=stocks)
            ep_covar_s = pd.Series(np.nan, index=stocks)

            ev = ep_stocks.values  # (T, N)
            emv = ep_mkt.values    # (T,)
            valid_em = np.isfinite(emv)
            valid_es = np.isfinite(ev)
            valid_ep = valid_es & valid_em[:, None]
            n_ep = valid_ep.sum(axis=0)

            ev_c = np.where(valid_ep, ev, 0)
            emv_c = np.where(valid_em, emv, 0)

            em_sum = (emv_c[:, None] * valid_ep).sum(axis=0)
            es_sum = (ev_c * valid_ep).sum(axis=0)
            em_mean = np.where(n_ep > 0, em_sum / n_ep, 0)
            es_mean = np.where(n_ep > 0, es_sum / n_ep, 0)

            d_em = (emv_c[:, None] - em_mean) * valid_ep
            d_es = (ev_c - es_mean) * valid_ep

            cov_ep = (d_em * d_es).sum(axis=0) / np.maximum(n_ep - 1, 1)
            var_ep = (d_em ** 2).sum(axis=0) / np.maximum(n_ep - 1, 1)
            std_es = np.sqrt((d_es ** 2).sum(axis=0) / np.maximum(n_ep - 1, 1))

            covar_vals = np.where((var_ep > 1e-12) & (n_ep >= 12), cov_ep / var_ep, np.nan)
            var_vals = np.where((std_es > 1e-10) & (n_ep >= 12), np.log(std_es), np.nan)

            ep_covar_s.loc[common_pe] = covar_vals
            ep_var_s.loc[common_pe] = var_vals
            feat['EP_COVAR'] = ep_covar_s.values
            feat['EP_VAR'] = ep_var_s.values

    # MACRO: risk-free rate and term proxy
    if mdate in rf.index:
        feat['RF'] = rf.loc[mdate]
    rf_start = mdate - pd.DateOffset(months=12)
    rf_w = rf.loc[(rf.index >= rf_start) & (rf.index <= mdate)]
    if len(rf_w) > 1:
        feat['TERM'] = rf_w.iloc[-1] - rf_w.iloc[0]

    # Industry dummies
    common_ind = stk_idx.intersection(ind_df.index)
    if len(common_ind) > 0:
        ind_s = ind_df.reindex(stocks)
        dummies = pd.get_dummies(ind_s, prefix='IND')
        for c in dummies.columns:
            feat[c] = dummies[c].values

    # ── Pack ──
    feat['realized_beta'] = rb.values
    feat['weight'] = mc.values
    feat['date'] = mdate
    feat['stock'] = stocks
    feat.index = range(len(feat))
    rows.append(feat)

    if (idx + 1) % 24 == 0:
        print(f"  {mdate.strftime('%Y-%m')} ({idx+1}/{len(month_ends)}) stocks={n}")

panel = pd.concat(rows, ignore_index=True)
print(f"  Panel: {panel.shape}, {time.time()-t0:.1f}s")

# ── 4. Preprocess ────────────────────────────────────────────
print("\nPreprocessing ...")
meta = ['stock', 'date', 'realized_beta', 'weight']
feat_cols = [c for c in panel.columns if c not in meta]
non_ind = [c for c in feat_cols if not c.startswith('IND_')]

# Winsorise 0.5/99.5 cross-sectionally
for col in non_ind:
    panel[col] = panel.groupby('date')[col].transform(
        lambda x: x.clip(x.quantile(0.005), x.quantile(0.995)))

# Rank-transform to (-1,1) cross-sectionally
for col in non_ind:
    panel[col] = panel.groupby('date')[col].transform(
        lambda x: 2 * x.rank(pct=True) - 1)

# Fill NaN with 0
panel[feat_cols] = panel[feat_cols].fillna(0)

# Normalise weights
panel['weight'] = panel.groupby('date')['weight'].transform(
    lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x))

print(f"  Features: {len(feat_cols)} ({len(non_ind)} non-industry)")

# ── 5. Models ────────────────────────────────────────────────
print("\n" + "="*60)
print("Training models ...")
print("="*60)

dates_sorted = sorted(panel['date'].unique())
# Paper: 9yr train + 1yr val + 1yr test, roll 1yr
TRAIN_Y, VAL_Y = 9, 1
min_before = (TRAIN_Y + VAL_Y) * 12

# Build test periods
test_periods = []
i = min_before
while i < len(dates_sorted):
    end_i = min(i + 11, len(dates_sorted) - 1)
    test_periods.append({
        'train': dates_sorted[max(0, i - min_before): i - VAL_Y * 12],
        'val':   dates_sorted[i - VAL_Y * 12: i],
        'test':  dates_sorted[i: end_i + 1],
    })
    i += 12

print(f"  Test periods: {len(test_periods)}")

model_names_ml = ['LM', 'ELANET', 'RF', 'GBRT', 'XGBoost', 'LightGBM', 'NN_1']
all_preds = {n: [] for n in model_names_ml}
all_actual, all_weight, all_date, all_stock = [], [], [], []
# Also store per-period feature importances for RF/XGB/LGBM
fi_rf, fi_xgb, fi_lgb = [], [], []
# Store per-period loss history for ML models (train_loss, val_loss per period)
loss_history = {n: [] for n in model_names_ml}  # list of (train_mse, val_mse) per period


class BetaNet(nn.Module):
    """Simple feedforward NN for beta prediction."""
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


def train_nn_gpu(Xtr, ytr, Xva, yva, hidden_sizes, lr, weight_decay, epochs=300, batch_size=2048):
    """Train NN on GPU with early stopping. Returns (model, best_val, train_losses, val_losses)."""
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.float32, device=DEVICE)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=DEVICE)
    yva_t = torch.tensor(yva, dtype=torch.float32, device=DEVICE)

    model = BetaNet(Xtr.shape[1], hidden_sizes).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    ds = TensorDataset(Xtr_t, ytr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val, patience_cnt, best_state = np.inf, 0, None
    train_losses, val_losses = [], []
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batch = 0
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
        train_losses.append(epoch_loss / max(n_batch, 1))

        model.eval()
        with torch.no_grad():
            val_loss = nn.MSELoss()(model(Xva_t), yva_t).item()
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 20:
                break

    model.load_state_dict(best_state)
    return model, best_val, train_losses, val_losses


def predict_nn_gpu(model, X):
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model(Xt).cpu().numpy()


def fit_predict(mtype, Xtr, ytr, Xva, yva, Xte):
    """Returns (predictions, loss_info_dict).
    loss_info_dict has keys: train_mse, val_mse, and optionally
    nn_train_curve / nn_val_curve for NN epoch-level curves.
    """
    train_mse_val = mean_squared_error(ytr, np.full(len(ytr), np.mean(ytr)))  # baseline

    if mtype == 'LM':
        m = LinearRegression().fit(Xtr, ytr)
        pred = m.predict(Xte)
        info = {'train_mse': mean_squared_error(ytr, m.predict(Xtr)),
                'val_mse': mean_squared_error(yva, m.predict(Xva))}
        return pred, info
    elif mtype == 'ELANET':
        Xc = np.vstack([Xtr, Xva]); yc = np.concatenate([ytr, yva])
        m = ElasticNetCV(l1_ratio=[.1,.5,.9,.99], cv=5, max_iter=5000, n_jobs=-1)
        m.fit(Xc, yc)
        info = {'train_mse': mean_squared_error(yc, m.predict(Xc)),
                'val_mse': mean_squared_error(yva, m.predict(Xva))}
        return m.predict(Xte), info
    elif mtype == 'RF':
        best, bm = np.inf, None
        # Use XGBRFRegressor (RF mode) to avoid sklearn+torch Cython bug
        for ne in [200, 400]:
            for md in [8, 12]:
                m = xgb.XGBRFRegressor(n_estimators=ne, max_depth=md,
                    colsample_bynode=0.5, subsample=0.8, min_child_weight=5,
                    device='cuda', random_state=42, verbosity=0)
                m.fit(Xtr, ytr)
                s = mean_squared_error(yva, m.predict(Xva))
                if s < best: best, bm = s, m
        fi_rf.append(bm.feature_importances_)
        info = {'train_mse': mean_squared_error(ytr, bm.predict(Xtr)), 'val_mse': best}
        return bm.predict(Xte), info
    elif mtype == 'GBRT':
        best, bm = np.inf, None
        best_staged = None
        for ne in [200, 400]:
            for md in [3, 5]:
                for lr in [.05, .1]:
                    m = HistGradientBoostingRegressor(max_iter=ne, max_depth=md,
                        learning_rate=lr, min_samples_leaf=5, random_state=42,
                        validation_fraction=None, early_stopping=False)
                    m.fit(Xtr, ytr)
                    s = mean_squared_error(yva, m.predict(Xva))
                    if s < best: best, bm = s, m
        info = {'train_mse': mean_squared_error(ytr, bm.predict(Xtr)), 'val_mse': best}
        return bm.predict(Xte), info
    elif mtype == 'XGBoost':
        best, bm = np.inf, None
        best_cfg = None
        for ne in [200, 400]:
            for md in [3, 5, 7]:
                for lr in [.05, .1]:
                    m = xgb.XGBRegressor(n_estimators=ne, max_depth=md,
                        learning_rate=lr, subsample=.8, colsample_bytree=.8,
                        min_child_weight=5, reg_alpha=.1, reg_lambda=1.,
                        device='cuda', random_state=42, verbosity=0,
                        eval_metric='rmse')
                    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)], verbose=False)
                    s = mean_squared_error(yva, m.predict(Xva))
                    if s < best:
                        best, bm = s, m
                        # Extract evals_result
                        res = m.evals_result()
                        best_cfg = {
                            'xgb_train': [v**2 for v in res['validation_0']['rmse']],
                            'xgb_val':   [v**2 for v in res['validation_1']['rmse']],
                        }
        fi_xgb.append(bm.feature_importances_)
        info = {'train_mse': mean_squared_error(ytr, bm.predict(Xtr)), 'val_mse': best}
        if best_cfg:
            info['staged_train'] = best_cfg['xgb_train']
            info['staged_val']   = best_cfg['xgb_val']
        return bm.predict(Xte), info
    elif mtype == 'LightGBM':
        best, bm = np.inf, None
        best_cb = None
        for ne in [200, 400]:
            for nl in [31, 63]:
                for lr in [.05, .1]:
                    cb = {}
                    m = lgb.LGBMRegressor(n_estimators=ne, num_leaves=nl,
                        learning_rate=lr, subsample=.8, colsample_bytree=.8,
                        min_child_samples=5, reg_alpha=.1, reg_lambda=1.,
                        n_jobs=-1, random_state=42, verbose=-1)
                    m.fit(Xtr, ytr, eval_set=[(Xtr, ytr), (Xva, yva)],
                          eval_metric='mse', callbacks=[lgb.record_evaluation(cb)])
                    s = mean_squared_error(yva, m.predict(Xva))
                    if s < best:
                        best, bm = s, m
                        best_cb = cb.copy()
        fi_lgb.append(bm.feature_importances_)
        info = {'train_mse': mean_squared_error(ytr, bm.predict(Xtr)), 'val_mse': best}
        if best_cb:
            # LightGBM callback keys vary; extract from best_cb
            keys = list(best_cb.keys())
            if len(keys) >= 2:
                tr_key, va_key = keys[0], keys[1]
                metric_name = list(best_cb[tr_key].keys())[0]
                info['staged_train'] = best_cb[tr_key][metric_name]
                info['staged_val']   = best_cb[va_key][metric_name]
        return bm.predict(Xte), info
    elif mtype == 'NN_1':
        best_val, best_model = np.inf, None
        best_curves = None
        for h, lr, wd in [((64, 32), 1e-3, 1e-3),
                           ((128, 64), 1e-3, 1e-3),
                           ((128, 64), 5e-4, 1e-2),
                           ((128, 64, 32), 1e-3, 1e-3)]:
            model, val, tr_losses, va_losses = train_nn_gpu(Xtr, ytr, Xva, yva, h, lr, wd)
            if val < best_val:
                best_val, best_model = val, model
                best_curves = (tr_losses, va_losses)
        info = {'train_mse': best_curves[0][-1], 'val_mse': best_val,
                'nn_train_curve': best_curves[0], 'nn_val_curve': best_curves[1]}
        return predict_nn_gpu(best_model, Xte), info


for pi, per in enumerate(test_periods):
    tr_mask = panel['date'].isin(per['train'])
    va_mask = panel['date'].isin(per['val'])
    te_mask = panel['date'].isin(per['test'])

    Xtr = np.ascontiguousarray(np.nan_to_num(panel.loc[tr_mask, feat_cols].values.astype(np.float64)))
    ytr = np.ascontiguousarray(panel.loc[tr_mask, 'realized_beta'].values.astype(np.float64))
    Xva = np.ascontiguousarray(np.nan_to_num(panel.loc[va_mask, feat_cols].values.astype(np.float64)))
    yva = np.ascontiguousarray(panel.loc[va_mask, 'realized_beta'].values.astype(np.float64))
    Xte = np.ascontiguousarray(np.nan_to_num(panel.loc[te_mask, feat_cols].values.astype(np.float64)))
    yte = np.ascontiguousarray(panel.loc[te_mask, 'realized_beta'].values.astype(np.float64))
    wte = np.ascontiguousarray(panel.loc[te_mask, 'weight'].values.astype(np.float64))
    dte = panel.loc[te_mask, 'date'].values
    ste = panel.loc[te_mask, 'stock'].values

    if len(Xtr) < 100 or len(Xte) < 50:
        continue

    # Ensure clean float64 arrays with no inf/nan
    Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
    Xva = np.nan_to_num(Xva, nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(Xte, nan=0.0, posinf=0.0, neginf=0.0)
    sc = StandardScaler()
    Xtr_s = np.ascontiguousarray(sc.fit_transform(Xtr), dtype=np.float64)
    Xva_s = np.ascontiguousarray(sc.transform(Xva), dtype=np.float64)
    Xte_s = np.ascontiguousarray(sc.transform(Xte), dtype=np.float64)
    # Replace any inf from StandardScaler (zero-variance cols)
    Xtr_s = np.nan_to_num(Xtr_s, nan=0.0, posinf=0.0, neginf=0.0)
    Xva_s = np.nan_to_num(Xva_s, nan=0.0, posinf=0.0, neginf=0.0)
    Xte_s = np.nan_to_num(Xte_s, nan=0.0, posinf=0.0, neginf=0.0)
    if pi == 0:
        print(f"    DEBUG: Xtr_s dtype={Xtr_s.dtype} shape={Xtr_s.shape} "
              f"C={Xtr_s.flags['C_CONTIGUOUS']} nan={np.any(np.isnan(Xtr_s))} "
              f"inf={np.any(np.isinf(Xtr_s))}", flush=True)
    ytr = np.ascontiguousarray(ytr, dtype=np.float64)
    yva = np.ascontiguousarray(yva, dtype=np.float64)

    all_actual.append(yte)
    all_weight.append(wte)
    all_date.append(dte)
    all_stock.append(ste)

    print(f"\n  Period {pi+1}/{len(test_periods)}: "
          f"test {per['test'][0].strftime('%Y-%m')}~{per['test'][-1].strftime('%Y-%m')} "
          f"(tr={len(Xtr)}, te={len(Xte)})")

    for mn in model_names_ml:
        t1 = time.time()
        try:
            pred, info = fit_predict(mn, Xtr_s, ytr, Xva_s, yva, Xte_s)
        except Exception as e:
            import traceback
            print(f"    {mn} ERROR: {e}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            pred = np.full(len(yte), np.nanmean(ytr))
            info = {'train_mse': np.nan, 'val_mse': np.nan}
        all_preds[mn].append(pred)
        loss_history[mn].append(info)
        print(f"    {mn}: {time.time()-t1:.1f}s  train_mse={info['train_mse']:.4f} val_mse={info['val_mse']:.4f}")

# Concatenate
actual = np.concatenate(all_actual)
weight = np.concatenate(all_weight)
dates_arr = np.concatenate(all_date)
stocks_arr = np.concatenate(all_stock)
for mn in model_names_ml:
    all_preds[mn] = np.concatenate(all_preds[mn])

# Save checkpoint after training
print("\nSaving training checkpoint ...", flush=True)
checkpoint = {
    'actual': actual, 'weight': weight, 'dates_arr': dates_arr, 'stocks_arr': stocks_arr,
    'all_preds': {k: v for k, v in all_preds.items()},
    'loss_history': loss_history,
    'fi_rf': fi_rf, 'fi_xgb': fi_xgb, 'fi_lgb': fi_lgb,
    'feat_cols': feat_cols, 'model_names_ml': model_names_ml,
    'beta_3m_index': beta_3m.index.tolist(), 'beta_3m_columns': beta_3m.columns.tolist(),
}
with open(f'{OUT}/checkpoint.pkl', 'wb') as f:
    pickle.dump(checkpoint, f)
print("  checkpoint.pkl saved", flush=True)

# ── Benchmark predictions (vectorised lookup) ──
print("\nBuilding benchmark predictions ...")
t_bench = time.time()
dates_ts = pd.to_datetime(dates_arr)

def lookup_beta_fast(beta_df, dates_ts, stocks_arr):
    """Vectorised beta lookup: multi-index join instead of iterrows."""
    mi = pd.MultiIndex.from_arrays([dates_ts, stocks_arr], names=['date', 'stock'])
    stacked = beta_df.stack(dropna=False)
    stacked.index.names = ['date', 'stock']
    return stacked.reindex(mi).values.astype(float)

bench_preds = {}
bench_preds['OLS_1Y_D'] = lookup_beta_fast(beta_1y, dates_ts, stocks_arr)
bench_preds['OLS_5Y_M'] = lookup_beta_fast(beta_5y, dates_ts, stocks_arr)

# BSW: winsorize beta to [0.5, 1.5]
bench_preds['BSW'] = np.clip(bench_preds['OLS_1Y_D'], 0.5, 1.5)

# HYBRID: Blume shrinkage toward cross-sectional mean
b1y_arr = bench_preds['OLS_1Y_D'].copy()
hybrid_df = pd.DataFrame({'date': dates_ts, 'b': b1y_arr})
cs_mean = hybrid_df.groupby('date')['b'].transform('mean')
bench_preds['HYBRID'] = (0.33 * cs_mean + 0.67 * b1y_arr).values

# FAMA_FRENCH: decile portfolio beta
ff = np.full_like(b1y_arr, np.nan)
ff_df = pd.DataFrame({'date': dates_ts, 'b': b1y_arr, 'idx': np.arange(len(b1y_arr))})
for d, grp in ff_df.groupby('date'):
    valid = grp['b'].notna()
    if valid.sum() >= 10:
        b_valid = grp.loc[valid, 'b']
        decile = pd.qcut(b_valid, 10, labels=False, duplicates='drop')
        for q in decile.unique():
            qm = decile == q
            mean_b = b_valid[qm].mean()
            ff[grp.loc[valid].loc[qm, 'idx'].values] = mean_b
        # Assign NaN stocks to overall mean
        ff[grp.loc[~valid, 'idx'].values] = b_valid.mean()
    else:
        ff[grp['idx'].values] = grp['b'].values
bench_preds['FAMA_FRENCH'] = ff

# LONG_MEMO: weighted average of 3m, 1y, 5y betas
b3m_arr = lookup_beta_fast(beta_3m, dates_ts, stocks_arr)
b5y_arr = bench_preds['OLS_5Y_M']
bench_preds['LONG_MEMO'] = 0.2 * b3m_arr + 0.3 * b1y_arr + 0.5 * b5y_arr
print(f"  Benchmarks done in {time.time()-t_bench:.1f}s")

# Fill NaN in benchmarks with cross-sectional mean
for k in bench_preds:
    arr = bench_preds[k]
    for d in np.unique(dates_arr):
        mask = dates_arr == d
        cs_mean = np.nanmean(arr[mask])
        arr[mask] = np.where(np.isnan(arr[mask]), cs_mean, arr[mask])
    bench_preds[k] = arr

# Merge all predictions
all_model_preds = {**bench_preds, **all_preds}

# ── 6. Evaluation ────────────────────────────────────────────
print("\n" + "="*60)
print("Evaluation")
print("="*60)

unique_dates = sorted(set(dates_arr))
monthly_mse = {}
for mn, pred in all_model_preds.items():
    mse_list = []
    for d in unique_dates:
        mask = dates_arr == d
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

best = sorted_models[0][0]
print(f"\n  DM test vs {best}:")
dm_results = {}
for mn in avg_mse:
    if mn != best:
        t = dm_test(monthly_mse[mn], monthly_mse[best])
        sig = "***" if abs(t)>2.576 else "**" if abs(t)>1.96 else "*" if abs(t)>1.645 else ""
        dm_results[mn] = t
        print(f"    {mn:15s}: t={t:6.2f} {sig}")

# ── 7. Plots ─────────────────────────────────────────────────
print("\n" + "="*60)
print("Plotting ...")
print("="*60)

unique_ts = [pd.Timestamp(d) for d in unique_dates]

# ---- Fig 1: Bar chart of average MSE ----
bench_order = ['OLS_5Y_M','OLS_1Y_D','BSW','HYBRID','FAMA_FRENCH','LONG_MEMO']
ml_order = ['LM','ELANET','RF','GBRT','XGBoost','LightGBM','NN_1']
order = [m for m in bench_order+ml_order if m in avg_mse]

fig, ax = plt.subplots(figsize=(14, 6))
vals = [avg_mse[m]*100 for m in order]
colors = ['#4472C4' if m in bench_order else '#ED7D31' for m in order]
bars = ax.bar(range(len(order)), vals, color=colors, edgecolor='k', lw=.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+.02, f'{v:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_xticks(range(len(order)))
ax.set_xticklabels(order, rotation=45, ha='right')
ax.set_ylabel('Value-Weighted MSE (%)')
ax.set_title('Figure 1: Average Forecast Errors', fontweight='bold', fontsize=14)
ax.legend(handles=[Patch(fc='#4472C4', label='Benchmark'),
                    Patch(fc='#ED7D31', label='Machine Learning')], fontsize=11)
ax.grid(axis='y', alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig1_avg_mse.png', dpi=200); plt.close()
print("  fig1_avg_mse.png")

# ---- Fig 2: MSE over time ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
c_bench = {'OLS_1Y_D':'#4472C4','OLS_5Y_M':'#ED7D31','BSW':'grey','HYBRID':'#FFC000','LONG_MEMO':'#70AD47'}
for mn, c in c_bench.items():
    if mn in monthly_mse:
        ax1.plot(unique_ts, [v*100 for v in monthly_mse[mn]], label=mn, color=c, alpha=.7, lw=1)
# Best ML
best_ml = min([(m, avg_mse[m]) for m in ml_order if m in avg_mse], key=lambda x:x[1])[0]
ax1.plot(unique_ts, [v*100 for v in monthly_mse[best_ml]], label=f'{best_ml}(Best ML)',
         color='red', lw=2, ls='--')
ax1.set_title('Panel A: Benchmarks vs Best ML', fontweight='bold')
ax1.set_ylabel('MSE (%)'); ax1.legend(fontsize=9); ax1.grid(alpha=.3)

c_ml = {'LM':'#4472C4','ELANET':'#ED7D31','RF':'grey','GBRT':'#FFC000',
        'XGBoost':'#5B9BD5','LightGBM':'#70AD47','NN_1':'#FF6384'}
for mn in ml_order:
    if mn in monthly_mse:
        # Rolling 12-month average
        s = pd.Series([v*100 for v in monthly_mse[mn]], index=unique_ts).rolling(12, min_periods=1).mean()
        ax2.plot(s.index, s.values, label=mn, color=c_ml.get(mn,'k'), lw=1.5)
ax2.set_title('Panel B: ML Models (12-month rolling avg)', fontweight='bold')
ax2.set_ylabel('MSE (%)'); ax2.set_xlabel('Year'); ax2.legend(fontsize=9); ax2.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig2_mse_time.png', dpi=200); plt.close()
print("  fig2_mse_time.png")

# ---- Fig 3: Relative forecast error (best ML vs OLS_1Y_D and vs LM) ----
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))
for ax, comp, clr, panel_label in [(ax1, 'OLS_1Y_D', '#4472C4', 'A'),
                                     (ax2, 'LM', '#ED7D31', 'B')]:
    if comp in monthly_mse:
        rel = [1 - monthly_mse[best_ml][i] / monthly_mse[comp][i]
               if monthly_mse[comp][i] > 0 else 0
               for i in range(len(unique_ts))]
        rel100 = [r * 100 for r in rel]
        ax.bar(unique_ts, rel100, width=25, alpha=.5, color=clr, label='Monthly')
        # Yearly average
        df_r = pd.DataFrame({'date': unique_ts, 'rel': rel100})
        df_r['year'] = df_r['date'].dt.year
        yr = df_r.groupby('year').agg({'date': 'first', 'rel': 'mean'})
        ax.plot(yr['date'], yr['rel'], 'ro-', ms=5, lw=2, label='Yearly Avg')
        ax.axhline(0, color='k', lw=.5)
        ax.set_title(f'Panel {panel_label}: {best_ml} vs {comp}', fontweight='bold')
        ax.set_ylabel('Relative MSE Reduction (%)')
        ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{OUT}/fig3_relative_error.png', dpi=200); plt.close()
print("  fig3_relative_error.png")

# ---- Fig 4: Variable importance ----
fig, axes = plt.subplots(1, 3, figsize=(20, 8))
imp_data = {}
if fi_rf:
    imp_data['RF'] = pd.Series(np.mean(fi_rf, axis=0), index=feat_cols)
if fi_xgb:
    imp_data['XGBoost'] = pd.Series(np.mean(fi_xgb, axis=0), index=feat_cols)
if fi_lgb:
    s = pd.Series(np.mean(fi_lgb, axis=0), index=feat_cols)
    imp_data['LightGBM'] = s / s.sum()

for i, (mname, imp) in enumerate(imp_data.items()):
    ax = axes[i]
    non_ind_imp = imp[[c for c in imp.index if not c.startswith('IND_')]]
    top = non_ind_imp.sort_values(ascending=True).tail(20)
    cat_colors = []
    for f in top.index:
        if f.startswith('OLS_'):  cat_colors.append('#4472C4')
        elif f in ['COSKEW','IVOL','KURT','MOM','TURNOVER','EP_COVAR','EP_VAR']:
            cat_colors.append('#70AD47')
        elif f in ['RF','TERM']: cat_colors.append('#FFC000')
        else:                    cat_colors.append('#ED7D31')
    ax.barh(range(len(top)), top.values, color=cat_colors)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=9)
    ax.set_title(mname, fontweight='bold', fontsize=13)
    ax.set_xlabel('Importance')
fig.legend(handles=[Patch(fc='#4472C4', label='Beta Est.'), Patch(fc='#ED7D31', label='Accounting'),
                     Patch(fc='#70AD47', label='Technical'), Patch(fc='#FFC000', label='Macro')],
           loc='lower center', ncol=4, fontsize=11)
fig.suptitle('Figure 4: Variable Importance (Top 20)', fontweight='bold', fontsize=14, y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/fig4_var_importance.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig4_var_importance.png")

# ---- Fig 5: DM heatmap ----
all_mn = [m for m in bench_order+ml_order if m in monthly_mse]
n = len(all_mn)
dm_mat = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        if i != j:
            dm_mat[i, j] = dm_test(monthly_mse[all_mn[i]], monthly_mse[all_mn[j]])

fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(dm_mat, cmap='RdYlGn_r', vmin=-5, vmax=5, aspect='auto')
ax.set_xticks(range(n)); ax.set_xticklabels(all_mn, rotation=45, ha='right')
ax.set_yticks(range(n)); ax.set_yticklabels(all_mn)
for i in range(n):
    for j in range(n):
        if i != j:
            ax.text(j, i, f'{dm_mat[i,j]:.1f}', ha='center', va='center', fontsize=8,
                    color='white' if abs(dm_mat[i,j])>3 else 'black')
plt.colorbar(im, label='DM Statistic', shrink=.8)
ax.set_title('Figure 5: Diebold-Mariano Test\n(+: row has higher MSE)', fontweight='bold')
plt.tight_layout(); plt.savefig(f'{OUT}/fig5_dm_heatmap.png', dpi=200); plt.close()
print("  fig5_dm_heatmap.png")

# ---- Fig 6: MSE by size & beta quintile ----
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
te_df = pd.DataFrame({'date': dates_arr, 'stock': stocks_arr, 'actual': actual, 'weight': weight})
# Merge panel features for grouping
panel_idx = panel.set_index(['date', 'stock'])
te_idx = te_df.set_index(['date', 'stock'])
for col in ['LOGME', 'OLS_1Y_D']:
    if col in panel_idx.columns:
        te_idx[col] = panel_idx[col]
te_df = te_idx.reset_index()

compare_models = ['OLS_1Y_D', 'RF', 'XGBoost', 'LightGBM']
compare_models = [m for m in compare_models if m in all_model_preds]

# Panel A: by size
ax = axes[0, 0]
if 'LOGME' in te_df.columns:
    def safe_qcut5(x, labels):
        try:
            return pd.qcut(x, 5, labels=labels, duplicates='drop')
        except ValueError:
            return pd.qcut(x.rank(method='first'), 5, labels=labels, duplicates='drop')
    te_df['sq'] = te_df.groupby('date')['LOGME'].transform(
        lambda x: safe_qcut5(x, ['Q1(S)','Q2','Q3','Q4','Q5(L)']))
    x = np.arange(5); w = .15
    for ci, mn in enumerate(compare_models):
        se = (actual - all_model_preds[mn])**2
        te_df['se'] = se
        qm = te_df.groupby('sq', observed=True).apply(
            lambda g: np.average(g['se'], weights=g['weight'])*100 if g['weight'].sum()>0 else 0)
        if len(qm) == 5:
            ax.bar(x + ci*w, qm.values, w, label=mn, alpha=.85)
    ax.set_xticks(x + w*len(compare_models)/2)
    ax.set_xticklabels(['Q1(S)','Q2','Q3','Q4','Q5(L)'])
    ax.set_ylabel('MSE (%)'); ax.set_title('Panel A: By Size', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

# Panel B: by beta
ax = axes[0, 1]
if 'OLS_1Y_D' in te_df.columns:
    te_df['bq'] = te_df.groupby('date')['OLS_1Y_D'].transform(
        lambda x: safe_qcut5(x, ['Q1(Low)','Q2','Q3','Q4','Q5(High)']))
    x = np.arange(5); w = .15
    for ci, mn in enumerate(compare_models):
        se = (actual - all_model_preds[mn])**2
        te_df['se'] = se
        qm = te_df.groupby('bq', observed=True).apply(
            lambda g: np.average(g['se'], weights=g['weight'])*100 if g['weight'].sum()>0 else 0)
        if len(qm) == 5:
            ax.bar(x + ci*w, qm.values, w, label=mn, alpha=.85)
    ax.set_xticks(x + w*len(compare_models)/2)
    ax.set_xticklabels(['Q1(Low)','Q2','Q3','Q4','Q5(High)'])
    ax.set_ylabel('MSE (%)'); ax.set_title('Panel B: By Beta', fontweight='bold')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=.3)

# Panel C: Cumulative MSE improvement
ax = axes[1, 0]
for mn in ['RF','GBRT','XGBoost','LightGBM']:
    if mn in monthly_mse and 'OLS_1Y_D' in monthly_mse:
        cum = np.cumsum([monthly_mse['OLS_1Y_D'][i]-monthly_mse[mn][i]
                          for i in range(len(unique_ts))]) * 100
        ax.plot(unique_ts, cum, label=f'{mn}', lw=1.5)
ax.axhline(0, color='k', lw=.5)
ax.set_title('Panel C: Cumul. MSE Improvement vs OLS_1Y_D', fontweight='bold')
ax.set_ylabel('Cumul. Δ MSE (%)'); ax.legend(fontsize=9); ax.grid(alpha=.3)

# Panel D: Rolling MSE of ML
ax = axes[1, 1]
for mn in ml_order:
    if mn in monthly_mse:
        s = pd.Series([v*100 for v in monthly_mse[mn]], index=unique_ts).rolling(12, min_periods=1).mean()
        ax.plot(s.index, s.values, label=mn, color=c_ml.get(mn,'k'), lw=1.5)
ax.set_title('Panel D: Rolling 12m MSE (ML)', fontweight='bold')
ax.set_ylabel('MSE (%)'); ax.legend(fontsize=9); ax.grid(alpha=.3)
fig.suptitle('Figure 6: Forecast Error Analysis', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f'{OUT}/fig6_error_analysis.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig6_error_analysis.png")

# ---- Fig 7: Category importance stacked bar ----
if imp_data:
    categories = {
        'Beta Est.': [c for c in feat_cols if c.startswith('OLS_')],
        'Accounting': [c for c in feat_cols if c in
            ['AGE','AT','BM','CAPTURN','FINLEV','FXDCOS','GM','LEV','LOGME','NOA','NI','OPLEV','ROA','ROE','NWCAP']],
        'Technical': [c for c in feat_cols if c in ['COSKEW','IVOL','KURT','MOM','TURNOVER','EP_COVAR','EP_VAR']],
        'Macro': [c for c in feat_cols if c in ['RF','TERM']],
        'Industry': [c for c in feat_cols if c.startswith('IND_')],
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    mns = list(imp_data.keys())
    x = np.arange(len(mns))
    bot = np.zeros(len(mns))
    cat_cols = ['#4472C4','#ED7D31','#70AD47','#FFC000','#A5A5A5']
    for ci, (cat, feats) in enumerate(categories.items()):
        vals = [sum(imp_data[m].get(f, 0) for f in feats) for m in mns]
        ax.bar(x, vals, .5, bottom=bot, label=cat, color=cat_cols[ci])
        bot += np.array(vals)
    ax.set_xticks(x); ax.set_xticklabels(mns, fontsize=12)
    ax.set_ylabel('Cumul. Importance'); ax.set_title('Figure 7: Category Importance', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(axis='y', alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig7_category_imp.png', dpi=200); plt.close()
    print("  fig7_category_imp.png")

# ---- Fig 8: Predicted vs Realized scatter ----
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
axes = axes.flatten()
plot_models = ['OLS_1Y_D','RF','GBRT','XGBoost','LightGBM','NN_1']
plot_models = [m for m in plot_models if m in all_model_preds]
np.random.seed(42)
sidx = np.random.choice(len(actual), min(5000, len(actual)), replace=False)
for i, mn in enumerate(plot_models):
    ax = axes[i]
    p = all_model_preds[mn]
    ax.scatter(actual[sidx], p[sidx], alpha=.1, s=5, c='#4472C4')
    lims = [max(-2, min(actual.min(), p.min())), min(5, max(actual.max(), p.max()))]
    ax.plot(lims, lims, 'r--', lw=1)
    r2 = 1 - np.sum((actual-p)**2)/np.sum((actual-np.mean(actual))**2)
    corr = np.corrcoef(actual, p)[0, 1]
    ax.set_title(f'{mn}  R²={r2:.4f}  ρ={corr:.4f}', fontweight='bold')
    ax.set_xlabel('Realized β'); ax.set_ylabel('Predicted β')
    ax.set_xlim(lims); ax.set_ylim(lims); ax.grid(alpha=.3)
for j in range(len(plot_models), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Figure 8: Predicted vs Realized Beta', fontweight='bold', fontsize=14, y=1.01)
plt.tight_layout(); plt.savefig(f'{OUT}/fig8_scatter.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig8_scatter.png")

# ---- Fig 9: Variable importance over time (RF only) ----
if fi_rf and len(fi_rf) > 1:
    key_feats = ['OLS_1Y_D','OLS_3M_D','OLS_5Y_M','LOGME','TURNOVER','BM','MOM','IVOL','AGE','FINLEV']
    key_feats = [f for f in key_feats if f in feat_cols]
    fi_arr = np.array(fi_rf)  # (n_periods, n_features)
    fig, ax = plt.subplots(figsize=(14, 7))
    period_dates = [unique_ts[min(i*12, len(unique_ts)-1)] for i in range(len(fi_rf))]
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_feats)))
    for fi, feat_name in enumerate(key_feats):
        ci = feat_cols.index(feat_name) if feat_name in feat_cols else -1
        if ci >= 0:
            ax.plot(period_dates, fi_arr[:, ci], label=feat_name, color=colors[fi], lw=1.5, marker='o', ms=3)
    ax.set_title('Figure 9: RF Variable Importance Over Time', fontweight='bold', fontsize=14)
    ax.set_ylabel('Importance'); ax.set_xlabel('Year')
    ax.legend(fontsize=9, ncol=2); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(f'{OUT}/fig9_imp_time.png', dpi=200); plt.close()
    print("  fig9_imp_time.png")

# ---- Fig 10: Summary table as figure ----
fig, ax = plt.subplots(figsize=(14, 8))
ax.axis('off')
col_labels = ['Model', 'MSE (%)', 'Rank', f'DM vs {best}']
table_data = []
for rank, (mn, mse) in enumerate(sorted_models, 1):
    dm_val = f'{dm_results.get(mn, 0):.2f}' if mn != best else '-'
    table_data.append([mn, f'{mse*100:.4f}', str(rank), dm_val])
tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1, 1.5)
# Color header
for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#4472C4')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
# Highlight top 3
for i in range(min(3, len(table_data))):
    for j in range(len(col_labels)):
        tbl[i+1, j].set_facecolor('#E2EFDA')
ax.set_title('Figure 10: Model Performance Summary', fontweight='bold', fontsize=14, pad=20)
plt.tight_layout(); plt.savefig(f'{OUT}/fig10_summary.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig10_summary.png")

# ---- Fig 11: ML Training Loss Curves ----
# Panel A: Train/Val MSE across rolling periods for each model
# Panel B: Epoch-level loss curves for boosting & NN (last period)
fig, axes = plt.subplots(2, 2, figsize=(18, 14))

# Panel A: Train MSE across periods
ax = axes[0, 0]
for mn in model_names_ml:
    tr = [h['train_mse'] for h in loss_history[mn]]
    if tr:
        ax.plot(range(1, len(tr)+1), tr, 'o-', label=mn, ms=4, lw=1.5)
ax.set_xlabel('Rolling Period'); ax.set_ylabel('Train MSE')
ax.set_title('Panel A: Training MSE by Rolling Period', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

# Panel B: Val MSE across periods
ax = axes[0, 1]
for mn in model_names_ml:
    va = [h['val_mse'] for h in loss_history[mn]]
    if va:
        ax.plot(range(1, len(va)+1), va, 's-', label=mn, ms=4, lw=1.5)
ax.set_xlabel('Rolling Period'); ax.set_ylabel('Validation MSE')
ax.set_title('Panel B: Validation MSE by Rolling Period', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

# Panel C: Boosting staged loss (last period) for GBRT / XGBoost / LightGBM
ax = axes[1, 0]
staged_models = {'GBRT': '#FFC000', 'XGBoost': '#5B9BD5', 'LightGBM': '#70AD47'}
for mn, clr in staged_models.items():
    if loss_history[mn] and 'staged_train' in loss_history[mn][-1]:
        st = loss_history[mn][-1]['staged_train']
        sv = loss_history[mn][-1]['staged_val']
        iters = range(1, len(st)+1)
        ax.plot(iters, st, '-', color=clr, alpha=.5, lw=1, label=f'{mn} train')
        ax.plot(iters, sv, '--', color=clr, lw=1.5, label=f'{mn} val')
ax.set_xlabel('Boosting Iteration'); ax.set_ylabel('MSE')
ax.set_title('Panel C: Boosting Loss Curve (Last Period)', fontweight='bold')
ax.legend(fontsize=8); ax.grid(alpha=.3)

# Panel D: NN epoch loss (last period)
ax = axes[1, 1]
if loss_history['NN_1'] and 'nn_train_curve' in loss_history['NN_1'][-1]:
    tr_c = loss_history['NN_1'][-1]['nn_train_curve']
    va_c = loss_history['NN_1'][-1]['nn_val_curve']
    epochs = range(1, len(tr_c)+1)
    ax.plot(epochs, tr_c, '-', color='#4472C4', lw=1.5, label='Train')
    ax.plot(epochs, va_c, '--', color='#ED7D31', lw=1.5, label='Validation')
    ax.set_xlabel('Epoch'); ax.set_ylabel('MSE')
    ax.set_title('Panel D: Neural Network Loss Curve (Last Period)', fontweight='bold')
    ax.legend(fontsize=10); ax.grid(alpha=.3)
fig.suptitle('Figure 11: Machine Learning Training Loss', fontweight='bold', fontsize=15, y=1.01)
plt.tight_layout(); plt.savefig(f'{OUT}/fig11_loss_curves.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig11_loss_curves.png")

# ---- Fig 12: Predicted vs Realized Beta Time Series ----
# For each ML model: cross-sectional mean predicted beta and mean realized beta, over time
fig, axes = plt.subplots(3, 3, figsize=(22, 16))
axes = axes.flatten()
plot_ml = [m for m in model_names_ml if m in all_model_preds]

for i, mn in enumerate(plot_ml):
    ax = axes[i]
    pred = all_model_preds[mn]

    # Compute cross-sectional weighted mean per month
    ts_actual, ts_pred, ts_dates = [], [], []
    for d in unique_dates:
        mask = dates_arr == d
        a_m = actual[mask]; p_m = pred[mask]; w_m = weight[mask]
        w_norm = w_m / w_m.sum() if w_m.sum() > 0 else np.ones_like(w_m) / len(w_m)
        ts_actual.append(np.sum(w_norm * a_m))
        ts_pred.append(np.sum(w_norm * p_m))
        ts_dates.append(pd.Timestamp(d))

    ax.plot(ts_dates, ts_actual, '-', color='#4472C4', lw=1.2, alpha=.7, label='Realized β')
    ax.plot(ts_dates, ts_pred, '--', color='#ED7D31', lw=1.2, alpha=.7, label='Predicted β')

    # Correlation
    corr = np.corrcoef(ts_actual, ts_pred)[0, 1]
    ax.set_title(f'{mn}  (ρ={corr:.3f})', fontweight='bold', fontsize=12)
    ax.set_ylabel('β')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=.3)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

for j in range(len(plot_ml), len(axes)):
    axes[j].set_visible(False)
fig.suptitle('Figure 12: Predicted vs Realized Beta Over Time\n(Value-weighted cross-sectional mean)',
             fontweight='bold', fontsize=15, y=1.02)
plt.tight_layout(); plt.savefig(f'{OUT}/fig12_pred_vs_actual_ts.png', dpi=200, bbox_inches='tight')
plt.close(); print("  fig12_pred_vs_actual_ts.png")

# Save results
with open(f'{OUT}/results.pkl', 'wb') as f:
    pickle.dump({
        'avg_mse': avg_mse, 'monthly_mse': monthly_mse,
        'dm_results': dm_results, 'sorted_models': sorted_models,
        'imp_data': {k: v.to_dict() for k, v in imp_data.items()},
    }, f)

# Save table as CSV
pd.DataFrame(sorted_models, columns=['Model', 'MSE']).to_csv(f'{OUT}/results_table.csv', index=False)

print("\n" + "="*60)
print("DONE! All results in results/")
print("="*60)
