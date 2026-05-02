#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage A: Load data, compute rolling betas, build feature panel, preprocess.
Output: results/stage_prep.pkl
"""
import os, time, gc, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

PKL = "data/pkl/"
OUT = os.environ.get('OUT', 'results/')
os.makedirs(OUT, exist_ok=True)

def load(name):
    return pd.read_pickle(os.path.join(PKL, name + ".pkl"))

# ── 1. Load data ─────────────────────────────────────────────
print("Loading pkl data ...")
t0 = time.time()
daily_ret   = load("daily_ret") / 100
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

if '000001.SH' in daily_ret.columns:
    daily_ret.drop(columns=['000001.SH'], inplace=True, errors='ignore')

# ── Build excess returns (return - risk-free rate) ──────────
# Daily rf: spread monthly rf evenly across trading days in each month
rf_period = rf.copy()
rf_period.index = rf_period.index.to_period('M')
daily_periods = daily_ret.index.to_period('M')
rf_daily = pd.Series(0.0, index=daily_ret.index)
for p in daily_periods.unique():
    if p in rf_period.index:
        mask = daily_periods == p
        rf_daily[mask] = rf_period[p] / mask.sum()

daily_ret_ex = daily_ret.sub(rf_daily, axis=0)
mkt_d_ex = mkt_d - rf_daily

# Monthly excess returns
rf_m_aligned = rf.reindex(monthly_ret.index).fillna(0)
monthly_ret_ex = monthly_ret.sub(rf_m_aligned, axis=0)
mkt_m_ex = mkt_m - rf.reindex(mkt_m.index).fillna(0)

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Daily: {daily_ret.shape}, Monthly: {monthly_ret.shape}")
print(f"  Excess returns built (daily rf range: {rf_daily.min():.6f}~{rf_daily.max():.6f})")

all_stocks = sorted(set(daily_ret.columns) & set(monthly_ret.columns))
print(f"  Common stocks: {len(all_stocks)}")

# ── 2. Rolling beta computation ──────────────────────────────
print("\nComputing rolling betas ...")
t0 = time.time()
month_ends = monthly_ret.index.tolist()


def compute_betas_at_months(daily_ret_df, mkt_daily, month_ends, window_months, min_obs):
    result = pd.DataFrame(index=month_ends, columns=daily_ret_df.columns, dtype=float)
    for i, mdate in enumerate(month_ends):
        start = mdate - pd.DateOffset(months=window_months)
        mask = (daily_ret_df.index > start) & (daily_ret_df.index <= mdate)
        s = daily_ret_df.loc[mask]
        m = mkt_daily.reindex(s.index)
        if len(s) < min_obs:
            continue
        sv = s.values; mv = m.values
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
        dm = mv_c[:, None] - m_mean[None, :]
        ds = sv_c - s_mean[None, :]
        dm = dm * valid; ds = ds * valid
        cov = (dm * ds).sum(axis=0) / np.maximum(n - 1, 1)
        var_m = (dm * dm).sum(axis=0) / np.maximum(n - 1, 1)
        beta = np.where((var_m > 1e-12) & (n >= min_obs), cov / var_m, np.nan)
        result.iloc[i] = beta
        if (i + 1) % 48 == 0:
            print(f"    {mdate.strftime('%Y-%m')} ({i+1}/{len(month_ends)})")
    return result


print("  OLS_3M_D ...")
beta_3m = compute_betas_at_months(daily_ret_ex, mkt_d_ex, month_ends, 3, 30)
print("  OLS_1Y_D ...")
beta_1y = compute_betas_at_months(daily_ret_ex, mkt_d_ex, month_ends, 12, 120)

print("  OLS_5Y_M ...")
beta_5y = pd.DataFrame(index=month_ends, columns=monthly_ret_ex.columns, dtype=float)
for i, mdate in enumerate(month_ends):
    start = mdate - pd.DateOffset(months=60)
    mask = (monthly_ret_ex.index > start) & (monthly_ret_ex.index <= mdate)
    s = monthly_ret_ex.loc[mask]; m = mkt_m_ex.reindex(s.index)
    if len(s) < 24: continue
    sv = s.values; mv = m.values
    valid_m = np.isfinite(mv); valid_s = np.isfinite(sv)
    valid = valid_s & valid_m[:, None]; n = valid.sum(axis=0)
    sv_c = np.where(valid, sv, 0.0); mv_c = np.where(valid_m, mv, 0.0)
    m_sum = (mv_c[:, None] * valid).sum(axis=0); s_sum = (sv_c * valid).sum(axis=0)
    m_mean = np.where(n > 0, m_sum / n, 0); s_mean = np.where(n > 0, s_sum / n, 0)
    dm = mv_c[:, None] - m_mean[None, :]; ds = sv_c - s_mean[None, :]
    dm *= valid; ds *= valid
    cov = (dm * ds).sum(axis=0) / np.maximum(n - 1, 1)
    var_m = (dm * dm).sum(axis=0) / np.maximum(n - 1, 1)
    beta_5y.iloc[i] = np.where((var_m > 1e-12) & (n >= 24), cov / var_m, np.nan)

print("  Realized betas (1-year forward, excess returns) ...")
realized_beta = pd.DataFrame(index=month_ends, columns=daily_ret_ex.columns, dtype=float)
for i, mdate in enumerate(month_ends):
    end = mdate + pd.DateOffset(months=12)
    mask = (daily_ret_ex.index > mdate) & (daily_ret_ex.index <= end)
    s = daily_ret_ex.loc[mask]; m = mkt_d_ex.reindex(s.index)
    if len(s) < 120: continue
    sv = s.values; mv = m.values
    valid_m = np.isfinite(mv); valid_s = np.isfinite(sv)
    valid = valid_s & valid_m[:, None]; n = valid.sum(axis=0)
    sv_c = np.where(valid, sv, 0.0); mv_c = np.where(valid_m, mv, 0.0)
    m_sum = (mv_c[:, None] * valid).sum(axis=0); s_sum = (sv_c * valid).sum(axis=0)
    m_mean = np.where(n > 0, m_sum / n, 0); s_mean = np.where(n > 0, s_sum / n, 0)
    dm = mv_c[:, None] - m_mean[None, :]; ds = sv_c - s_mean[None, :]
    dm *= valid; ds *= valid
    cov = (dm * ds).sum(axis=0) / np.maximum(n - 1, 1)
    var_m = (dm * dm).sum(axis=0) / np.maximum(n - 1, 1)
    realized_beta.iloc[i] = np.where((var_m > 1e-12) & (n >= 120), cov / var_m, np.nan)

print(f"  Betas done in {time.time()-t0:.1f}s")

# ── 3. Build features ────────────────────────────────────────
print("\nBuilding features ...")
t0 = time.time()
rows = []

for idx, mdate in enumerate(month_ends):
    rb = realized_beta.loc[mdate].dropna()
    if len(rb) < 50:
        continue
    stocks = rb.index.tolist()
    feat = pd.DataFrame(index=stocks)

    feat['OLS_3M_D'] = beta_3m.loc[mdate, stocks] if mdate in beta_3m.index else np.nan
    feat['OLS_1Y_D'] = beta_1y.loc[mdate, stocks] if mdate in beta_1y.index else np.nan
    feat['OLS_5Y_M'] = beta_5y.loc[mdate, stocks] if mdate in beta_5y.index else np.nan

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
    mc = mcap.loc[mdate, stocks].astype(float) if mdate in mcap.index else pd.Series(np.nan, index=stocks)

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
    feat['GM'] = (gm / 100).values
    feat['LEV'] = np.log((ta / eq).clip(lower=1e-6)).values
    feat['LOGME'] = np.log(mc.clip(lower=1)).values
    feat['NOA'] = (noa / ta).values
    feat['NI'] = (ni / ta).values
    feat['OPLEV'] = (fa / ta).values
    feat['ROA'] = (roa / 100).values
    feat['ROE'] = (roe / 100).values
    feat['NWCAP'] = (nwcap / ta).values

    d1y_start = mdate - pd.DateOffset(months=12)
    d1y = daily_ret_ex.loc[(daily_ret_ex.index > d1y_start) & (daily_ret_ex.index <= mdate), stocks]
    m1y = mkt_d_ex.reindex(d1y.index)
    if len(d1y) >= 120:
        sv = d1y.values; mv = m1y.values
        valid_m = np.isfinite(mv); valid_s = np.isfinite(sv)
        valid = valid_s & valid_m[:, None]; n_v = valid.sum(axis=0)
        sv_c = np.where(valid, sv, 0.0); mv_c = np.where(valid_m, mv, 0.0)
        m_sum = (mv_c[:, None] * valid).sum(axis=0); s_sum = (sv_c * valid).sum(axis=0)
        m_mean = np.where(n_v > 0, m_sum / n_v, 0); s_mean = np.where(n_v > 0, s_sum / n_v, 0)
        dm = (mv_c[:, None] - m_mean) * valid; ds = (sv_c - s_mean) * valid
        cov_sm = (dm * ds).sum(axis=0) / np.maximum(n_v - 1, 1)
        var_mk = (dm * dm).sum(axis=0) / np.maximum(n_v - 1, 1)
        b1y = np.where((var_mk > 1e-12) & (n_v >= 60), cov_sm / var_mk, 0)
        alpha = s_mean - b1y * m_mean
        resid = (sv_c - alpha - b1y * mv_c[:, None]) * valid
        resid_sq = (resid ** 2).sum(axis=0) / np.maximum(n_v - 1, 1)
        ivol = np.where(n_v >= 60, np.log(np.sqrt(resid_sq).clip(1e-10)), np.nan)
        feat['IVOL'] = ivol
        se = ds; me = dm
        coskew_num = (se * me**2).sum(axis=0) / np.maximum(n_v, 1)
        coskew_den = np.sqrt((se**2).sum(axis=0) / np.maximum(n_v, 1)) * \
                     ((me**2).sum(axis=0) / np.maximum(n_v, 1))
        feat['COSKEW'] = np.where((coskew_den > 1e-12) & (n_v >= 60),
                                   coskew_num / coskew_den, np.nan)
        s2 = (ds**2).sum(axis=0) / np.maximum(n_v, 1)
        s4 = (ds**4).sum(axis=0) / np.maximum(n_v, 1)
        feat['KURT'] = np.where((s2 > 1e-12) & (n_v >= 60), s4 / s2**2 - 3, np.nan)

    mom_start = mdate - pd.DateOffset(months=12)
    mom_end = mdate - pd.DateOffset(months=1)
    mom_w = monthly_ret_ex.loc[(monthly_ret_ex.index >= mom_start) & (monthly_ret_ex.index <= mom_end), stocks]
    if len(mom_w) >= 6:
        feat['MOM'] = ((1 + mom_w).prod() - 1).values

    vol_start = mdate - pd.DateOffset(months=12)
    vol_w = vol.loc[(vol.index >= vol_start) & (vol.index <= mdate)]
    common_v = stk_idx.intersection(vol_w.columns)
    if len(vol_w) > 0 and len(common_v) > 0:
        avg_vol = vol_w[common_v].mean()
        turnover = pd.Series(np.nan, index=stocks)
        turnover.loc[common_v] = np.log(avg_vol.clip(lower=1))
        feat['TURNOVER'] = turnover.values

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
            ev = ep_stocks.values; emv = ep_mkt.values
            valid_em = np.isfinite(emv); valid_es = np.isfinite(ev)
            valid_ep = valid_es & valid_em[:, None]; n_ep = valid_ep.sum(axis=0)
            ev_c = np.where(valid_ep, ev, 0); emv_c = np.where(valid_em, emv, 0)
            em_sum = (emv_c[:, None] * valid_ep).sum(axis=0)
            es_sum = (ev_c * valid_ep).sum(axis=0)
            em_mean = np.where(n_ep > 0, em_sum / n_ep, 0)
            es_mean = np.where(n_ep > 0, es_sum / n_ep, 0)
            d_em = (emv_c[:, None] - em_mean) * valid_ep
            d_es = (ev_c - es_mean) * valid_ep
            cov_ep = (d_em * d_es).sum(axis=0) / np.maximum(n_ep - 1, 1)
            var_ep = (d_em ** 2).sum(axis=0) / np.maximum(n_ep - 1, 1)
            std_es = np.sqrt((d_es ** 2).sum(axis=0) / np.maximum(n_ep - 1, 1))
            ep_covar_s.loc[common_pe] = np.where((var_ep > 1e-12) & (n_ep >= 12), cov_ep / var_ep, np.nan)
            ep_var_s.loc[common_pe] = np.where((std_es > 1e-10) & (n_ep >= 12), np.log(std_es), np.nan)
            feat['EP_COVAR'] = ep_covar_s.values
            feat['EP_VAR'] = ep_var_s.values

    # RF removed as feature; TERM (change in rf) retained
    rf_start = mdate - pd.DateOffset(months=12)
    rf_w = rf.loc[(rf.index >= rf_start) & (rf.index <= mdate)]
    if len(rf_w) > 1:
        feat['TERM'] = rf_w.iloc[-1] - rf_w.iloc[0]

    common_ind = stk_idx.intersection(ind_df.index)
    if len(common_ind) > 0:
        ind_s = ind_df.reindex(stocks)
        dummies = pd.get_dummies(ind_s, prefix='IND')
        for c in dummies.columns:
            feat[c] = dummies[c].values

    feat['realized_beta'] = rb.values
    feat['weight'] = mc.values
    feat['date'] = mdate
    feat['stock'] = stocks
    feat.index = range(len(feat))
    rows.append(feat)

    if (idx + 1) % 24 == 0:
        print(f"  {mdate.strftime('%Y-%m')} ({idx+1}/{len(month_ends)}) stocks={len(stocks)}")

panel = pd.concat(rows, ignore_index=True)
print(f"  Panel: {panel.shape}, {time.time()-t0:.1f}s")

# Free memory
del daily_ret, monthly_ret, daily_ret_ex, monthly_ret_ex, vol, pe, fund, realized_beta
gc.collect()

# ── 4. Preprocess ────────────────────────────────────────────
print("\nPreprocessing ...")
t0 = time.time()
meta = ['stock', 'date', 'realized_beta', 'weight']
feat_cols = [c for c in panel.columns if c not in meta]
non_ind = [c for c in feat_cols if not c.startswith('IND_')]

# Winsorise 0.5/99.5 cross-sectionally
for col in non_ind:
    panel[col] = panel.groupby('date')[col].transform(
        lambda x: x.clip(x.quantile(0.005), x.quantile(0.995)))

# Rank-transform to (-1,1)
for col in non_ind:
    panel[col] = panel.groupby('date')[col].transform(
        lambda x: 2 * x.rank(pct=True) - 1)

panel[feat_cols] = panel[feat_cols].fillna(0)
panel['weight'] = panel.groupby('date')['weight'].transform(
    lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x))

print(f"  Features: {len(feat_cols)} ({len(non_ind)} non-industry)")
print(f"  Preprocessing done in {time.time()-t0:.1f}s")

# ── 5. Build test periods ────────────────────────────────────
dates_sorted = sorted(panel['date'].unique())
TRAIN_Y, VAL_Y = 9, 1
min_before = (TRAIN_Y + VAL_Y) * 12

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

# ── 6. Save ──────────────────────────────────────────────────
print("\nSaving ...")
import pickle
save_data = {
    'panel': panel,
    'feat_cols': feat_cols,
    'non_ind': non_ind,
    'test_periods': test_periods,
    'beta_1y': beta_1y,
    'beta_3m': beta_3m,
    'beta_5y': beta_5y,
    'mkt_d': mkt_d,
    'mkt_m': mkt_m,
    'mkt_pe': mkt_pe,
    'rf': rf,
}
with open(f'{OUT}/stage_prep.pkl', 'wb') as f:
    pickle.dump(save_data, f, protocol=4)
print(f"  Saved to {OUT}/stage_prep.pkl")
print("\n[Stage A Complete]")
