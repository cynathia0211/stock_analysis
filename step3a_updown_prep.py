#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage A (Up/Down Beta): Build separate panels for upside and downside beta.

Classification:
  For each trading day t, compute:
    avg_fwd = mean(mkt_excess[t+1 : t+12M])
  If mkt_excess[t] > avg_fwd => "up" day, else "down" day.

Rolling betas and realized betas use only the relevant sub-sample of days.
All other features remain unchanged.

Output: results_up/stage_prep.pkl, results_down/stage_prep.pkl
"""
import os, time, gc, warnings, pickle
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

PKL = "data/pkl/"
SNAP = 'panel_snapshots/'
os.makedirs(SNAP, exist_ok=True)

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

# ── Build excess returns ─────────────────────────────────────
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

rf_m_aligned = rf.reindex(monthly_ret.index).fillna(0)
monthly_ret_ex = monthly_ret.sub(rf_m_aligned, axis=0)
mkt_m_ex = mkt_m - rf.reindex(mkt_m.index).fillna(0)

print(f"  Done in {time.time()-t0:.1f}s")
print(f"  Daily: {daily_ret.shape}, Monthly: {monthly_ret.shape}")

all_stocks = sorted(set(daily_ret.columns) & set(monthly_ret.columns))
print(f"  Common stocks: {len(all_stocks)}")

# ── 2. Compute future 12-month average & classify days ───────
print("\nClassifying up/down days ...")
t0 = time.time()

mkt_dates = mkt_d_ex.index
mkt_vals = mkt_d_ex.values
n_days = len(mkt_vals)

vals_clean = np.where(np.isnan(mkt_vals), 0, mkt_vals)
is_valid = ~np.isnan(mkt_vals)
cumsum = np.cumsum(vals_clean)
cumcount = np.cumsum(is_valid.astype(int))

future_avg = np.full(n_days, np.nan)
for i in range(n_days):
    end_date = mkt_dates[i] + pd.DateOffset(months=12)
    j = mkt_dates.searchsorted(end_date, side='right') - 1
    if j > i and j < n_days:
        cnt = cumcount[j] - cumcount[i]
        if cnt >= 120:
            future_avg[i] = (cumsum[j] - cumsum[i]) / cnt

future_avg_s = pd.Series(future_avg, index=mkt_dates)
has_future = ~np.isnan(future_avg)
is_up = pd.Series(False, index=mkt_dates)
is_down = pd.Series(False, index=mkt_dates)
is_up[has_future] = mkt_vals[has_future] > future_avg[has_future]
is_down[has_future] = ~is_up[has_future]

n_up = is_up.sum()
n_down = is_down.sum()
n_classified = has_future.sum()
print(f"  Classified {n_classified}/{n_days} days (last ~12M lack forward data)")
print(f"  Up days: {n_up} ({n_up/n_classified*100:.1f}%)")
print(f"  Down days: {n_down} ({n_down/n_classified*100:.1f}%)")
print(f"  Classification done in {time.time()-t0:.1f}s")

# ── 3. Compute betas ────────────────────────────────────────
month_ends = monthly_ret.index.tolist()

def compute_betas_filtered(daily_ret_df, mkt_daily, month_ends, window_months,
                           min_obs, day_mask):
    result = pd.DataFrame(index=month_ends, columns=daily_ret_df.columns, dtype=float)
    mask_arr = day_mask.reindex(daily_ret_df.index).fillna(False).values
    for i, mdate in enumerate(month_ends):
        start = mdate - pd.DateOffset(months=window_months)
        time_idx = (daily_ret_df.index > start) & (daily_ret_df.index <= mdate)
        combined = np.asarray(time_idx) & mask_arr
        if combined.sum() < min_obs:
            continue
        s = daily_ret_df.iloc[combined]
        m = mkt_daily.reindex(s.index)
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
            print(f"      {mdate.strftime('%Y-%m')} ({i+1}/{len(month_ends)})")
    return result

def compute_realized_filtered(daily_ret_df, mkt_daily, month_ends, min_obs, day_mask):
    result = pd.DataFrame(index=month_ends, columns=daily_ret_df.columns, dtype=float)
    mask_arr = day_mask.reindex(daily_ret_df.index).fillna(False).values
    for i, mdate in enumerate(month_ends):
        end = mdate + pd.DateOffset(months=12)
        time_idx = (daily_ret_df.index > mdate) & (daily_ret_df.index <= end)
        combined = np.asarray(time_idx) & mask_arr
        if combined.sum() < min_obs:
            continue
        s = daily_ret_df.iloc[combined]
        m = mkt_daily.reindex(s.index)
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
        result.iloc[i] = np.where((var_m > 1e-12) & (n >= min_obs), cov / var_m, np.nan)
    return result

# OLS_5Y_M: unfiltered (monthly frequency, not affected by daily split)
print("\n  OLS_5Y_M (unfiltered, monthly) ...")
t0 = time.time()
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
print(f"  Done in {time.time()-t0:.1f}s")

# Compute filtered betas for both regimes
betas = {}
for regime, day_mask in [('up', is_up), ('down', is_down)]:
    print(f"\n  Computing betas for {regime} regime ...")
    t0 = time.time()
    print(f"    OLS_3M_D_{regime} ...")
    b3m = compute_betas_filtered(daily_ret_ex, mkt_d_ex, month_ends, 3, 15, day_mask)
    print(f"    OLS_1Y_D_{regime} ...")
    b1y = compute_betas_filtered(daily_ret_ex, mkt_d_ex, month_ends, 12, 60, day_mask)
    print(f"    Realized beta ({regime}) ...")
    rb = compute_realized_filtered(daily_ret_ex, mkt_d_ex, month_ends, 60, day_mask)
    betas[regime] = {'beta_3m': b3m, 'beta_1y': b1y, 'realized': rb}
    print(f"    {regime} betas done in {time.time()-t0:.1f}s")
    n_valid = rb.notna().sum().sum()
    print(f"    Realized beta valid entries: {n_valid}")

# ── 4. Build features for each regime ────────────────────────
def build_panel(regime, beta_3m, beta_1y, realized_beta):
    print(f"\n{'='*60}")
    print(f"Building features for {regime} regime ...")
    print(f"{'='*60}")
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

        # IVOL, COSKEW, KURT: use ALL daily returns (unfiltered)
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

    if not rows:
        print(f"  WARNING: no valid months for {regime} regime!")
        return None

    panel = pd.concat(rows, ignore_index=True)
    print(f"  Panel ({regime}): {panel.shape}, {time.time()-t0:.1f}s")
    return panel


def preprocess_and_save(panel, regime, beta_3m, beta_1y):
    OUT = f'results_{regime}/'
    os.makedirs(OUT, exist_ok=True)

    print(f"\nPreprocessing {regime} panel ...")
    t0 = time.time()
    meta = ['stock', 'date', 'realized_beta', 'weight']
    feat_cols = [c for c in panel.columns if c not in meta]
    non_ind = [c for c in feat_cols if not c.startswith('IND_')]

    for col in non_ind:
        panel[col] = panel.groupby('date')[col].transform(
            lambda x: x.clip(x.quantile(0.005), x.quantile(0.995)))

    feat_stats = {}
    for col in feat_cols:
        vals = panel[col].values
        feat_stats[col] = {
            'mean': float(np.nanmean(vals)),
            'var':  float(np.nanvar(vals)),
            'max':  float(np.nanmax(vals)),
            'min':  float(np.nanmin(vals)),
        }

    panel_cs = panel.copy()
    for col in non_ind:
        panel_cs[col] = panel_cs.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8))
    feat_stats_cs = {}
    for col in feat_cols:
        vals = panel_cs[col].values
        feat_stats_cs[col] = {
            'mean': float(np.nanmean(vals)),
            'var':  float(np.nanvar(vals)),
            'max':  float(np.nanmax(vals)),
            'min':  float(np.nanmin(vals)),
        }
    del panel_cs

    for col in non_ind:
        panel[col] = panel.groupby('date')[col].transform(
            lambda x: 2 * x.rank(pct=True) - 1)

    panel[feat_cols] = panel[feat_cols].fillna(0)

    panel['weight'] = panel.groupby('date')['weight'].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x))

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

    print(f"  Features: {len(feat_cols)} ({len(non_ind)} non-industry)")
    print(f"  Test periods: {len(test_periods)}")
    print(f"  Preprocessing done in {time.time()-t0:.1f}s")

    save_data = {
        'panel': panel,
        'feat_cols': feat_cols,
        'non_ind': non_ind,
        'feat_stats': feat_stats,
        'feat_stats_cs': feat_stats_cs,
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
    return test_periods


# ── Build panels for both regimes ────────────────────────────
for regime in ['up', 'down']:
    b = betas[regime]
    panel = build_panel(regime, b['beta_3m'], b['beta_1y'], b['realized'])
    if panel is not None:
        preprocess_and_save(panel, regime, b['beta_3m'], b['beta_1y'])
    gc.collect()

print("\n[Stage A Up/Down Complete]")
