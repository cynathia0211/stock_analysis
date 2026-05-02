#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Step 1: Convert xlsx -> pkl (one file at a time, only run once)."""
import os, time, datetime
import pandas as pd
import numpy as np

SRC = "data/按指标拆分的文件/"
DST = "data/pkl/"
os.makedirs(DST, exist_ok=True)


def clean_panel(df):
    """Clean panel data: remove repeated header rows, set date index."""
    cols = list(df.columns)
    stock_codes = cols[1:]
    df = df.iloc[2:].copy()
    df.columns = ['date'] + stock_codes
    # Drop repeated header rows embedded in data
    df = df[df['date'].apply(lambda x: isinstance(x, (datetime.datetime, pd.Timestamp)))].copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def convert_one(name, src_file, is_panel=True):
    dst = os.path.join(DST, name + ".pkl")
    if os.path.exists(dst):
        print(f"  [skip] {name}")
        return
    t0 = time.time()
    print(f"  Loading {src_file} ...", end="", flush=True)
    df = pd.read_excel(os.path.join(SRC, src_file))
    if is_panel:
        df = clean_panel(df)
    print(f" -> {df.shape}, saving ...", end="", flush=True)
    df.to_pickle(dst)
    print(f" done ({time.time()-t0:.1f}s)")


def convert_fundamental(name, src_file):
    """Fundamental data: rows=stocks, cols=years."""
    import re
    dst = os.path.join(DST, name + ".pkl")
    if os.path.exists(dst):
        print(f"  [skip] {name}")
        return
    t0 = time.time()
    print(f"  Loading {src_file} ...", end="", flush=True)
    df = pd.read_excel(os.path.join(SRC, src_file))
    stock_codes = df.iloc[:, 0].values
    years = []
    for col in df.columns[2:]:
        m = re.search(r'(\d{4})年报', str(col))
        years.append(int(m.group(1)) if m else None)
    result = pd.DataFrame(index=stock_codes)
    result.index.name = 'stock_code'
    for i, yr in enumerate(years):
        if yr is not None:
            result[yr] = pd.to_numeric(df.iloc[:, i+2].replace('--', np.nan), errors='coerce').values
    print(f" -> {result.shape}, saving ...", end="", flush=True)
    result.to_pickle(dst)
    print(f" done ({time.time()-t0:.1f}s)")


def convert_meta(name, src_file, columns, parse_dates=None):
    dst = os.path.join(DST, name + ".pkl")
    if os.path.exists(dst):
        print(f"  [skip] {name}")
        return
    t0 = time.time()
    print(f"  Loading {src_file} ...", end="", flush=True)
    df = pd.read_excel(os.path.join(SRC, src_file))
    df.columns = columns[:len(df.columns)]
    if parse_dates:
        for c in parse_dates:
            df[c] = pd.to_datetime(df[c], errors='coerce')
    print(f" -> {df.shape}, saving ...", end="", flush=True)
    df.to_pickle(dst)
    print(f" done ({time.time()-t0:.1f}s)")


print("=== Converting data files to pkl ===")

# Panel data (date × stocks)
convert_one("daily_ret",        "日收益率.xlsx")
convert_one("monthly_ret",      "月收益率.xlsx")
convert_one("market_daily_ret", "中证全指日度收益率.xlsx")
convert_one("market_monthly_ret","中证全指月度收益率.xlsx")
convert_one("market_cap",       "所有A股市值.xlsx")
convert_one("volume",           "所有A股成交量.xlsx")
convert_one("pe_ratio",         "所有A股市盈率.xlsx")
convert_one("market_pe",        "中证全指月度市盈率.xlsx")

# Risk-free rate
dst = os.path.join(DST, "rf_monthly.pkl")
if not os.path.exists(dst):
    print("  Loading rf ...", end="", flush=True)
    rf = pd.read_excel(os.path.join(SRC, "1年期国债到期收益率月数据.xlsx"))
    rf.columns = ['date', 'rf']
    rf = rf[pd.to_datetime(rf['date'], errors='coerce').notna()].copy()
    rf['date'] = pd.to_datetime(rf['date'])
    rf = rf.set_index('date').sort_index()
    rf['rf'] = rf['rf'] / 100 / 12  # annualized -> monthly
    rf.to_pickle(dst)
    print(" done")
else:
    print("  [skip] rf_monthly")

# Metadata
convert_meta("listing_dates", "C列_上市日期.xlsx",
             ['stock_code', 'stock_name', 'listing_date'], parse_dates=['listing_date'])
convert_meta("industry", "D列_所属申万行业[行业级别]一级行业[截止日期]最新.xlsx",
             ['stock_code', 'stock_name', 'industry'])

# Fundamentals
fund_map = {
    'fund_total_assets':   '指标_账面资产总额.xlsx',
    'fund_equity':         '指标_所有者权益合计.xlsx',
    'fund_net_sales':      '指标_净销售额.xlsx',
    'fund_net_income':     '指标_净利润.xlsx',
    'fund_roe':            '指标_净资产收益率ROE.xlsx',
    'fund_roa':            '指标_总资产报酬率ROA.xlsx',
    'fund_fixed_cost':     '指标_固定成本.xlsx',
    'fund_fixed_assets':   '指标_固定资产.xlsx',
    'fund_gross_margin':   '指标_销售毛利率.xlsx',
    'fund_ebit':           '指标_息税前利润EBIT.xlsx',
    'fund_cogs':           '指标_营业成本.xlsx',
    'fund_noa':            '指标_净运营资产.xlsx',
    'fund_nwcap':          '指标_非现金营运资本.xlsx',
}
for name, fname in fund_map.items():
    convert_fundamental(name, fname)

print("\n=== All conversions done ===")
