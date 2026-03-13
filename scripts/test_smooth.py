"""测试5因子IC加权不同smooth_window的效果"""
import time, sys
sys.path.insert(0, "D:/signal_valuation")

import pandas as pd
import numpy as np
from core.data_prepare import (
    get_index_component, get_market_data,
    get_trading_calendar, get_st_stocks, get_notrade_stocks, next_workday,
)
from core.factor_combine import combine_factors_for_index
from core.report import calc_daily_ic, calc_ic_summary
from core.backtest import SignalAnalysis, INDEX_CODE_MAP

start_date = "2023-07-01"
end_date = "2026-03-10"
stock_number = 500
ic_window = 60
date_mode = "available_date"

print("加载数据...")
t0 = time.time()
mkt_end = next_workday(end_date)
market_data = get_market_data(start_date, mkt_end)
df_stock = market_data[0]
df_index_ret = market_data[6]
df_calendar = get_trading_calendar(start_date, mkt_end)
df_st = get_st_stocks(start_date, mkt_end)
df_notrade = get_notrade_stocks(start_date, mkt_end)
df_index_comp = get_index_component(start_date, end_date, "hs300")
print(f"数据加载完成 [{time.time()-t0:.1f}s]")

index_code = INDEX_CODE_MAP.get("沪深300")
df_idx_ret = df_index_ret[df_index_ret["code"] == index_code][["valuation_date", "pct_chg"]].copy()
df_idx_ret["valuation_date"] = df_idx_ret["valuation_date"].astype(str)

clusters = {
    "估值": {"factors": [
        {"name": "PE", "direction": 1},
        {"name": "PCF", "direction": 1},
        {"name": "PB", "direction": 1},
        {"name": "PS", "direction": 1},
    ]},
    "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
}

results = []
for smooth in [1, 2, 3, 5]:
    print(f"\n===== smooth_window={smooth} =====")
    t0 = time.time()
    index_cfg = {"output_name": "test", "clusters": clusters}
    result = combine_factors_for_index(
        start_date, end_date, index_name="hs300", index_cfg=index_cfg,
        df_index_comp=df_index_comp, df_stock=df_stock,
        weight_method="two_level_ic", ic_window=ic_window,
        date_mode=date_mode, df_calendar=df_calendar,
        top_n_extra=stock_number, smooth_window=smooth,
        df_st=df_st, df_notrade=df_notrade,
    )
    if isinstance(result, tuple):
        df_combined, df_fm_scores = result
    else:
        df_combined, df_fm_scores = result, None
    elapsed = time.time() - t0

    # IC
    df_fm = df_combined.rename(columns={"final_score": "factor_value"})
    df_ic = calc_daily_ic(df_fm, df_stock)
    ics = calc_ic_summary(df_ic) if not df_ic.empty else {}

    # 分层
    sa = SignalAnalysis(
        signal_name="test", index_name="hs300", n_groups=5,
        df_factor=df_combined, df_index_comp=df_index_comp,
        df_stock=df_stock, df_index_ret=df_index_ret,
        df_calendar=df_calendar, df_st=df_st, df_notrade=df_notrade,
        date_mode=date_mode,
    )
    df_info = sa.run()
    df_info = df_info[df_info["valuation_date"].astype(str) <= end_date]
    gs = {}
    for gn in ["group_1", "group_2", "group_3", "group_4", "group_5"]:
        sub = df_info[df_info["portfolio_name"] == gn]
        cum = (1 + sub["excess_paper_return"]).prod() - 1
        cum_net = (1 + sub["excess_net_return"]).prod() - 1
        gs[gn] = {"pre": cum * 100, "net": cum_net * 100}

    # 多空夏普
    g5_sub = df_info[df_info["portfolio_name"] == "group_5"].sort_values("valuation_date")
    g1_sub = df_info[df_info["portfolio_name"] == "group_1"].sort_values("valuation_date")
    if len(g5_sub) == len(g1_sub):
        ls = g5_sub["excess_paper_return"].values - g1_sub["excess_paper_return"].values
        ls_cum = ((1 + pd.Series(ls)).prod() - 1) * 100
        ls_vol = pd.Series(ls).std() * np.sqrt(252) * 100
        ls_ann = ls_cum * 252 / len(ls)
        ls_sharpe = ls_ann / ls_vol if ls_vol > 0 else 0
    else:
        ls_cum = ls_sharpe = 0

    # Top
    comp_codes_by_date = df_combined.groupby("valuation_date")["code"].apply(set).to_dict()
    dates = sorted(df_combined["valuation_date"].unique())
    parts = []
    for d in dates:
        df_fm_day = df_fm_scores[df_fm_scores["valuation_date"] == d]
        if df_fm_day.empty:
            continue
        comp_codes = comp_codes_by_date.get(d, set())
        df_comp = df_fm_day[df_fm_day["code"].isin(comp_codes)]
        df_non_comp = df_fm_day[~df_fm_day["code"].isin(comp_codes)]
        n_extra = max(0, stock_number - len(df_comp))
        if n_extra > 0 and not df_non_comp.empty:
            df_day = pd.concat([df_comp, df_non_comp.nlargest(n_extra, "final_score")], ignore_index=True)
        else:
            df_day = df_comp.copy()
        mean, std = df_day["final_score"].mean(), df_day["final_score"].std()
        if std > 0:
            df_day["final_score"] = (df_day["final_score"] - mean) / std
        parts.append(df_day)
    df_fixed = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    comp_count = df_combined.groupby("valuation_date")["code"].count().to_dict()
    df_top = (
        df_fixed.groupby("valuation_date", group_keys=False)
        .apply(lambda g: g.nlargest(max(0, stock_number - comp_count.get(g.name, 0)), "final_score"))
        .reset_index(drop=True)
    )
    df_hold = df_top[["valuation_date", "code"]].copy()
    df_hold["valuation_date"] = df_hold["valuation_date"].astype(str)
    df_hold["weight"] = df_hold.groupby("valuation_date")["code"].transform(lambda x: 1.0 / len(x))
    df_ret = df_stock[["valuation_date", "code", "close", "pre_close"]].copy()
    df_ret["valuation_date"] = df_ret["valuation_date"].astype(str)
    df_ret["pct_chg"] = (df_ret["close"] - df_ret["pre_close"]) / df_ret["pre_close"]
    df_calc = df_hold.merge(df_ret[["valuation_date", "code", "pct_chg"]], on=["valuation_date", "code"], how="left")
    df_calc.dropna(subset=["pct_chg"], inplace=True)
    df_calc["wr"] = df_calc["weight"] * df_calc["pct_chg"]
    df_daily = df_calc.groupby("valuation_date")["wr"].sum().reset_index()
    df_daily.columns = ["valuation_date", "port_ret"]
    df_daily = df_daily[df_daily["valuation_date"] <= end_date].sort_values("valuation_date")
    df_daily = df_daily.merge(df_idx_ret, on="valuation_date", how="left")
    df_daily["pct_chg"].fillna(0, inplace=True)
    df_daily["excess"] = df_daily["port_ret"] - df_daily["pct_chg"]
    df_daily["cum_excess"] = (1 + df_daily["excess"]).cumprod()
    top_cum = (df_daily["cum_excess"].iloc[-1] - 1) * 100
    n_days = len(df_daily)
    top_ann = top_cum * 252 / n_days
    top_vol = df_daily["excess"].std() * np.sqrt(252) * 100
    top_sharpe = top_ann / top_vol if top_vol > 0 else 0

    # 换手
    ds = sorted(df_hold["valuation_date"].unique())
    tvs = []
    for i in range(1, len(ds)):
        prev = set(df_hold[df_hold["valuation_date"] == ds[i - 1]]["code"])
        curr = set(df_hold[df_hold["valuation_date"] == ds[i]]["code"])
        tvs.append(len(curr - prev) / len(curr) if curr else 0)
    ann_tv = np.mean(tvs) * 252 if tvs else 0

    # 年度G5
    df_info["year"] = df_info["valuation_date"].astype(str).str[:4]
    yr_g5 = {}
    for yr, sub in df_info[df_info["portfolio_name"] == "group_5"].groupby("year"):
        yr_g5[yr] = round(((1 + sub["excess_paper_return"]).prod() - 1) * 100, 1)

    r = {
        "MA": smooth,
        "IR": round(ics.get("IR (IC\u5747\u503c/IC\u6807\u51c6\u5dee)", 0), 3),
        "G1": round(gs["group_1"]["pre"], 1),
        "G2": round(gs["group_2"]["pre"], 1),
        "G3": round(gs["group_3"]["pre"], 1),
        "G4": round(gs["group_4"]["pre"], 1),
        "G5": round(gs["group_5"]["pre"], 1),
        "G5net": round(gs["group_5"]["net"], 1),
        "G5-G1": round(gs["group_5"]["pre"] - gs["group_1"]["pre"], 1),
        "LS_sharpe": round(ls_sharpe, 2),
        "Top%": round(top_cum, 1),
        "TopSharpe": round(top_sharpe, 2),
        "TV": round(ann_tv, 1),
        "G5_23": yr_g5.get("2023", "-"),
        "G5_24": yr_g5.get("2024", "-"),
        "G5_25": yr_g5.get("2025", "-"),
        "G5_26": yr_g5.get("2026", "-"),
    }
    results.append(r)
    print(f"  IR={r['IR']} G5={r['G5']}% G5net={r['G5net']}% G5-G1={r['G5-G1']}% LS_sharpe={r['LS_sharpe']}")
    print(f"  G1={r['G1']} G2={r['G2']} G3={r['G3']} G4={r['G4']} G5={r['G5']}")
    print(f"  Top={r['Top%']}% TopSharpe={r['TopSharpe']} TV={r['TV']}")
    print(f"  G5年度: 23={r['G5_23']} 24={r['G5_24']} 25={r['G5_25']} 26={r['G5_26']}")

print(f"\n{'='*140}")
print("5因子IC加权 smooth_window扫描:")
print(f"{'='*140}")
df_r = pd.DataFrame(results)
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
print(df_r.to_string(index=False))
