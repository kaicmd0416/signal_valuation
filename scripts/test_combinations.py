"""
批量测试不同因子组合方案 - 只跑hs300
直接调用核心逻辑，不生成PDF，只输出关键指标对比
"""
import time, sys
sys.path.insert(0, "D:/signal_valuation")

import pandas as pd
import numpy as np
from core.data_prepare import (
    load_config, get_factor_data, get_index_component, get_market_data,
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

# 加载共享数据
print("加载共享数据...")
t0 = time.time()
mkt_end = next_workday(end_date)
market_data = get_market_data(start_date, mkt_end)
df_stock = market_data[0]
df_index_ret = market_data[6]
df_calendar = get_trading_calendar(start_date, mkt_end)
df_st = get_st_stocks(start_date, mkt_end)
df_notrade = get_notrade_stocks(start_date, mkt_end)
df_index_comp = get_index_component(start_date, end_date, "hs300")
print(f"共享数据加载完成 [{time.time()-t0:.1f}s]")

# 指数收益
index_code = INDEX_CODE_MAP.get("沪深300")
df_idx_ret = df_index_ret[df_index_ret["code"] == index_code][
    ["valuation_date", "pct_chg"]
].copy()
df_idx_ret["valuation_date"] = df_idx_ret["valuation_date"].astype(str)

# 定义测试方案
experiments = {
    "A_9因子_MA1": {
        "smooth": 1,
        "clusters": {
            "估值": {"factors": [
                {"name": "PCF", "direction": 1},
                {"name": "PB", "direction": 1},
                {"name": "PE", "direction": 1},
                {"name": "PS", "direction": 1},
            ]},
            "换手率": {"factors": [
                {"name": "turnoverRateAvg20d", "direction": 1},
                {"name": "turnoverRateAvg120d", "direction": 1},
            ]},
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "CTA": {"factors": [{"name": "CTA", "direction": -1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
        },
    },
    "B_5精选_MA1": {
        "smooth": 1,
        "clusters": {
            "估值": {"factors": [
                {"name": "PCF", "direction": 1},
                {"name": "PB", "direction": 1},
            ]},
            "换手率": {"factors": [{"name": "turnoverRateAvg20d", "direction": 1}]},
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
        },
    },
    "C_3强因子_MA1": {
        "smooth": 1,
        "clusters": {
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "估值": {"factors": [{"name": "PCF", "direction": 1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
        },
    },
    "D_6因子+TRC_MA1": {
        "smooth": 1,
        "clusters": {
            "估值": {"factors": [
                {"name": "PCF", "direction": 1},
                {"name": "PB", "direction": 1},
            ]},
            "换手率": {"factors": [{"name": "turnoverRateAvg20d", "direction": 1}]},
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
            "换手变化": {"factors": [{"name": "TurnoverRateChange", "direction": -1}]},
        },
    },
    "E_TopSharpe6_MA1": {
        "smooth": 1,
        "clusters": {
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "估值": {"factors": [
                {"name": "PCF", "direction": 1},
                {"name": "PB", "direction": 1},
                {"name": "PS", "direction": 1},
            ]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
            "换手变化": {"factors": [{"name": "TurnoverRateChange", "direction": -1}]},
        },
    },
    "F_4独立_MA1": {
        "smooth": 1,
        "clusters": {
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "估值": {"factors": [{"name": "PCF", "direction": 1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
            "换手变化": {"factors": [{"name": "TurnoverRateChange", "direction": -1}]},
        },
    },
    "G_9因子_MA5": {
        "smooth": 5,
        "clusters": {
            "估值": {"factors": [
                {"name": "PCF", "direction": 1},
                {"name": "PB", "direction": 1},
                {"name": "PE", "direction": 1},
                {"name": "PS", "direction": 1},
            ]},
            "换手率": {"factors": [
                {"name": "turnoverRateAvg20d", "direction": 1},
                {"name": "turnoverRateAvg120d", "direction": 1},
            ]},
            "流动性": {"factors": [{"name": "ILLIQ", "direction": -1}]},
            "CTA": {"factors": [{"name": "CTA", "direction": -1}]},
            "市值": {"factors": [{"name": "LnFloatCap1", "direction": -1}]},
        },
    },
}


def run_experiment(exp_name, exp_cfg):
    """运行单个实验，返回结果dict"""
    print(f"\n{'='*60}")
    print(f"  实验: {exp_name}")
    print(f"{'='*60}")

    t0 = time.time()
    index_cfg = {"output_name": "test", "clusters": exp_cfg["clusters"]}

    result = combine_factors_for_index(
        start_date, end_date,
        index_name="hs300",
        index_cfg=index_cfg,
        df_index_comp=df_index_comp,
        df_stock=df_stock,
        weight_method="two_level_ic",
        ic_window=ic_window,
        date_mode=date_mode,
        df_calendar=df_calendar,
        top_n_extra=stock_number,
        smooth_window=exp_cfg["smooth"],
        df_st=df_st,
        df_notrade=df_notrade,
    )

    if isinstance(result, tuple):
        df_combined, df_fm_scores = result
    else:
        df_combined, df_fm_scores = result, None

    elapsed = time.time() - t0

    if df_combined.empty:
        print("  空结果，跳过")
        return None

    # 构建固定维度
    comp_codes_by_date = (
        df_combined.groupby("valuation_date")["code"].apply(set).to_dict()
    )
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
            df_top_extra = df_non_comp.nlargest(n_extra, "final_score")
            df_day = pd.concat([df_comp, df_top_extra], ignore_index=True)
        else:
            df_day = df_comp.copy()
        mean, std = df_day["final_score"].mean(), df_day["final_score"].std()
        if std > 0:
            df_day["final_score"] = (df_day["final_score"] - mean) / std
        parts.append(df_day)

    df_fixed = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    # IC (成分股)
    df_factor_merged = df_combined.rename(columns={"final_score": "factor_value"})
    df_ic = calc_daily_ic(df_factor_merged, df_stock)
    ic_summary = calc_ic_summary(df_ic) if not df_ic.empty else {}

    # 分层回测
    sa = SignalAnalysis(
        signal_name="test", index_name="hs300", n_groups=5,
        df_factor=df_combined, df_index_comp=df_index_comp,
        df_stock=df_stock, df_index_ret=df_index_ret,
        df_calendar=df_calendar, df_st=df_st, df_notrade=df_notrade,
        date_mode=date_mode,
    )
    df_info = sa.run()
    df_info = df_info[df_info["valuation_date"].astype(str) <= end_date]

    g1_cum = g5_cum = 0
    if not df_info.empty:
        for gname, sub in df_info.groupby("portfolio_name"):
            cum = (1 + sub["excess_paper_return"]).prod() - 1
            if gname == "group_1":
                g1_cum = cum * 100
            if gname == "group_5":
                g5_cum = cum * 100

    # Top组合
    comp_count = df_combined.groupby("valuation_date")["code"].count().to_dict()
    df_top = (
        df_fixed.groupby("valuation_date", group_keys=False)
        .apply(
            lambda g: g.nlargest(
                max(0, stock_number - comp_count.get(g.name, 0)), "final_score"
            )
        )
        .reset_index(drop=True)
    )

    # Top回测
    df_hold = df_top[["valuation_date", "code"]].copy()
    df_hold["valuation_date"] = df_hold["valuation_date"].astype(str)
    df_hold["weight"] = df_hold.groupby("valuation_date")["code"].transform(
        lambda x: 1.0 / len(x)
    )

    df_ret = df_stock[["valuation_date", "code", "close", "pre_close"]].copy()
    df_ret["valuation_date"] = df_ret["valuation_date"].astype(str)
    df_ret["pct_chg"] = (df_ret["close"] - df_ret["pre_close"]) / df_ret["pre_close"]

    df_calc = df_hold.merge(
        df_ret[["valuation_date", "code", "pct_chg"]],
        on=["valuation_date", "code"],
        how="left",
    )
    df_calc.dropna(subset=["pct_chg"], inplace=True)
    df_calc["wr"] = df_calc["weight"] * df_calc["pct_chg"]
    df_daily = df_calc.groupby("valuation_date")["wr"].sum().reset_index()
    df_daily.columns = ["valuation_date", "port_ret"]
    df_daily = df_daily[df_daily["valuation_date"] <= end_date].sort_values(
        "valuation_date"
    )

    df_daily = df_daily.merge(df_idx_ret, on="valuation_date", how="left")
    df_daily["pct_chg"].fillna(0, inplace=True)
    df_daily["excess"] = df_daily["port_ret"] - df_daily["pct_chg"]
    df_daily["cum_excess"] = (1 + df_daily["excess"]).cumprod()

    top_cum_excess = (df_daily["cum_excess"].iloc[-1] - 1) * 100
    n_days = len(df_daily)
    top_ann = top_cum_excess * 252 / n_days
    top_vol = df_daily["excess"].std() * np.sqrt(252) * 100
    top_sharpe = top_ann / top_vol if top_vol > 0 else 0

    # 换手
    dates_sorted = sorted(df_hold["valuation_date"].unique())
    turnovers = []
    for i in range(1, len(dates_sorted)):
        prev = set(df_hold[df_hold["valuation_date"] == dates_sorted[i - 1]]["code"])
        curr = set(df_hold[df_hold["valuation_date"] == dates_sorted[i]]["code"])
        turnovers.append(len(curr - prev) / len(curr) if curr else 0)
    ann_turnover = np.mean(turnovers) * 252 if turnovers else 0

    r = {
        "方案": exp_name,
        "因子数": sum(len(c["factors"]) for c in exp_cfg["clusters"].values()),
        "IC": ic_summary.get("Rank IC 均值", 0),
        "IR": ic_summary.get("IR (IC均值/IC标准差)", 0),
        "G1%": round(g1_cum, 1),
        "G5%": round(g5_cum, 1),
        "G5-G1": round(g5_cum - g1_cum, 1),
        "Top累计%": round(top_cum_excess, 1),
        "Top年化%": round(top_ann, 1),
        "Top夏普": round(top_sharpe, 2),
        "Top换手": round(ann_turnover, 1),
        "耗时": f"{elapsed:.0f}s",
    }
    print(f"  IC={r['IC']:.4f}, IR={r['IR']:.4f}")
    print(f"  G1={r['G1%']}%, G5={r['G5%']}%, G5-G1={r['G5-G1']}%")
    print(
        f"  Top: 累计={r['Top累计%']}%, 年化={r['Top年化%']}%, "
        f"夏普={r['Top夏普']}, 换手={r['Top换手']}"
    )
    return r


# 执行
results = []
for exp_name, exp_cfg in experiments.items():
    r = run_experiment(exp_name, exp_cfg)
    if r:
        results.append(r)

# 汇总
print(f"\n{'='*120}")
print("方案对比汇总:")
print(f"{'='*120}")
df_r = pd.DataFrame(results)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
print(df_r.to_string(index=False))
