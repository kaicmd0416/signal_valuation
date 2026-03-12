"""
按指数合成因子模块
==================
支持三种合成方式: 等权 / IC加权 / 两层IC加权

核心函数: combine_factors_for_index()

流程:
1. 批量加载因子 → 成分股内 z-score 标准化
2. IC加权合成 → 时序平滑(MA)
3. 全市场评分(用成分股IC权重) → top选股
"""

import time
import pandas as pd
import numpy as np

from .data_prepare import (
    load_combine_by_index_config, get_factor_data_batch, get_index_component,
    last_workday,
)


# ============================================================
# 标准化
# ============================================================

def zscore_cross_section(series: pd.Series) -> pd.Series:
    """截面 z-score 标准化: (x - mean) / std"""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - mean) / std


# ============================================================
# 按指数合成因子 - 辅助函数
# ============================================================

def _calc_factor_daily_ic(df_factor: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    """
    计算单因子逐日 Rank IC

    因子日期已统一为 target_date（调用前已完成转换），
    IC(t) = corr(factor(t), return(t))  — 同日匹配

    Parameters
    ----------
    df_factor : DataFrame[valuation_date, code, z_score]  (方向已处理, 日期=target_date)
    df_stock  : 股票行情 DataFrame[valuation_date, code, close, pre_close]

    Returns
    -------
    DataFrame[valuation_date, rank_IC]
    """
    df_ret = df_stock[["valuation_date", "code", "close", "pre_close"]].copy()
    df_ret["valuation_date"] = df_ret["valuation_date"].astype(str)
    df_ret["pct_chg"] = (df_ret["close"] - df_ret["pre_close"]) / df_ret["pre_close"]
    df_ret = df_ret[["valuation_date", "code", "pct_chg"]].drop_duplicates(
        subset=["valuation_date", "code"]
    )

    # 同日匹配: factor(t) vs return(t)
    df_merged = df_factor.merge(
        df_ret, on=["valuation_date", "code"], how="inner",
    )
    df_merged.dropna(subset=["z_score", "pct_chg"], inplace=True)

    # 过滤每日至少10只股票
    counts = df_merged.groupby("valuation_date")["code"].transform("count")
    df_merged = df_merged[counts >= 10]

    if df_merged.empty:
        return pd.DataFrame(columns=["valuation_date", "rank_IC"])

    # Rank IC (Spearman)
    df_merged["factor_rank"] = df_merged.groupby("valuation_date")["z_score"].rank()
    df_merged["ret_rank"] = df_merged.groupby("valuation_date")["pct_chg"].rank()

    ic_rank = df_merged.groupby("valuation_date").apply(
        lambda g: g["factor_rank"].corr(g["ret_rank"])
    ).rename("rank_IC").reset_index()
    ic_rank.dropna(inplace=True)
    return ic_rank


def _build_ic_weights(ic_map: dict, ic_window: int) -> pd.DataFrame:
    """
    构建滚动IC权重表

    Parameters
    ----------
    ic_map   : {name: DataFrame[valuation_date, rank_IC]}
    ic_window: 滚动窗口天数

    Returns
    -------
    DataFrame[valuation_date, factor_name, rolling_ic, weight]
    """
    all_ic = []
    for fname, ic_df in ic_map.items():
        tmp = ic_df[["valuation_date", "rank_IC"]].copy()
        tmp = tmp.sort_values("valuation_date")
        # shift(1): 日期t的权重只用t-1及之前的IC，避免前瞻偏差
        tmp["rolling_ic"] = tmp["rank_IC"].rolling(
            ic_window, min_periods=5
        ).mean().abs().shift(1)
        tmp["factor_name"] = fname
        all_ic.append(tmp[["valuation_date", "factor_name", "rolling_ic"]].dropna())

    if not all_ic:
        return pd.DataFrame(columns=["valuation_date", "factor_name", "rolling_ic", "weight"])

    df_ic_weights = pd.concat(all_ic, ignore_index=True)

    # 逐日归一化权重 (各因子权重之和=1)
    df_ic_weights["weight"] = df_ic_weights.groupby("valuation_date")["rolling_ic"].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
    )
    return df_ic_weights


def _ic_weighted_combine(zscores_map: dict, ic_map: dict,
                          ic_window: int, label: str = "") -> pd.DataFrame:
    """
    对一组 z-score 做 IC 加权合成

    Parameters
    ----------
    zscores_map : {name: DataFrame[valuation_date, code, z_score]}
    ic_map      : {name: DataFrame[valuation_date, rank_IC]}
    ic_window   : 滚动窗口
    label       : 日志标签 (空字符串时不打印)

    Returns
    -------
    DataFrame[valuation_date, code, combined_score]
    """
    if not zscores_map:
        return pd.DataFrame(columns=["valuation_date", "code", "combined_score"])

    df_weights = _build_ic_weights(ic_map, ic_window)

    if df_weights.empty:
        # 回退等权
        if label:
            print(f"    [{label}] IC权重为空，回退等权")
        all_z = list(zscores_map.values())
        df_all = pd.concat(all_z, ignore_index=True)
        result = df_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
        result.rename(columns={"z_score": "combined_score"}, inplace=True)
        return result

    # 打印平均权重
    if label:
        avg_w = df_weights.groupby("factor_name")["weight"].mean()
        print(f"    [{label}] 平均IC权重:")
        for fname, w in avg_w.sort_values(ascending=False).items():
            print(f"      {fname}: {w:.4f}")

    weighted_parts = []
    for fname, df_z in zscores_map.items():
        if fname not in ic_map:
            continue
        w_df = df_weights[df_weights["factor_name"] == fname][["valuation_date", "weight"]]
        df_zw = df_z.merge(w_df, on="valuation_date", how="inner")
        df_zw["weighted_z"] = df_zw["z_score"] * df_zw["weight"]
        weighted_parts.append(df_zw[["valuation_date", "code", "weighted_z"]])

    if not weighted_parts:
        if label:
            print(f"    [{label}] 无可加权因子，回退等权")
        all_z = list(zscores_map.values())
        df_all = pd.concat(all_z, ignore_index=True)
        result = df_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
        result.rename(columns={"z_score": "combined_score"}, inplace=True)
        return result

    df_all = pd.concat(weighted_parts, ignore_index=True)
    result = df_all.groupby(["valuation_date", "code"])["weighted_z"].sum().reset_index()
    result.rename(columns={"weighted_z": "combined_score"}, inplace=True)
    return result


# ============================================================
# 按指数合成因子 - 主函数
# ============================================================

def combine_factors_for_index(start_date: str, end_date: str,
                               index_name: str,
                               index_cfg: dict = None,
                               df_index_comp: pd.DataFrame = None,
                               df_stock: pd.DataFrame = None,
                               weight_method: str = "equal",
                               ic_window: int = 20,
                               date_mode: str = "target_date",
                               df_calendar: pd.DataFrame = None,
                               top_n_extra: int = 0,
                               smooth_window: int = 1,
                               df_st=None,
                               df_notrade=None) -> pd.DataFrame:
    """
    针对单个指数合成因子（支持等权 / IC加权 / 两层IC加权）

    z-score 在指数成分股内部做截面标准化，而非全市场，
    确保因子排序仅反映成分股内部的相对差异。

    weight_method 说明:
      - "equal":         所有因子等权
      - "ic_weight":     所有因子单层IC加权
      - "two_level_ic":  簇内IC加权 → 簇间IC加权 (需要config中有clusters字段)

    Parameters
    ----------
    start_date    : 开始日期 (target_date)
    end_date      : 结束日期 (target_date)
    index_name    : 指数简称, 如 "hs300"
    index_cfg     : 该指数的配置 dict，含 output_name 和 factors/clusters
    df_index_comp : 指数成分股 DataFrame[valuation_date, code]
    df_stock      : 股票行情 (IC加权时需要), 含 valuation_date, code, close, pre_close
    weight_method : "equal" | "ic_weight" | "two_level_ic"
    ic_window     : IC加权滚动窗口天数 (默认20)
    date_mode     : "target_date" | "available_date"
    df_calendar   : 交易日历 [valuation_date, next_workday], available_date模式必须传入
    top_n_extra   : >0 时，额外对全市场评分，返回 (成分股分数, 全市场分数) 元组
    smooth_window : 合成分数时序平滑窗口 (MA), 1=不平滑, 5=MA5
    df_st         : ST股票 DataFrame[valuation_date, code], 传入则复用外部数据，否则内部查询DB
    df_notrade    : 涨跌停股票 DataFrame[valuation_date, code], 传入则复用外部数据，否则内部查询DB

    Returns
    -------
    top_n_extra=0 : DataFrame[valuation_date, code, score_name, final_score]
    top_n_extra>0 : (DataFrame_成分股, DataFrame_全市场) 两个同格式的DataFrame
    """
    if index_cfg is None:
        cfg = load_combine_by_index_config()
        index_cfg = cfg["indices"][index_name]

    # 解析配置: 支持 clusters (新) 或 factors (旧) 格式
    cluster_map = None  # {cluster_name: [factor_cfgs]}
    if "clusters" in index_cfg:
        cluster_map = {}
        factor_list = []
        for cluster_name, cluster_cfg in index_cfg["clusters"].items():
            cluster_factors = cluster_cfg["factors"]
            cluster_map[cluster_name] = cluster_factors
            factor_list.extend(cluster_factors)
    else:
        factor_list = index_cfg["factors"]

    output_name = index_cfg.get("output_name", f"combine_{index_name}")

    # --- 步骤0: 获取指数成分股 ---
    if df_index_comp is None:
        print(f"\n加载指数成分: {index_name}")
        t0 = time.time()
        df_index_comp = get_index_component(start_date, end_date, index_name)
        print(f"  成分股加载完成: {len(df_index_comp)} 条, 耗时 {time.time()-t0:.1f}s")

    # 构建成分股 (date, code) 用于过滤 — 日期维度: available_date
    comp_keys = df_index_comp[["valuation_date", "code"]].copy()
    comp_keys["valuation_date"] = comp_keys["valuation_date"].astype(str)
    n_comp_dates = comp_keys["valuation_date"].nunique()
    n_comp_stocks = int(comp_keys.groupby("valuation_date")["code"].count().median())
    print(f"  成分股: {n_comp_dates} 个交易日, 每日中位数 {n_comp_stocks} 只")

    # --- 步骤1: 批量加载因子, 内部统一用 available_date ---
    factor_names = [f["name"] for f in factor_list]
    print(f"\n批量加载 {len(factor_names)} 个因子 (单次SQL)...")
    t0 = time.time()

    if date_mode == "available_date":
        # available_date 模式: DB日期就是 available_date, 直接查询不映射
        query_start = last_workday(start_date)
        df_all_raw = get_factor_data_batch(query_start, end_date, factor_names)
    else:
        # target_date 模式: DB日期是 target_date, 映射为 available_date (前一交易日)
        df_all_raw = get_factor_data_batch(start_date, end_date, factor_names)

    t_db = time.time() - t0
    print(f"  DB查询完成: {len(df_all_raw)} 条, 耗时 {t_db:.1f}s")

    # target_date 模式: 映射为 available_date
    if date_mode == "target_date":
        df_all_raw["valuation_date"] = df_all_raw["valuation_date"].astype(str)
        if df_calendar is not None and not df_calendar.empty:
            # target_date → available_date (前一交易日)
            reverse_cal = df_calendar.set_index(
                df_calendar["next_workday"].astype(str)
            )["valuation_date"].astype(str)
            df_all_raw["valuation_date"] = df_all_raw["valuation_date"].map(reverse_cal)
        else:
            all_dates = sorted(df_all_raw["valuation_date"].unique())
            date_shift = {all_dates[i + 1]: all_dates[i]
                          for i in range(len(all_dates) - 1)}
            df_all_raw["valuation_date"] = df_all_raw["valuation_date"].map(date_shift)
        df_all_raw.dropna(subset=["valuation_date"], inplace=True)
        print(f"  target_date → available_date 映射完成: {len(df_all_raw)} 条")

    # --- 步骤2: 过滤成分股 + 剔除ST/涨跌停 ---
    print(f"\n过滤到 {index_name} 成分股...")
    t0 = time.time()
    df_all_raw["valuation_date"] = df_all_raw["valuation_date"].astype(str)

    # 保留全市场数据副本（用于 top_n_extra 全市场评分）
    df_full_market_raw = df_all_raw.copy() if top_n_extra > 0 else None

    n_before_filter = len(df_all_raw)
    df_all_raw = df_all_raw.merge(comp_keys, on=["valuation_date", "code"], how="inner")
    n_after_filter = len(df_all_raw)
    print(f"  成分股过滤: {n_before_filter} -> {n_after_filter} 条")

    # 加载ST和涨跌停数据（若外部未传入则从DB加载）
    if df_st is None or df_notrade is None:
        print(f"加载ST和涨跌停数据...")
        if df_st is None:
            from data_prepare import get_st_stocks
            df_st = get_st_stocks(start_date, end_date)
        if df_notrade is None:
            from data_prepare import get_notrade_stocks
            df_notrade = get_notrade_stocks(start_date, end_date)
    else:
        print(f"使用外部传入的ST和涨跌停数据...")
    n_before = len(df_all_raw)

    # ST/涨跌停: DB日期 = available_date, 与因子日期同维度, 直接匹配剔除
    if not df_st.empty:
        _df_st = df_st[["valuation_date", "code"]].copy()
        _df_st["valuation_date"] = _df_st["valuation_date"].astype(str)
        df_all_raw = df_all_raw.merge(
            _df_st,
            on=["valuation_date", "code"],
            how="left", indicator="_st"
        )
        df_all_raw = df_all_raw[df_all_raw["_st"] == "left_only"].drop(columns=["_st"])

    if not df_notrade.empty:
        _df_notrade = df_notrade[["valuation_date", "code"]].copy()
        _df_notrade["valuation_date"] = _df_notrade["valuation_date"].astype(str)
        df_all_raw = df_all_raw.merge(
            _df_notrade,
            on=["valuation_date", "code"],
            how="left", indicator="_notrade"
        )
        df_all_raw = df_all_raw[df_all_raw["_notrade"] == "left_only"].drop(columns=["_notrade"])

    n_removed = n_before - len(df_all_raw)
    print(f"  剔除ST和涨跌停: {n_removed} 条")
    print(f"  剩余: {len(df_all_raw)} 条, 耗时 {time.time()-t0:.1f}s")

    # 全市场数据也剔除ST和涨跌停（复用上面已映射的 _df_st/_df_notrade）
    if df_full_market_raw is not None:
        n_fm_before = len(df_full_market_raw)
        if not df_st.empty:
            df_full_market_raw = df_full_market_raw.merge(
                _df_st,
                on=["valuation_date", "code"],
                how="left", indicator="_st"
            )
            df_full_market_raw = df_full_market_raw[df_full_market_raw["_st"] == "left_only"].drop(columns=["_st"])
        if not df_notrade.empty:
            df_full_market_raw = df_full_market_raw.merge(
                _df_notrade,
                on=["valuation_date", "code"],
                how="left", indicator="_notrade"
            )
            df_full_market_raw = df_full_market_raw[df_full_market_raw["_notrade"] == "left_only"].drop(columns=["_notrade"])
        print(f"  全市场剔除ST和涨跌停: {n_fm_before - len(df_full_market_raw)} 条, 剩余 {len(df_full_market_raw)} 条")

    factor_data_map = {name: grp for name, grp in df_all_raw.groupby("score_name")}

    # --- 步骤3: 逐因子z-score + 方向处理 (成分股) ---
    factor_zscores = {}  # {factor_name: DataFrame[valuation_date, code, z_score]}
    print(f"\n处理 {len(factor_list)} 个因子 (成分股内部z-score)...")

    for i, fcfg in enumerate(factor_list):
        t1 = time.time()
        name = fcfg["name"]
        direction = fcfg.get("direction", 1)
        dir_label = "正向" if direction == 1 else "反向"

        if name not in factor_data_map:
            print(f"  [{i+1}/{len(factor_list)}] {name} - 无数据，跳过")
            continue

        df = factor_data_map[name].copy()

        # 在成分股内部做截面 z-score
        df["z_score"] = df.groupby("valuation_date")["final_score"].transform(
            zscore_cross_section
        )
        df = df[["valuation_date", "code", "z_score"]].copy()

        # 直接应用方向
        df["z_score"] = df["z_score"] * direction
        t_proc = time.time() - t1

        factor_zscores[name] = df
        print(f"  [{i+1}/{len(factor_list)}] {name}  {dir_label}"
              f"  ({len(df)}条, 处理:{t_proc:.1f}s)")

    if not factor_zscores:
        print("错误: 没有加载到任何因子数据")
        empty = pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])
        return (empty, empty) if top_n_extra > 0 else empty

    # --- 步骤3b: 逐因子z-score + 方向处理 (全市场, top选股用) ---
    fm_factor_zscores = {}
    if df_full_market_raw is not None:
        fm_factor_data_map = {name: grp for name, grp in df_full_market_raw.groupby("score_name")}
        print(f"\n处理 {len(factor_list)} 个因子 (全市场z-score, 用于top选股)...")
        for i, fcfg in enumerate(factor_list):
            name = fcfg["name"]
            direction = fcfg.get("direction", 1)
            if name not in fm_factor_data_map:
                continue
            df = fm_factor_data_map[name].copy()
            df["z_score"] = df.groupby("valuation_date")["final_score"].transform(
                zscore_cross_section
            )
            df = df[["valuation_date", "code", "z_score"]].copy()
            df["z_score"] = df["z_score"] * direction
            fm_factor_zscores[name] = df
        print(f"  全市场因子处理完成: {len(fm_factor_zscores)} 个")

    # --- 步骤4: 合成 (等权 / IC加权 / 两层IC加权) ---
    t0 = time.time()
    factor_ic_map = {}  # 所有因子的IC（单层和两层共用）
    cluster_ic_map = {}  # 簇级IC（两层模式用）
    df_combined = None

    # two_level_ic 需要 cluster_map 和 df_stock，否则降级
    if weight_method == "two_level_ic":
        if cluster_map is None:
            print("警告: two_level_ic 需要 clusters 配置，降级为 ic_weight")
            weight_method = "ic_weight"
        elif df_stock is None:
            print("警告: two_level_ic 需要行情数据(df_stock)，降级为 equal")
            weight_method = "equal"

    if weight_method == "two_level_ic":
        # ========================================
        # 两层IC加权合成
        # Level 1: 簇内IC加权 → 簇因子
        # Level 2: 簇间IC加权 → 最终分数
        # ========================================
        print(f"\n两层IC加权合成 (窗口={ic_window}天)...")

        # --- Level 1: 簇内IC加权 ---
        print(f"\n  Level 1: 簇内IC加权...")
        cluster_scores = {}  # {cluster_name: DataFrame[valuation_date, code, z_score]}

        for cluster_name, cluster_factors in cluster_map.items():
            member_names = [f["name"] for f in cluster_factors]
            member_zscores = {n: factor_zscores[n] for n in member_names if n in factor_zscores}

            if not member_zscores:
                print(f"    簇 [{cluster_name}]: 无可用因子，跳过")
                continue

            if len(member_zscores) == 1:
                # 单因子簇: 直接透传
                fname = list(member_zscores.keys())[0]
                df_cs = list(member_zscores.values())[0].copy()
                # 计算IC备用（level 2 需要）
                ic_df = _calc_factor_daily_ic(df_cs, df_stock)
                if not ic_df.empty:
                    factor_ic_map[fname] = ic_df
                cluster_scores[cluster_name] = df_cs
                mean_ic = ic_df["rank_IC"].mean() if not ic_df.empty else 0
                print(f"    簇 [{cluster_name}]: 单因子 {fname}, 平均IC={mean_ic:.4f}")
            else:
                # 多因子簇: IC加权合成
                member_ics = {}
                for fname, df_z in member_zscores.items():
                    ic_df = _calc_factor_daily_ic(df_z, df_stock)
                    if not ic_df.empty:
                        member_ics[fname] = ic_df
                        factor_ic_map[fname] = ic_df

                print(f"    簇 [{cluster_name}]: {len(member_zscores)}个因子, {len(member_ics)}个有IC")
                df_cluster = _ic_weighted_combine(
                    member_zscores, member_ics, ic_window, label=cluster_name
                )
                # rename combined_score → z_score for consistency
                df_cluster.rename(columns={"combined_score": "z_score"}, inplace=True)
                cluster_scores[cluster_name] = df_cluster

        if not cluster_scores:
            print("错误: 所有簇合成失败")
            empty = pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])
            return (empty, empty) if top_n_extra > 0 else empty

        # --- Level 2: 簇间IC加权 ---
        print(f"\n  Level 2: 簇间IC加权 ({len(cluster_scores)}个簇)...")
        cluster_ic_map = {}
        for cname, df_cs in cluster_scores.items():
            ic_df = _calc_factor_daily_ic(df_cs, df_stock)
            if not ic_df.empty:
                cluster_ic_map[cname] = ic_df
                mean_ic = ic_df["rank_IC"].mean()
                print(f"    簇 [{cname}]: 平均IC={mean_ic:.4f}, 共{len(ic_df)}天")
            else:
                print(f"    簇 [{cname}]: IC计算无数据")

        df_final = _ic_weighted_combine(
            cluster_scores, cluster_ic_map, ic_window, label="Level2-簇间"
        )
        df_combined = df_final.rename(columns={"combined_score": "final_score"})
        method_label = f"两层IC加权(窗口{ic_window}天, {len(cluster_scores)}簇)"

    elif weight_method == "ic_weight" and df_stock is not None:
        # ---- 单层IC加权合成 ----
        print(f"\nIC加权合成 (窗口={ic_window}天)...")

        for fname, df_z in factor_zscores.items():
            ic_df = _calc_factor_daily_ic(df_z, df_stock)
            if not ic_df.empty:
                factor_ic_map[fname] = ic_df
                mean_ic = ic_df["rank_IC"].mean()
                print(f"  {fname}: 平均IC={mean_ic:.4f}, 共{len(ic_df)}天")
            else:
                print(f"  {fname}: IC计算无数据")

        if factor_ic_map:
            df_final = _ic_weighted_combine(
                factor_zscores, factor_ic_map, ic_window, label="单层IC"
            )
            df_combined = df_final.rename(columns={"combined_score": "final_score"})
            method_label = f"IC加权(窗口{ic_window}天)"
        else:
            print("警告: 所有因子IC计算失败，回退到等权合成")
            weight_method = "equal"

    # 兜底: 等权合成 (包括降级到 equal 的情况)
    if df_combined is None:
        print(f"\n等权合成中...")
        all_zscores = list(factor_zscores.values())
        df_all = pd.concat(all_zscores, ignore_index=True)
        df_combined = df_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
        df_combined.rename(columns={"z_score": "final_score"}, inplace=True)
        method_label = "等权"

    # --- 步骤4b: 时序平滑(MA) ---
    if smooth_window and smooth_window > 1:
        df_combined.sort_values(["code", "valuation_date"], inplace=True)
        df_combined["final_score"] = df_combined.groupby("code")["final_score"].transform(
            lambda x: x.rolling(smooth_window, min_periods=1).mean()
        )
        print(f"  成分股因子平滑: MA{smooth_window}")

    df_combined["score_name"] = output_name
    df_combined.sort_values(["valuation_date", "code"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    t_merge = time.time() - t0
    n_dates = df_combined["valuation_date"].nunique()
    n_stocks = int(df_combined.groupby("valuation_date")["code"].count().median())
    print(f"合成完成: {output_name} [{method_label}] (合并耗时: {t_merge:.1f}s)")
    print(f"  交易日数: {n_dates}")
    print(f"  每日股票数(中位数): {n_stocks}")
    print(f"  总记录数: {len(df_combined)}")

    # --- 步骤5: 全市场合成 (top选股用, 用成分股IC权重) ---
    if top_n_extra > 0 and fm_factor_zscores:
        print(f"\n全市场合成 (用成分股权重, 选非成分股top {top_n_extra})...")
        t_fm = time.time()

        if weight_method == "two_level_ic" and cluster_map and cluster_ic_map:
            # 两层: 用成分股的簇内IC权重 + 簇间IC权重 对全市场评分
            # Level 1: 簇内合成（全市场z-score + 成分股IC权重）
            fm_cluster_scores = {}
            for cluster_name, cluster_factors in cluster_map.items():
                member_names = [f["name"] for f in cluster_factors]
                member_fm_z = {n: fm_factor_zscores[n] for n in member_names if n in fm_factor_zscores}
                if not member_fm_z:
                    continue

                if len(member_fm_z) == 1:
                    df_cs = list(member_fm_z.values())[0].copy()
                    fm_cluster_scores[cluster_name] = df_cs
                else:
                    # 用成分股算出的因子IC做加权
                    member_ics = {n: factor_ic_map[n] for n in member_fm_z if n in factor_ic_map}
                    df_cluster = _ic_weighted_combine(
                        member_fm_z, member_ics, ic_window, label=""
                    )
                    df_cluster.rename(columns={"combined_score": "z_score"}, inplace=True)
                    fm_cluster_scores[cluster_name] = df_cluster

            if fm_cluster_scores:
                # Level 2: 用成分股的簇IC做簇间加权
                df_fm_final = _ic_weighted_combine(
                    fm_cluster_scores, cluster_ic_map, ic_window, label=""
                )
                df_fm_combined = df_fm_final.rename(columns={"combined_score": "final_score"})
            else:
                # 降级: 全市场等权
                fm_all_zscores = list(fm_factor_zscores.values())
                df_fm_all = pd.concat(fm_all_zscores, ignore_index=True)
                df_fm_combined = df_fm_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
                df_fm_combined.rename(columns={"z_score": "final_score"}, inplace=True)

        elif weight_method == "ic_weight" and factor_ic_map:
            df_fm_final = _ic_weighted_combine(
                fm_factor_zscores, factor_ic_map, ic_window, label=""
            )
            df_fm_combined = df_fm_final.rename(columns={"combined_score": "final_score"})
        else:
            fm_all_zscores = list(fm_factor_zscores.values())
            df_fm_all = pd.concat(fm_all_zscores, ignore_index=True)
            df_fm_combined = df_fm_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
            df_fm_combined.rename(columns={"z_score": "final_score"}, inplace=True)

        # 全市场也做时序平滑
        if smooth_window and smooth_window > 1:
            df_fm_combined.sort_values(["code", "valuation_date"], inplace=True)
            df_fm_combined["final_score"] = df_fm_combined.groupby("code")["final_score"].transform(
                lambda x: x.rolling(smooth_window, min_periods=1).mean()
            )
            print(f"  全市场因子平滑: MA{smooth_window}")

        df_fm_combined["score_name"] = output_name
        df_fm_combined.sort_values(["valuation_date", "code"], inplace=True)
        df_fm_combined.reset_index(drop=True, inplace=True)

        n_fm_dates = df_fm_combined["valuation_date"].nunique()
        n_fm_stocks = int(df_fm_combined.groupby("valuation_date")["code"].count().median())
        print(f"  全市场合成完成: {n_fm_dates}天, 每日{n_fm_stocks}只, 耗时{time.time()-t_fm:.1f}s")

        return (
            df_combined[["valuation_date", "code", "score_name", "final_score"]],
            df_fm_combined[["valuation_date", "code", "score_name", "final_score"]],
        )

    return df_combined[["valuation_date", "code", "score_name", "final_score"]]


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    import sys
    from core.data_prepare import load_config

    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        bt = load_config()["backtest"]
        start_date = bt["start_date"]
        end_date = bt["end_date"]

    print(f"因子合成: {start_date} ~ {end_date}")
    print("=" * 50)

    # 演示: 对 zz500 等权合成
    index_name = "zz500"
    df_combined = combine_factors_for_index(
        start_date, end_date,
        index_name=index_name,
        weight_method="equal",
    )

    if not df_combined.empty:
        print(f"\n前10条数据:")
        print(df_combined.head(10).to_string(index=False))
