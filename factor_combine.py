"""
因子合成模块
============
从 config_combine.yaml 读取因子列表及方向配置，等权合成全市场因子。

核心流程:
1. 单次 SQL 批量加载所有因子原始数据
2. 逐因子截面 z-score 标准化
3. 应用方向: 正向因子 ×1, 反向因子 ×(-1)
   - 支持 direction_override: 同一因子在不同指数成分股上使用不同方向
     例如 RTN1 在沪深300成分股上为正向(动量), 其余股票为反向(反转)
4. 所有因子等权取均值 → 一个全市场合成因子

用法:
    python factor_combine.py                        # 默认日期
    python factor_combine.py 2023-06-30 2025-12-31  # 指定区间
"""

import os
import sys
import time
import pandas as pd
import numpy as np

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv

from data_prepare import load_combine_config, load_combine_by_index_config, get_factor_data_batch, get_index_component, get_st_stocks, get_notrade_stocks


# ============================================================
# 标准化与方向处理
# ============================================================

def zscore_cross_section(series: pd.Series) -> pd.Series:
    """截面 z-score 标准化: (x - mean) / std"""
    mean = series.mean()
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - mean) / std


def _build_index_member_set(start_date: str, end_date: str,
                            index_names: list) -> dict:
    """
    构建指数成分股集合，用于 direction_override

    Returns
    -------
    dict : {index_name: set((date_str, code), ...)}
    """
    member_sets = {}
    for idx in index_names:
        print(f"  加载指数成分: {idx}")
        df = get_index_component(start_date, end_date, idx)
        if not df.empty:
            member_sets[idx] = set(
                zip(df["valuation_date"].astype(str), df["code"])
            )
        else:
            member_sets[idx] = set()
    return member_sets


def _apply_direction(df_zscore: pd.DataFrame, factor_cfg: dict,
                     member_sets: dict) -> pd.DataFrame:
    """
    对单因子 z-score 应用方向

    无 direction_override 时: 全部乘以默认方向
    有 direction_override 时: 按股票当日是否属于覆盖指数，逐行决定方向
      - 属于覆盖指数的股票 → 使用覆盖方向
      - 其余股票 → 使用默认方向
    """
    default_dir = factor_cfg.get("direction", 1)
    overrides = factor_cfg.get("direction_override", {})

    if not overrides:
        df_zscore["z_score"] = df_zscore["z_score"] * default_dir
        return df_zscore

    # 先设默认方向，再按覆盖指数修改
    df_zscore["direction"] = default_dir
    date_str = df_zscore["valuation_date"].astype(str)

    for idx_name, idx_dir in overrides.items():
        if idx_name not in member_sets:
            continue
        idx_set = member_sets[idx_name]
        mask = pd.Series(
            [((d, c) in idx_set) for d, c in zip(date_str, df_zscore["code"])],
            index=df_zscore.index,
        )
        n_override = mask.sum()
        if n_override > 0:
            df_zscore.loc[mask, "direction"] = idx_dir
            dir_label = "正向" if idx_dir == 1 else "反向"
            print(f"    {idx_name} 成分股 {n_override} 条 -> {dir_label}")

    df_zscore["z_score"] = df_zscore["z_score"] * df_zscore["direction"]
    df_zscore.drop(columns=["direction"], inplace=True)
    return df_zscore


# ============================================================
# 因子合成主函数
# ============================================================

def combine_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    等权合成全市场因子

    Returns
    -------
    DataFrame[valuation_date, code, score_name, final_score]
        格式与 get_factor_data 一致，可直接传入回测
    """
    combine_cfg = load_combine_config()
    factor_list = combine_cfg["factors"]
    output_name = combine_cfg.get("output_name", "CombinedFactor")

    # --- 步骤1: 加载 direction_override 涉及的指数成分 ---
    override_indices = set()
    for fcfg in factor_list:
        for idx in fcfg.get("direction_override", {}):
            override_indices.add(idx)

    member_sets = {}
    if override_indices:
        t0 = time.time()
        print(f"加载 direction_override 涉及的指数成分...")
        member_sets = _build_index_member_set(start_date, end_date,
                                              list(override_indices))
        print(f"  指数成分加载耗时: {time.time()-t0:.1f}s")

    # --- 步骤2: 单次SQL批量加载所有因子 ---
    factor_names = [f["name"] for f in factor_list]
    print(f"\n批量加载 {len(factor_names)} 个因子 (单次SQL)...")
    t0 = time.time()
    df_all_raw = get_factor_data_batch(start_date, end_date, factor_names)
    t_db = time.time() - t0
    print(f"  DB查询完成: {len(df_all_raw)} 条, 耗时 {t_db:.1f}s")

    # --- 步骤2.5: 剔除ST和涨跌停股票 ---
    print(f"\n加载ST和涨跌停数据...")
    t0 = time.time()
    df_st = get_st_stocks(start_date, end_date)
    df_notrade = get_notrade_stocks(start_date, end_date)
    n_before = len(df_all_raw)

    if not df_st.empty:
        df_all_raw = df_all_raw.merge(
            df_st[["valuation_date", "code"]],
            on=["valuation_date", "code"],
            how="left", indicator="_st"
        )
        df_all_raw = df_all_raw[df_all_raw["_st"] == "left_only"].drop(columns=["_st"])

    if not df_notrade.empty:
        df_all_raw = df_all_raw.merge(
            df_notrade[["valuation_date", "code"]],
            on=["valuation_date", "code"],
            how="left", indicator="_notrade"
        )
        df_all_raw = df_all_raw[df_all_raw["_notrade"] == "left_only"].drop(columns=["_notrade"])

    n_removed = n_before - len(df_all_raw)
    print(f"  剔除ST和涨跌停: {n_removed} 条 (ST {len(df_st)} 条, 涨跌停 {len(df_notrade)} 条)")
    print(f"  剩余: {len(df_all_raw)} 条, 耗时 {time.time()-t0:.1f}s")

    factor_data_map = {name: grp for name, grp in df_all_raw.groupby("score_name")}

    # --- 步骤3: 逐因子 z-score + 方向处理 ---
    all_zscores = []
    print(f"\n处理 {len(factor_list)} 个因子...")

    for i, fcfg in enumerate(factor_list):
        t1 = time.time()
        name = fcfg["name"]
        default_dir = fcfg.get("direction", 1)
        has_override = bool(fcfg.get("direction_override"))
        dir_label = "正向" if default_dir == 1 else "反向"
        extra = " (有指数级方向覆盖)" if has_override else ""

        if name not in factor_data_map:
            print(f"  [{i+1}/{len(factor_list)}] {name} - 无数据，跳过")
            continue

        df = factor_data_map[name].copy()

        # 截面 z-score
        df["z_score"] = df.groupby("valuation_date")["final_score"].transform(
            zscore_cross_section
        )
        df = df[["valuation_date", "code", "z_score"]].copy()

        # 应用方向
        df = _apply_direction(df, fcfg, member_sets)
        t_proc = time.time() - t1

        all_zscores.append(df)
        print(f"  [{i+1}/{len(factor_list)}] {name}  默认{dir_label}{extra}"
              f"  ({len(df)}条, 处理:{t_proc:.1f}s)")

    if not all_zscores:
        print("错误: 没有加载到任何因子数据")
        return pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])

    # --- 步骤4: 等权合成 ---
    t0 = time.time()
    print(f"\n合成中...")
    df_all = pd.concat(all_zscores, ignore_index=True)
    df_combined = df_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
    df_combined.rename(columns={"z_score": "final_score"}, inplace=True)
    df_combined["score_name"] = output_name
    df_combined.sort_values(["valuation_date", "code"], inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    t_merge = time.time() - t0
    n_dates = df_combined["valuation_date"].nunique()
    n_stocks = int(df_combined.groupby("valuation_date")["code"].count().median())
    print(f"合成完成: {output_name} (合并耗时: {t_merge:.1f}s)")
    print(f"  交易日数: {n_dates}")
    print(f"  每日股票数(中位数): {n_stocks}")
    print(f"  总记录数: {len(df_combined)}")

    return df_combined[["valuation_date", "code", "score_name", "final_score"]]


# ============================================================
# 按指数合成因子
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


def combine_factors_for_index(start_date: str, end_date: str,
                               index_name: str,
                               index_cfg: dict = None,
                               df_index_comp: pd.DataFrame = None,
                               df_stock: pd.DataFrame = None,
                               weight_method: str = "equal",
                               ic_window: int = 20,
                               date_mode: str = "target_date",
                               df_calendar: pd.DataFrame = None) -> pd.DataFrame:
    """
    针对单个指数合成因子（支持等权 / IC加权）

    z-score 在指数成分股内部做截面标准化，而非全市场，
    确保因子排序仅反映成分股内部的相对差异。

    start_date/end_date 永远是 target_date（目标日期）。
    date_mode 只影响信号取法:
      - target_date:    DB date = target_date, 直接取
      - available_date: DB date = available_date(T-1), 取出后映射为 target_date(T)

    Parameters
    ----------
    start_date    : 开始日期 (target_date)
    end_date      : 结束日期 (target_date)
    index_name    : 指数简称, 如 "hs300"
    index_cfg     : 该指数的配置 dict，含 output_name 和 factors 列表
    df_index_comp : 指数成分股 DataFrame[valuation_date, code]
    df_stock      : 股票行情 (IC加权时需要), 含 valuation_date, code, close, pre_close
    weight_method : "equal" 等权 | "ic_weight" 滚动IC加权
    ic_window     : IC加权滚动窗口天数 (默认20)
    date_mode     : "target_date" | "available_date"
    df_calendar   : 交易日历 [valuation_date, next_workday], available_date模式必须传入

    Returns
    -------
    DataFrame[valuation_date, code, score_name, final_score]  (valuation_date = target_date)
    """
    if index_cfg is None:
        cfg = load_combine_by_index_config()
        index_cfg = cfg["indices"][index_name]

    factor_list = index_cfg["factors"]
    output_name = index_cfg.get("output_name", f"combine_{index_name}")

    # --- 步骤0: 获取指数成分股 ---
    if df_index_comp is None:
        print(f"\n加载指数成分: {index_name}")
        t0 = time.time()
        df_index_comp = get_index_component(start_date, end_date, index_name)
        print(f"  成分股加载完成: {len(df_index_comp)} 条, 耗时 {time.time()-t0:.1f}s")

    # 构建成分股 (date, code) 用于过滤
    comp_keys = df_index_comp[["valuation_date", "code"]].copy()
    comp_keys["valuation_date"] = comp_keys["valuation_date"].astype(str)
    n_comp_dates = comp_keys["valuation_date"].nunique()
    n_comp_stocks = int(comp_keys.groupby("valuation_date")["code"].count().median())
    print(f"  成分股: {n_comp_dates} 个交易日, 每日中位数 {n_comp_stocks} 只")

    # --- 步骤1: 批量加载所有因子 ---
    factor_names = [f["name"] for f in factor_list]
    print(f"\n批量加载 {len(factor_names)} 个因子 (单次SQL)...")
    t0 = time.time()

    if date_mode == "available_date":
        # available_date 模式: DB date = available_date(T-1)
        # 查询范围向前扩展，确保覆盖 start_date 对应的 T-1 信号
        import global_tools as gt
        query_start = gt.previous_workday_calculate(start_date)
        df_all_raw = get_factor_data_batch(query_start, end_date, factor_names)
    else:
        df_all_raw = get_factor_data_batch(start_date, end_date, factor_names)

    t_db = time.time() - t0
    print(f"  DB查询完成: {len(df_all_raw)} 条, 耗时 {t_db:.1f}s")

    # available_date → target_date 映射 (取完信号后立即转换，后续统一用 target_date)
    if date_mode == "available_date":
        df_all_raw["valuation_date"] = df_all_raw["valuation_date"].astype(str)
        if df_calendar is not None and not df_calendar.empty:
            cal_map = df_calendar.set_index(
                df_calendar["valuation_date"].astype(str)
            )["next_workday"].astype(str)
            df_all_raw["valuation_date"] = df_all_raw["valuation_date"].map(cal_map)
        else:
            # 无日历时用因子自身日期序列近似
            all_dates = sorted(df_all_raw["valuation_date"].unique())
            date_shift = {all_dates[i]: all_dates[i + 1]
                          for i in range(len(all_dates) - 1)}
            df_all_raw["valuation_date"] = df_all_raw["valuation_date"].map(date_shift)
        df_all_raw.dropna(subset=["valuation_date"], inplace=True)
        n_mapped = len(df_all_raw)
        print(f"  available_date → target_date 映射完成: {n_mapped} 条")

    # --- 步骤2: 过滤到指数成分股 + 剔除ST和涨跌停 ---
    print(f"\n过滤到 {index_name} 成分股...")
    t0 = time.time()
    df_all_raw["valuation_date"] = df_all_raw["valuation_date"].astype(str)
    n_before_filter = len(df_all_raw)
    df_all_raw = df_all_raw.merge(comp_keys, on=["valuation_date", "code"], how="inner")
    n_after_filter = len(df_all_raw)
    print(f"  成分股过滤: {n_before_filter} -> {n_after_filter} 条")

    print(f"加载ST和涨跌停数据...")
    df_st = get_st_stocks(start_date, end_date)
    df_notrade = get_notrade_stocks(start_date, end_date)
    n_before = len(df_all_raw)

    # ST/涨跌停的DB日期 = 事件发生日(T-1), 影响下一交易日(T)的交易
    # 将事件日期映射到 target_date(T) 后再剔除
    if df_calendar is not None and not df_calendar.empty:
        _cal_map = df_calendar.set_index(
            df_calendar["valuation_date"].astype(str)
        )["next_workday"].astype(str)
    else:
        _cal_map = None

    if not df_st.empty:
        df_st["valuation_date"] = df_st["valuation_date"].astype(str)
        if _cal_map is not None:
            df_st["valuation_date"] = df_st["valuation_date"].map(_cal_map)
            df_st.dropna(subset=["valuation_date"], inplace=True)
        df_all_raw = df_all_raw.merge(
            df_st[["valuation_date", "code"]],
            on=["valuation_date", "code"],
            how="left", indicator="_st"
        )
        df_all_raw = df_all_raw[df_all_raw["_st"] == "left_only"].drop(columns=["_st"])

    if not df_notrade.empty:
        df_notrade["valuation_date"] = df_notrade["valuation_date"].astype(str)
        if _cal_map is not None:
            df_notrade["valuation_date"] = df_notrade["valuation_date"].map(_cal_map)
            df_notrade.dropna(subset=["valuation_date"], inplace=True)
        df_all_raw = df_all_raw.merge(
            df_notrade[["valuation_date", "code"]],
            on=["valuation_date", "code"],
            how="left", indicator="_notrade"
        )
        df_all_raw = df_all_raw[df_all_raw["_notrade"] == "left_only"].drop(columns=["_notrade"])

    n_removed = n_before - len(df_all_raw)
    print(f"  剔除ST和涨跌停: {n_removed} 条")
    print(f"  剩余: {len(df_all_raw)} 条, 耗时 {time.time()-t0:.1f}s")

    factor_data_map = {name: grp for name, grp in df_all_raw.groupby("score_name")}

    # --- 步骤3: 逐因子在成分股内部做 z-score + 方向处理 ---
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
        return pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])

    # --- 步骤4: 合成 ---
    t0 = time.time()

    if weight_method == "ic_weight" and df_stock is not None:
        # ---- IC加权合成 ----
        print(f"\nIC加权合成 (窗口={ic_window}天)...")

        # 4a. 计算每个因子的逐日 Rank IC
        factor_ic_map = {}  # {factor_name: DataFrame[valuation_date, rank_IC]}
        for fname, df_z in factor_zscores.items():
            ic_df = _calc_factor_daily_ic(df_z, df_stock)
            if not ic_df.empty:
                factor_ic_map[fname] = ic_df
                mean_ic = ic_df["rank_IC"].mean()
                print(f"  {fname}: 平均IC={mean_ic:.4f}, 共{len(ic_df)}天")
            else:
                print(f"  {fname}: IC计算无数据")

        if not factor_ic_map:
            print("警告: 所有因子IC计算失败，回退到等权合成")
            weight_method = "equal"

    if weight_method == "ic_weight" and df_stock is not None and factor_ic_map:
        # 4b. 构建滚动IC权重表: 每个因子每天一个权重
        #     权重 = 过去 ic_window 天的平均 |IC|（取绝对值，方向已在z-score中处理）
        all_ic = []
        for fname, ic_df in factor_ic_map.items():
            tmp = ic_df[["valuation_date", "rank_IC"]].copy()
            tmp = tmp.sort_values("valuation_date")
            # shift(1): 日期t的权重只用t-1及之前的IC，避免前瞻偏差
            # IC(t-1) = corr(factor(t-1), return(t-1))，在t日前已知
            tmp["rolling_ic"] = tmp["rank_IC"].rolling(ic_window, min_periods=5).mean().abs().shift(1)
            tmp["factor_name"] = fname
            all_ic.append(tmp[["valuation_date", "factor_name", "rolling_ic"]].dropna())

        df_ic_weights = pd.concat(all_ic, ignore_index=True)

        # 逐日归一化权重 (各因子权重之和=1)
        df_ic_weights["weight"] = df_ic_weights.groupby("valuation_date")["rolling_ic"].transform(
            lambda x: x / x.sum() if x.sum() > 0 else 1.0 / len(x)
        )

        # 打印平均权重
        avg_weights = df_ic_weights.groupby("factor_name")["weight"].mean()
        print(f"\n  各因子平均IC权重:")
        for fname, w in avg_weights.sort_values(ascending=False).items():
            print(f"    {fname}: {w:.4f}")

        # 4c. 逐因子加权: z_score * weight
        weighted_parts = []
        for fname, df_z in factor_zscores.items():
            if fname not in factor_ic_map:
                continue
            w_df = df_ic_weights[df_ic_weights["factor_name"] == fname][["valuation_date", "weight"]]
            df_zw = df_z.merge(w_df, on="valuation_date", how="inner")
            df_zw["weighted_z"] = df_zw["z_score"] * df_zw["weight"]
            weighted_parts.append(df_zw[["valuation_date", "code", "weighted_z"]])

        df_all = pd.concat(weighted_parts, ignore_index=True)
        df_combined = df_all.groupby(["valuation_date", "code"])["weighted_z"].sum().reset_index()
        df_combined.rename(columns={"weighted_z": "final_score"}, inplace=True)
        method_label = f"IC加权(窗口{ic_window}天)"

    else:
        # ---- 等权合成 ----
        print(f"\n等权合成中...")
        all_zscores = list(factor_zscores.values())
        df_all = pd.concat(all_zscores, ignore_index=True)
        df_combined = df_all.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
        df_combined.rename(columns={"z_score": "final_score"}, inplace=True)
        method_label = "等权"

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

    return df_combined[["valuation_date", "code", "score_name", "final_score"]]


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    from data_prepare import load_config
    glv.init()

    if len(sys.argv) >= 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        bt = load_config()["backtest"]
        start_date = bt["start_date"]
        end_date = bt["end_date"]

    print(f"因子合成: {start_date} ~ {end_date}")
    print("=" * 50)

    df_combined = combine_factors(start_date, end_date)

    if not df_combined.empty:
        print(f"\n前10条数据:")
        print(df_combined.head(10).to_string(index=False))
