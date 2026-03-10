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

from data_prepare import load_combine_config, get_factor_data_batch, get_index_component, get_st_stocks, get_notrade_stocks


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
