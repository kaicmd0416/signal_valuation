"""
因子分组层次合成模块
====================
自动根据因子截面相关性进行层次聚类分组，然后层次合成：
  组内等权 → 二级因子 → 组间等权 → 一级因子

核心流程:
1. 批量加载因子数据，逐因子截面 z-score + 方向处理
2. 采样多个交易日计算因子间截面相关性矩阵
3. 层次聚类(ward linkage) + silhouette score 自动选择最优分组数
4. 组内等权合成 → 二级因子
5. 二级因子截面 z-score → 等权合成 → 一级因子

优势:
- 高相关因子自动归组，避免冗余因子占过多权重
- 每个组在最终合成中权重相等，真正实现多元化
- 分组数自动判别，无需手动指定

用法:
    python factor_group_combine.py
    python factor_group_combine.py 2023-06-30 2025-12-31
"""

import os
import sys
import time
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv

from data_prepare import (
    load_combine_config, get_factor_data_batch, get_index_component,
    get_st_stocks, get_notrade_stocks,
)
from factor_combine import zscore_cross_section, _build_index_member_set, _apply_direction


# ============================================================
# 相关性计算
# ============================================================

def calc_factor_corr_matrix(factor_zscores: dict, n_sample_dates: int = 80) -> pd.DataFrame:
    """
    采样多个交易日，计算因子间方向调整后的平均截面相关性矩阵

    Parameters
    ----------
    factor_zscores : dict {factor_name: DataFrame[valuation_date, code, z_score]}
    n_sample_dates : 采样交易日数量

    Returns
    -------
    DataFrame : 因子间平均相关性矩阵 (index=columns=factor_names)
    """
    factor_names = list(factor_zscores.keys())

    # 合并为宽表
    frames = []
    for name in factor_names:
        df = factor_zscores[name][["valuation_date", "code", "z_score"]].copy()
        df.rename(columns={"z_score": name}, inplace=True)
        frames.append(df)

    from functools import reduce
    df_wide = reduce(
        lambda l, r: l.merge(r, on=["valuation_date", "code"], how="outer"),
        frames,
    )

    # 均匀采样交易日
    dates = sorted(df_wide["valuation_date"].unique())
    step = max(1, len(dates) // n_sample_dates)
    sample_dates = dates[::step][:n_sample_dates]

    corr_sum = None
    n = 0
    for d in sample_dates:
        sub = df_wide[df_wide["valuation_date"] == d][factor_names].dropna()
        if len(sub) < 50:
            continue
        c = sub.corr()
        if corr_sum is None:
            corr_sum = c.copy()
        else:
            corr_sum = corr_sum + c
        n += 1

    if n == 0:
        return pd.DataFrame(np.eye(len(factor_names)),
                            index=factor_names, columns=factor_names)

    avg_corr = corr_sum / n
    return avg_corr


# ============================================================
# 自动分组
# ============================================================

def auto_group_factors(corr_matrix: pd.DataFrame,
                       min_groups: int = 2,
                       max_groups: int = None,
                       forced_k: int = None) -> dict:
    """
    基于相关性矩阵，层次聚类 + silhouette score 自动分组

    Parameters
    ----------
    corr_matrix  : 因子间相关性矩阵
    min_groups   : 最少分组数
    max_groups   : 最多分组数（默认 N-1）
    forced_k     : 强制指定分组数（不为None时跳过自动选择）

    Returns
    -------
    dict : {
        "labels": {factor_name: group_id},
        "n_groups": int,
        "silhouette_scores": {k: score},
        "groups": {group_id: [factor_names]},
    }
    """
    factor_names = corr_matrix.index.tolist()
    n = len(factor_names)

    if n <= 2:
        # 2个因子无需聚类
        labels = {name: i for i, name in enumerate(factor_names)}
        groups = {i: [name] for i, name in enumerate(factor_names)}
        return {
            "labels": labels,
            "n_groups": n,
            "silhouette_scores": {},
            "groups": groups,
        }

    if max_groups is None:
        max_groups = min(n - 1, n)
    max_groups = min(max_groups, n - 1)
    min_groups = max(min_groups, 2)

    # 相关性 → 距离矩阵: distance = 1 - |corr|
    dist_matrix = 1 - corr_matrix.abs().values
    np.fill_diagonal(dist_matrix, 0)
    # 保证对称且非负
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    dist_matrix = np.clip(dist_matrix, 0, None)

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method="ward")

    # 尝试不同分组数，用 silhouette score 选最优
    scores = {}
    for k in range(min_groups, max_groups + 1):
        cluster_labels = fcluster(Z, t=k, criterion="maxclust")
        if len(set(cluster_labels)) < 2:
            continue
        sc = silhouette_score(dist_matrix, cluster_labels, metric="precomputed")
        scores[k] = sc

    if not scores:
        # fallback: 每个因子一组
        labels = {name: i for i, name in enumerate(factor_names)}
        groups = {i: [name] for i, name in enumerate(factor_names)}
        return {
            "labels": labels,
            "n_groups": n,
            "silhouette_scores": scores,
            "groups": groups,
        }

    if forced_k is not None:
        best_k = forced_k
    else:
        best_k = max(scores, key=scores.get)
    best_labels = fcluster(Z, t=best_k, criterion="maxclust")

    labels = {name: int(lbl) for name, lbl in zip(factor_names, best_labels)}
    groups = {}
    for name, lbl in labels.items():
        groups.setdefault(lbl, []).append(name)

    return {
        "labels": labels,
        "n_groups": best_k,
        "silhouette_scores": scores,
        "groups": groups,
    }


# ============================================================
# 层次合成主函数
# ============================================================

def group_combine_factors(start_date: str, end_date: str) -> pd.DataFrame:
    """
    层次合成: 自动分组 → 组内等权(二级因子) → 组间等权(一级因子)

    Returns
    -------
    DataFrame[valuation_date, code, score_name, final_score]
    """
    combine_cfg = load_combine_config()
    factor_list = combine_cfg["factors"]
    output_name = combine_cfg.get("output_name", "CombinedFactor")

    # === 步骤1: 加载 direction_override 涉及的指数成分 ===
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

    # === 步骤2: 批量加载因子数据 ===
    factor_names = [f["name"] for f in factor_list]
    factor_cfg_map = {f["name"]: f for f in factor_list}
    print(f"\n批量加载 {len(factor_names)} 个因子 (单次SQL)...")
    t0 = time.time()
    df_all_raw = get_factor_data_batch(start_date, end_date, factor_names)
    print(f"  DB查询完成: {len(df_all_raw)} 条, 耗时 {time.time()-t0:.1f}s")

    # === 步骤2.5: 剔除ST和涨跌停 ===
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
    print(f"  剔除ST和涨跌停: {n_removed} 条")
    print(f"  剩余: {len(df_all_raw)} 条, 耗时 {time.time()-t0:.1f}s")

    factor_data_map = {name: grp for name, grp in df_all_raw.groupby("score_name")}

    # === 步骤3: 逐因子 z-score + 方向处理 ===
    factor_zscores = {}  # {name: DataFrame[valuation_date, code, z_score]}
    print(f"\n处理 {len(factor_list)} 个因子 (z-score + 方向)...")

    for i, fcfg in enumerate(factor_list):
        name = fcfg["name"]
        default_dir = fcfg.get("direction", 1)
        has_override = bool(fcfg.get("direction_override"))
        dir_label = "正向" if default_dir == 1 else "反向"
        extra = " (有指数级方向覆盖)" if has_override else ""

        if name not in factor_data_map:
            print(f"  [{i+1}/{len(factor_list)}] {name} - 无数据，跳过")
            continue

        df = factor_data_map[name].copy()
        df["z_score"] = df.groupby("valuation_date")["final_score"].transform(
            zscore_cross_section
        )
        df = df[["valuation_date", "code", "z_score"]].copy()
        df = _apply_direction(df, fcfg, member_sets)

        factor_zscores[name] = df
        print(f"  [{i+1}/{len(factor_list)}] {name}  默认{dir_label}{extra}  ({len(df)}条)")

    if not factor_zscores:
        print("错误: 没有加载到任何因子数据")
        return pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])

    # === 步骤4: 计算因子间相关性矩阵 ===
    print(f"\n计算因子间截面相关性矩阵 (采样80个交易日)...")
    t0 = time.time()
    corr_matrix = calc_factor_corr_matrix(factor_zscores, n_sample_dates=80)
    print(f"  耗时: {time.time()-t0:.1f}s")
    print(f"\n相关性矩阵:")
    print(corr_matrix.round(3).to_string())

    # === 步骤5: 自动分组 ===
    forced_k = combine_cfg.get("n_groups", None)
    if forced_k:
        print(f"\n聚类分组 (强制 K={forced_k})...")
    else:
        print(f"\n自动聚类分组...")
    result = auto_group_factors(corr_matrix, forced_k=forced_k)
    n_groups = result["n_groups"]
    groups = result["groups"]
    scores = result["silhouette_scores"]

    print(f"  各分组数的 silhouette score:")
    for k, sc in sorted(scores.items()):
        marker = " ← 选中" if k == n_groups else ""
        print(f"    K={k}: {sc:.4f}{marker}")

    print(f"\n  最优分组数: {n_groups}")
    for gid, members in sorted(groups.items()):
        print(f"  组{gid}: {members}")

    # === 步骤6: 组内等权合成 → 二级因子 ===
    print(f"\n组内等权合成 → 二级因子...")
    t0 = time.time()
    level2_zscores = []

    for gid, members in sorted(groups.items()):
        print(f"  组{gid} ({len(members)}个因子): {members}")

        # 合并组内所有因子
        group_frames = []
        for name in members:
            df = factor_zscores[name].copy()
            df.rename(columns={"z_score": name}, inplace=True)
            group_frames.append(df)

        from functools import reduce
        df_group = reduce(
            lambda l, r: l.merge(r, on=["valuation_date", "code"], how="outer"),
            group_frames,
        )

        # 组内等权取均值（不再做z-score，保留原始scale差异以体现分组权重）
        df_group["level2_score"] = df_group[members].mean(axis=1)

        df_l2 = df_group[["valuation_date", "code", "level2_score"]].copy()
        df_l2.rename(columns={"level2_score": "z_score"}, inplace=True)
        df_l2["group_id"] = gid
        level2_zscores.append(df_l2)

        n_stocks = int(df_l2.groupby("valuation_date")["code"].count().median())
        print(f"    → 二级因子: 每日 {n_stocks} 只股票")

    print(f"  组内合成耗时: {time.time()-t0:.1f}s")

    # === 步骤7: 组间等权合成 → 一级因子 ===
    print(f"\n组间等权合成 → 一级因子 ({n_groups}组等权)...")
    t0 = time.time()
    df_all_l2 = pd.concat(level2_zscores, ignore_index=True)
    df_combined = df_all_l2.groupby(["valuation_date", "code"])["z_score"].mean().reset_index()
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

    print(f"因子分组层次合成: {start_date} ~ {end_date}")
    print("=" * 60)

    df_combined = group_combine_factors(start_date, end_date)

    if not df_combined.empty:
        print(f"\n前10条数据:")
        print(df_combined.head(10).to_string(index=False))
