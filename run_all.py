"""
Signal Valuation 统一入口
========================
三大功能:
1. run_single_backtest()  — 单因子分层回测
2. run_combine_history()  — 历史区间因子合成 (可选回测)
3. run_combine_update()   — 单日因子合成更新

用法:
    # 命令行
    python run_all.py single                           # 全部单因子回测
    python run_all.py single --factors PE PB           # 指定因子
    python run_all.py combine                          # 全部指数合成+回测
    python run_all.py combine --index zz500 --no-backtest
    python run_all.py update                            # 自动决策日期更新
    python run_all.py update 2026-03-10                # 指定日期更新
    python run_all.py update 2026-03-10 --index zz500

    # 作为模块导入
    from run_all import run_combine_history, run_combine_update
    results = run_combine_history(backtest=False)
    df_today = run_combine_update("2026-03-10")
"""

import sys
import time
import argparse
from datetime import datetime, timedelta

import pandas as pd

from core.data_prepare import (
    load_config, load_signals_config, load_combine_by_index_config,
    get_factor_data, get_factor_data_batch, get_index_component,
    get_market_data, get_trading_calendar, get_st_stocks,
    get_notrade_stocks, next_workday, last_workday, _ensure_calendar,
)
from core.report import generate_report
from core.db_writer import save_combine_score
from core.factor_combine import combine_factors_for_index


# ============================================================
# 共享数据加载
# ============================================================

def _fmt_elapsed(seconds):
    """格式化耗时"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"


def _load_shared_data(start_date, end_date, index_list=None,
                      need_index_comp=True):
    """
    统一加载共享数据 (行情/日历/ST/涨跌停/指数成分)

    Parameters
    ----------
    start_date     : 回测开始日期
    end_date       : 回测结束日期
    index_list     : 需要加载成分股的指数列表, None则跳过
    need_index_comp: 是否加载指数成分股

    Returns
    -------
    dict : {
        "df_stock", "df_index_ret", "df_calendar",
        "df_st", "df_notrade", "index_comps"
    }
    """
    mkt_end = next_workday(end_date)
    result = {}

    # 行情数据
    print(f"正在获取行情数据: {start_date} ~ {mkt_end}")
    t0 = time.time()
    market_data = get_market_data(start_date, mkt_end)
    result["df_stock"] = market_data[0]
    result["df_index_ret"] = market_data[6]
    print(f"行情数据获取完成 [{_fmt_elapsed(time.time() - t0)}]")

    # 交易日历
    print("正在获取交易日历...")
    t0 = time.time()
    result["df_calendar"] = get_trading_calendar(start_date, mkt_end)
    print(f"交易日历: {len(result['df_calendar'])} 条 [{_fmt_elapsed(time.time() - t0)}]")

    # ST / 涨跌停
    print("正在获取ST/涨跌停数据...")
    t0 = time.time()
    result["df_st"] = get_st_stocks(start_date, mkt_end)
    result["df_notrade"] = get_notrade_stocks(start_date, mkt_end)
    print(f"ST: {len(result['df_st'])} 条, 涨跌停: {len(result['df_notrade'])} 条"
          f" [{_fmt_elapsed(time.time() - t0)}]")

    # 指数成分股
    result["index_comps"] = {}
    if need_index_comp and index_list:
        for idx_name in index_list:
            print(f"正在获取指数成分: {idx_name}")
            t0 = time.time()
            result["index_comps"][idx_name] = get_index_component(
                start_date, end_date, idx_name
            )
            print(f"  {idx_name} 成分股获取完成 [{_fmt_elapsed(time.time() - t0)}]")

    return result


def _calc_lookback_start(target_date, n_trading_days):
    """
    计算 target_date 前 n_trading_days 个交易日的起始日期

    先用自然日粗略估算范围，查日历后精确定位。
    """
    rough_start = (
        datetime.strptime(target_date, "%Y-%m-%d")
        - timedelta(days=int(n_trading_days * 1.6))
    ).strftime("%Y-%m-%d")

    df_cal = get_trading_calendar(rough_start, target_date)
    all_dates = sorted(df_cal["valuation_date"].astype(str).unique())

    if len(all_dates) >= n_trading_days:
        return all_dates[-n_trading_days]
    return all_dates[0] if all_dates else rough_start


def target_date_decision():
    """
    自动确定目标日期（target_date）

    根据当前时间和交易日情况，确定应该处理的目标日期：
    - 如果是交易日且当前时间 >= 16:00，返回下一个交易日
    - 如果是交易日且当前时间 < 16:00，返回当天
    - 如果不是交易日，返回下一个交易日
    """
    cal = _ensure_calendar()
    today = datetime.now().strftime("%Y-%m-%d")
    time_now = datetime.now().strftime("%H:%M")
    is_trading_day = today in cal

    if is_trading_day:
        if time_now >= "16:00":
            target = next_workday(today)
        else:
            target = today
    else:
        target = next_workday(today)

    print(f"  自动日期决策: 今天={today}, 时间={time_now}, "
          f"交易日={'是' if is_trading_day else '否'} → target_date={target}")
    return target


# ============================================================
# 功能1: 单因子分层回测
# ============================================================

def run_single_backtest(factors=None, index_list=None,
                        start_date=None, end_date=None,
                        n_groups=None, date_mode="available_date",
                        top_n=None):
    """
    单因子分层回测

    逐因子获取DB数据 → 在每个指数上做分层回测 → 生成PDF/Excel报告

    Parameters
    ----------
    factors    : 因子名称列表, None则从 config_signals.yaml 读取全部
    index_list : 指数列表, None则从 config.yaml 读取全部
    start_date : 开始日期, None则从 config.yaml 读取
    end_date   : 结束日期, None则从 config.yaml 读取
    n_groups   : 分层数, None则从 config.yaml 读取
    date_mode  : "available_date" (默认) 或 "target_date"
    top_n      : Top组合选股数(成分内), None则从 config.yaml 读取, 0=不跑
    """
    # 从 config_single_signal.yaml 读取默认参数
    signals_cfg = load_signals_config()
    bt = signals_cfg["backtest"]
    start_date = start_date or bt["start_date"]
    end_date = end_date or bt["end_date"]
    n_groups = n_groups or bt["n_groups"]
    index_list = index_list or bt["index_list"]
    if top_n is None:
        top_n = bt.get("top_n", 0)

    if factors is None:
        factors = signals_cfg["signals"]

    print(f"{'='*60}")
    print(f"  单因子分层回测")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  日期模式: {date_mode}")
    print(f"  因子数: {len(factors)}")
    print(f"  指数: {index_list}")
    if top_n > 0:
        print(f"  Top组合: 成分内top{top_n}等权")
    print(f"{'='*60}\n")

    # 加载共享数据
    shared = _load_shared_data(start_date, end_date, index_list)

    # 逐因子回测
    for i, factor_name in enumerate(factors):
        print(f"\n{'#'*60}")
        print(f"  [{i+1}/{len(factors)}] 回测: {factor_name} x {index_list}"
              f" (mode={date_mode})")
        print(f"{'#'*60}")

        t0 = time.time()
        try:
            df_factor = get_factor_data(start_date, end_date, factor_name)
            print(f"  因子数据: {len(df_factor)} 条")

            output_name = (f"{factor_name}_available"
                           if date_mode == "available_date"
                           else factor_name)

            generate_report(
                signal_name=output_name,
                start_date=start_date,
                end_date=end_date,
                n_groups=n_groups,
                df_factor=df_factor,
                index_data={idx: shared["index_comps"][idx]
                            for idx in index_list},
                df_stock=shared["df_stock"],
                df_index_ret=shared["df_index_ret"],
                df_calendar=shared["df_calendar"],
                df_st=shared["df_st"],
                df_notrade=shared["df_notrade"],
                date_mode=date_mode,
                top_n=top_n,
            )
        except Exception as e:
            print(f"  {factor_name} 回测失败: {e}")

        print(f"  耗时: {_fmt_elapsed(time.time() - t0)}")

    print(f"\n全部单因子回测完成!")


# ============================================================
# 功能2: 历史区间因子合成
# ============================================================


def run_combine_history(index_list=None, start_date=None, end_date=None,
                        backtest=True, mode="test", is_sql=False):
    """
    历史区间因子合成

    按 config_combine_by_index_{mode}.yaml 配置，逐指数做两层IC加权合成。
    backtest=True 时额外做分层回测+生成报告。

    Parameters
    ----------
    index_list  : 指数列表, None则从配置读取全部
    start_date  : 开始日期, None则从配置读取
    end_date    : 结束日期, None则从配置读取
    backtest    : True时做分层回测+生成PDF/Excel报告
    mode        : "test" 或 "prod"，决定读取哪个配置文件
    is_sql      : True时入库combine_score表 (需配合mode="prod")

    Returns
    -------
    dict : {index_name: DataFrame[valuation_date, code, score_name, final_score]}
    """
    # 加载配置
    combine_cfg = load_combine_by_index_config(mode=mode)
    bt = combine_cfg["backtest"]
    start_date = start_date or bt["start_date"]
    end_date = end_date or bt["end_date"]
    n_groups = bt["n_groups"]
    stock_number = bt["stock_number"]
    weight_method = combine_cfg.get("weight_method", "equal")
    ic_window = combine_cfg.get("ic_window", 60)
    smooth_window = combine_cfg.get("smooth_window", 5)
    date_mode = combine_cfg.get("date_mode", "target_date")

    # 确定要跑的指数
    all_indices = list(combine_cfg["indices"].keys())
    if index_list is None:
        run_indices = all_indices
    else:
        run_indices = [idx for idx in index_list if idx in combine_cfg["indices"]]
        invalid = [idx for idx in index_list if idx not in combine_cfg["indices"]]
        if invalid:
            print(f"警告: 以下指数不在配置中，已跳过: {invalid}")
        if not run_indices:
            print(f"错误: 没有有效的指数，可选: {all_indices}")
            return {}

    print(f"{'='*60}")
    print(f"  因子合成 (历史模式 | {mode})")
    print(f"  回测区间: {start_date} ~ {end_date}")
    print(f"  日期模式: {date_mode}")
    print(f"  合成方法: {weight_method}"
          + (f" (IC窗口={ic_window}天)" if "ic" in weight_method else ""))
    if smooth_window > 1:
        print(f"  因子平滑: MA{smooth_window}")
    print(f"  待处理指数: {run_indices}")
    print(f"  回测报告: {'是' if backtest else '否'}")
    print(f"{'='*60}\n")

    # 加载共享数据
    shared = _load_shared_data(start_date, end_date, run_indices)

    # 逐指数合成
    results = {}
    timings = {}

    for index_name in run_indices:
        index_cfg = combine_cfg["indices"][index_name]
        output_name = index_cfg.get("output_name", f"combine_{index_name}")

        # 打印因子/簇信息
        if "clusters" in index_cfg:
            n_clusters = len(index_cfg["clusters"])
            n_factors = sum(len(c["factors"])
                            for c in index_cfg["clusters"].values())
            cluster_info = f" | 簇数: {n_clusters}"
        else:
            n_factors = len(index_cfg["factors"])
            cluster_info = ""

        print(f"\n{'#'*60}")
        print(f"  指数: {index_name} | 合成因子: {output_name}"
              f" | 因子数: {n_factors}{cluster_info}")
        print(f"{'#'*60}")

        # 合成
        t0 = time.time()
        result = combine_factors_for_index(
            start_date, end_date,
            index_name=index_name,
            index_cfg=index_cfg,
            df_index_comp=shared["index_comps"][index_name],
            df_stock=shared["df_stock"],
            weight_method=weight_method,
            ic_window=ic_window,
            date_mode=date_mode,
            df_calendar=shared["df_calendar"],
            top_n_extra=stock_number if (backtest or is_sql) else 0,
            smooth_window=smooth_window,
            df_st=shared["df_st"],
            df_notrade=shared["df_notrade"],
        )
        timings[f"{index_name}.因子合成"] = time.time() - t0

        # 拆分返回值
        if isinstance(result, tuple):
            df_combined, df_fm_scores = result
        else:
            df_combined, df_fm_scores = result, None

        if df_combined.empty:
            print(f"  {output_name} 合成因子为空，跳过")
            continue

        results[index_name] = df_combined

        # 构建固定维度输出: 全部从全市场打分取 (统一z-score基准)
        # 成分股 + 非成分股top(stock_number - 成分股数) → 截面标准化
        df_fixed = None
        if df_fm_scores is not None and not df_fm_scores.empty:
            # 预计算每日成分股code集合
            comp_codes_by_date = (
                df_combined.groupby("valuation_date")["code"]
                .apply(set).to_dict()
            )

            dates = sorted(df_combined["valuation_date"].unique())
            parts = []
            for d in dates:
                df_fm_day = df_fm_scores[df_fm_scores["valuation_date"] == d]
                if df_fm_day.empty:
                    continue
                comp_codes = comp_codes_by_date.get(d, set())
                # 从全市场打分中提取: 成分股 + 非成分股top
                df_comp = df_fm_day[df_fm_day["code"].isin(comp_codes)]
                df_non_comp = df_fm_day[~df_fm_day["code"].isin(comp_codes)]
                n_extra = max(0, stock_number - len(df_comp))
                if n_extra > 0 and not df_non_comp.empty:
                    df_top_extra = df_non_comp.nlargest(n_extra, "final_score")
                    df_day = pd.concat([df_comp, df_top_extra], ignore_index=True)
                else:
                    df_day = df_comp.copy()
                # 截面标准化: 对这 stock_number 只做 z-score
                mean = df_day["final_score"].mean()
                std = df_day["final_score"].std()
                if std > 0:
                    df_day["final_score"] = (df_day["final_score"] - mean) / std
                parts.append(df_day)

            if parts:
                df_fixed = pd.concat(parts, ignore_index=True)
                df_fixed["score_name"] = output_name
                n_fixed_median = int(
                    df_fixed.groupby("valuation_date")["code"].count().median()
                )
                print(f"  固定维度输出: 每日约{n_fixed_median}只 (全市场基准+截面标准化)")

        # 回测 + 报告
        if backtest:
            # top组合: 从固定维度输出(df_fixed)中取打分最高的 (stock_number - 成分股数) 只等权
            df_top = None
            if df_fixed is not None and not df_fixed.empty:
                # 预计算每日成分股数, 避免重复扫描
                comp_count_per_day = (
                    df_combined.groupby("valuation_date")["code"].count()
                    .to_dict()
                )
                df_top = (
                    df_fixed
                    .groupby("valuation_date", group_keys=False)
                    .apply(lambda g: g.nlargest(
                        max(0, stock_number - comp_count_per_day.get(g.name, 0)),
                        "final_score"
                    ))
                    .reset_index(drop=True)
                )
                n_top_dates = df_top["valuation_date"].nunique()
                n_top_median = int(
                    df_top.groupby("valuation_date")["code"].count().median()
                )
                print(f"  top组合: {n_top_dates}天, 每日{n_top_median}只 (stock_number-成分股数 等权)")

            # 生成报告
            print(f"\n{'='*50}")
            print(f"  开始回测: {output_name} x {index_name}")
            print(f"{'='*50}")
            t0 = time.time()
            generate_report(
                signal_name=output_name,
                start_date=start_date,
                end_date=end_date,
                n_groups=n_groups,
                df_factor=df_combined,
                index_data={index_name: shared["index_comps"][index_name]},
                df_stock=shared["df_stock"],
                df_index_ret=shared["df_index_ret"],
                df_calendar=shared["df_calendar"],
                df_st=shared["df_st"],
                df_notrade=shared["df_notrade"],
                date_mode=date_mode,
                df_top=df_top,
                top_n_extra=stock_number,
                mode=mode,
            )
            timings[f"{index_name}.回测+报告"] = time.time() - t0

        # 入库: 用固定维度数据
        if is_sql:
            t0 = time.time()
            table_label = "combine_score" if mode == "prod" else "combine_score_test"
            df_to_save = df_fixed if df_fixed is not None else df_combined
            df_to_save = df_to_save.copy()
            df_to_save["score_name"] = output_name
            n_save = len(df_to_save)
            print(f"\n  入库: {output_name} → {table_label} 表 ({n_save}行)")
            save_combine_score(df_to_save, mode=mode)
            timings[f"{index_name}.入库"] = time.time() - t0

    # 耗时汇总
    if timings:
        total = sum(timings.values())
        print(f"\n{'='*60}")
        print(f"  耗时统计")
        print(f"{'='*60}")
        for step, elapsed in timings.items():
            pct = elapsed / total * 100 if total > 0 else 0
            print(f"  {step:<45} {_fmt_elapsed(elapsed):>8}  ({pct:.1f}%)")
        print(f"  {'─'*55}")
        print(f"  {'总计':<45} {_fmt_elapsed(total):>8}")

    return results


# ============================================================
# 功能3: 单日因子合成更新
# ============================================================

def run_combine_update(target_date=None, index_list=None, mode="prod", is_sql=False):
    """
    单日因子合成更新

    根据 config_combine_by_index_{mode}.yaml 配置，计算指定日期的合成因子。
    内部会自动回溯足够的历史数据以计算IC权重和MA平滑。

    Parameters
    ----------
    target_date : 目标持仓日 (T日), 格式 "YYYY-MM-DD", None则自动决策
    index_list  : 指数列表, None则从配置读取全部
    mode        : "test" 或 "prod"，决定读取哪个配置文件 (默认prod)
    is_sql      : True时入库combine_score表 (需配合mode="prod")

    Returns
    -------
    dict : {index_name: DataFrame[valuation_date, code, score_name, final_score]}
           只包含 target_date 当天的数据
    """
    if target_date is None:
        target_date = target_date_decision()
    combine_cfg = load_combine_by_index_config(mode=mode)
    bt = combine_cfg["backtest"]
    stock_number = bt["stock_number"]
    weight_method = combine_cfg.get("weight_method", "equal")
    ic_window = combine_cfg.get("ic_window", 60)
    smooth_window = combine_cfg.get("smooth_window", 5)
    date_mode = combine_cfg.get("date_mode", "available_date")

    # 确定要跑的指数
    all_indices = list(combine_cfg["indices"].keys())
    if index_list is None:
        run_indices = all_indices
    else:
        run_indices = [idx for idx in index_list if idx in combine_cfg["indices"]]
        if not run_indices:
            print(f"错误: 没有有效的指数，可选: {all_indices}")
            return {}

    # 计算回溯起始日 (IC窗口 + MA窗口 + 缓冲)
    total_lookback = ic_window + smooth_window + 10
    print(f"计算回溯范围: IC窗口={ic_window}, MA窗口={smooth_window},"
          f" 总回溯≈{total_lookback}个交易日")
    lookback_start = _calc_lookback_start(target_date, total_lookback)

    print(f"{'='*60}")
    print(f"  因子合成 (更新模式)")
    print(f"  目标日期: {target_date}")
    print(f"  回溯起始: {lookback_start}")
    print(f"  合成方法: {weight_method}"
          + (f" (IC窗口={ic_window}天)" if "ic" in weight_method else ""))
    if smooth_window > 1:
        print(f"  因子平滑: MA{smooth_window}")
    print(f"  待处理指数: {run_indices}")
    print(f"{'='*60}\n")

    # ============================================================
    # 第一步: 加载所有数据
    # ============================================================
    shared = _load_shared_data(lookback_start, target_date, run_indices)

    # 收集所有指数需要的因子名
    all_factor_names = set()
    for index_name in run_indices:
        index_cfg = combine_cfg["indices"][index_name]
        if "clusters" in index_cfg:
            for cluster_cfg in index_cfg["clusters"].values():
                for f in cluster_cfg["factors"]:
                    all_factor_names.add(f["name"])
        else:
            for f in index_cfg["factors"]:
                all_factor_names.add(f["name"])
    all_factor_names = sorted(all_factor_names)

    # 预加载因子数据
    print(f"\n预加载因子数据: {len(all_factor_names)} 个因子...")
    t0 = time.time()
    if date_mode == "available_date":
        query_start = last_workday(lookback_start)
        df_all_factors = get_factor_data_batch(query_start, target_date, all_factor_names)
    else:
        df_all_factors = get_factor_data_batch(lookback_start, target_date, all_factor_names)
    print(f"  因子数据加载完成: {len(df_all_factors)} 条, "
          f"耗时 {_fmt_elapsed(time.time() - t0)}")

    # ============================================================
    # 第二步: 数据完整性检查
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  数据完整性检查")
    print(f"{'='*60}")
    errors = []

    # 2a. 检查因子数据的 available_date 是否覆盖 target_date
    factor_dates = sorted(df_all_factors["valuation_date"].astype(str).unique())
    if not factor_dates:
        errors.append("因子数据为空，没有任何日期的数据")
    else:
        # available_date 模式下, target_date 的前一个交易日应该有数据
        expected_date = last_workday(target_date)
        if expected_date not in factor_dates:
            errors.append(
                f"因子数据缺少 target_date 对应的 available_date: "
                f"期望 {expected_date}, 实际最新日期 {factor_dates[-1]}"
            )
        else:
            print(f"  [OK] 因子 available_date 覆盖正常 (最新: {factor_dates[-1]})")

    # 2b. 检查每个因子在 target_date 对应的 available_date 是否有数据
    if factor_dates:
        expected_date = last_workday(target_date)
        df_latest = df_all_factors[
            df_all_factors["valuation_date"].astype(str) == expected_date
        ]
        factors_with_data = set(df_latest["score_name"].unique())
        missing_factors = set(all_factor_names) - factors_with_data
        if missing_factors:
            errors.append(
                f"以下因子在 {expected_date} 缺少数据: {sorted(missing_factors)}"
            )
        else:
            print(f"  [OK] 全部 {len(all_factor_names)} 个因子在 {expected_date} 有数据")

    # 2c. 检查 IC 回溯数据是否充足
    n_factor_dates = len(factor_dates)
    min_required = ic_window + smooth_window
    if n_factor_dates < min_required:
        errors.append(
            f"因子数据交易日数不足: 需要至少 {min_required} 天 "
            f"(IC窗口{ic_window} + MA窗口{smooth_window}), "
            f"实际只有 {n_factor_dates} 天"
        )
    else:
        print(f"  [OK] IC回溯数据充足: {n_factor_dates} 天 "
              f"(需要 >= {min_required} 天)")

    # 2d. 检查行情数据是否覆盖
    stock_dates = sorted(
        shared["df_stock"]["valuation_date"].astype(str).unique()
    )
    if not stock_dates:
        errors.append("行情数据为空")
    else:
        if stock_dates[-1] < last_workday(target_date):
            errors.append(
                f"行情数据最新日期 {stock_dates[-1]} "
                f"早于 target 对应的 available_date {last_workday(target_date)}"
            )
        else:
            print(f"  [OK] 行情数据覆盖正常 (最新: {stock_dates[-1]})")

    # 2e. 检查指数成分股是否覆盖
    for index_name in run_indices:
        comp_dates = sorted(
            shared["index_comps"][index_name]["valuation_date"].astype(str).unique()
        )
        if not comp_dates:
            errors.append(f"{index_name} 成分股数据为空")
        elif comp_dates[-1] < last_workday(target_date):
            errors.append(
                f"{index_name} 成分股最新日期 {comp_dates[-1]} "
                f"早于 {last_workday(target_date)}"
            )
        else:
            print(f"  [OK] {index_name} 成分股覆盖正常 (最新: {comp_dates[-1]})")

    # 检查结果
    if errors:
        print(f"\n{'!'*60}")
        print(f"  数据完整性检查失败！发现 {len(errors)} 个问题:")
        for i, err in enumerate(errors, 1):
            print(f"  {i}. {err}")
        print(f"{'!'*60}")
        raise RuntimeError(f"数据完整性检查失败: {len(errors)} 个问题, 详见上方日志")

    print(f"\n  数据检查全部通过，开始合成...\n")

    # ============================================================
    # 第三步: 逐指数合成 (使用预加载数据)
    # ============================================================
    results = {}

    for index_name in run_indices:
        index_cfg = combine_cfg["indices"][index_name]
        output_name = index_cfg.get("output_name", f"combine_{index_name}")

        print(f"\n合成: {output_name} (回溯 {lookback_start} ~ {target_date})")
        t0 = time.time()

        result = combine_factors_for_index(
            lookback_start, target_date,
            index_name=index_name,
            index_cfg=index_cfg,
            df_index_comp=shared["index_comps"][index_name],
            df_stock=shared["df_stock"],
            weight_method=weight_method,
            ic_window=ic_window,
            date_mode=date_mode,
            df_calendar=shared["df_calendar"],
            top_n_extra=stock_number,
            smooth_window=smooth_window,
            df_st=shared["df_st"],
            df_notrade=shared["df_notrade"],
            df_all_factors=df_all_factors,
        )
        elapsed = time.time() - t0

        # 拆分返回值
        if isinstance(result, tuple):
            df_combined, df_fm_scores = result
        else:
            df_combined, df_fm_scores = result, None

        if df_combined.empty:
            print(f"  {output_name} 合成为空，跳过")
            continue

        # 只保留最新日期
        latest_date = df_combined["valuation_date"].max()

        # 从全市场打分取固定维度: 成分股 + 非成分股top + 截面标准化
        comp_codes = set(
            df_combined[df_combined["valuation_date"] == latest_date]["code"]
        )
        n_comp = len(comp_codes)

        df_output = pd.DataFrame()
        if df_fm_scores is not None and not df_fm_scores.empty:
            df_fm_latest = df_fm_scores[
                df_fm_scores["valuation_date"] == latest_date
            ]
            if not df_fm_latest.empty:
                df_comp = df_fm_latest[df_fm_latest["code"].isin(comp_codes)]
                df_non_comp = df_fm_latest[~df_fm_latest["code"].isin(comp_codes)]
                n_extra = max(0, stock_number - len(df_comp))
                if n_extra > 0 and not df_non_comp.empty:
                    df_top_extra = df_non_comp.nlargest(n_extra, "final_score")
                    df_output = pd.concat([df_comp, df_top_extra], ignore_index=True)
                else:
                    df_output = df_comp.copy()
                # 截面标准化
                mean = df_output["final_score"].mean()
                std = df_output["final_score"].std()
                if std > 0:
                    df_output["final_score"] = (df_output["final_score"] - mean) / std

        df_output["score_name"] = output_name
        df_output = df_output[["valuation_date", "code", "score_name", "final_score"]].reset_index(drop=True)

        n_total = len(df_output)
        n_extra = n_total - n_comp
        print(f"  {output_name}: 日期={latest_date}, 成分股={n_comp} + 非成分股={n_extra} = {n_total}"
              f" [{_fmt_elapsed(elapsed)}]")

        results[index_name] = df_output

        # 入库
        if is_sql:
            table_label = "combine_score" if mode == "prod" else "combine_score_test"
            print(f"  入库: {output_name} → {table_label} 表 ({n_total}只)")
            save_combine_score(df_output, mode=mode)

    # 汇总
    print(f"\n{'='*60}")
    print(f"  更新完成")
    for idx_name, df in results.items():
        print(f"  {idx_name}: {len(df)} 只股票")
    print(f"{'='*60}")

    return results


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Signal Valuation 统一入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_all.py single                              # 全部单因子回测
  python run_all.py single --factors PE PB ROA          # 指定因子
  python run_all.py single --index zz500 zz1000         # 指定指数

  python run_all.py combine                             # 全部指数合成+回测
  python run_all.py combine --index hs300               # 只跑沪深300
  python run_all.py combine --no-backtest               # 只合成不回测

  python run_all.py update 2026-03-10                   # 单日更新全部指数
  python run_all.py update 2026-03-10 --index zz500     # 单日更新指定指数
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="功能选择")

    # --- single: 单因子回测 ---
    p_single = subparsers.add_parser("single", help="单因子分层回测")
    p_single.add_argument("--factors", nargs="+", default=None,
                          help="因子名称列表 (默认从config_signals.yaml读取)")
    p_single.add_argument("--index", nargs="+", default=None,
                          help="指数列表 (默认从config.yaml读取)")
    p_single.add_argument("--start", default=None, help="开始日期")
    p_single.add_argument("--end", default=None, help="结束日期")
    p_single.add_argument("--date-mode", default="available_date",
                          choices=["available_date", "target_date"],
                          help="日期模式 (默认available_date)")
    p_single.add_argument("--top-n", type=int, default=None,
                          help="Top组合选股数-成分内 (默认从config.yaml读取, 0=不跑)")

    # --- combine: 历史因子合成 ---
    p_combine = subparsers.add_parser("combine", help="历史因子合成(+可选回测)")
    p_combine.add_argument("--index", nargs="+", default=None,
                           help="指数列表 (默认从combine配置读取)")
    p_combine.add_argument("--start", default=None, help="开始日期")
    p_combine.add_argument("--end", default=None, help="结束日期")
    p_combine.add_argument("--no-backtest", action="store_true",
                           help="只合成不做回测")
    p_combine.add_argument("--mode", default="test", choices=["test", "prod"],
                           help="配置模式: test或prod (默认test)")
    p_combine.add_argument("--sql", action="store_true",
                           help="入库combine_score表 (需配合--mode prod)")

    # --- update: 单日更新 ---
    p_update = subparsers.add_parser("update", help="单日因子合成更新")
    p_update.add_argument("target_date", nargs="?", default=None,
                          help="目标日期 YYYY-MM-DD (不指定则自动决策)")
    p_update.add_argument("--index", nargs="+", default=None,
                          help="指数列表 (默认从combine配置读取)")
    p_update.add_argument("--mode", default="prod", choices=["test", "prod"],
                          help="配置模式: test或prod (默认prod)")
    p_update.add_argument("--sql", action="store_true",
                          help="入库combine_score表 (需配合--mode prod)")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    total_start = time.time()

    if args.command == "single":
        run_single_backtest(
            factors=args.factors,
            index_list=args.index,
            start_date=args.start,
            end_date=args.end,
            date_mode=args.date_mode,
            top_n=args.top_n,
        )

    elif args.command == "combine":
        results = run_combine_history(
            index_list=args.index,
            start_date=args.start,
            end_date=args.end,
            backtest=not args.no_backtest,
            mode=args.mode,
            is_sql=args.sql,
        )
        # 打印返回结果摘要
        print(f"\n返回结果:")
        for idx_name, df in results.items():
            n_dates = df["valuation_date"].nunique()
            n_stocks = int(df.groupby("valuation_date")["code"].count().median())
            print(f"  {idx_name}: {n_dates}天, 每日{n_stocks}只")

    elif args.command == "update":
        results = run_combine_update(
            target_date=args.target_date,
            index_list=args.index,
            mode=args.mode,
            is_sql=args.sql,
        )

    print(f"\n总耗时: {_fmt_elapsed(time.time() - total_start)}")
