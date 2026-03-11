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
    python run_all.py update 2026-03-10                # 单日更新
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
    get_factor_data, get_index_component, get_market_data,
    get_trading_calendar, get_st_stocks, get_notrade_stocks,
    next_workday,
)
from core.report import generate_report
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
    # 补全默认参数
    cfg = load_config()
    bt = cfg["backtest"]
    start_date = start_date or bt["start_date"]
    end_date = end_date or bt["end_date"]
    n_groups = n_groups or bt["n_groups"]
    index_list = index_list or bt["index_list"]
    if top_n is None:
        top_n = bt.get("top_n", 0)

    if factors is None:
        signals_cfg = load_signals_config()
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

TOP_N_DEFAULT = 200


def run_combine_history(index_list=None, start_date=None, end_date=None,
                        backtest=True, n_groups=None,
                        top_n_extra=TOP_N_DEFAULT):
    """
    历史区间因子合成

    按 config_combine_by_index.yaml 配置，逐指数做两层IC加权合成。
    backtest=True 时额外做分层回测+生成报告。

    Parameters
    ----------
    index_list  : 指数列表, None则从 combine 配置读取全部
    start_date  : 开始日期, None则从 config.yaml 读取
    end_date    : 结束日期, None则从 config.yaml 读取
    backtest    : True时做分层回测+生成PDF/Excel报告
    n_groups    : 分层数, None则从 config.yaml 读取
    top_n_extra : 全市场top选股数(backtest=True时生效)

    Returns
    -------
    dict : {index_name: DataFrame[valuation_date, code, score_name, final_score]}
    """
    # 加载配置
    cfg = load_config()
    bt = cfg["backtest"]
    start_date = start_date or bt["start_date"]
    end_date = end_date or bt["end_date"]
    n_groups = n_groups or bt["n_groups"]

    combine_cfg = load_combine_by_index_config()
    weight_method = combine_cfg.get("weight_method", "equal")
    ic_window = combine_cfg.get("ic_window", 20)
    smooth_window = combine_cfg.get("smooth_window", 1)
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
    print(f"  因子合成 (历史模式)")
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
            top_n_extra=top_n_extra if backtest else 0,
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

        # 回测 + 报告
        if backtest:
            # 构建 top 组合
            df_top = None
            if df_fm_scores is not None and not df_fm_scores.empty:
                print(f"\n  构建 top 组合: 全市场打分最高 top{top_n_extra}")
                df_top = (
                    df_fm_scores
                    .groupby("valuation_date", group_keys=False)
                    .apply(lambda g: g.nlargest(top_n_extra, "final_score"))
                    .reset_index(drop=True)
                )
                n_top_dates = df_top["valuation_date"].nunique()
                n_top_median = int(
                    df_top.groupby("valuation_date")["code"].count().median()
                )
                print(f"  top组合: {n_top_dates}天, 每日{n_top_median}只")

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
                top_n_extra=top_n_extra,
            )
            timings[f"{index_name}.回测+报告"] = time.time() - t0

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

def run_combine_update(target_date, index_list=None):
    """
    单日因子合成更新

    根据 config_combine_by_index.yaml 配置，计算指定日期的合成因子。
    内部会自动回溯足够的历史数据以计算IC权重和MA平滑。

    Parameters
    ----------
    target_date : 目标持仓日 (T日), 格式 "YYYY-MM-DD"
    index_list  : 指数列表, None则从 combine 配置读取全部

    Returns
    -------
    dict : {index_name: DataFrame[valuation_date, code, score_name, final_score]}
           只包含 target_date 当天的数据
    """
    combine_cfg = load_combine_by_index_config()
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

    # 加载共享数据 (只加载回溯范围)
    shared = _load_shared_data(lookback_start, target_date, run_indices)

    # 逐指数合成
    results = {}

    for index_name in run_indices:
        index_cfg = combine_cfg["indices"][index_name]
        output_name = index_cfg.get("output_name", f"combine_{index_name}")

        print(f"\n合成: {output_name} (回溯 {lookback_start} ~ {target_date})")
        t0 = time.time()

        df_combined = combine_factors_for_index(
            lookback_start, target_date,
            index_name=index_name,
            index_cfg=index_cfg,
            df_index_comp=shared["index_comps"][index_name],
            df_stock=shared["df_stock"],
            weight_method=weight_method,
            ic_window=ic_window,
            date_mode=date_mode,
            df_calendar=shared["df_calendar"],
            top_n_extra=0,  # 更新模式不需要全市场评分
            smooth_window=smooth_window,
            df_st=shared["df_st"],
            df_notrade=shared["df_notrade"],
        )
        elapsed = time.time() - t0

        if df_combined.empty:
            print(f"  {output_name} 合成为空，跳过")
            continue

        # 只保留目标日期
        df_target = df_combined[
            df_combined["valuation_date"] == target_date
        ].copy().reset_index(drop=True)

        if df_target.empty:
            # target_date 可能不是交易日，尝试取最近的日期
            latest_date = df_combined["valuation_date"].max()
            print(f"  警告: {target_date} 无数据，取最近日期 {latest_date}")
            df_target = df_combined[
                df_combined["valuation_date"] == latest_date
            ].copy().reset_index(drop=True)

        n_stocks = len(df_target)
        print(f"  {output_name}: {target_date} 共 {n_stocks} 只股票"
              f" [{_fmt_elapsed(elapsed)}]")

        results[index_name] = df_target

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
    p_combine.add_argument("--top-n", type=int, default=TOP_N_DEFAULT,
                           help=f"Top组合选股数 (默认{TOP_N_DEFAULT})")

    # --- update: 单日更新 ---
    p_update = subparsers.add_parser("update", help="单日因子合成更新")
    p_update.add_argument("target_date", help="目标日期 YYYY-MM-DD")
    p_update.add_argument("--index", nargs="+", default=None,
                          help="指数列表 (默认从combine配置读取)")

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
            top_n_extra=args.top_n,
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
        )

    print(f"\n总耗时: {_fmt_elapsed(time.time() - total_start)}")
