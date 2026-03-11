"""
按指数合成因子回测入口
=======================
每个指数独立合成因子 + 独立回测，互不干扰。

流程 (对每个指数):
1. 从 config_combine_by_index.yaml 读取该指数的因子配置
2. 合成因子 (DB查询 + z-score + 方向处理 + 等权合成)
3. 获取该指数的成分股
4. 只跑该指数的分层回测 + 生成PDF报告

用法:
    python run_combine.py                  # 跑全部指数
    python run_combine.py hs300            # 只跑沪深300
    python run_combine.py hs300 zz500      # 跑沪深300和中证500
"""

import os
import sys
import time

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv
import global_tools as gt

from data_prepare import (
    load_config, load_combine_by_index_config,
    get_index_component, get_market_data, get_trading_calendar,
    get_st_stocks, get_notrade_stocks,
)
from report import generate_report
from factor_combine import combine_factors_for_index


def fmt_elapsed(seconds):
    """格式化耗时"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"


# ============================================================

if __name__ == "__main__":
    total_start = time.time()
    timings = {}

    glv.init()

    # 加载配置
    cfg = load_config()
    bt = cfg["backtest"]
    START_DATE = bt["start_date"]
    END_DATE = bt["end_date"]
    N_GROUPS = bt["n_groups"]

    combine_cfg = load_combine_by_index_config()
    all_indices = list(combine_cfg["indices"].keys())

    # 命令行参数: 指定要跑的指数（可选）
    if len(sys.argv) > 1:
        run_indices = [idx for idx in sys.argv[1:] if idx in combine_cfg["indices"]]
        invalid = [idx for idx in sys.argv[1:] if idx not in combine_cfg["indices"]]
        if invalid:
            print(f"警告: 以下指数不在配置中，已跳过: {invalid}")
        if not run_indices:
            print(f"错误: 没有有效的指数，可选: {all_indices}")
            sys.exit(1)
    else:
        run_indices = all_indices

    weight_method_display = combine_cfg.get("weight_method", "equal")
    ic_window_display = combine_cfg.get("ic_window", 20)
    date_mode_display = combine_cfg.get("date_mode", "target_date")
    print(f"{'='*60}")
    print(f"  按指数合成因子回测")
    print(f"  回测区间: {START_DATE} ~ {END_DATE}")
    print(f"  日期模式: {date_mode_display}")
    print(f"  合成方法: {weight_method_display}" +
          (f" (窗口={ic_window_display}天)" if weight_method_display == "ic_weight" else ""))
    print(f"  待处理指数: {run_indices}")
    print(f"{'='*60}\n")

    # 1. 共享数据: 行情 + 日历 + ST/涨跌停
    mkt_end = gt.next_workday_calculate(END_DATE)

    print(f"正在获取行情数据: {START_DATE} ~ {mkt_end}")
    t0 = time.time()
    market_data = get_market_data(START_DATE, mkt_end)
    df_stock = market_data[0]
    df_index_ret = market_data[6]
    timings["0.行情数据"] = time.time() - t0
    print(f"行情数据获取完成 [{fmt_elapsed(timings['0.行情数据'])}]\n")

    t0 = time.time()
    print("正在获取交易日历...")
    df_calendar = get_trading_calendar(START_DATE, mkt_end)
    timings["0.交易日历"] = time.time() - t0
    print(f"交易日历: {len(df_calendar)} 条 [{fmt_elapsed(timings['0.交易日历'])}]\n")

    print("正在获取ST股票数据...")
    t0 = time.time()
    df_st = get_st_stocks(START_DATE, mkt_end)
    print(f"ST股票: {len(df_st)} 条")
    df_notrade = get_notrade_stocks(START_DATE, mkt_end)
    print(f"涨跌停股票: {len(df_notrade)} 条")
    timings["0.ST与涨跌停数据"] = time.time() - t0
    print(f"ST与涨跌停数据获取完成 [{fmt_elapsed(timings['0.ST与涨跌停数据'])}]\n")

    # 2. 逐指数: 合成 + 回测
    for index_name in run_indices:
        index_cfg = combine_cfg["indices"][index_name]
        output_name = index_cfg.get("output_name", f"combine_{index_name}")
        n_factors = len(index_cfg["factors"])

        print(f"\n{'#'*60}")
        print(f"  指数: {index_name} | 合成因子: {output_name} | 因子数: {n_factors}")
        print(f"{'#'*60}")

        # 2a. 获取指数成分（提前，供合成函数内部z-score使用）
        print(f"\n正在获取指数成分: {index_name}")
        t0 = time.time()
        df_index_comp = get_index_component(START_DATE, END_DATE, index_name)
        t_idx = time.time() - t0
        timings[f"{index_name}.指数成分"] = t_idx
        print(f"  {index_name} 成分股获取完成 [{fmt_elapsed(t_idx)}]")

        # 2b. 合成因子（在成分股内部做z-score）
        weight_method = combine_cfg.get("weight_method", "equal")
        ic_window = combine_cfg.get("ic_window", 20)
        date_mode = combine_cfg.get("date_mode", "target_date")

        t0 = time.time()
        df_combined = combine_factors_for_index(
            START_DATE, END_DATE,
            index_name=index_name,
            index_cfg=index_cfg,
            df_index_comp=df_index_comp,
            df_stock=df_stock,
            weight_method=weight_method,
            ic_window=ic_window,
            date_mode=date_mode,
            df_calendar=df_calendar,
        )
        t_combine = time.time() - t0
        timings[f"{index_name}.因子合成"] = t_combine

        if df_combined.empty:
            print(f"  {output_name} 合成因子为空，跳过")
            continue

        # 2c. 回测 + 报告 (只跑这一个指数)
        print(f"\n{'='*50}")
        print(f"  开始回测: {output_name} x {index_name}")
        print(f"{'='*50}")
        t0 = time.time()
        generate_report(
            signal_name=output_name,
            start_date=START_DATE,
            end_date=END_DATE,
            n_groups=N_GROUPS,
            df_factor=df_combined,
            index_data={index_name: df_index_comp},
            df_stock=df_stock,
            df_index_ret=df_index_ret,
            df_calendar=df_calendar,
            df_st=df_st,
            df_notrade=df_notrade,
            date_mode=date_mode,
        )
        timings[f"{index_name}.回测+报告"] = time.time() - t0

    # 耗时汇总
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  耗时统计")
    print(f"{'='*60}")
    for step, elapsed in timings.items():
        pct = elapsed / total_elapsed * 100
        print(f"  {step:<45} {fmt_elapsed(elapsed):>8}  ({pct:.1f}%)")
    print(f"  {'─'*55}")
    print(f"  {'总计':<45} {fmt_elapsed(total_elapsed):>8}")
