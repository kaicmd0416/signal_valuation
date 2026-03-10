"""
合成因子回测入口
================
仅运行合成因子的回测（跳过单因子），适合快速迭代调试。

流程:
1. 合成因子 (内含DB查询 + z-score + 方向处理 + 等权合成)
2. 获取行情、日历、指数成分
3. 一份PDF报告包含全部指数的回测结果

用法:
    python run_combine.py
"""

import os
import sys
import time

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv
import global_tools as gt

from data_prepare import load_config, load_combine_config, get_index_component, get_market_data, get_trading_calendar
from report import generate_report
from factor_combine import combine_factors


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
    INDEX_LIST = bt["index_list"]

    combine_cfg = load_combine_config()
    output_name = combine_cfg.get("output_name", "combine_1")

    # 1. 合成因子
    print(f"{'#'*60}")
    print(f"  步骤1: 合成因子 {output_name}")
    print(f"{'#'*60}\n")
    t0 = time.time()
    df_combined = combine_factors(START_DATE, END_DATE)
    timings["1.因子合成(DB查询+zscore+方向处理+等权合成)"] = time.time() - t0

    if df_combined.empty:
        print("合成因子为空，退出")
        sys.exit(1)

    # 2. 行情数据
    mkt_end = gt.next_workday_calculate(END_DATE)
    print(f"\n正在获取行情数据: {START_DATE} ~ {mkt_end}")
    t0 = time.time()
    market_data = get_market_data(START_DATE, mkt_end)
    df_stock = market_data[0]
    df_index_ret = market_data[6]
    timings["2.行情数据"] = time.time() - t0
    print(f"行情数据获取完成 [{fmt_elapsed(timings['2.行情数据'])}]\n")

    # 3. 交易日历
    t0 = time.time()
    print("正在获取交易日历...")
    df_calendar = get_trading_calendar(START_DATE, mkt_end)
    timings["3.交易日历"] = time.time() - t0
    print(f"交易日历: {len(df_calendar)} 条 [{fmt_elapsed(timings['3.交易日历'])}]\n")

    # 4. 指数成分
    print("正在获取指数成分数据...")
    t0 = time.time()
    index_data = {}
    for index_name in INDEX_LIST:
        t_idx = time.time()
        index_data[index_name] = get_index_component(START_DATE, END_DATE, index_name)
        print(f"  {index_name} [{fmt_elapsed(time.time() - t_idx)}]")
    timings["4.指数成分数据"] = time.time() - t0
    print(f"指数成分数据获取完成 [{fmt_elapsed(timings['4.指数成分数据'])}]\n")

    # 5. 回测 + 报告
    print(f"{'#'*60}")
    print(f"  步骤2: 回测 {output_name}")
    print(f"{'#'*60}")
    t0 = time.time()
    generate_report(
        signal_name=output_name,
        start_date=START_DATE,
        end_date=END_DATE,
        n_groups=N_GROUPS,
        df_factor=df_combined,
        index_data=index_data,
        df_stock=df_stock,
        df_index_ret=df_index_ret,
        df_calendar=df_calendar,
    )
    timings["5.回测+报告生成"] = time.time() - t0

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
