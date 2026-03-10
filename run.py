"""
全量回测入口
============
执行所有单因子的分层回测 + 合成因子回测。

流程:
1. 获取行情数据、交易日历、指数成分（共享数据，只取一次）
2. 逐个因子获取数据 → 生成分层回测PDF报告
3. 合成因子（等权） → 生成合成因子PDF报告

用法:
    python run.py
"""

import os
import sys

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv
import global_tools as gt

from data_prepare import (
    load_config, load_signals_config, load_combine_config,
    get_factor_data, get_index_component, get_market_data, get_trading_calendar,
)
from report import generate_report
from factor_combine import combine_factors

# ============================================================

if __name__ == "__main__":
    glv.init()

    # 加载配置
    cfg = load_config()
    bt = cfg["backtest"]
    START_DATE = bt["start_date"]
    END_DATE = bt["end_date"]
    N_GROUPS = bt["n_groups"]
    INDEX_LIST = bt["index_list"]

    signals_cfg = load_signals_config()
    SIGNAL_LIST = signals_cfg["signals"]

    # 1. 行情数据（只取一次）
    mkt_end = gt.next_workday_calculate(END_DATE)
    print(f"正在获取行情数据: {START_DATE} ~ {mkt_end}")
    market_data = get_market_data(START_DATE, mkt_end)
    df_stock = market_data[0]
    df_index_ret = market_data[6]
    print("行情数据获取完成\n")

    # 2. 交易日历
    print("正在获取交易日历...")
    df_calendar = get_trading_calendar(START_DATE, mkt_end)
    print(f"交易日历: {len(df_calendar)} 条\n")

    # 3. 指数成分（只取一次）
    print("正在获取指数成分数据...")
    index_data = {}
    for index_name in INDEX_LIST:
        print(f"  {index_name}...")
        index_data[index_name] = get_index_component(START_DATE, END_DATE, index_name)
    print("指数成分数据获取完成\n")

    # 4. 逐个单因子回测
    for signal_name in SIGNAL_LIST:
        print(f"\n{'#'*60}")
        print(f"  开始回测因子: {signal_name}")
        print(f"{'#'*60}")

        df_factor = get_factor_data(START_DATE, END_DATE, signal_name)
        print(f"  因子数据: {len(df_factor)} 条")

        generate_report(
            signal_name=signal_name,
            start_date=START_DATE,
            end_date=END_DATE,
            n_groups=N_GROUPS,
            df_factor=df_factor,
            index_data=index_data,
            df_stock=df_stock,
            df_index_ret=df_index_ret,
            df_calendar=df_calendar,
        )

    # 5. 合成因子回测
    combine_cfg = load_combine_config()
    output_name = combine_cfg.get("output_name", "combine_1")

    print(f"\n{'#'*60}")
    print(f"  开始合成因子回测: {output_name}")
    print(f"{'#'*60}")

    df_combined = combine_factors(START_DATE, END_DATE)
    if not df_combined.empty:
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
