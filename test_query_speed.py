"""
查询速度测试
============
测试 data_score 表索引优化后的查询性能。

对比:
1. 单因子查询 (get_factor_data)
2. 批量查询 UNION ALL (get_factor_data_batch)
3. 原始 IN(...) 方式作为对照
4. EXPLAIN 分析执行计划
"""

import os
import sys
import time

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv

glv.init()

import pandas as pd
import sqlalchemy
from data_prepare import get_engine, get_factor_data, get_factor_data_batch

START_DATE = "2023-06-30"
END_DATE = "2025-12-31"

SINGLE_FACTORS = ["PE", "PB", "turnoverRateAvg20d"]
BATCH_FACTORS = [
    "turnoverRateAvg20d", "turnoverRateAvg120d", "ILLIQ",
    "PB", "PCF", "PE", "OCFPR", "OCFTD", "GPG",
    "RTN1", "RTN3", "RTN6", "LnFloatCap",
]


def fmt(seconds):
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    return f"{int(m)}m{s:.1f}s"


def test_explain():
    """查看 EXPLAIN 确认索引使用情况"""
    engine = get_engine("factor")
    print("=" * 60)
    print("  EXPLAIN 分析")
    print("=" * 60)

    # 单因子查询
    print("\n[1] 单因子查询 EXPLAIN:")
    sql = f"""
        EXPLAIN SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name = 'PE'
          AND valuation_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    df = pd.read_sql(sql, engine)
    print(df.to_string(index=False))

    # IN 查询
    print("\n[2] IN(...) 查询 EXPLAIN:")
    names_str = ", ".join([f"'{n}'" for n in BATCH_FACTORS])
    sql = f"""
        EXPLAIN SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name IN ({names_str})
          AND valuation_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    df = pd.read_sql(sql, engine)
    print(df.to_string(index=False))

    # UNION ALL 单条
    print("\n[3] UNION ALL 单条 EXPLAIN:")
    sql = f"""
        EXPLAIN SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name = 'PE'
          AND valuation_date BETWEEN '{START_DATE}' AND '{END_DATE}'
        UNION ALL
        SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name = 'PB'
          AND valuation_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    df = pd.read_sql(sql, engine)
    print(df.to_string(index=False))


def test_single_factor():
    """测试单因子查询速度"""
    print("\n" + "=" * 60)
    print("  单因子查询速度测试")
    print("=" * 60)

    for name in SINGLE_FACTORS:
        t0 = time.time()
        df = get_factor_data(START_DATE, END_DATE, name)
        elapsed = time.time() - t0
        print(f"  {name:<25} {len(df):>8} 条  {fmt(elapsed):>10}")


def test_batch_union_all():
    """测试 UNION ALL 批量查询速度"""
    print("\n" + "=" * 60)
    print("  批量查询 UNION ALL (13因子)")
    print("=" * 60)

    t0 = time.time()
    df = get_factor_data_batch(START_DATE, END_DATE, BATCH_FACTORS)
    elapsed = time.time() - t0
    print(f"  总条数: {len(df):>10}")
    print(f"  因子数: {df['score_name'].nunique()}")
    print(f"  耗时:   {fmt(elapsed)}")


def test_batch_in_clause():
    """对照: 原始 IN(...) 方式"""
    print("\n" + "=" * 60)
    print("  对照: IN(...) 方式 (13因子)")
    print("=" * 60)

    engine = get_engine("factor")
    names_str = ", ".join([f"'{n}'" for n in BATCH_FACTORS])
    sql = f"""
        SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name IN ({names_str})
          AND valuation_date BETWEEN '{START_DATE}' AND '{END_DATE}'
    """
    t0 = time.time()
    df = pd.read_sql(sql, engine)
    elapsed = time.time() - t0
    print(f"  总条数: {len(df):>10}")
    print(f"  因子数: {df['score_name'].nunique()}")
    print(f"  耗时:   {fmt(elapsed)}")


if __name__ == "__main__":
    print("数据库查询速度测试")
    print(f"日期范围: {START_DATE} ~ {END_DATE}")
    print(f"批量因子数: {len(BATCH_FACTORS)}")

    test_explain()
    test_single_factor()
    test_batch_union_all()
    test_batch_in_clause()

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)
