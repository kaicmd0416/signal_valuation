"""
数据准备模块
============
负责所有外部数据的获取，包括：
- 因子数据（data_score 表）
- 指数成分数据（data_indexcomponent 表）
- 交易日历（chinesevaluationdate 表）
- 行情数据（股票、指数等）
- 交易日工具（next_workday / last_workday）

所有数据均来自远程 MySQL 数据库（阿里云 RDS），
连接信息配置在 config.yaml 的 database 段。
"""

import yaml
import pandas as pd
import sqlalchemy
from pathlib import Path

_CONFIG_DIR = Path(__file__).parent.parent / "config"


# ============================================================
# 配置加载
# ============================================================

def load_config() -> dict:
    """加载 config.yaml（数据库 + 输出路径 + 回测参数）"""
    with open(_CONFIG_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_signals_config() -> dict:
    """加载 config_single_signal.yaml（单因子回测配置）"""
    with open(_CONFIG_DIR / "config_single_signal.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_combine_by_index_config(mode: str = "test") -> dict:
    """
    加载按指数分别合成配置

    Parameters
    ----------
    mode : "test" 或 "prod"
        test → config_combine_by_index_test.yaml
        prod → config_combine_by_index_prod.yaml
    """
    filename = f"config_combine_by_index_{mode}.yaml"
    with open(_CONFIG_DIR / filename, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_engine(db_key: str = "factor"):
    """
    创建数据库连接引擎

    Parameters
    ----------
    db_key : "factor"(因子库) 或 "market"(行情库)
    """
    cfg = load_config()["database"][db_key]
    url = (
        f"mysql+pymysql://{cfg['user']}:{cfg['password']}"
        f"@{cfg['host']}:{cfg['port']}/{cfg['database']}"
    )
    return sqlalchemy.create_engine(url)


# ============================================================
# 因子数据
# ============================================================

def get_factor_data(start_date: str, end_date: str, signal_name: str) -> pd.DataFrame:
    """
    获取单个因子数据

    Parameters
    ----------
    start_date  : 开始日期, 如 '2023-06-30'
    end_date    : 结束日期, 如 '2025-12-31'
    signal_name : 因子名称 (score_name 字段)

    Returns
    -------
    DataFrame[valuation_date, code, score_name, final_score]
    """
    engine = get_engine("factor")
    # 利用 idx_scorename_date (score_name, valuation_date) 索引
    # 去掉 ORDER BY 让 DB 只做索引扫描，排序交给 pandas
    query = sqlalchemy.text("""
        SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name = :signal_name
          AND valuation_date BETWEEN :start_date AND :end_date
    """)
    params = {
        "signal_name": signal_name,
        "start_date": start_date,
        "end_date": end_date,
    }
    df = pd.read_sql(query, engine, params=params)
    df.sort_values(["valuation_date", "code"], inplace=True, ignore_index=True)
    return df


def get_factor_data_batch(start_date: str, end_date: str,
                          signal_names: list) -> pd.DataFrame:
    """
    批量获取多个因子数据（单次 SQL 查询，减少连接开销）

    Parameters
    ----------
    start_date   : 开始日期
    end_date     : 结束日期
    signal_names : 因子名称列表, 如 ["PE", "PB", "ROA"]

    Returns
    -------
    DataFrame[valuation_date, code, score_name, final_score]
    """
    if not signal_names:
        return pd.DataFrame(columns=["valuation_date", "code", "score_name", "final_score"])

    engine = get_engine("factor")
    # IN(...) 大批量时 MySQL 优化器自动选择 PRIMARY 聚簇索引顺序扫描
    # 比 UNION ALL 或 FORCE INDEX(idx_scorename_date) 回表更快
    # 去掉 ORDER BY 避免 DB 端 filesort，排序交给 pandas
    placeholders = ", ".join([f":name_{i}" for i in range(len(signal_names))])
    query = sqlalchemy.text(f"""
        SELECT valuation_date, code, score_name, final_score
        FROM data_score
        WHERE score_name IN ({placeholders})
          AND valuation_date BETWEEN :start_date AND :end_date
    """)
    params = {"start_date": start_date, "end_date": end_date}
    for i, name in enumerate(signal_names):
        params[f"name_{i}"] = name

    df = pd.read_sql(query, engine, params=params)
    df.sort_values(["score_name", "valuation_date", "code"], inplace=True, ignore_index=True)
    return df


# ============================================================
# 指数与行情数据
# ============================================================

def get_index_component(start_date: str, end_date: str, index_name: str) -> pd.DataFrame:
    """
    获取指数成分股数据

    Parameters
    ----------
    index_name : 指数简称, 如 "hs300", "zz500", "zz1000", "zz2000"

    Returns
    -------
    DataFrame[valuation_date, code, weight, status, organization]
    """
    engine = get_engine("market")
    query = sqlalchemy.text("""
        SELECT valuation_date, code, weight, status, organization
        FROM data_indexcomponent
        WHERE organization = :index_name
          AND valuation_date BETWEEN :start_date AND :end_date
        ORDER BY valuation_date, code
    """)
    params = {
        "index_name": index_name,
        "start_date": start_date,
        "end_date": end_date,
    }
    df = pd.read_sql(query, engine, params=params)
    return df


def get_trading_calendar(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取交易日历，并构建 next_workday 映射

    Returns
    -------
    DataFrame[valuation_date, next_workday]
        用于将因子可用日期映射到实际持仓日期
    """
    engine = get_engine("market")
    query = sqlalchemy.text("""
        SELECT valuation_date
        FROM chinesevaluationdate
        WHERE valuation_date BETWEEN :start_date AND :end_date
        ORDER BY valuation_date
    """)
    params = {"start_date": start_date, "end_date": end_date}
    df = pd.read_sql(query, engine, params=params)
    df["next_workday"] = df["valuation_date"].shift(-1)
    df.dropna(subset=["next_workday"], inplace=True)
    return df


def get_st_stocks(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取ST股票列表

    Returns
    -------
    DataFrame[valuation_date, code]
    """
    engine = get_engine("market")
    query = sqlalchemy.text("""
        SELECT valuation_date, code
        FROM st_stock
        WHERE valuation_date BETWEEN :start_date AND :end_date
    """)
    params = {"start_date": start_date, "end_date": end_date}
    df = pd.read_sql(query, engine, params=params)
    return df


def get_notrade_stocks(start_date: str, end_date: str) -> pd.DataFrame:
    """
    获取涨跌停（不可交易）股票列表

    Returns
    -------
    DataFrame[valuation_date, code]
    """
    engine = get_engine("market")
    query = sqlalchemy.text("""
        SELECT valuation_date, code
        FROM data_stocknotrade
        WHERE valuation_date BETWEEN :start_date AND :end_date
    """)
    params = {"start_date": start_date, "end_date": end_date}
    df = pd.read_sql(query, engine, params=params)
    return df


def get_market_data(start_date: str, end_date: str):
    """
    获取行情数据 (股票 + 指数)

    Returns
    -------
    tuple : (df_stock, None, None, None, None, None, df_index)
        [0]=股票行情, [6]=指数行情, 其余位置保留None以兼容旧调用方式
    """
    engine = get_engine("market")

    # 股票行情: valuation_date, code, close, pre_close
    query_stock = sqlalchemy.text("""
        SELECT valuation_date, code, close, pre_close
        FROM data_stock
        WHERE valuation_date BETWEEN :start_date AND :end_date
    """)
    df_stock = pd.read_sql(query_stock, engine,
                           params={"start_date": start_date, "end_date": end_date})
    df_stock.sort_values(["code", "valuation_date"], inplace=True, ignore_index=True)

    # 指数行情: valuation_date, code, pct_chg
    query_index = sqlalchemy.text("""
        SELECT valuation_date, code, pct_chg
        FROM data_index
        WHERE valuation_date BETWEEN :start_date AND :end_date
    """)
    df_index = pd.read_sql(query_index, engine,
                           params={"start_date": start_date, "end_date": end_date})
    df_index.sort_values(["code", "valuation_date"], inplace=True, ignore_index=True)

    return (df_stock, None, None, None, None, None, df_index)


# ============================================================
# 交易日工具
# ============================================================

# 模块级缓存: 完整交易日历 (首次调用时加载)
_FULL_CALENDAR = None


def _ensure_calendar():
    """确保交易日历已加载到缓存"""
    global _FULL_CALENDAR
    if _FULL_CALENDAR is None:
        engine = get_engine("market")
        query = sqlalchemy.text("""
            SELECT valuation_date
            FROM chinesevaluationdate
            ORDER BY valuation_date
        """)
        df = pd.read_sql(query, engine)
        _FULL_CALENDAR = sorted(df["valuation_date"].astype(str).tolist())
    return _FULL_CALENDAR


def next_workday(date_str: str) -> str:
    """
    取下一个交易日

    Parameters
    ----------
    date_str : 日期字符串, 如 "2025-12-31"

    Returns
    -------
    str : 下一个交易日, 如 "2026-01-02"
    """
    cal = _ensure_calendar()
    # 如果 date_str 是交易日，取其后一天
    # 如果不是交易日，取第一个 >= date_str 的交易日
    try:
        idx = cal.index(date_str)
        if idx + 1 < len(cal):
            return cal[idx + 1]
        return date_str
    except ValueError:
        # date_str 不在日历中，取第一个 > date_str 的交易日
        for d in cal:
            if d > date_str:
                return d
        return date_str


def last_workday(date_str: str) -> str:
    """
    取上一个交易日

    Parameters
    ----------
    date_str : 日期字符串, 如 "2025-01-02"

    Returns
    -------
    str : 上一个交易日, 如 "2024-12-31"
    """
    cal = _ensure_calendar()
    # 取最后一个 < date_str 的交易日
    result = None
    for d in cal:
        if d < date_str:
            result = d
        else:
            break
    return result if result else date_str
