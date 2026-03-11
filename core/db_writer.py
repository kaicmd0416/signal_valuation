"""
数据库入库模块
============
从 globalToolsFunc/sql_saving.py 移植精简而来。
提供 REPLACE INTO 方式写入 MySQL 的能力。
"""

import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from sqlalchemy import (
    create_engine, inspect, MetaData, Table, Column, text,
    String, Float, Integer, DateTime, Date,
)

_CONFIG_DIR = Path(__file__).parent.parent / "config"


def _load_db_saving_config(task_name: str) -> dict:
    """加载 config_db_saving.yaml 中指定任务的配置"""
    with open(_CONFIG_DIR / "config_db_saving.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    task_cfg = cfg.get(task_name)
    if not task_cfg:
        raise ValueError(f"config_db_saving.yaml 中未找到任务: {task_name}")
    return task_cfg


# ============================================================
# 表管理
# ============================================================

_TYPE_MAP = {
    "String": lambda info: String(info.get("length", 50)),
    "Float": lambda info: Float(),
    "Integer": lambda info: Integer(),
    "DateTime": lambda info: DateTime(),
    "Date": lambda info: Date(),
}


def _ensure_table(engine, table_name: str, schema: dict, private_keys: list):
    """如果表不存在则按 schema 创建"""
    insp = inspect(engine)
    if insp.has_table(table_name):
        return
    metadata = MetaData()
    columns = []
    for col_name, col_info in schema.items():
        col_type = col_info["type"]
        if col_type not in _TYPE_MAP:
            raise ValueError(f"未知列类型: {col_type}")
        coltype_obj = _TYPE_MAP[col_type](col_info)
        is_pk = col_name in (private_keys or [])
        columns.append(Column(col_name, coltype_obj, primary_key=is_pk))
    Table(table_name, metadata, *columns)
    metadata.create_all(engine)
    print(f"  已创建表: {table_name}")


# ============================================================
# REPLACE INTO 写入
# ============================================================

def _replace_into_method(table, conn, keys, data_iter):
    """pandas to_sql 的自定义 method，使用 REPLACE INTO"""
    l_columns = [f"`{col}`" for col in keys]
    columns = ", ".join(l_columns)
    placeholders = ", ".join([f":{k}" for k in keys])
    sql = f"REPLACE INTO {table.name} ({columns}) VALUES ({placeholders})"

    data = []
    for row in data_iter:
        processed_row = {}
        for i, val in enumerate(row):
            processed_row[keys[i]] = None if val is None else str(val)
        data.append(processed_row)

    stmt = text(sql)
    conn.execute(stmt, data)


def _standardize_df(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """按 schema 标准化 DataFrame 类型"""
    df = df.copy()
    if "valuation_date" in df.columns:
        df["valuation_date"] = (
            pd.to_datetime(df["valuation_date"].astype(str))
            .apply(lambda x: x.strftime("%Y-%m-%d"))
        )
    for col, col_info in schema.items():
        if col not in df.columns:
            continue
        col_type = col_info.get("type")
        try:
            if col_type == "String":
                df[col] = df[col].astype(str)
            elif col_type == "Float":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
            elif col_type == "Integer":
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
            elif col_type == "DateTime":
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif col_type == "Date":
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date
        except Exception as e:
            print(f"  警告: 列 {col} 类型转换为 {col_type} 失败: {e}")
    return df


# ============================================================
# 对外接口
# ============================================================

def save_combine_score(df: pd.DataFrame,
                       delete_before_insert: bool = True):
    """
    将合成因子打分写入 combine_score 表

    Parameters
    ----------
    df : DataFrame[valuation_date, code, score_name, final_score]
    delete_before_insert : True 则先按 score_name + 日期 删除旧数据
    """
    cfg = _load_db_saving_config("CombineScore")
    db_url = cfg["db_url"]
    table_name = cfg["table_name"].lower()
    schema = cfg["schema"]
    private_keys = cfg["private_keys"]
    chunk_size = cfg.get("chunk_size", 20000)

    # 构建写入 DataFrame
    df_write = df[["valuation_date", "code", "score_name", "final_score"]].copy()
    df_write["update_time"] = datetime.now()

    # 标准化
    df_write = _standardize_df(df_write, schema)

    score_name = df_write["score_name"].iloc[0]

    # 创建引擎
    engine = create_engine(
        db_url, pool_size=5, max_overflow=10,
        pool_timeout=30, pool_recycle=3600, echo=False,
    )

    try:
        # 确保表存在
        _ensure_table(engine, table_name, schema, private_keys)

        # 删除旧数据
        if delete_before_insert:
            val_list = df_write["valuation_date"].unique().tolist()
            if val_list:
                with engine.connect() as conn:
                    from sqlalchemy import bindparam
                    delete_sql = text(
                        f"DELETE FROM {table_name} "
                        f"WHERE score_name = :sname "
                        f"AND valuation_date IN :val_list"
                    ).bindparams(bindparam("val_list", expanding=True))
                    conn.execute(delete_sql, {
                        "sname": score_name,
                        "val_list": val_list,
                    })
                    conn.commit()
                    print(f"  已删除 {score_name} 旧数据: {len(val_list)}天")

        # 写入
        df_write.to_sql(
            name=table_name,
            con=engine,
            if_exists="append",
            index=False,
            chunksize=chunk_size,
            method=_replace_into_method,
        )
        print(f"  已入库 {score_name}: {len(df_write)}行")
    finally:
        engine.dispose()
