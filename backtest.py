"""
分层回测模块
============
核心类 SignalAnalysis，实现因子分层回测的完整流程：
1. 因子与指数成分取交集，按因子值分组
2. 组内按指数权重归一化
3. 计算组合日收益、换手率、手续费
4. 计算相对指数的超额收益（扣费前/后）

分层逻辑:
- group_1 = 因子值最低组, group_N = 因子值最高组
- 正向因子: group_N 应跑赢 group_1
- 反向因子: group_1 应跑赢 group_N
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_tools as gt

# 指数中文名映射
INDEX_CN_MAP = {
    "hs300": "沪深300",
    "zz500": "中证500",
    "zz1000": "中证1000",
    "zzA500": "中证A500",
    "zz2000": "中证2000",
    "sz50": "上证50",
    "gz2000": "国证2000",
}

# 指数代码映射（用于匹配指数收益率数据）
INDEX_CODE_MAP = {
    "沪深300": "000300.SH",
    "中证500": "000905.SH",
    "中证1000": "000852.SH",
    "中证2000": "932000.CSI",
    "中证A500": "000510.CSI",
    "上证50": "000016.SH",
}


class SignalAnalysis:
    """
    因子分层回测分析（向量化计算）

    Parameters
    ----------
    signal_name   : 因子名称
    index_name    : 指数简称, 如 "hs300"
    n_groups      : 分层数量, 默认10
    df_factor     : 因子数据 [valuation_date, code, score_name, final_score]
    df_index_comp : 指数成分数据 [valuation_date, code, weight]
    df_stock      : 股票行情 [valuation_date, code, close, pre_close]
    df_index_ret  : 指数收益率 [valuation_date, code, pct_chg]
    df_calendar   : 交易日历 [valuation_date, next_workday]（可选，加速持仓日映射）

    Attributes
    ----------
    df_factor_merged : 因子与指数成分合并后的数据（用于IC计算）
    df_holding       : 分层持仓明细
    df_info          : 回测结果（日收益、超额收益、换手率等）
    """

    # 双边手续费率: 万分之8.5
    FEE_RATE = 0.00085

    def __init__(self, signal_name: str, index_name: str,
                 n_groups: int = 10, df_factor: pd.DataFrame = None,
                 df_index_comp: pd.DataFrame = None,
                 df_stock: pd.DataFrame = None,
                 df_index_ret: pd.DataFrame = None,
                 df_calendar: pd.DataFrame = None,
                 df_st: pd.DataFrame = None,
                 df_notrade: pd.DataFrame = None):

        self.signal_name = signal_name
        self.index_name = index_name
        self.index_cn = INDEX_CN_MAP.get(index_name, index_name)
        self.n_groups = n_groups

        self.df_factor = df_factor if df_factor is not None else pd.DataFrame()
        self.df_index_comp = df_index_comp if df_index_comp is not None else pd.DataFrame()
        self.df_stock = df_stock if df_stock is not None else pd.DataFrame()
        self.df_index_ret = df_index_ret if df_index_ret is not None else pd.DataFrame()
        self.df_calendar = df_calendar if df_calendar is not None else pd.DataFrame()
        self.df_st = df_st if df_st is not None else pd.DataFrame()
        self.df_notrade = df_notrade if df_notrade is not None else pd.DataFrame()

        self.df_factor_merged = pd.DataFrame()
        self.df_holding = pd.DataFrame()
        self.df_info = pd.DataFrame()

    def build_holding(self) -> pd.DataFrame:
        """
        构建分层持仓

        流程:
        1. 因子数据 × 指数成分取交集
        2. available_date → next_workday 映射为实际持仓日
        3. 按因子值分位数分组 (qcut)
        4. 组内按指数权重归一化
        """
        df_factor = self.df_factor.rename(columns={
            "valuation_date": "available_date",
            "final_score": "factor_value",
        })
        df_idx = self.df_index_comp[["valuation_date", "code", "weight"]].copy()

        # 因子 × 指数成分交集
        df_merged = df_factor.merge(
            df_idx,
            left_on=["available_date", "code"],
            right_on=["valuation_date", "code"],
            how="inner",
        )

        # 持仓日期 = next_workday(available_date)
        if not self.df_calendar.empty:
            cal_map = self.df_calendar.set_index("valuation_date")["next_workday"]
            df_merged["valuation_date"] = df_merged["available_date"].map(cal_map)
            df_merged.dropna(subset=["valuation_date"], inplace=True)
        else:
            df_merged["valuation_date"] = df_merged["available_date"].apply(
                gt.next_workday_calculate
            )
        print(f"  指数内因子数据: {len(df_merged)} 条")

        # 剔除ST股票
        if not self.df_st.empty:
            n_before = len(df_merged)
            df_merged = df_merged.merge(
                self.df_st[["valuation_date", "code"]],
                on=["valuation_date", "code"],
                how="left", indicator="_st"
            )
            df_merged = df_merged[df_merged["_st"] == "left_only"].drop(columns=["_st"])
            print(f"  剔除ST股票: {n_before - len(df_merged)} 条, 剩余 {len(df_merged)} 条")

        # 剔除涨跌停（不可交易）股票
        if not self.df_notrade.empty:
            n_before = len(df_merged)
            df_merged = df_merged.merge(
                self.df_notrade[["valuation_date", "code"]],
                on=["valuation_date", "code"],
                how="left", indicator="_notrade"
            )
            df_merged = df_merged[df_merged["_notrade"] == "left_only"].drop(columns=["_notrade"])
            print(f"  剔除涨跌停股票: {n_before - len(df_merged)} 条, 剩余 {len(df_merged)} 条")

        self.df_factor_merged = df_merged[["valuation_date", "code", "factor_value"]].copy()

        # 按因子值分组 (group_1=最低, group_N=最高)
        df_merged["group"] = df_merged.groupby("valuation_date")["factor_value"].transform(
            lambda x: pd.qcut(x, self.n_groups, labels=False, duplicates="drop")
        )
        df_merged.dropna(subset=["group"], inplace=True)
        df_merged["group"] = df_merged["group"].astype(int)

        # 组内权重归一化
        df_merged["norm_weight"] = df_merged.groupby(
            ["valuation_date", "group"]
        )["weight"].transform(lambda x: x / x.sum())

        df_merged["portfolio_name"] = "group_" + (df_merged["group"] + 1).astype(str)

        self.df_holding = df_merged[[
            "valuation_date", "code", "norm_weight", "portfolio_name", "factor_value"
        ]].rename(columns={"norm_weight": "weight"})

        print(f"  持仓记录: {len(self.df_holding)} 条, 共 {self.n_groups} 组")
        return self.df_holding

    def calc_turnover(self) -> pd.DataFrame:
        """
        向量化计算每个组合每日的单边换手率

        计算逻辑:
        - 连续交易日: 换手 = |今日权重 - 昨日权重| 之和
        - 非连续日(如假期后): 视为新建仓，全额计入
        - 补充上期持有但本期消失的股票（全部卖出）
        - 最终单边换手率 = L1距离 / 2
        """
        df = self.df_holding[["valuation_date", "code", "weight", "portfolio_name"]].copy()
        df.sort_values(["portfolio_name", "code", "valuation_date"], inplace=True)

        # 日期序号映射
        dates_sorted = sorted(df["valuation_date"].unique())
        date_rank = {d: i for i, d in enumerate(dates_sorted)}
        df["date_rank"] = df["valuation_date"].map(date_rank)

        # 同 group+code 内取上一日权重
        df["prev_weight"] = df.groupby(["portfolio_name", "code"])["weight"].shift(1)
        df["prev_date_rank"] = df.groupby(["portfolio_name", "code"])["date_rank"].shift(1)

        # 连续交易日才算调仓
        is_consecutive = (df["date_rank"] - df["prev_date_rank"]) == 1
        df["weight_chg"] = np.where(
            is_consecutive,
            (df["weight"] - df["prev_weight"]).abs(),
            df["weight"]
        )

        df_turnover = df.groupby(
            ["valuation_date", "portfolio_name"]
        )["weight_chg"].sum().reset_index()
        df_turnover.rename(columns={"weight_chg": "turnover"}, inplace=True)

        # 补充: 上期持有但本期消失的股票
        df_prev = df[["valuation_date", "code", "weight", "portfolio_name", "date_rank"]].copy()
        df_prev["next_date_rank"] = df_prev["date_rank"] + 1
        df_next_dates = df[["date_rank", "valuation_date"]].drop_duplicates()
        df_prev = df_prev.merge(
            df_next_dates.rename(columns={"valuation_date": "next_date", "date_rank": "next_date_rank"}),
            on="next_date_rank", how="inner"
        )
        df_next_holding = df[["valuation_date", "code", "portfolio_name"]].copy()
        df_prev_check = df_prev.merge(
            df_next_holding,
            left_on=["next_date", "code", "portfolio_name"],
            right_on=["valuation_date", "code", "portfolio_name"],
            how="left", indicator=True
        )
        df_exited = df_prev_check[df_prev_check["_merge"] == "left_only"]
        if not df_exited.empty:
            df_exit_tv = df_exited.groupby(
                ["next_date", "portfolio_name"]
            )["weight"].sum().reset_index()
            df_exit_tv.rename(columns={"next_date": "valuation_date", "weight": "exit_turnover"}, inplace=True)
            df_turnover = df_turnover.merge(df_exit_tv, on=["valuation_date", "portfolio_name"], how="left")
            df_turnover["exit_turnover"].fillna(0, inplace=True)
            df_turnover["turnover"] = df_turnover["turnover"] + df_turnover["exit_turnover"]
            df_turnover.drop(columns=["exit_turnover"], inplace=True)

        # L1距离 / 2 = 单边换手率
        df_turnover["turnover"] = df_turnover["turnover"] / 2

        return df_turnover

    def run(self):
        """
        执行完整回测流程

        Returns
        -------
        DataFrame : 每日每组的回测结果，包含:
            - paper_return    : 组合日收益
            - net_return      : 扣费后日收益
            - index_return    : 指数日收益
            - excess_paper_return : 扣费前超额收益
            - excess_net_return   : 扣费后超额收益
            - turnover / fee  : 换手率 / 手续费
        """
        if self.df_factor.empty or self.df_index_comp.empty:
            print("  数据为空，跳过")
            return self.df_info

        # 1. 构建持仓
        self.build_holding()
        if self.df_holding.empty:
            print("  持仓为空，跳过")
            return self.df_info

        # 2. 个股收益率
        df_ret = self.df_stock.copy()
        df_ret["pct_chg"] = (df_ret["close"] - df_ret["pre_close"]) / df_ret["pre_close"]

        # 3. 持仓匹配收益率
        df_calc = self.df_holding.merge(
            df_ret[["valuation_date", "code", "pct_chg"]],
            on=["valuation_date", "code"],
            how="left",
        )
        df_calc.dropna(subset=["pct_chg"], inplace=True)

        # 4. 组合日收益 = Σ(weight × pct_chg)
        df_calc["weighted_return"] = df_calc["weight"] * df_calc["pct_chg"]
        df_portfolio = df_calc.groupby(
            ["valuation_date", "portfolio_name"]
        )["weighted_return"].sum().reset_index()
        df_portfolio.rename(columns={"weighted_return": "paper_return"}, inplace=True)

        # 5. 换手率与手续费
        df_turnover = self.calc_turnover()
        df_portfolio = df_portfolio.merge(
            df_turnover, on=["valuation_date", "portfolio_name"], how="left"
        )
        df_portfolio["turnover"].fillna(0, inplace=True)
        df_portfolio["fee"] = df_portfolio["turnover"] * self.FEE_RATE
        df_portfolio["net_return"] = df_portfolio["paper_return"] - df_portfolio["fee"]

        # 6. 指数收益率
        index_code = INDEX_CODE_MAP.get(self.index_cn)
        if index_code and not self.df_index_ret.empty:
            df_idx_ret = self.df_index_ret[
                self.df_index_ret["code"] == index_code
            ][["valuation_date", "pct_chg"]].rename(columns={"pct_chg": "index_return"})
            df_portfolio = df_portfolio.merge(df_idx_ret, on="valuation_date", how="left")
            df_portfolio["index_return"].fillna(0, inplace=True)
        else:
            df_portfolio["index_return"] = 0

        # 7. 超额收益
        df_portfolio["excess_paper_return"] = df_portfolio["paper_return"] - df_portfolio["index_return"]
        df_portfolio["excess_net_return"] = df_portfolio["net_return"] - df_portfolio["index_return"]
        df_portfolio.sort_values(["portfolio_name", "valuation_date"], inplace=True)

        self.df_info = df_portfolio
        print("  回测完成!")
        return self.df_info
