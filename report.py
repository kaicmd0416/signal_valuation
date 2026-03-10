"""
报告生成模块
============
根据回测结果生成 PDF 分层回测报告。

一份 PDF 报告包含所有指数的完整回测分析:
- IC/IR 分析（截面 Rank IC、年度 IC 表现）
- 分层超额收益统计（扣费前 / 扣费后）
- 换手率与手续费统计
- 分层超额净值走势图
- 多空组合净值分析
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats

matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

path = os.getenv('GLOBAL_TOOLSFUNC_new')
sys.path.append(path)
import global_dic as glv
import global_tools as gt
from PDF.PDFCreator import PDFCreator

from backtest import SignalAnalysis
from data_prepare import load_config

# 指数简称 → 中文名映射
INDEX_MAP = {
    "hs300": "沪深300",
    "zz500": "中证500",
    "zz1000": "中证1000",
    "zz2000": "中证2000",
}


# ============================================================
# 分层回测相关
# ============================================================

def calc_yearly_excess(df_info: pd.DataFrame) -> pd.DataFrame:
    """计算每一年各 group 的累计超额收益(%)"""
    df = df_info[["valuation_date", "excess_paper_return", "portfolio_name"]].copy()
    df["year"] = df["valuation_date"].astype(str).str[:4]

    result = {}
    for (year, group), sub in df.groupby(["year", "portfolio_name"]):
        cum_ret = (1 + sub["excess_paper_return"]).prod() - 1
        result.setdefault(year, {})[group] = round(cum_ret * 100, 2)

    df_yearly = pd.DataFrame(result).T.sort_index()
    df_yearly = df_yearly[sorted(df_yearly.columns, key=lambda x: int(x.split("_")[1]))]
    return df_yearly


def calc_excess_nav(df_info: pd.DataFrame) -> pd.DataFrame:
    """计算各组超额净值曲线"""
    df = df_info[["valuation_date", "excess_paper_return", "portfolio_name"]].copy()
    df_pivot = df.pivot_table(
        index="valuation_date", columns="portfolio_name",
        values="excess_paper_return",
    )
    df_pivot.sort_index(inplace=True)
    df_nav = (1 + df_pivot).cumprod()
    df_nav = df_nav[sorted(df_nav.columns, key=lambda x: int(x.split("_")[1]))]
    return df_nav


def plot_excess_nav(df_nav: pd.DataFrame, title: str, save_path: str):
    """画分层超额净值走势图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = pd.to_datetime(df_nav.index)
    for col in df_nav.columns:
        ax.plot(dates, df_nav[col], label=col)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("日期", fontsize=13)
    ax.set_ylabel("超额净值", fontsize=13)
    ax.legend(loc="best", fontsize=13, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# IC / IR 相关
# ============================================================

def calc_daily_ic(df_factor_merged: pd.DataFrame, df_stock: pd.DataFrame) -> pd.DataFrame:
    """
    计算每日截面 Rank IC (Spearman)

    Parameters
    ----------
    df_factor_merged : 包含 valuation_date, code, factor_value (持仓日期已对齐)
    df_stock         : 股票行情，包含 valuation_date, code, close, pre_close

    Returns
    -------
    pd.DataFrame : columns = [valuation_date, IC, rank_IC]
    """
    df_ret = df_stock.copy()
    df_ret["pct_chg"] = (df_ret["close"] - df_ret["pre_close"]) / df_ret["pre_close"]
    df_ret = df_ret[["valuation_date", "code", "pct_chg"]].drop_duplicates(
        subset=["valuation_date", "code"]
    )
    df_merged = df_factor_merged.merge(df_ret, on=["valuation_date", "code"], how="inner")
    df_merged.dropna(subset=["factor_value", "pct_chg"], inplace=True)

    # 向量化: 按日期分组计算 rank 相关系数
    df_merged["factor_rank"] = df_merged.groupby("valuation_date")["factor_value"].rank()
    df_merged["ret_rank"] = df_merged.groupby("valuation_date")["pct_chg"].rank()
    counts = df_merged.groupby("valuation_date")["code"].transform("count")
    df_merged = df_merged[counts >= 10]

    # Pearson IC
    ic_pearson = df_merged.groupby("valuation_date").apply(
        lambda g: g["factor_value"].corr(g["pct_chg"])
    ).rename("IC")

    # Rank IC (Spearman via rank correlation)
    ic_rank = df_merged.groupby("valuation_date").apply(
        lambda g: g["factor_rank"].corr(g["ret_rank"])
    ).rename("rank_IC")

    df_ic = pd.concat([ic_pearson, ic_rank], axis=1).reset_index()
    df_ic.dropna(inplace=True)
    return df_ic


def calc_ic_summary(df_ic: pd.DataFrame) -> dict:
    """汇总 IC 指标"""
    rank_ic = df_ic["rank_IC"]
    return {
        "Rank IC 均值": round(rank_ic.mean(), 4),
        "Rank IC 标准差": round(rank_ic.std(), 4),
        "IR (IC均值/IC标准差)": round(rank_ic.mean() / rank_ic.std(), 4) if rank_ic.std() > 0 else 0,
        "IC 胜率(%)": round((rank_ic > 0).sum() / len(rank_ic) * 100, 2),
        "IC > 0.02 占比(%)": round((rank_ic > 0.02).sum() / len(rank_ic) * 100, 2),
        "IC < -0.02 占比(%)": round((rank_ic < -0.02).sum() / len(rank_ic) * 100, 2),
        "|IC| 均值": round(rank_ic.abs().mean(), 4),
    }


def calc_yearly_ic(df_ic: pd.DataFrame) -> pd.DataFrame:
    """计算每一年的 IC 均值、IR、胜率"""
    df = df_ic.copy()
    df["year"] = df["valuation_date"].astype(str).str[:4]

    rows = []
    for year, sub in df.groupby("year"):
        ic_mean = sub["rank_IC"].mean()
        ic_std = sub["rank_IC"].std()
        ir = ic_mean / ic_std if ic_std > 0 else 0
        win_rate = (sub["rank_IC"] > 0).sum() / len(sub) * 100
        rows.append({
            "year": year,
            "Rank IC均值": round(ic_mean, 4),
            "IC标准差": round(ic_std, 4),
            "IR": round(ir, 4),
            "IC胜率(%)": round(win_rate, 2),
        })

    # 全区间
    ic_mean = df_ic["rank_IC"].mean()
    ic_std = df_ic["rank_IC"].std()
    ir = ic_mean / ic_std if ic_std > 0 else 0
    win_rate = (df_ic["rank_IC"] > 0).sum() / len(df_ic) * 100
    rows.append({
        "year": "全区间",
        "Rank IC均值": round(ic_mean, 4),
        "IC标准差": round(ic_std, 4),
        "IR": round(ir, 4),
        "IC胜率(%)": round(win_rate, 2),
    })

    return pd.DataFrame(rows)


def plot_ic(df_ic: pd.DataFrame, title: str, save_path: str):
    """画 IC 柱状图 + 累计IC曲线"""
    fig, ax1 = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(df_ic["valuation_date"])

    # IC 柱状图
    colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in df_ic["rank_IC"]]
    ax1.bar(dates, df_ic["rank_IC"], color=colors, alpha=0.6, width=1.5, label="Rank IC")
    ax1.set_ylabel("Rank IC", fontsize=13)
    ax1.axhline(y=0, color="black", linewidth=0.5)

    # 累计IC曲线（右轴）
    ax2 = ax1.twinx()
    cum_ic = df_ic["rank_IC"].cumsum()
    ax2.plot(dates, cum_ic.values, color="#3498db", linewidth=2, label="累计IC")
    ax2.set_ylabel("累计IC", fontsize=13)

    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel("日期", fontsize=13)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=12)

    ax1.xaxis.set_major_locator(mdates.MonthLocator())
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 换手率相关
# ============================================================

def calc_yearly_turnover(df_info: pd.DataFrame) -> dict:
    """
    计算每年各组的年化换手率和年化手续费

    Returns
    -------
    dict with keys:
        "annualized_turnover" : 年化换手率(倍) — year × group
        "annualized_fee"      : 年化手续费(%) — year × group
    """
    df = df_info[["valuation_date", "turnover", "fee", "portfolio_name"]].copy()
    df["year"] = df["valuation_date"].astype(str).str[:4]

    def _pivot_annualized(df_src, agg_col, scale):
        """scale: 年化时的显示系数 (换手率用1显示倍数, 手续费用100显示百分比)"""
        # 每年: 日均 × 252 年化
        yearly = df_src.groupby(["year", "portfolio_name"])[agg_col].mean().reset_index()
        yearly[agg_col] = (yearly[agg_col] * 252 * scale).round(2)
        pt = yearly.pivot(index="year", columns="portfolio_name", values=agg_col)
        # 全区间: 日均 × 252
        all_row = df_src.groupby("portfolio_name")[agg_col].mean() * 252 * scale
        all_row = all_row.round(2)
        pt.loc["全区间"] = all_row
        pt = pt[sorted(pt.columns, key=lambda x: int(x.split("_")[1]))]
        return pt

    return {
        "annualized_turnover": _pivot_annualized(df, "turnover", 1),   # 倍数
        "annualized_fee": _pivot_annualized(df, "fee", 100),            # 百分比
    }


def calc_yearly_excess_net(df_info: pd.DataFrame) -> pd.DataFrame:
    """计算每一年各 group 的扣费后累计超额收益(%)"""
    df = df_info[["valuation_date", "excess_net_return", "portfolio_name"]].copy()
    df["year"] = df["valuation_date"].astype(str).str[:4]

    result = {}
    for (year, group), sub in df.groupby(["year", "portfolio_name"]):
        cum_ret = (1 + sub["excess_net_return"]).prod() - 1
        result.setdefault(year, {})[group] = round(cum_ret * 100, 2)

    df_yearly = pd.DataFrame(result).T.sort_index()
    df_yearly = df_yearly[sorted(df_yearly.columns, key=lambda x: int(x.split("_")[1]))]
    return df_yearly


def calc_excess_nav_net(df_info: pd.DataFrame) -> pd.DataFrame:
    """计算各组扣费后超额净值曲线"""
    df = df_info[["valuation_date", "excess_net_return", "portfolio_name"]].copy()
    df_pivot = df.pivot_table(
        index="valuation_date", columns="portfolio_name",
        values="excess_net_return",
    )
    df_pivot.sort_index(inplace=True)
    df_nav = (1 + df_pivot).cumprod()
    df_nav = df_nav[sorted(df_nav.columns, key=lambda x: int(x.split("_")[1]))]
    return df_nav


def plot_excess_nav_compare(df_nav_gross: pd.DataFrame, df_nav_net: pd.DataFrame,
                            title: str, save_path: str):
    """画扣费前后超额净值对比图（只画 top 和 bottom 组）"""
    fig, ax = plt.subplots(figsize=(14, 7))
    dates = pd.to_datetime(df_nav_gross.index)

    top_col = df_nav_gross.columns[-1]
    bottom_col = df_nav_gross.columns[0]

    ax.plot(dates, df_nav_gross[top_col], color="#e74c3c", linewidth=2,
            label=f"{top_col} 扣费前")
    ax.plot(dates, df_nav_net[top_col], color="#e74c3c", linewidth=2,
            linestyle="--", label=f"{top_col} 扣费后")
    ax.plot(dates, df_nav_gross[bottom_col], color="#2ecc71", linewidth=2,
            label=f"{bottom_col} 扣费前")
    ax.plot(dates, df_nav_net[bottom_col], color="#2ecc71", linewidth=2,
            linestyle="--", label=f"{bottom_col} 扣费后")

    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("日期", fontsize=13)
    ax.set_ylabel("超额净值", fontsize=13)
    ax.legend(loc="best", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# 多空收益
# ============================================================

def calc_long_short(df_info: pd.DataFrame, n_groups: int) -> pd.DataFrame:
    """计算多空组合日收益 (top组 - bottom组)"""
    df = df_info[["valuation_date", "excess_paper_return", "portfolio_name"]].copy()
    top_name = f"group_{n_groups}"
    bottom_name = "group_1"

    df_top = df[df["portfolio_name"] == top_name][["valuation_date", "excess_paper_return"]]
    df_top = df_top.rename(columns={"excess_paper_return": "top"})
    df_bottom = df[df["portfolio_name"] == bottom_name][["valuation_date", "excess_paper_return"]]
    df_bottom = df_bottom.rename(columns={"excess_paper_return": "bottom"})

    df_ls = df_top.merge(df_bottom, on="valuation_date")
    df_ls["long_short"] = df_ls["top"] - df_ls["bottom"]
    df_ls.sort_values("valuation_date", inplace=True)
    df_ls["ls_nav"] = (1 + df_ls["long_short"]).cumprod()
    return df_ls


def plot_long_short(df_ls: pd.DataFrame, title: str, save_path: str):
    """画多空净值曲线"""
    fig, ax = plt.subplots(figsize=(14, 6))
    dates = pd.to_datetime(df_ls["valuation_date"])
    ax.plot(dates, df_ls["ls_nav"], color="#e67e22", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("日期", fontsize=13)
    ax.set_ylabel("多空净值", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ============================================================
# PDF 报告生成
# ============================================================

def generate_report(signal_name: str, start_date: str, end_date: str,
                    n_groups: int = 5,
                    df_factor: pd.DataFrame = None,
                    index_data: dict = None,
                    df_stock: pd.DataFrame = None,
                    df_index_ret: pd.DataFrame = None,
                    output_base: str = None,
                    df_calendar: pd.DataFrame = None):
    """
    生成因子分层回测 PDF 报告

    一份PDF包含所有指数的回测结果。每个指数包含:
    IC/IR分析、分层超额收益(扣费前/后)、换手率、超额净值走势、多空组合。

    Parameters
    ----------
    signal_name  : 因子名称（也用作输出目录名）
    start_date   : 回测开始日期
    end_date     : 回测结束日期
    n_groups     : 分层数量
    df_factor    : 因子数据 [valuation_date, code, score_name, final_score]
    index_data   : {index_name: df_index_comp} 指数成分数据字典
    df_stock     : 股票行情 [valuation_date, code, close, pre_close]
    df_index_ret : 指数收益率 [valuation_date, code, pct_chg]
    output_base  : 输出根目录（默认从config.yaml读取）
    df_calendar  : 交易日历（可选，加速持仓日映射）
    """
    if output_base is None:
        output_base = load_config().get("output", {}).get(
            "base_dir", os.path.join(os.path.dirname(__file__), "output")
        )

    index_list = list(index_data.keys()) if index_data else []

    output_dir = os.path.join(output_base, signal_name)
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, f"{signal_name}_分层回测报告.pdf")
    pdf = PDFCreator(pdf_path)
    pdf.title(f"<b>{signal_name} 因子分层回测报告</b>")
    pdf.text(f"回测区间: {start_date} ~ {end_date}，分组数: {n_groups}")

    for index_name in index_list:
        index_cn = INDEX_MAP.get(index_name, index_name)
        print(f"\n{'='*50}")
        print(f"正在回测: {signal_name} x {index_cn}")
        print(f"{'='*50}")

        # 运行回测
        sa = SignalAnalysis(
            signal_name=signal_name,
            index_name=index_name,
            n_groups=n_groups,
            df_factor=df_factor,
            df_index_comp=index_data[index_name],
            df_stock=df_stock,
            df_index_ret=df_index_ret,
            df_calendar=df_calendar,
        )
        df_info = sa.run()

        if not df_info.empty:
            # 过滤掉超出回测区间的溢出日期（如 next_workday(2025-12-31)=2026-01-05）
            df_info = df_info[df_info["valuation_date"].astype(str) <= end_date].copy()

        if df_info.empty:
            pdf.h1(f"<b>{index_cn}</b>")
            pdf.text("无回测数据")
            continue

        # ========== 1. IC / IR 分析 ==========
        pdf.h1(f"<b>{index_cn} - IC/IR 分析</b>")

        # 过滤 factor_merged 的溢出日期
        df_factor_merged = sa.df_factor_merged[
            sa.df_factor_merged["valuation_date"].astype(str) <= end_date
        ].copy()
        df_ic = calc_daily_ic(df_factor_merged, df_stock)
        if not df_ic.empty:
            # IC 汇总表
            ic_summary = calc_ic_summary(df_ic)
            summary_data = [["指标", "数值"]]
            for k, v in ic_summary.items():
                summary_data.append([k, str(v)])
            pdf.table(summary_data, highlight_first_row=False)

            # 年度IC表
            pdf.h2("<b>年度 IC 表现</b>")
            df_yearly_ic = calc_yearly_ic(df_ic)
            header = df_yearly_ic.columns.tolist()
            table_data = [header]
            for _, row in df_yearly_ic.iterrows():
                table_data.append([str(v) for v in row.tolist()])
            pdf.table(table_data, highlight_first_row=False)

            # IC 图
            fig_ic_path = os.path.join(output_dir, f"{signal_name}_{index_name}_ic.png")
            plot_ic(df_ic, f"{signal_name} - {index_cn} Rank IC", fig_ic_path)
            pdf.image(fig_ic_path)

        # ========== 2. 年度分层超额收益（扣费前）==========
        pdf.h1(f"<b>{index_cn} - 年度分层超额收益-扣费前(%)</b>")
        df_yearly = calc_yearly_excess(df_info)

        # 全区间汇总行
        df_all = df_info[["excess_paper_return", "portfolio_name"]].copy()
        all_row = {}
        for group, sub in df_all.groupby("portfolio_name"):
            cum_ret = (1 + sub["excess_paper_return"]).prod() - 1
            all_row[group] = round(cum_ret * 100, 2)
        df_yearly.loc["全区间"] = all_row

        header = ["year"] + df_yearly.columns.tolist()
        table_data = [header]
        for idx, row in df_yearly.iterrows():
            table_data.append([str(idx)] + [str(v) for v in row.tolist()])
        pdf.table(table_data, highlight_first_row=False)

        # ========== 2.5 年度分层超额收益（扣费后）==========
        pdf.h1(f"<b>{index_cn} - 年度分层超额收益-扣费后(%)</b>")
        pdf.text("手续费: 双边万分之8.5")
        df_yearly_net = calc_yearly_excess_net(df_info)

        df_all_net = df_info[["excess_net_return", "portfolio_name"]].copy()
        all_row_net = {}
        for group, sub in df_all_net.groupby("portfolio_name"):
            cum_ret = (1 + sub["excess_net_return"]).prod() - 1
            all_row_net[group] = round(cum_ret * 100, 2)
        df_yearly_net.loc["全区间"] = all_row_net

        header_net = ["year"] + df_yearly_net.columns.tolist()
        table_data_net = [header_net]
        for idx, row in df_yearly_net.iterrows():
            table_data_net.append([str(idx)] + [str(v) for v in row.tolist()])
        pdf.table(table_data_net, highlight_first_row=False)

        # ========== 3. 换手率统计 ==========
        tv_tables = calc_yearly_turnover(df_info)

        def _df_to_table(df_pivot, title_suffix):
            header = [""] + df_pivot.columns.tolist()
            rows = [header]
            for idx, row in df_pivot.iterrows():
                rows.append([str(idx)] + [str(v) for v in row.tolist()])
            return rows

        pdf.h1(f"<b>{index_cn} - 年化单边换手率(倍)</b>")
        pdf.table(_df_to_table(tv_tables["annualized_turnover"], "年化换手率"),
                  highlight_first_row=False)

        pdf.h1(f"<b>{index_cn} - 年化手续费(%, 双边万分之8.5)</b>")
        pdf.table(_df_to_table(tv_tables["annualized_fee"], "年化手续费"),
                  highlight_first_row=False)

        # ========== 4. 分层超额净值走势图（扣费前）==========
        pdf.h1(f"<b>{index_cn} - 分层超额净值走势（扣费前）</b>")
        df_nav = calc_excess_nav(df_info)
        fig_nav_path = os.path.join(output_dir, f"{signal_name}_{index_name}_excess_nav.png")
        plot_excess_nav(df_nav, f"{signal_name} - {index_cn} 分层超额净值（扣费前）", fig_nav_path)
        pdf.image(fig_nav_path)

        # ========== 4.5 分层超额净值走势图（扣费后）==========
        pdf.h1(f"<b>{index_cn} - 分层超额净值走势（扣费后）</b>")
        df_nav_net = calc_excess_nav_net(df_info)
        fig_nav_net_path = os.path.join(output_dir, f"{signal_name}_{index_name}_excess_nav_net.png")
        plot_excess_nav(df_nav_net, f"{signal_name} - {index_cn} 分层超额净值（扣费后）", fig_nav_net_path)
        pdf.image(fig_nav_net_path)

        # ========== 4.6 扣费前后对比图（Top vs Bottom）==========
        fig_cmp_path = os.path.join(output_dir, f"{signal_name}_{index_name}_nav_compare.png")
        plot_excess_nav_compare(df_nav, df_nav_net,
                                f"{signal_name} - {index_cn} 扣费前后超额净值对比", fig_cmp_path)
        pdf.image(fig_cmp_path)

        # ========== 5. 多空组合净值 ==========
        pdf.h1(f"<b>{index_cn} - 多空组合 (Top-Bottom)</b>")
        df_ls = calc_long_short(df_info, n_groups)
        if df_ls.empty:
            pdf.text("多空数据不足，跳过")
        else:
            ls_ret = df_ls["ls_nav"].iloc[-1] - 1
            ls_annual = ls_ret * 252 / len(df_ls)
            ls_vol = df_ls["long_short"].std() * np.sqrt(252)
            ls_sharpe = ls_annual / ls_vol if ls_vol > 0 else 0
            ls_data = [
                ["指标", "数值"],
                ["累计多空收益(%)", str(round(ls_ret * 100, 2))],
                ["年化多空收益(%)", str(round(ls_annual * 100, 2))],
                ["年化波动率(%)", str(round(ls_vol * 100, 2))],
                ["夏普比率", str(round(ls_sharpe, 2))],
            ]
            pdf.table(ls_data, highlight_first_row=False)

            fig_ls_path = os.path.join(output_dir, f"{signal_name}_{index_name}_long_short.png")
            plot_long_short(df_ls, f"{signal_name} - {index_cn} 多空净值 (group_{n_groups} - group_1)", fig_ls_path)
            pdf.image(fig_ls_path)

    pdf.build()
    print(f"\n报告已生成: {pdf_path}")
    return pdf_path


if __name__ == "__main__":
    from data_prepare import get_factor_data, get_index_component, get_market_data

    glv.init()

    start_date, end_date = "2025-01-02", "2025-12-31"
    index_list = ["hs300", "zz500", "zz1000"]

    mkt_end = gt.next_workday_calculate(end_date)
    market_data = get_market_data(start_date, mkt_end)
    df_stock, df_index_ret = market_data[0], market_data[6]
    index_data = {idx: get_index_component(start_date, end_date, idx) for idx in index_list}
    df_factor = get_factor_data(start_date, end_date, "PE")

    generate_report(
        signal_name="PE",
        start_date=start_date,
        end_date=end_date,
        n_groups=5,
        df_factor=df_factor,
        index_data=index_data,
        df_stock=df_stock,
        df_index_ret=df_index_ret,
    )
