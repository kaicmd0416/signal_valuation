# Signal Valuation - A股多因子分层回测系统

按指数分别合成因子 + 分层回测框架，支持两层IC加权合成、因子平滑、全市场top选股。

## 项目结构

```
signal_valuation/
├── run_all.py                         # 统一入口（单因子回测 / 历史合成 / 单日更新）
├── README.md
├── config/                            # 配置文件
│   ├── config.yaml                    #   数据库连接 + 输出路径 + 回测参数
│   ├── config_combine_by_index.yaml   #   按指数因子合成配置（簇/方向/IC窗口）
│   └── config_signals.yaml            #   单因子回测列表
└── core/                              # 核心模块
    ├── data_prepare.py                #   数据层（DB查询 + 配置加载 + 交易日工具）
    ├── factor_combine.py              #   因子合成（z-score + IC加权 + MA平滑）
    ├── backtest.py                    #   分层回测引擎（SignalAnalysis类）
    ├── report.py                      #   PDF/Excel报告生成
    ├── test_query_speed.py            #   DB查询性能测试
    └── PDF/
        └── PDFCreator.py              #   PDF生成工具类
```

输出目录: `D:/Signal_backtesting/output/`

## 快速开始

### 环境依赖

- Python 3.11+
- 依赖库: `pandas`, `numpy`, `matplotlib`, `sqlalchemy`, `pymysql`, `pyyaml`, `reportlab`, `Pillow`

### 运行方式

```bash
# 单因子回测
python run_all.py single                           # 全部因子
python run_all.py single --factors PE PB ROA       # 指定因子
python run_all.py single --index zz500 zz1000      # 指定指数

# 因子合成 + 回测
python run_all.py combine                          # 全部指数合成+回测
python run_all.py combine --index hs300            # 只跑沪深300
python run_all.py combine --no-backtest            # 只合成不回测，返回DataFrame

# 单日更新（生产用）
python run_all.py update 2026-03-10                # 全部指数
python run_all.py update 2026-03-10 --index zz500  # 指定指数
```

```python
# 作为模块导入
from run_all import run_combine_history, run_combine_update
results = run_combine_history(index_list=["zz500"], backtest=False)
df_today = run_combine_update("2026-03-10")
```

## 核心模块说明

### core/data_prepare.py - 数据层

所有外部数据获取 + 交易日工具，数据源为阿里云 RDS MySQL：

| 函数 | 说明 |
|------|------|
| `get_factor_data()` | 获取单个因子数据 |
| `get_factor_data_batch()` | 批量获取多个因子（单次SQL） |
| `get_index_component()` | 获取指数成分股及权重 |
| `get_trading_calendar()` | 交易日历（含 next_workday 映射） |
| `get_market_data()` | 股票/指数行情数据 |
| `get_st_stocks()` | ST股票列表 |
| `get_notrade_stocks()` | 涨跌停股票列表 |
| `next_workday(date)` | 取下一个交易日 |
| `last_workday(date)` | 取上一个交易日 |

### core/factor_combine.py - 因子合成

核心函数 `combine_factors_for_index()`，支持三种合成方式：

- **equal** — 所有因子等权
- **ic_weight** — 所有因子单层IC加权
- **two_level_ic** — 簇内IC加权 → 簇间IC加权（推荐）

流程：
1. 批量加载因子 → 成分股内 z-score 标准化
2. 方向处理（direction: 1=正向, -1=反向）
3. IC加权合成（滚动窗口, shift(1)避免前瞻偏差）
4. 时序平滑 MA（降低换手率）
5. 可选: 全市场评分（用成分股IC权重）→ top选股

### core/backtest.py - 分层回测引擎

`SignalAnalysis` 类：因子 × 指数成分取交集 → 分N组 → 计算组合收益/换手率/手续费/超额收益

- `group_1` = 因子值最低组，`group_N` = 因子值最高组
- 手续费: 双边万分之8.5

### core/report.py - 报告生成

一份 PDF/Excel 报告包含：
- IC/IR 分析（Rank IC、年度IC、累计IC曲线）
- 分层超额收益（扣费前/后）
- 换手率与手续费统计
- 分层超额净值走势图
- 多空组合净值（Top组 - Bottom组）
- Top组合回测（全市场打分最高的top N等权组合）

### run_all.py - 统一入口

三大功能：

| 函数 | 用途 | 返回值 |
|------|------|--------|
| `run_single_backtest()` | 单因子分层回测 | None（生成报告文件） |
| `run_combine_history()` | 历史区间因子合成 | `{index: DataFrame}` |
| `run_combine_update()` | 单日因子合成更新 | `{index: DataFrame}` |

DataFrame 统一格式: `[valuation_date, code, score_name, final_score]`

## 配置说明

### config/config.yaml — 基础配置

```yaml
database:
  factor:    # 因子库 (data_prepared_hg)
  market:    # 行情库 (data_prepared_new)

output:
  base_dir: "D:/Signal_backtesting/output"

backtest:
  start_date: "2023-07-01"
  end_date: "2026-03-10"
  n_groups: 5
  index_list: ["hs300", "zz500", "zz1000", "zz2000"]
```

### config/config_combine_by_index.yaml — 按指数因子合成配置

```yaml
weight_method: "two_level_ic"
ic_window: 60          # IC加权滚动窗口
smooth_window: 5       # 因子平滑 MA5
date_mode: "available_date"

indices:
  hs300:
    output_name: "combine_hs300_available"
    clusters:
      估值:
        factors:
          - name: "PB"
            direction: 1
          - name: "PCF"
            direction: 1
          # ...
      换手率:
        factors:
          - name: "turnoverRateAvg20d"
            direction: 1
          # ...
  zz500:
    # ...
  zz1000:
    # ...
```

**簇定义原则**: 簇内因子相关性 > 0.5，簇内至少一个因子 |IR| >= 0.10

### config/config_signals.yaml — 单因子回测配置

```yaml
signals:
  - "turnoverRateAvg20d"
  - "PB"
  - "PE"
  # ... 修改此列表控制 run_all.py single 跑哪些因子
```

## 日期模式说明

- **available_date**: DB中因子日期 = T-1（信号可用日），映射到 T（持仓日）再做回测。这是标准模式。
- **target_date**: DB中因子日期直接作为持仓日。

## 数据库表

| 库 | 表 | 用途 |
|----|-----|------|
| data_prepared_hg | data_score | 因子数据 |
| data_prepared_new | data_stock | 股票行情 |
| data_prepared_new | data_index | 指数行情 |
| data_prepared_new | data_indexcomponent | 指数成分股 |
| data_prepared_new | chinesevaluationdate | 交易日历 |
| data_prepared_new | st_stock | ST股票 |
| data_prepared_new | data_stocknotrade | 涨跌停股票 |
