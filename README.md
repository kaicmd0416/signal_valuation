# Signal Valuation - A股多因子分层回测系统

按指数分别合成因子 + 分层回测框架，支持等权/IC加权/两层IC加权合成、因子平滑、全市场top选股、数据库入库。

## 项目结构

```
signal_valuation/
├── run_all.py                         # 统一入口（单因子回测 / 历史合成 / 单日更新）
├── README.md
├── config/                            # 配置文件
│   ├── config.yaml                    #   数据库连接 + 输出路径
│   ├── config_single_signal.yaml      #   单因子回测配置（因子列表 + 回测参数）
│   ├── config_combine_by_index_test.yaml  #   合成因子测试配置
│   ├── config_combine_by_index_prod.yaml  #   合成因子生产配置
│   └── config_db_saving.yaml          #   入库配置（表结构 + 连接）
└── core/                              # 核心模块
    ├── data_prepare.py                #   数据层（DB查询 + 配置加载 + 交易日工具）
    ├── factor_combine.py              #   因子合成（z-score + IC加权 + MA平滑）
    ├── backtest.py                    #   分层回测引擎（SignalAnalysis类）
    ├── db_writer.py                   #   数据库入库（DELETE + INSERT 事务安全）
    ├── report.py                      #   PDF/Excel报告生成
    └── PDF/
        └── PDFCreator.py              #   PDF生成工具类
```

输出目录: `D:/Signal_backtesting/output/`（prod/test 子目录分开存放）

## 快速开始

### 环境依赖

- Python 3.11+
- SQLAlchemy 2.0+
- 依赖库: `pandas`, `numpy`, `matplotlib`, `sqlalchemy`, `pymysql`, `pyyaml`, `reportlab`, `Pillow`

### 运行方式

```bash
# 单因子回测（生成 PDF + Excel 报告）
python -m run_all single                           # 配置中全部因子
python -m run_all single --factors PE PB ROA       # 指定因子
python -m run_all single --index zz500 zz1000      # 指定指数

# 因子合成 + 回测（历史模式）
python -m run_all combine --mode test              # 测试配置
python -m run_all combine --mode prod              # 生产配置
python -m run_all combine --mode prod --sql        # 生产配置 + 入库
python -m run_all combine --mode prod --no-backtest  # 只合成不回测

# 单日更新（默认 prod 模式，自动入库）
python -m run_all update                           # 自动决定日期
python -m run_all update 2026-03-16                # 指定目标日期
python -m run_all update --mode test               # 测试模式（入库到 test 表）
python -m run_all update --no-sql                  # 不入库
```

```python
# 作为模块导入
from run_all import run_combine_history, run_combine_update
results = run_combine_history(index_list=["hs300"], mode="prod", backtest=False)
df_today = run_combine_update("2026-03-16", mode="prod")
```

## 核心模块说明

### core/data_prepare.py - 数据层

所有外部数据获取 + 交易日工具，数据源为阿里云 RDS MySQL。
Engine 带模块级缓存，每个 db_key 只创建一次连接池。

| 函数 | 说明 |
|------|------|
| `get_factor_data()` | 获取单个因子数据（表名从 config.yaml 的 sheet_name 读取） |
| `get_factor_data_batch()` | 批量获取多个因子（单次 SQL，IN 查询） |
| `get_index_component()` | 获取指数成分股及权重 |
| `get_trading_calendar()` | 交易日历（含 next_workday 映射） |
| `get_market_data()` | 股票/指数行情数据 |
| `get_st_stocks()` | ST 股票列表 |
| `get_notrade_stocks()` | 涨跌停股票列表 |
| `next_workday(date)` | 取下一个交易日（bisect 二分查找，找不到返回 None） |
| `last_workday(date)` | 取上一个交易日（bisect 二分查找，找不到返回 None） |

### core/factor_combine.py - 因子合成

核心函数 `combine_factors_for_index()`，支持三种合成方式：

- **equal** — 所有因子等权平均
- **ic_weight** — 所有因子单层 IC 加权
- **two_level_ic** — 簇内 IC 加权 -> 簇间 IC 加权

流程：
1. 批量加载因子 -> 成分股内 z-score 截面标准化
2. 方向处理（direction: 1=正向, -1=反向）
3. IC 加权合成（滚动窗口 60 天）或等权合成
4. 时序平滑 MA（降低换手率，smooth_window=1 时不平滑）
5. 可选: 全市场评分（用成分股 IC 权重对全市场 z-score 加权）-> top 选股

### core/backtest.py - 分层回测引擎

`SignalAnalysis` 类：因子 x 指数成分取交集 -> 按因子值分 N 组 -> 计算组合收益/换手率/手续费/超额收益

- `group_1` = 因子值最低组，`group_N` = 因子值最高组
- 组内按成分股权重归一化加权
- 换手率: L1 距离 / 2（单边换手率）
- 手续费: 双边万分之 8.5（单边换手率 x 0.00085）

### core/db_writer.py - 数据库入库

`save_combine_score()` 函数：
- DELETE + INSERT 在同一个数据库事务中执行，失败自动回滚，防止数据丢失
- 支持多个 score_name 的批量 DELETE
- 表名经过正则白名单校验，防止 SQL 注入

### core/report.py - 报告生成

一份 PDF/Excel 报告包含：
- IC/IR 分析（Rank IC、年度 IC、累计 IC 曲线）
- 分层超额收益统计（扣费前/后，年化使用复合年化 CAGR）
- 换手率与手续费统计
- 分层超额净值走势图
- 多空组合净值（Top 组 - Bottom 组）
- Top 组合回测（全市场打分最高的 top N 等权组合）

### run_all.py - 统一入口

三大功能：

| 函数 | 用途 | 返回值 |
|------|------|--------|
| `run_single_backtest()` | 单因子分层回测 | None（生成报告文件） |
| `run_combine_history()` | 历史区间因子合成+回测 | `{index: DataFrame}` |
| `run_combine_update()` | 单日因子合成更新（带完整性检查） | `{index: DataFrame}` |

DataFrame 统一格式: `[valuation_date, code, score_name, final_score]`

## 配置说明

### config/config.yaml - 基础配置

```yaml
database:
  factor:                  # 因子库
    host: ...
    database: data_prepared_hg
    sheet_name: data_score  # 因子表名（可配置，代码通过此字段读取）
  market:                  # 行情库
    host: ...
    database: data_prepared_new

output:
  base_dir: "D:/Signal_backtesting/output"
```

### config/config_combine_by_index_{test,prod}.yaml - 合成因子配置

test/prod 双模式，分别对应不同的因子组合和参数：

```yaml
backtest:
  start_date: "2023-07-01"
  end_date: "2026-03-13"
  n_groups: 5              # 分层数
  stock_number: 500        # 固定维度输出的股票数

weight_method: "equal"     # 合成方法: equal / ic_weight / two_level_ic
ic_window: 60              # IC 加权滚动窗口（天）
smooth_window: 1           # 因子平滑 MA 窗口（1=不平滑）
date_mode: "available_date" # 日期模式

indices:
  hs300:
    output_name: "fm04_hs300"
    factors:                # 平铺格式（等权时使用）
      - name: "ATER"
        direction: 1
      - name: "PE"
        direction: 1
      - name: "ROA"
        direction: 1
  # 也支持 clusters 格式（两层IC加权时使用）:
  # zz500:
  #   clusters:
  #     估值:
  #       factors:
  #         - name: "PB"
  #           direction: 1
```

### config/config_db_saving.yaml - 入库配置

- **CombineScoreProd**: 写入 `data_score` 表（prod 模式）
- **CombineScoreTest**: 写入 `combine_score_test` 表（test 模式）

读因子用 `kai` 用户（只读），写入库用 `prod` 用户（读写分离）。

## 日期模式说明

- **available_date**（推荐）: DB 中因子日期 = 因子可用日（基于 T-1 收盘数据计算，T 日盘前可得），持仓日 = next_workday(available_date)。时间链: T-1 收盘 -> T 日因子可用 -> T+1 日持仓交易。
- **target_date**: DB 中因子日期直接作为持仓日，内部自动映射回前一交易日做 z-score/IC 计算。

## 数据库表

| 库 | 表 | 用途 |
|----|-----|------|
| data_prepared_hg | data_score | 原始因子数据（读取） |
| data_prepared_new | data_stock | 股票行情 |
| data_prepared_new | data_index | 指数行情 |
| data_prepared_new | data_indexcomponent | 指数成分股 |
| data_prepared_new | chinesevaluationdate | 交易日历 |
| data_prepared_new | st_stock | ST 股票 |
| data_prepared_new | data_stocknotrade | 涨跌停股票 |
| data_prepared_new | data_score | 合成因子打分（prod 入库） |
| data_prepared_new | combine_score_test | 合成因子打分（test 入库） |

## 注意事项

- **vp08 和 a3 是异常数据，禁止参与任何计算和回测**
- combine 命令入库需要显式加 `--sql` 参数；update 命令默认入库，用 `--no-sql` 禁用
- 年化收益使用复合年化公式: `(1 + 累计收益) ^ (252 / 交易天数) - 1`
- 入库采用事务安全的 DELETE + INSERT 模式，失败自动回滚
