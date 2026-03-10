# Signal Backtesting - 因子分层回测系统

A股多因子分层回测框架，支持单因子回测与等权合成因子回测。

## 项目结构

```
Signal_backtesting/
├── config.yaml            # 基础配置（数据库连接、输出路径、回测参数）
├── config_combine.yaml    # 因子合成配置（因子列表、方向、合成方式）
├── config_signals.yaml    # 单因子回测配置（因子列表）
├── data_prepare.py        # 数据准备模块（DB查询 + 配置加载）
├── backtest.py            # 分层回测核心引擎
├── report.py              # PDF报告生成模块
├── factor_combine.py      # 因子合成模块（z-score + 方向处理 + 等权合成）
├── run.py                 # 全量回测入口（单因子 + 合成因子）
├── run_combine.py         # 合成因子回测入口（仅合成因子，快速迭代）
├── factor_evaluation.md   # 因子评估报告
└── output/                # 回测输出目录（PDF报告 + 图表）
```

## 快速开始

### 环境依赖

- Python 3.11+
- 依赖库: `pandas`, `numpy`, `matplotlib`, `scipy`, `sqlalchemy`, `pymysql`, `pyyaml`
- 内部工具库: `global_tools`, `global_dic`, `PDFCreator`（通过环境变量 `GLOBAL_TOOLSFUNC_new` 引用）

### 运行方式

```bash
# 仅运行合成因子回测（推荐，快速迭代）
python run_combine.py

# 全量回测（25个单因子 + 合成因子）
python run.py

# 单独运行因子合成（不回测，仅输出合成结果）
python factor_combine.py
```

## 核心模块说明

### data_prepare.py - 数据准备

配置加载与远程 MySQL (阿里云 RDS) 数据获取：
- `load_config()` — 加载 config.yaml（数据库 + 输出路径 + 回测参数）
- `load_combine_config()` — 加载 config_combine.yaml（因子合成配置）
- `load_signals_config()` — 加载 config_signals.yaml（单因子回测列表）
- `get_factor_data()` — 获取单个因子数据
- `get_factor_data_batch()` — 批量获取多个因子（单次 SQL，`WHERE score_name IN (...)`）
- `get_index_component()` — 获取指数成分股及权重
- `get_trading_calendar()` — 交易日历（含 next_workday 映射）
- `get_market_data()` — 股票/指数行情数据

### backtest.py - 分层回测引擎

`SignalAnalysis` 类实现完整回测流程：
1. **构建持仓** — 因子 × 指数成分取交集 → 按因子值分位数分 N 组 → 组内按指数权重归一化
2. **计算换手率** — 向量化计算每日单边换手率（含退出股票处理）
3. **计算收益** — 组合日收益、扣费收益、指数收益、超额收益

分层逻辑：
- `group_1` = 因子值最低组，`group_N` = 因子值最高组
- 正向因子：`group_N` 应跑赢 `group_1`
- 反向因子：`group_1` 应跑赢 `group_N`
- 手续费：双边万分之 8.5

### factor_combine.py - 因子合成

等权合成全市场因子的流程：
1. **批量 SQL** 加载所有因子原始数据（单次查询）
2. **截面 z-score** 标准化（逐因子、逐交易日）
3. **方向处理** — 正向因子 ×1，反向因子 ×(-1)
   - 支持 `direction_override`：同一因子在不同指数成分股上使用不同方向
   - 例如 RTN1 在沪深300成分股上为正向（动量），其余股票为反向（反转）
4. **等权合成** — 所有因子取均值，输出一个全市场合成因子

### report.py - 报告生成

一份 PDF 包含所有指数（沪深300/中证500/中证1000/中证2000）的回测结果：
- IC/IR 分析（截面 Rank IC、年度 IC 表现、累计 IC 曲线）
- 年度分层超额收益（扣费前 / 扣费后）
- 年化换手率与手续费统计
- 分层超额净值走势图（扣费前 / 扣费后 / 对比图）
- 多空组合净值（Top 组 - Bottom 组）

## 配置说明

配置拆分为 3 个文件，职责分明：

### config.yaml — 基础配置

```yaml
database:
  factor:    # 因子库（阿里云 RDS）
  market:    # 行情库（阿里云 RDS）

output:
  base_dir: "D:/Signal_backtesting/output"

backtest:
  start_date: "2023-06-30"
  end_date: "2025-12-31"
  n_groups: 5
  index_list: ["hs300", "zz500", "zz1000", "zz2000"]
```

### config_combine.yaml — 因子合成配置

```yaml
method: "equal_weight"
output_name: "combine_1"
factors:
  - name: "turnoverRateAvg20d"
    direction: 1              # 1=正向, -1=反向
  - name: "RTN1"
    direction: -1             # 默认反向（中小盘反转）
    direction_override:
      hs300: 1                # 沪深300成分股上为正向（动量）
```

### config_signals.yaml — 单因子回测配置

```yaml
signals:
  - "PE"
  - "PB"
  - "ROE"
  # ... 修改此列表控制 run.py 跑哪些因子
```

### 因子方向说明

- `direction: 1`（正向）：因子值越大 → 预期收益越高
- `direction: -1`（反向）：因子值越小 → 预期收益越高
- `direction_override`：覆盖特定指数成分股的方向。非覆盖指数的股票使用默认方向；未出现在任何指数中的股票也使用默认方向

## 当前合成因子（combine_1）

包含 13 个评分 ≥ 5 的因子：

| 类别 | 因子 | 方向 | 评分 |
|------|------|------|------|
| 流动性 | turnoverRateAvg20d | 正向 | 10 |
| 流动性 | turnoverRateAvg120d | 正向 | 8 |
| 流动性 | ILLIQ | 反向 | 7 |
| 估值 | PB | 正向 | 8 |
| 估值 | PCF | 正向 | 7 |
| 估值 | PE | 正向 | 6 |
| 盈利质量 | OCFPR | 正向 | 6 |
| 盈利质量 | OCFTD | 正向 | 5 |
| 盈利质量 | GPG | 正向 | 5 |
| 动量/反转 | RTN1 | 反向(hs300正向) | 7 |
| 动量/反转 | RTN3 | 反向(hs300正向) | 6 |
| 动量/反转 | RTN6 | 反向(hs300正向) | 6 |
| 市值 | LnFloatCap | 反向 | 5 |
