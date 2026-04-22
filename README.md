# QuantTradingLucky

一个从零开始、渐进式学习量化交易的个人项目。定位：计算机专业程序员视角，覆盖 A 股 / 美股 / 加密货币 / 期货期权四大市场，理论与实战并重。

> ⚠️ 本项目为个人学习用途，不构成任何投资建议。

## 📘 学习计划

详细的 12 个月渐进式学习计划见：**[`docs/LEARNING_PLAN.md`](docs/LEARNING_PLAN.md)**

| 阶段 | 月份 | 主题 | 核心交付物 |
|------|------|------|------------|
| 筑基 | M1–M2 | Python 金融工程 + 市场基础 | 数据管道、EDA 报告 |
| 分析 | M3–M4 | 时间序列 + 因子投资入门 | 指标库、单因子测试框架 |
| 回测 | M5–M6 | 回测引擎 + 经典策略 | 自研回测引擎、3 个策略复现 |
| 深化 | M7–M8 | 多因子模型 + 衍生品 CTA | 多因子模型 v1、期权定价器、CTA |
| 拓展 | M9–M10 | 加密货币 + 机器学习 | 加密套利、ML 选股模型 |
| 实战 | M11–M12 | 风控 + 模拟实盘 | 风控模块、端到端系统 |

## 🚀 快速开始

需要：Python 3.11+、Git、make。推荐使用 [uv](https://github.com/astral-sh/uv) 管理依赖（Makefile 中已自动安装）。

```bash
# 一键初始化：安装 uv → 同步依赖 → 装 pre-commit hook
make bootstrap

# 配置环境变量（Tushare token、交易所 API 等）
cp .env.example .env
# 然后编辑 .env

# 验证环境
make test

# 启动 Jupyter Lab 开始研究
make notebook
```

常用命令：

| 命令 | 作用 |
|------|------|
| `make lint` | ruff 检查 |
| `make format` | ruff 自动格式化 + 修复 |
| `make typecheck` | mypy 类型检查 |
| `make test` | 跑测试 + 覆盖率 |
| `make test-fast` | 仅跑快速测试（排除 slow/integration/live） |
| `make precommit` | 在全部文件上跑 pre-commit |
| `make clean` | 清理缓存 |

## 🗂️ 项目结构

```
QuantTradingLucky/
├── docs/
│   ├── LEARNING_PLAN.md         # 12 个月学习计划（核心文件）
│   ├── pitfalls.md              # 踩坑实录
│   └── monthly-review/          # 每月复盘（模板 + 逐月归档）
├── src/quant_lucky/             # Python 包
│   ├── data/                    # 数据采集与存储
│   ├── universe/                # 标的池（point-in-time）
│   ├── costs/                   # 手续费 / 滑点 / 冲击成本
│   ├── indicators/              # 技术指标（手写 + 测试）
│   ├── factors/                 # 因子库 + 单因子测试框架
│   ├── alpha/                   # 多因子合成 / 信号
│   ├── portfolio/               # 组合优化（cvxpy）
│   ├── backtest/                # 回测引擎（向量化 + 事件驱动）
│   ├── risk/                    # 风控（VaR / 仓位 / 止损）
│   ├── derivatives/             # 期权 / 期货工具
│   ├── crypto/                  # 加密市场专属
│   ├── ml/                      # 金融 ML（含 Purged K-Fold）
│   └── utils/                   # 配置 / 日志 / IO
├── tests/                       # pytest 测试套件
├── strategies/                  # 具体策略实现
├── notebooks/                   # 研究用 Jupyter Notebook
├── reports/                     # 报告产出（Markdown 源）
├── data/                        # 本地数据缓存（大部分 gitignored）
├── pyproject.toml               # 依赖 + ruff / mypy / pytest 配置
├── .pre-commit-config.yaml      # 提交前自动化检查
├── .github/workflows/ci.yml     # CI（lint、test、precommit）
├── Makefile                     # 开发常用命令
└── .env.example                 # 环境变量模板
```

## ✅ 当前进度

- [x] 创建学习计划（`docs/LEARNING_PLAN.md`）
- [x] 初始化项目骨架（pyproject / ruff / mypy / pytest / CI / pre-commit）
- [x] M1：数据管道 v0.1（Tushare + yfinance + ccxt）
- [x] M2：市场基础与成本模型（标的池、手续费与滑点模型）
- [ ] **M3 进行中**：时间序列与技术分析（指标库构建、ARMA/GARCH、协整分析）
- [ ] M4：因子投资与 Alpha 研究入门
- [ ] M5：回测系统原理与实现
- [ ] ...

## 📝 惯例

- **提交前**：`make precommit` 必须全绿。
- **新增模块**：先写 docstring 说明职责与边界，再写实现。
- **新增依赖**：加进 `pyproject.toml` 相应分组（核心 / dev / ml / backtest）。
- **Notebook**：提交前 output 自动被 `nbstripout` 清理。
- **每月末**：复制 `docs/monthly-review/TEMPLATE.md` 写当月复盘。
