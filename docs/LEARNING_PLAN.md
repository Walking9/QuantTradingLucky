# 量化交易 12 个月渐进式学习计划

> 起点：有计算机专业背景的普通程序员（熟悉 Python 或其他通用语言、Git、基本算法数据结构），金融与量化几乎零基础。
> 终点：能独立完成「数据→因子→策略→回测→风控→复盘」全流程，产出可解释、可复现的策略报告。
> 总时长：约 12 个月，每周 10–15 小时（理论 40% + 编码实战 60%）。
> 覆盖市场：A 股（主战场：多因子/中低频）、美股（对标国际方法论与 ETF 轮动）、加密货币（24/7 高频/套利）、期货期权（衍生品与 CTA）。

---

## 0. 学习方法论（开篇必读）

### 0.1 指导原则

1. **先广后深，螺旋上升**：第一轮覆盖全链路基础；第二轮选定一个方向深入（多因子 / CTA / 统计套利 / 加密做市）。
2. **以可交付物为导向**：每月必须产出一个可运行、可复现、可被他人读懂的代码仓库或 Notebook。
3. **永远怀疑自己的回测**：幸存者偏差、前视偏差、过拟合是三大杀手，贯穿始终。
4. **读论文、读源码、读复盘**：比看课程更重要。优先级：实盘复盘 > 经典论文 > 书籍 > 视频课程。
5. **先追求可解释，再追求高收益**：无法解释的 Alpha 上实盘 = 自杀。
6. **工程先行**：量化 80% 的工作是数据与工程，20% 才是模型。前期就把数据管道、版本管理、可复现性做好。

### 0.2 学习节奏建议

- **工作日**：每晚 1–2 小时，以理论、论文、代码精读为主。
- **周末**：集中 4–6 小时编码、回测、写交付物。
- **每月末**：1 天复盘 + 写月报（上传到 `docs/monthly-review/YYYY-MM.md`）。
- **每季度**：1 周 Buffer，用于补课、整理、休息，避免累积债务。

### 0.3 评估标准

每月结束时用以下三问自检：
1. 我能**白板讲清楚**本月核心概念吗？（给不懂量化的同事讲 15 分钟）
2. 我的代码**别人能 clone 下来一键复现**吗？
3. 我能**定位到至少一个自己踩过的坑**并写进 `docs/pitfalls.md` 吗？

---

## 1. 总体路线图

```
阶段           月份    主题                          市场聚焦         关键交付物
────────────────────────────────────────────────────────────────────────────────
阶段一 筑基     M1      Python 金融工程 + 数学统计    A股 + 美股       数据管道 v0.1
              M2      金融市场与品种基础知识         A股 + 美股       EDA 报告集
阶段二 分析     M3      时间序列与技术分析            A股             技术指标库
              M4      因子投资与Alpha研究入门        A股              单因子测试框架
阶段三 回测     M5      回测系统原理与实现            A股 + 美股       自研回测引擎 MVP
              M6      经典策略复现与评价             美股 + A股       3 个经典策略复现
阶段四 深化     M7      多因子模型与组合优化          A股             多因子模型 v1
              M8      衍生品定价与CTA策略           期货期权          期权定价器 + CTA 策略
阶段五 拓展     M9      加密货币与另类市场            加密货币         加密套利策略
              M10     机器学习在量化中的应用         A股+加密          ML 选股/择时模型
阶段六 实战     M11     风险管理与组合管理            全市场           风控模块 + 组合报告
              M12     综合项目 + 模拟实盘            全市场           端到端量化系统
```

---

## 2. 每月详细计划

### 📅 M1：Python 金融工程与数学统计基础

**目标**：把量化所需的语言、库、数学工具打扎实，能独立获取并清洗金融数据。

**理论学习（约 15h）**
- Python 进阶：装饰器、生成器、类型注解、`asyncio` 基本用法。
- 科学计算栈：`NumPy`（向量化思维）、`Pandas`（时间序列索引、重采样、多级索引）、`Matplotlib` / `Plotly`。
- 数学复习：描述统计、概率分布、假设检验、相关/协方差、线性回归。
- 金融数据类型：OHLCV、分红除权、停牌、复权因子（前复权 vs 后复权）。

**实战任务（约 25h）**
1. 搭建开发环境：`uv` 或 `poetry` 管理依赖、`pre-commit`（black、ruff、mypy）、`pytest` 基础。
2. 数据源调研与对接：
   - A 股：Tushare Pro / AKShare。
   - 美股：yfinance / Alpha Vantage / Polygon（免费档）。
   - 加密：`ccxt` 拉取 Binance/OKX 历史 K 线。
3. 实现统一的数据下载器 `src/data/downloader.py`，落盘为 Parquet，按 `市场/品种/频率` 分区。
4. EDA：对沪深 300、标普 500、BTC 分别做收益率分布、波动率、相关性可视化。

**交付物**
- `src/data/` 数据管道 v0.1（A 股日频 + 美股日频 + 加密小时级）。
- `notebooks/M01_eda.ipynb`：三个市场的 EDA 对比报告。
- `docs/monthly-review/2026-05.md`：月报。

**推荐资源**
- 📖 《Python for Data Analysis, 3rd》— Wes McKinney
- 📖 《概率论与数理统计》浙大第五版（复习用）
- 🎥 StatQuest with Josh Starmer（YouTube）— 统计直觉
- ⚠️ **坑点**：A 股复权因子错误是第一大新手坑；务必统一用前复权或原始价+复权因子。

---

### 📅 M2：金融市场与品种基础知识

**目标**：理解四大市场的交易机制、费用结构、监管差异，建立「金融直觉」。

**理论学习（约 20h）**
- 股票：交易机制（集合竞价、连续竞价）、涨跌停、T+1 vs T+0、融资融券。
- ETF / 指数 / 行业分类（GICS、申万）。
- 期货：合约乘数、保证金、交割、主力合约与换月、展期收益。
- 期权：看涨/看跌、内在价值、时间价值、Greeks（先感性认识）。
- 加密：现货、合约（U 本位 / 币本位）、资金费率、链上 vs 中心化。
- 费用模型：佣金、印花税、过户费、点差、滑点、冲击成本。

**实战任务（约 20h）**
1. 在数据管道中加入「手续费 + 滑点」成本模型（可配置）。
2. 写一个「回测成本计算器」，输入换手率与策略频率，输出年化成本消耗。
3. 用 `src/universe.py` 定义标的池构建器（沪深 300、标普 500、BTC/ETH + Top 20 币种）。
4. 阅读并整理：A 股涨跌停对策略的影响、美股盘前盘后的数据处理、加密交易所 API 差异。

**交付物**
- `src/universe/` 标的池模块。
- `src/costs/` 成本模型模块。
- `docs/markets-101.md`：个人总结的市场对比 Cheatsheet。

**推荐资源**
- 📖 《Options, Futures, and Other Derivatives》— John Hull（第 1–5 章）
- 📖 《一本书读懂中国股市》入门级普及
- 🌐 [Investopedia](https://www.investopedia.com/) 查术语
- ⚠️ **坑点**：期货主力合约切换若没处理，曲线会出现假跳空；加密资金费率若不计入会严重高估收益。

---

### 📅 M3：时间序列分析与技术分析

**目标**：掌握时序数据的分析与建模工具，能实现常见技术指标并理解其统计含义。

**理论学习（约 18h）**
- 平稳性与单位根检验（ADF）、差分、对数收益率。
- ACF / PACF、ARMA / ARIMA、GARCH（波动率建模）。
- 协整与配对交易理论（Engle-Granger）。
- 技术指标的数学本质：MA（低通滤波）、MACD（差分）、RSI（动量归一化）、Bollinger Bands（±σ 通道）、ATR（波动率）。
- 区分：**技术指标 ≠ 因子**；技术指标是特征，因子需经过有效性检验。

**实战任务（约 22h）**
1. 用 `statsmodels` 做一次完整的时序建模：检验 → 拟合 → 残差分析 → 预测。
2. 自己实现（不调库）10 个常见技术指标 `src/indicators/`，并写单元测试与 TA-Lib 对比。
3. 协整检验：在美股中找一对真协整的股票（如 KO/PEP），做配对交易可视化。
4. 波动率建模：用 GARCH(1,1) 拟合标普 500，预测未来 20 日波动率。

**交付物**
- `src/indicators/` 技术指标库（含测试覆盖率 ≥ 80%）。
- `notebooks/M03_timeseries.ipynb`：ARIMA / GARCH / 协整完整案例。

**推荐资源**
- 📖 《Analysis of Financial Time Series, 3rd》— Ruey Tsay（挑第 1, 2, 3, 8 章）
- 📖 《Python for Finance, 2nd》— Yves Hilpisch
- 📄 论文：Engle & Granger (1987), Cointegration and Error Correction
- ⚠️ **坑点**：不要直接对价格序列做回归，要先取对数收益率或差分。

---

### 📅 M4：因子投资与 Alpha 研究入门

**目标**：理解「因子」作为 Alpha 研究的核心抽象，掌握单因子测试标准流程。

**理论学习（约 18h）**
- CAPM → Fama-French 3 因子 → 5 因子 → Barra 多因子框架。
- 因子分类：价值、动量、质量、低波、成长、规模。
- 因子评价指标：IC（信息系数）、IR、分组收益、多空收益、换手率、衰减。
- 中性化：行业中性、市值中性、Barra 风险中性。
- 横截面回归 vs 时间序列回归。

**实战任务（约 22h）**
1. 构造至少 5 个单因子（估值 PE/PB、动量 20d/60d、波动率、换手率、质量 ROE）。
2. 搭建单因子测试框架 `src/factors/`，输入因子值，输出：
   - IC 时序 + IC 均值 / ICIR
   - 5 分组累计收益曲线
   - 多空年化 + 最大回撤
   - 行业/市值分布
3. 用 `alphalens`（或自研等价物）在 A 股沪深 300 上跑出完整因子报告。
4. 用 Barra 风格因子做回归，拆解一只基金的风险暴露。

**交付物**
- `src/factors/tester.py` 单因子测试框架。
- `reports/factors/`：至少 5 份因子报告（PDF 或 HTML）。

**推荐资源**
- 📖 《Quantitative Equity Portfolio Management》— Chincarini & Kim
- 📖 《主动投资组合管理》— Grinold & Kahn（经典，难但值得啃）
- 📄 Fama & French (1993, 2015)
- 📄 《Barra Risk Model Handbook》
- 🌐 [WorldQuant Alphas 101](https://arxiv.org/abs/1601.00991) — 101 个公式 Alpha
- ⚠️ **坑点**：因子值要做截面标准化 + 去极值，否则 IC 会被异常值污染。

---

### 📅 M5：回测系统原理与实现

**目标**：从 0 实现一个向量化回测引擎，理解回测的每一个细节和陷阱。

**理论学习（约 15h）**
- 回测两大范式：**向量化**（快、适合因子）vs **事件驱动**（真实、适合订单簿与实盘）。
- 偏差陷阱：幸存者偏差、前视偏差（look-ahead bias）、数据窥探、过拟合、生存偏差、未来函数。
- 成交撮合：T+1 / T+0、涨跌停撮合、流动性约束、滑点建模。
- 绩效指标：年化收益、年化波动、Sharpe、Sortino、Calmar、MDD、胜率、盈亏比、换手率。
- 业绩归因：Brinson 归因、因子归因。

**实战任务（约 25h）**
1. 从零实现向量化回测引擎 `src/backtest/vector_engine.py`，支持：
   - 调仓频率可配置（日/周/月）
   - 持仓权重计算
   - 手续费 + 滑点 + 冲击成本
   - 绩效报告自动生成
2. 对比自研引擎与 `vectorbt` / `backtrader` 在同一策略上的结果，分析差异。
3. 专门写一个「陷阱演示」Notebook：演示前视偏差会如何把一个噪声策略算出 Sharpe 3.0。

**交付物**
- `src/backtest/` MVP（含 ≥ 20 个单元测试）。
- `notebooks/M05_backtest_traps.ipynb`：回测陷阱演示。

**推荐资源**
- 📖 《Advances in Financial Machine Learning》— Marcos López de Prado（Ch.1–7 回测陷阱必读）
- 📖 《Algorithmic Trading》— Ernest Chan
- 🌐 [Quantopian Lectures](https://github.com/quantopian/research_public)（虽停运但资料仍是金矿）
- 🛠️ 参考实现：`vectorbt`、`backtrader`、`zipline-reloaded`
- ⚠️ **坑点**：Sharpe 比率 > 3 的日频策略，99% 是回测有 bug；优先怀疑自己的代码。

---

### 📅 M6：经典策略复现与评价

**目标**：复现 3 个不同风格的经典策略，建立「把论文/书本变成代码」的肌肉记忆。

**理论学习（约 12h）**
- 动量策略（Jegadeesh & Titman 1993）
- 均值回归 / 配对交易
- 双均线 / Turtle Trading（趋势跟踪）
- 行业/ETF 轮动（相对强度）

**实战任务（约 28h）**
1. 策略 A：美股动量策略（Cross-Sectional Momentum）。标普 500 成分股，月度调仓买入过去 12–1 月收益前 20%。
2. 策略 B：A 股双均线 + 波动率过滤（趋势类）。
3. 策略 C：BTC/ETH + 美股 SPY 的跨市场风险平价组合。
4. 每个策略产出标准报告：策略描述、假设、回测、稳健性测试（参数敏感性、不同区间）、归因、局限性。

**交付物**
- `strategies/momentum_us/`、`strategies/dual_ma_a/`、`strategies/risk_parity_multi/`
- `reports/strategies/`：3 份完整的策略研究报告。

**推荐资源**
- 📄 Jegadeesh & Titman (1993) Returns to Buying Winners…
- 📖 《Trading Systems and Methods, 6th》— Perry Kaufman
- ⚠️ **坑点**：参数调优不要用全样本；强制做样本外测试（IS/OOS 切分或 Walk-Forward）。

---

### 📅 M7：多因子模型与组合优化

**目标**：把多个单因子合成为组合，学会用优化器构建投资组合。

**理论学习（约 18h）**
- 因子合成：等权、IC 加权、最大化 ICIR、Lasso / Ridge。
- 组合优化：均值-方差（Markowitz）、Black-Litterman、风险平价、最大分散化。
- 约束：行业/风格中性、换手率约束、单票权重上限、做空限制（A 股无融券时）。
- 凸优化工具：`cvxpy`。

**实战任务（约 22h）**
1. 扩充因子库到 ≥ 15 个因子，做因子相关性与冗余分析（正交化或 PCA）。
2. 实现多因子打分模型 `src/alpha/multi_factor.py`。
3. 用 `cvxpy` 实现组合优化器 `src/portfolio/optimizer.py`，支持：
   - 均值方差 / 风险平价 / 最大化 IR
   - 行业中性约束
   - 换手率惩罚
4. 在沪深 300 上跑一次完整的多因子选股策略，对比不同优化目标的表现。

**交付物**
- `src/alpha/` 多因子 v1
- `src/portfolio/` 优化器
- `reports/multifactor_v1.md`：多因子模型研究报告

**推荐资源**
- 📖 《Active Portfolio Management》— Grinold & Kahn（重啃）
- 📖 《Machine Learning for Asset Managers》— López de Prado
- 🌐 [Robust Portfolio Optimization with CVXPY](https://www.cvxpy.org/examples/)
- ⚠️ **坑点**：协方差矩阵不可逆/病态，要做 Ledoit-Wolf 收缩或用因子协方差。

---

### 📅 M8：衍生品定价与 CTA 策略

**目标**：理解期权定价与对冲，实现一个期货 CTA 策略。

**理论学习（约 20h）**
- 期权：Black-Scholes 推导、Greeks（Delta、Gamma、Vega、Theta、Rho）、隐含波动率、波动率微笑。
- 期权策略：Covered Call、Protective Put、Straddle、Iron Condor。
- 期货：展期收益（Roll Yield）、Contango/Backwardation、CTA 的三大流派（趋势、反转、套利）。
- 统计套利：跨品种（豆油-棕榈油）、跨期（近月-远月）。

**实战任务（约 20h）**
1. 实现 BS 定价器 `src/derivatives/bs.py` + 蒙特卡洛定价器做对比。
2. 实现 Greeks 计算，并做一次 Delta 中性对冲的回测演示（不需要真实成交）。
3. 实现一个 CTA 趋势策略：海龟交易法则变种，跑在商品期货连续合约上（铁矿、沪铜、螺纹），处理主力合约切换。
4. 实现一个跨期套利策略：沪深 300 股指期货近远月价差回归。

**交付物**
- `src/derivatives/` 定价 + Greeks
- `strategies/cta_turtle/`、`strategies/arb_calendar/`
- `reports/cta_report.md`

**推荐资源**
- 📖 《Options, Futures, and Other Derivatives》— John Hull（重啃 Ch.10–19）
- 📖 《Volatility Trading》— Euan Sinclair
- 🌐 [QuantLib-Python](https://www.quantlib.org/) 学习工业级实现
- ⚠️ **坑点**：期货回测必须用「连续合约」并正确展期；不要直接拼接主力。

---

### 📅 M9：加密货币与另类市场

**目标**：理解加密市场的独特性，实现一个低风险的套利策略。

**理论学习（约 12h）**
- CEX vs DEX、订单簿机制、做市商、资金费率。
- 加密因子：链上数据（活跃地址、算力、MVRV）、资金费率、交易所净流量。
- 套利类型：搬砖（跨交易所）、三角套利、资金费率套利、现期套利。
- 稳定币与风险（USDT/USDC 脱锚风险）。

**实战任务（约 28h）**
1. 用 `ccxt` 实现多交易所（Binance + OKX + Bybit）行情聚合与订单簿采集（限频、异步）。
2. 实现资金费率套利策略（U 本位合约做空 + 现货多头，吃资金费），回测 + 理论收益测算。
3. 实现跨交易所搬砖价差监控（不必实盘，只记录机会）。
4. 用 Glassnode / CoinGecko 免费数据，构建 BTC 链上因子并做 IC 测试。

**交付物**
- `src/crypto/` 模块（数据采集 + 策略）
- `strategies/crypto_funding_arb/`
- `reports/crypto_research.md`

**推荐资源**
- 📖 《Mastering Bitcoin, 2nd》— Andreas Antonopoulos（了解底层）
- 🌐 [ccxt docs](https://docs.ccxt.com/)
- 🌐 [Glassnode Academy](https://academy.glassnode.com/)
- ⚠️ **坑点**：加密交易所 API 限频严格；回测要考虑提币时间、手续费差异、资金费结算时刻。

---

### 📅 M10：机器学习在量化中的应用

**目标**：掌握 ML/DL 在选股和择时中的应用，关键是避免过拟合和数据泄漏。

**理论学习（约 18h）**
- 特征工程：滚动统计、时间差分、组内标准化。
- 金融 ML 特有技术（de Prado）：
  - **Purged K-Fold** + **Embargo**（避免标签泄漏）
  - **Triple-Barrier Labeling**（标签构造）
  - **Meta-Labeling**（二次学习）
  - **Fractional Differentiation**（保留记忆的平稳化）
- 模型：Lasso、Ridge、GBDT（LightGBM / XGBoost）、Random Forest；时序用 LSTM / Transformer（谨慎）。
- 强化学习在交易中的现状与局限（了解即可，不实操）。

**实战任务（约 22h）**
1. 在 M7 多因子基础上，用 LightGBM 做选股模型，特征 = 因子值，标签 = 未来 20 日收益。
2. 用 Purged K-Fold 而非普通 TimeSeriesSplit；对比两者的过拟合差异。
3. 做一次 SHAP 可解释性分析，输出特征重要性 + 交互。
4. 思考并书面回答：ML 模型比线性多因子好在哪？差在哪？可解释性如何保障？

**交付物**
- `src/ml/` 模块（特征工程 + 训练 + 评估）
- `notebooks/M10_ml_selection.ipynb`
- `docs/ml-in-quant.md`：个人总结的经验与反思

**推荐资源**
- 📖 《Advances in Financial Machine Learning》— López de Prado（核心教材）
- 📖 《Machine Learning for Algorithmic Trading, 2nd》— Stefan Jansen
- 🌐 [Jansen 配套 GitHub](https://github.com/stefan-jansen/machine-learning-for-trading)
- ⚠️ **坑点**：Train/Test 没做 Purged 切分 → IS Sharpe 2.5，OOS -0.3，典型数据泄漏。

---

### 📅 M11：风险管理与组合管理

**目标**：构建生产级风控与组合管理模块，让策略真正「可上实盘」。

**理论学习（约 15h）**
- 风险度量：波动率、VaR（历史、参数、蒙特卡洛）、CVaR / ES、最大回撤。
- 仓位管理：固定比例、凯利公式（及其危险性）、波动率目标、风险平价。
- 止损/止盈设计：ATR 止损、移动止损、时间止损。
- 压力测试：历史情景（08 年、15 年股灾、2020 疫情、2022 LUNA）、蒙特卡洛。
- 流动性风险与清算成本。

**实战任务（约 25h）**
1. 实现风控模块 `src/risk/`：
   - 事前：仓位限额、集中度限额、杠杆限额。
   - 事中：实时 PnL、回撤监控、止损触发。
   - 事后：日度风险报告（VaR、压力情景）。
2. 将 M7/M8/M9 的策略通过风控模块统一管理，做资金配置。
3. 跑一次历史压力测试：把所有策略放在 2015/7、2020/3、2022/5 三个危机窗口下看表现。
4. 实现一份「组合日报」模板（HTML/PDF 自动生成）。

**交付物**
- `src/risk/` 风控模块
- `src/portfolio_manager/` 组合管理
- `reports/daily_portfolio_template.html`

**推荐资源**
- 📖 《Quantitative Risk Management》— McNeil, Frey, Embrechts
- 📖 《The Volatility Surface》— Jim Gatheral（进阶）
- ⚠️ **坑点**：凯利公式在估计偏差下会过度下注；实战常用 0.25–0.5 凯利。

---

### 📅 M12：综合项目 + 模拟实盘

**目标**：把前 11 个月的所有模块整合为一个端到端系统，并跑一段时间的模拟实盘。

**理论学习（约 8h）**
- 实盘与回测差异：延迟、撤单、部分成交、交易所故障、心理偏差。
- 运维基础：日志（结构化）、监控（Prometheus + Grafana 基础）、告警（钉钉/Telegram Bot）。
- 策略生命周期管理：上线前检查表、A/B 灰度、停机条件、退役流程。

**实战任务（约 32h）**
1. 整合所有模块为统一系统：`main.py` 一键跑整个研究流程。
2. 选定一个你研究下来最有信心的策略，接入**模拟盘**（A 股模拟、加密用 Binance Testnet、美股用 IBKR Paper）。
3. 跑至少 2 周模拟盘，每日生成对账报告，对比模拟盘与回测差异。
4. 写一份「年度学习总结」：你学到了什么、哪些方向值得深入、下一年的计划。
5. 将整个仓库整理到可以公开（或至少给同行评审）的状态：README、架构图、示例、测试、CI。

**交付物**
- 完整的量化研究系统（GitHub 仓库，含 CI、测试、文档）。
- `reports/annual_review_2026-2027.md`：年度总结。
- `docs/architecture.md`：系统架构图 + 模块说明。

**推荐资源**
- 📖 《Building Algorithmic Trading Systems》— Kevin Davey
- 🌐 [QuantInsti 实盘部署文章](https://blog.quantinsti.com/)
- ⚠️ **坑点**：第一次跑实盘，钱别多，账号别是主账号。模拟盘至少 2 周后再谈实盘。

---

## 3. 建议技术栈

| 层级 | 选型 | 备注 |
|------|------|------|
| 语言 | Python 3.11+ | 主力语言 |
| 依赖管理 | `uv` 或 `poetry` | 优先 `uv`（快） |
| 科学计算 | `numpy`, `pandas`, `polars` | Polars 值得学，大数据友好 |
| 可视化 | `matplotlib`, `plotly`, `seaborn` | |
| 时序建模 | `statsmodels`, `arch` | |
| 机器学习 | `scikit-learn`, `lightgbm`, `xgboost` | |
| 优化 | `cvxpy`, `scipy.optimize` | |
| 回测 | 自研 + `vectorbt` / `backtrader` 参考 | 自研为主 |
| 数据源 | `tushare`, `akshare`, `yfinance`, `ccxt` | |
| 存储 | Parquet + DuckDB | 入门阶段足够，不要上来就 TimeScaleDB |
| 工程 | `pytest`, `ruff`, `mypy`, `pre-commit` | 必须 |
| 笔记本 | Jupyter Lab + `nbstripout` | 提交前清理 outputs |
| 文档 | Markdown + Mermaid | 图用 Mermaid |

---

## 4. 建议项目结构

```
QuantTradingLucky/
├── README.md
├── pyproject.toml
├── docs/
│   ├── LEARNING_PLAN.md            # 本文件
│   ├── architecture.md
│   ├── markets-101.md
│   ├── pitfalls.md                 # 持续记录踩过的坑
│   ├── ml-in-quant.md
│   └── monthly-review/             # 每月复盘
│       ├── 2026-05.md
│       └── ...
├── src/
│   ├── data/                       # 数据管道
│   ├── universe/                   # 标的池
│   ├── costs/                      # 成本模型
│   ├── indicators/                 # 技术指标
│   ├── factors/                    # 因子库 + 测试框架
│   ├── alpha/                      # Alpha 模型
│   ├── portfolio/                  # 组合优化
│   ├── backtest/                   # 回测引擎
│   ├── risk/                       # 风控
│   ├── derivatives/                # 衍生品定价
│   ├── crypto/                     # 加密相关
│   ├── ml/                         # 机器学习
│   └── utils/
├── strategies/                     # 各具体策略
│   ├── momentum_us/
│   ├── dual_ma_a/
│   └── ...
├── notebooks/                      # 研究 Notebook
│   ├── M01_eda.ipynb
│   └── ...
├── reports/                        # 策略/因子报告
│   ├── factors/
│   └── strategies/
├── tests/                          # 单元测试
├── data/                           # 本地数据缓存（gitignore）
└── .github/workflows/              # CI
```

---

## 5. 核心阅读清单（贯穿 12 个月）

### 必读书籍（按优先级）
1. **《Advances in Financial Machine Learning》** — López de Prado（圣经级，难度高，适合 M5 开始反复读）
2. **《Active Portfolio Management》** — Grinold & Kahn（多因子方法论天花板）
3. **《Options, Futures, and Other Derivatives》** — Hull（衍生品不二选择）
4. **《Analysis of Financial Time Series》** — Tsay（时序金融）
5. **《Machine Learning for Algorithmic Trading》** — Stefan Jansen（ML 工程落地）
6. **《Algorithmic Trading》** — Ernest Chan（入门友好）

### 经典论文（选读）
- Fama & French (1993, 2015) — 多因子模型
- Jegadeesh & Titman (1993) — 动量效应
- Carhart (1997) — 4 因子模型
- WorldQuant (2016) — 101 Formulaic Alphas
- López de Prado (2018) — The 10 Reasons Most ML Funds Fail

### 博客与社区
- [QuantStart](https://www.quantstart.com/)
- [Quantpedia](https://quantpedia.com/) — 策略目录
- [Hudson & Thames (mlfinlab)](https://hudsonthames.org/)
- 知乎话题：量化交易（中文资源）
- 雪球专栏：量化 / 程序化交易

---

## 6. 常见陷阱汇总（持续更新 `docs/pitfalls.md`）

| 阶段 | 陷阱 | 对策 |
|------|------|------|
| 数据 | 复权处理错误 | 统一用前复权或原始价+复权因子 |
| 数据 | 幸存者偏差 | 使用包含退市股的全量历史池 |
| 回测 | 前视偏差 | 严格检查 t 时刻只能用 ≤ t-1 的数据 |
| 回测 | 过拟合 | Walk-Forward + 样本外 + 参数稳健性 |
| 回测 | 忽略成本 | 手续费 + 滑点 + 冲击成本三件套 |
| 因子 | 不做中性化 | 行业 + 市值中性是基础操作 |
| ML | 数据泄漏 | Purged K-Fold + Embargo |
| CTA | 主力换月 | 用连续合约，切换点做价格调整 |
| 期权 | 忽略 Gamma 风险 | Delta 中性只是一阶，要看 Gamma/Vega |
| 加密 | 资金费率未计入 | 合约回测必须把 funding 加进去 |
| 风控 | 凯利全押 | 最多 0.5 凯利，且要考虑估计误差 |
| 实盘 | 一把梭 | 永远先模拟盘 2 周以上 |

---

## 7. 里程碑检查表

在每个阶段结束时，对照以下清单自检。未完成的不要强推进度，回炉重造。

- [ ] **阶段一结束（M2）**：能一键下载三个市场的数据并做 EDA。
- [ ] **阶段二结束（M4）**：能独立做完一次单因子测试（IC、分组、多空、换手）。
- [ ] **阶段三结束（M6）**：自研回测引擎跑通 3 个策略，结果与成熟框架偏差 < 3%。
- [ ] **阶段四结束（M8）**：多因子模型 IC 均值 > 0.03、ICIR > 0.3；CTA 策略在样本外可交易。
- [ ] **阶段五结束（M10）**：有一份可解释的 ML 选股模型报告；加密策略样本外不是噪声。
- [ ] **阶段六结束（M12）**：模拟盘跑满 2 周，实盘-回测偏差可解释；有完整年度总结。

---

## 8. 写在最后

量化学习的真正难点不在数学、不在编程，而在**诚实**：
- 对回测结果的诚实（它真的是 Alpha 吗？）
- 对自己能力边界的诚实（我真的理解了吗？）
- 对市场的诚实（这个策略真的能穿越牛熊吗？）

12 个月只是一个开始。如果 M12 结束时你发现自己只想深挖某一个方向（比如多因子、或 CTA、或做市），那才是真正进入量化的起点。

祝学习愉快，Lucky。

— 本计划创建于 2026-04-19
