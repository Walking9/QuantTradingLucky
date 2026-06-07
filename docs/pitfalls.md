# 踩坑实录 (Pitfalls Journal)

> 每一个踩过的坑都值千金。**遇到问题 → 定位 → 修复 → 记录**。
> 记录格式越简单越好，保证会持续写下去。

## 使用说明

- 每次解决一个非平凡的 bug 或发现一个思维陷阱，添加一条记录。
- 分类参考《学习计划》第 6 节，但新类别也欢迎。
- 尽量带上：**现象、根因、修复、教训**四要素。
- 严重性分级：🔴 Critical / 🟠 Major / 🟡 Minor。

---

## 模板

```markdown
### YYYY-MM-DD · 🟠 短标题

**分类**：数据 / 回测 / 因子 / ML / 风控 / 实盘 / ...
**上下文**：在做什么？

**现象**
- ...

**根因**
- ...

**修复**
```python
# 关键代码或配置变更
```

**教训**
- 一句话能让未来的自己避免重蹈。
```

---

## 记录

<!-- 在此之下新增记录，最新的放最上面 -->

### 2026-06-07 · 🔴 合成面板的"静态截面离散度"会被动量排序误读成 Alpha

**分类**：回测 / 数据 / 过拟合
**上下文**：M6 写 `momentum_us_monthly_top20`。离线没有标普 500 横截面，用
`synthetic_price_panel` 兜底。默认想法是"`autocorr=0` → 无边 → 动量赚 ≈0"，
跑出来却是 **full Sharpe 2.47、OOS 2.60、agg WF OOS 2.34**。

**现象**
```
data: synthetic | autocorr=0.0
full   sharpe 2.47    out-sample sharpe 2.60
```
一个声称"无植入边"的面板上，月度多空动量跑出了 Sharpe 2.5。典型的"Sharpe > 2 先
怀疑自己"。

**根因**
`synthetic_price_panel` 的默认 generator 给每个资产**异质的静态期望收益**
（`drift_dispersion=0.10`，`beta ∈ [0.6, 1.4]`）。在持续正漂移的市场里，
按 trailing return 排序 ≈ 按"真实 drift / beta"排序——而真实 drift 是 generator
写死的、回测能"偷看"到的常数。于是"赢家持续赢"，动量机械地收割了这份静态离散度。
这**不是时间序列动量 Alpha**，是一个不可交易的静态 tilt（实盘里你不知道每个名字的
真实 drift）。

**验证**（同 seed，仅改 generator 参数）
```
A 默认 generator（静态离散），autocorr=0  : gross Sharpe 2.56   ← 假边
B 同质 drift/beta，autocorr=0（真零假设）  : gross Sharpe 0.19   ← 真无边
C 同质 + autocorr=0.12（植入真边）         : gross Sharpe 3.53   ← 真边
```

**修复**
- 动量包的合成配置改为**同质期望收益**：`drift_dispersion=0, beta_low=beta_high=1.0`。
  这样 `autocorr=0` 是真零假设（B），动量排序的是纯特异噪声，Sharpe≈0、DSR≈0。
- 用 `--set synthetic.autocorr=0.12` 注入真持续性（C），同一套机器 DSR 跳到 1.00。
- 把"零假设 DSR≈0 / 真边 DSR≈1"的对照写进 `reports/strategies/momentum_us.md`。

**教训**
- **合成数据的"无边"必须是构造出来的，不是默认的**：任何带静态截面离散度的 generator
  对横截面策略都不是零假设。做 null 检验前先问"我的 generator 里有没有策略能合法收割
  的结构"。
- 这是 §5「为什么回测会骗人」的最纯粹案例：回测好看可以纯粹因为不可交易的理由。
  attribution（beta≈0）和 DSR 都帮着拆穿它，但**最终是靠理解 generator 才定位的**。

---

### 2026-05-24 · 🟠 把 train slice 喂给 `strategy_fn` 等于把 lookback 砍掉

**分类**：回测 / Walk-Forward
**上下文**：M5 写 `backtest.validation.walk_forward`，第一版 API 设计成
`strategy_fn(param, train_prices) -> train_weights`，理由是「跟 sklearn 的
`fit(X_train)` 对齐」。跑 20 日动量策略时发现每个 split 前 20 个交易日都是
0 权重——明明 strategy_fn 里写的是 `pct_change(20).shift(1)`。

**现象**
- Walk-Forward OOS 净值在每个 split 头部都有一段平台期。
- 同一策略在 full panel 上跑出来没有这个平台期，仅 burn-in 阶段为 0。
- 单测里手算的 OOS Sharpe 和 driver 跑出来的对不上。

**根因**
- `pct_change(20).shift(1)` 要 21 个交易日的前置数据才能给出第一个非 NaN 权重。
- 把 200-bar 的 train slice 单独喂给 strategy_fn，相当于让它假装历史从 train slice
  开始，于是「最近 20 天的动量」前 20 个 train bar 全是 NaN → 0 权重。
- 每个 split 都重新付一遍 lookback 税 = 用错误的「冷启动样本」估指标。

**修复**
```python
# src/quant_lucky/backtest/validation.py 模块文档
# A strategy function takes a parameter dict + the FULL price panel and
# returns the weights it would have used on every date in that panel.
# Key contract: the strategy MUST be no-look-ahead — weight at ``t`` may
# only depend on prices ``≤ t-1``. The Walk-Forward driver does NOT
# enforce this; it trusts the strategy.
```
- API 改成 `strategy_fn(param, full_prices) -> full_weights_df`，driver 只
  对**权重 DataFrame** 切 train/test，不对**计算**切。
- 副作用：strategy_fn 必须自证无前视；driver 不帮你管。这条强制约束写在模块
  docstring 的「Performance note」段。

**教训**
- Walk-Forward 的切分点是「评估指标」，不是「策略计算」。一切需要 lookback 的
  指标都该一次性算到全样本上，再切窗口看表现。
- sklearn 的 `fit(X_train)` 类比在时间序列上是有毒的——「拟合」和「计算特征」
  在时间序列里是两件事。
- 测试 `test_walk_forward_concatenated_oos_covers_all_test_dates` 守住「OOS 长度
  必须等于 split test 长度之和」的不变量。

---

### 2026-05-24 · 🟠 月度调仓的 ffill 权重在两个引擎里是两个策略

**分类**：回测 / 引擎对照
**上下文**：M5 写 `scripts/verify_vectorbt_parity.py`，第一版把月度调仓策略
写成「调仓日权重 = 1/N、其它日 ffill」。`VectorEngine` 跑出来 Sharpe +0.376、
vectorbt 跑出来 Sharpe +0.353，10bps 成本下终值差 -0.14%。看似在 3% 容差内，
但同样的「无成本」版本却严格相等——异常。

**现象**
- `monthly_rebalance` + `cost_bps=10`，两个引擎终值偏差 -0.14% / Sharpe 差 -0.024。
- 把同一份权重 print 出来对齐到逐 bar 浮点完全相同。
- 把 cost 降到 0，偏差几乎消失（< 1e-4）—— 说明差异**只在成本侧**。

**根因**
- 自研引擎是「**权重态**」：以为用户传的就是真实持仓权重，于是 `Δw[t] = w[t] - w[t-1]`，
  ffill 后 Δw = 0 → 月内无交易、无成本。
- vectorbt 是「**股数态**」：每天用 `targetpercent` 把仓位**拉回** 1/N，月内
  天天调仓、天天付 10bps。
- 同一份 DataFrame 在两个引擎里表达的策略**不同**——自研引擎认为是
  「每月调一次、月内任由漂移」，vectorbt 认为是「每天都强制回到 1/N」。
- 0 成本下两条净值数学等价（连续股数 + 0 fees → 漂移路径终值一样），
  10bps 成本立刻把语义差异显形。

**修复**
- 把 parity 测试的月度调仓改成**显式构造漂移权重**：在每个月初按 NAV/N 算
  shares，月内固定 shares、显式算 `value[t] / NAV[t]` 当作真实权重路径。
  这样两个引擎收到的「权重」就是真实持仓比例，行为对齐。
- 留一个 `scenario_naive_monthly_ffill_trap`（不参与 parity 验收）专门展示
  这个陷阱，并把整段解释写进 `reports/m05_vectorbt_parity.md` 第 5 节。

```python
# scripts/verify_vectorbt_parity.py
def scenario_monthly_rebalance_drifted(prices):
    # ... per-month: 计算 shares = (nav / N) / price0
    # 月内权重 = shares * price[t] / NAV[t] —— 显式漂移
```

**教训**
- 「权重 DataFrame」不是一个无歧义的语言。同一张表至少有两种合法解释：
  「这是真实持仓」vs「这是每天的调仓目标」。两个引擎选了不同的解释。
- 在写跨引擎对照前，先用一句话钉死语义——「我传的是漂移后真实权重」还是
  「我传的是 rebalance target」。否则成本一开就翻车。
- M6 以后做月度/季度调仓策略**禁用 `ffill`**，要么显式构造漂移权重，要么在
  非调仓日填 NaN（让 vectorbt 也跳过那一天）。两种做法等价。
- 此坑的根因不是 bug，是规约不清；所以正确处理是写文档 + 留陷阱测试，
  不是「修引擎」。

---

### 2026-05-20 · 🟡 "完全恒定"的收益序列让 Sharpe 爆炸到 7e16

**分类**：回测 / 指标
**上下文**：M5 写 `compute_performance(pd.Series([0.001]*252))`，期望 Sharpe 是 NaN（标准差为 0 → 无定义），结果跑出来 `sharpe=7.3e16`。

**现象**
- 单测 `test_constant_positive_return_zero_volatility` 断言 `np.isnan(report.sharpe)` 失败。
- 实际 `annual_vol ≈ 3.45e-18`（不是 0），`annual_return / annual_vol ≈ 7.3e16`。

**根因**
- pandas `Series.std(ddof=1)` 对完全恒定的浮点序列**不返回精确 0** — 减均值时残留 ULP 量级误差，平方根放大后就是机器 epsilon 级。
- 我的判断写成 `if annual_vol > 0`，这对 1e-18 是 True，于是除法产生天文数。

**修复**
```python
# src/quant_lucky/backtest/report.py
vol_floor = 1e-12 * max(abs(mean), 1e-6)
sharpe = (
    annual_return / annual_vol
    if not np.isnan(annual_vol) and annual_vol > vol_floor
    else float("nan")
)
```
同样的守护加到 Sortino 上。

**教训**
- **永远不要拿浮点和 0 直接比较**，尤其是经过 `cumsum / mean / std` 这种累积运算后。
- 写"无定义"检查时，相对阈值 (`相对于 mean 的 1e-12`) 比绝对阈值更稳健。
- 这个 bug 在真实策略上也会出现：常数股息复利曲线、停牌后被前向填充的价格——任何"几乎不动"的序列都会触发。

---

### 2026-04-20 · 🟡 `settings.data_root` 默认值随 CWD 漂移

**分类**：工程 / 配置
**上下文**：M1 EDA notebook 从 `notebooks/` 目录执行时，本地 Parquet 缓存落到了
`notebooks/data/` 而不是 `data/`；再次执行根目录代码时又重新下载了一遍。

**现象**
- `Downloader` 写入路径是 `./data/raw/<provider>/<symbol>/<freq>.parquet`。
- Jupyter 默认把 notebook 所在目录作为 CWD，于是 `./data` 指向 `notebooks/data`。
- 从根目录跑测试/CLI 时找不到缓存，所有数据重新拉一次。

**根因**
- `Settings.data_root` 默认值 `Path("./data")` + `.env` 中 `QUANT_DATA_ROOT=./data`
  都是相对路径，`pydantic-settings` 不会替你锚定到项目根。

**修复**
```python
# src/quant_lucky/utils/config.py
def _find_project_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()

PROJECT_ROOT = _find_project_root()

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", ...)
    data_root: Path = Field(default=PROJECT_ROOT / "data", alias="QUANT_DATA_ROOT")

    @field_validator("data_root", mode="after")
    @classmethod
    def _anchor_data_root(cls, value: Path) -> Path:
        return value if value.is_absolute() else (PROJECT_ROOT / value).resolve()
```

**教训**
- 任何「默认路径」字段都应当在项目入口锚定一次，不要依赖 CWD。
- `.env` 里用相对路径要同时做 validator 兜底，用户习惯写 `./data` 我们挡不住。
- 写了测试 `tests/utils/test_config.py::test_data_root_stable_across_cwd`
  用 subprocess 切 CWD，防止回归。

---

### 2026-04-20 · 🟠 `CCXTProvider` 对带 tzinfo 的 `end` 会抛 `ValueError`

**分类**：数据 / provider
**上下文**：M1 EDA notebook 第一次执行，`ccxt_dl.download("BTC/USDT", START, END, ...)`
其中 `END = datetime(2025, 1, 1, tzinfo=UTC)` 直接报错。

**现象**
```
ValueError: Cannot pass a datetime or Timestamp with tzinfo with the tz parameter.
```

**根因**
- `ccxt_provider.py:106` 写的是 `pd.Timestamp(request.end, tz="UTC")`；
  当 `request.end` 已经带 `tzinfo` 时，pandas 2.x 明确禁止再传 `tz=`。
- 覆盖率报告早就提示 provider 只有 23% 覆盖 —— 所有通路只在 notebook 实跑时才暴露。

**修复**
```python
end_utc = pd.Timestamp(request.end)
if end_utc.tz is None:
    end_utc = end_utc.tz_localize("UTC")
else:
    end_utc = end_utc.tz_convert("UTC")
df = df[df["timestamp"] <= end_utc]
```

**教训**
- 所有「把用户传入的时间戳和 pandas 时区比较」的地方，都要判断一下 tz 是否已存在。
- provider 测试不能只停在抽象基类 —— 至少补一组 mock 掉 HTTP 层的集成测试。
  短期 TODO：`tests/data/test_providers.py`，覆盖 yfinance/ccxt/tushare 的 `_normalise` 与边界。
