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
