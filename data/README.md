# Data

本地数据缓存目录。**大部分子目录已被 `.gitignore` 忽略**，只有此 README 和 `tests/fixtures/` 中的极小样本文件会入库。

## 目录规划

```
data/
├── raw/              # 原始下载数据（不可变），按 provider 分区
│   ├── tushare/
│   ├── akshare/
│   ├── yfinance/
│   └── ccxt/
├── interim/          # 半成品（如合并了不同 provider 的同一标的）
├── processed/        # 清洗+复权+对齐后的规范化数据，供策略直接读取
├── cache/            # 临时缓存（Parquet/DuckDB），可任意删除
└── downloads/        # 手动下载的原始文件（CSV/ZIP），用于不便用 API 拿的数据
```

## 规范

1. **Raw 是不可变的**。如需重算，先清空 `processed/` 再从 `raw/` 构建。
2. **所有 Parquet 必须带元数据**：provider、下载时间、schema 版本。`src/quant_lucky/utils/io.py` 会统一封装。
3. **路径通过 `settings.data_root` 解析**，不要硬编码 `./data`。
4. **大文件（>10MB）不入库**；需要的话写脚本 `scripts/download_*.py` 让其他人可复现。
5. **敏感或付费数据**：额外在 `.env` 中声明路径并在 `.gitignore` 中显式忽略。

## 快速检查磁盘占用

```bash
du -sh data/*/ | sort -h
```
