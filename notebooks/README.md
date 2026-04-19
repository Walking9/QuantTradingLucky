# Notebooks

研究型 Notebook 目录。生产代码请放到 `src/quant_lucky/` 下并写单元测试。

## 命名约定

```
M{阶段月份}_{主题}.ipynb
```

示例：
- `M01_eda.ipynb` — M1 月初探性数据分析
- `M04_factor_momentum.ipynb` — M4 动量因子研究
- `M10_ml_selection.ipynb` — M10 ML 选股实验

## 规则

1. **Output 必须清除后再提交**。`pre-commit` 的 `nbstripout` 会自动处理，但本地也请养成习惯。
2. **数据路径不要硬编码绝对路径**。从 `quant_lucky.utils.config` 读取 `QUANT_DATA_ROOT`。
3. **实验性代码成熟后，迁移到 `src/` 并写测试**。Notebook 不是最终产物。
4. **大型结果图表**导出到 `reports/figures/`，并在 Notebook 中用相对路径引用。
5. **随机实验**必须设定 seed 并在首个 cell 标明，否则结果不可复现。

## 模板首 Cell

```python
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from quant_lucky.utils.config import settings  # 待 M1 实现

np.random.seed(42)
pd.set_option("display.max_columns", 50)
```
