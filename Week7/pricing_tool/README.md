# Week 7 BSM Option Pricing Tool

## 三模型 CallPrice 部署

实时区域现在会在同一个市场日期、同一份输入数据和同一个期权合约下，同时输出：

1. `LSTM + BSM`：LSTM 预测最新一期日波动率变化，再将预测波动率年化后输入 BSM；
2. `Linear Regression`：直接根据最新一行特征预测 CallPrice；
3. `BSM Baseline`：使用最新实际 `RollingVol20` 年化后进行 BSM 定价。

为了与训练数据保持一致，三模型实时比较固定为 JPM 欧式看涨期权、平值 `K=S`、到期时间 `T=1 year`。页面上方原有的手动 BSM 计算器仍可独立输入有效的 `S、K、r、sigma、T`，它与三模型比较区域不是同一项功能。

### 一次性生成模型文件

模型只在离线阶段训练一次，Streamlit 刷新页面时不会重新训练：

```powershell
$env:TF_ENABLE_ONEDNN_OPTS='0'
Week7\.venv\Scripts\python.exe -m Week7.model_deployment.artifact_training
```

成功后会在 `Week7/model_artifacts/` 生成：

```text
linear_all_vix.joblib
lstm_all_vix.keras
lstm_x_scaler.joblib
lstm_delta_scaler.joblib
manifest.json
```

`manifest.json` 保存训练口径、特征顺序、数据日期、验证/测试指标、训练特征范围和每个二进制文件的 SHA-256。只有重新计算的指标与 Week7 已报告指标一致时，新的模型文件才会发布。

### 运行测试与 App

```powershell
Week7\.venv\Scripts\python.exe -m unittest discover -s Week7\model_deployment\tests -p "test_*.py" -v
Week7\.venv\Scripts\python.exe -m unittest discover -s Week7\pricing_tool\tests -p "test_*.py" -v
Week7\.venv\Scripts\python.exe -m streamlit run Week7\pricing_tool\app.py
```

打开 App 后启用 `Enable automatic market updates`，即可在市场数据下方看到 `Live Three-Model CallPrice Comparison`。

### 如何理解页面警告

- `training range`：某个实时特征超出训练集最小值或最大值，模型正在外推；原始输入和预测不会被修改。
- `no-arbitrage`：模型的原始 CallPrice 超出欧式看涨期权的理论边界；页面仍显示原始值，便于研究模型风险。
- Linear Regression 直接预测价格，因此比较表中的 `Annual Volatility` 显示为 `N/A`。
- 所有结果仅用于模型比较和研究，不构成交易或投资建议。

### 扩展后的 3.6 端到端验证

`Week7/results/task_3_6_end_to_end_report.csv` 现在覆盖完整链路：

```text
market data -> cache/fallback -> feature engineering
-> artifact hash/shape validation -> Linear/LSTM/BSM inference
-> common contract checks -> three finite CallPrice outputs
```

报告中的 `ModelWarningCount` 是诊断信息数量。只要模型文件、输入、合约和输出均有效，存在训练区间警告不会把 `OverallStatus` 改为 `FAIL`。

## 任务目的

Tasks 2.1 和 2.2 把前期 Notebook 中的定价知识整理成一个可以交互使用的 Streamlit 原型。重点不只是得到价格，而是学习如何把金融计算、输入验证、用户界面和自动测试分开组织。

## 结构

```text
pricing_tool/
├── app.py                  # Streamlit 界面
├── pricing_service.py      # 纯 Python BSM 计算服务
├── requirements.txt       # UI 依赖
└── tests/
    ├── test_pricing_service.py
    └── test_app.py
```

数据流是：

```text
用户输入百分比和价格
    -> app.py
    -> 百分比转换与输入验证
    -> pricing_service.py
    -> Call / Put / Parity / Spot Curve
    -> Streamlit 页面展示
```

`pricing_service.py` 不导入 Streamlit，因此可以被测试、FastAPI 或未来的模型页面直接复用。

## BSM 原理

对无股息欧式期权：

```text
d1 = [ln(S/K) + (r + 0.5 sigma^2)T] / [sigma sqrt(T)]
d2 = d1 - sigma sqrt(T)

Call = S N(d1) - K exp(-rT) N(d2)
Put  = K exp(-rT) N(-d2) - S N(-d1)
```

其中：

- `S`：当前标的价格；
- `K`：执行价；
- `r`：连续复利无风险利率，服务层使用小数；
- `sigma`：年化波动率，服务层使用小数；
- `T`：距离到期的年数；
- `N()`：标准正态分布累计概率。

页面中 `r` 和 `sigma` 按百分比输入。输入 `5` 和 `20` 后，服务分别转换为 `0.05` 和 `0.20`。

## Put-Call Parity

工具额外检查：

```text
Call - Put = S - K exp(-rT)
```

这个恒等式为 Call 和 Put 计算提供独立的一致性检查。残差应接近零。

## 安装

在项目根目录执行：

```powershell
Week7\.venv\Scripts\python.exe -m pip install -r Week7\pricing_tool\requirements.txt
```

## 运行测试

```powershell
Week7\.venv\Scripts\python.exe -m unittest discover -s Week7\pricing_tool\tests -p "test_*.py" -v
```

经典测试参数为：

- `S = 100`
- `K = 100`
- `r = 5%`
- `sigma = 20%`
- `T = 1`

预期 Call 约为 `10.4506`，Put 约为 `5.5735`。

## 启动页面

```powershell
Week7\.venv\Scripts\python.exe -m streamlit run Week7\pricing_tool\app.py
```

浏览器打开 `http://localhost:8501`。

## 模型假设与限制

- 仅适用于欧式期权；
- 不考虑股息；
- 假设利率和波动率固定；
- 不包含波动率微笑、跳跃、交易成本和流动性；
- 这是 BSM 教学与工具框架原型，不是交易建议。

## 后续扩展

后续可以保留 `app.py` 的输入与展示结构，增加：

1. Linear Regression 或 LSTM 模型选择；
2. 从实时市场数据生成参数；
3. VIX 与压力测试页面；
4. FastAPI 接口。

这些扩展应通过新的服务模块接入，而不是把模型训练代码直接写进 Streamlit 页面。

## API Key 安全配置

实时数据接口使用项目根目录中的 `.env`，但 `.env` 已被 Git 忽略，不能提交。首次使用时：

1. 打开根目录 `.env.example` 查看变量名称；
2. 在根目录 `.env` 中填写新申请的 Key；
3. 不要把 Key 写进 Python、notebook、截图、聊天或终端命令；
4. 填写后只检查配置状态，不打印实际值。

```text
ALPHA_VANTAGE_API_KEY=在本地填写
FRED_API_KEY=在本地填写
```

应用通过 `config.py` 加载变量。缺少变量时只报告变量名称，不会在异常信息中包含已经配置的 Key。

## 3.2 统一市场数据接口

`market_data_service.py` 把 Alpha Vantage、Cboe 和 FRED 的不同响应格式转换成统一的日频数据结构：

```python
from Week7.pricing_tool.market_data_service import load_market_data

bundle = load_market_data()

print(bundle.jpm.tail())
print(bundle.vix.tail())
print(bundle.treasury.tail())
print(bundle.merged.tail())
print(bundle.metadata)
```

`MarketDataBundle` 包含：

- `jpm`：`Date, Open, High, Low, Close, Volume`；
- `vix`：`Date, VIX_Open, VIX_High, VIX_Low, VIX_Close`；
- `treasury`：`Treasury_Observation_Date, Treasury_Yield`；
- `merged`：以 JPM 交易日为主时间轴的完整数据；
- `metadata`：供应商、行数、日期范围和 UTC 获取时间。

JPM 与 VIX 必须同日匹配。FRED 使用当前 JPM 日期之前最近的有效 DGS10，并保留真实的 `Treasury_Observation_Date` 和 `Treasury_Staleness_Days`。`Treasury_Yield` 仍以百分数保存，例如 `4.56`，只有进入 BSM 定价时才除以100。

3.2 不计算滚动波动率或 VIX 衍生特征，不缓存网络结果，也不使用备用供应商。这三部分分别属于3.3、3.4和3.5。

## 3.3 实时特征工程

### 目的与原理

`feature_engineering_service.py` 把 3.2 的统一市场数据转换成模型训练时使用的相同特征。实时计算必须保持公式、单位、列顺序和阈值不变，否则模型看到的数据含义会与训练阶段不同，这种问题称为 **training-serving skew（训练与应用偏差）**。

特征只使用当前日期及其之前的数据：收益率使用前一日，5 日变化使用前 5 日，移动均值和标准差使用向后滚动窗口。因此，某一天之后的新行情不会改变该日已经计算出的特征。

### 使用方法

在项目根目录已经配置好忽略提交的 `.env` 后运行：

```python
from Week7.pricing_tool.feature_engineering_service import load_realtime_features

features = load_realtime_features()

print(features.history.tail())
print(features.linear_latest)          # 形状：(1, 10)
print(features.lstm_sequence.shape)    # 形状：(20, 13)
print(features.latest_market_date)
print(features.quality)
```

`RealTimeFeatureBundle` 包含：

- `history`：去除暖机期后的完整有效特征历史；
- `linear_latest`：最新交易日的 1 行、10 列 Linear Regression 原始输入；
- `lstm_sequence`：最近 20 个有效交易日、13 个特征的 LSTM 原始序列；
- `latest_market_date`：本次输入对应的最新市场日期；
- `quality`：原始行数、暖机行数、模型形状、缺失值和无穷值检查结果。

### 重要口径

- `RollingVol20` 是 20 个日对数收益率的样本标准差（`ddof=1`），仍是日波动率；进入 BSM 时才乘以 `sqrt(252)` 年化。
- `Treasury_Yield` 和 `RateMomentum` 保持百分数单位，例如 `4.56` 表示 4.56%。
- `VIX_20D_ZScore` 的滚动标准差使用 `ddof=0`，与第 1.2/1.3 阶段训练数据一致。
- `VIX_Regime` 和 `VIX_Spike` 使用 `Week7/data/vix_feature_thresholds.json` 中由训练集得到的固定阈值，不能根据最新 100 天重新计算。
- 第一个价格没有日收益率，所以 20 日收益率波动率需要 21 个价格观测；完整模型特征会移除前 20 行暖机数据。

本阶段输出的是**原始特征**。不能在实时数据上重新拟合 `StandardScaler`；LSTM 推理时必须加载训练阶段已经拟合并保存的 scaler，再把序列转换为 `(1, 20, 13)`。现有 LSTM 用前 20 行预测下一期波动率变化，因此 `lstm_sequence` 表示下一期预测所需的历史上下文。

## 3.4–3.5 自动更新、缓存与失败回退

### 使用方法

启动 Streamlit 后，在页面底部找到 `Real-Time Market Data`：

1. 打开 `Enable automatic market updates`；
2. 页面立即检查本地缓存或请求实时数据；
3. 开关保持打开时，数据区域每 60 分钟自动检查一次；
4. 点击 `Refresh now` 可以跳过 60 分钟新鲜期并立即尝试请求 API。

自动检查只在当前 Streamlit 页面保持打开且开关启用时运行。关闭页面后不会有后台进程继续请求 API，但磁盘缓存会保留，供下次启动使用。

### 三种数据状态

- `live`：本次成功调用 Alpha Vantage、Cboe 和 FRED，并取得实时可用数据；
- `fresh_cache`：缓存不超过 60 分钟，直接复用且不会调用 API；
- `stale_fallback`：缓存超过 60 分钟、实时刷新失败，但缓存不超过 7 天，因此临时回退使用。

`stale_fallback` 会在页面显示黄色警告、缓存年龄和最后成功更新时间，不会把旧数据伪装成实时数据。如果没有有效缓存，或缓存已经超过 7 天，工具会拒绝返回市场数据。

### 缓存文件

缓存位置：

```text
Week7/pricing_tool/.cache/market_data_cache.pkl
```

该目录已加入 `.gitignore`。缓存只保留最近一次成功获得的标准化 `MarketDataBundle` 和时间信息：

- 不保存 Alpha Vantage 或 FRED API Key；
- 不保存原始请求、完整 URL 或原始响应；
- 新缓存通过临时文件校验后原子覆盖旧缓存；
- 写入中断不会破坏上一次有效缓存；
- 缓存损坏、版本不一致或结构错误时不会被使用；
- 始终只有一个正式缓存文件，不会随刷新次数持续增加。

代码中也可以直接使用：

```python
from Week7.pricing_tool.market_data_cache import load_cached_market_data
from Week7.pricing_tool.feature_engineering_service import build_realtime_features

cached = load_cached_market_data()
features = build_realtime_features(cached.bundle)

print(cached.status.source)
print(cached.status.cached_at_utc)
print(cached.status.age_seconds)
print(features.linear_latest.shape)   # (1, 10)
print(features.lstm_sequence.shape)   # (20, 13)
```

缓存与失败回退是为了提高教学原型的稳定性，不代表旧行情仍具有实时性，也不构成交易建议。

## 3.6 端到端测试

### 测试范围

3.6验证第三阶段的完整实时数据链：

```text
Alpha Vantage / Cboe / FRED
→ 数据标准化与日期对齐
→ 磁盘缓存与失败回退
→ 实时特征工程
→ 模型文件哈希、特征顺序与输入形状验证
→ Linear、LSTM + BSM与BSM Baseline实际推理
→ 三模型共同日期与期权合约验证
→ 端到端CSV报告
```

它检查29项条件：原有21项数据和接口检查，加上模型清单、共同日期、共同合约、三个有限CallPrice和两个正波动率检查。

运行自动化端到端测试：

```powershell
Week7\.venv\Scripts\python.exe -m unittest Week7.pricing_tool.tests.test_end_to_end -v
```

测试包含：

- 三个模拟API首次获取数据并写入缓存；
- 60分钟内完全不调用API的 `fresh_cache`；
- API失败后的 `stale_fallback`；
- 故意破坏模型输入形状后必须得到 `FAIL`；
- 非有限模型输出必须得到稳定的 `FAIL` 检查名称；
- CSV报告的原子写入和旧文件保护。

真实API验证报告位于：

```text
Week7/results/task_3_6_end_to_end_report.csv
```

- `PASS`：29项检查全部通过；
- `FAIL`：`FailedChecks`列列出未通过的稳定检查名称；
- `DataSource=stale_fallback`：链路仍然可以通过，但报告明确表示数据来自失败回退，不代表实时API成功。
- `ModelWarningCount`：训练范围或无套利边界诊断数量；警告不会修改原始预测。
