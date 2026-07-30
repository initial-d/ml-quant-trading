# 中文社区定制发布稿

每篇只保留一个行动入口：零账号 Colab。发布前应再次检查对应社区的自荐规则，
并在发布后回复技术问题。不要在同一天把完全相同的正文群发到所有平台。

统一配图：[`assets/validation-cost-sensitivity.png`](assets/validation-cost-sensitivity.png)。
图中横轴是交易成本而非时间，发布时不要称为净值曲线。

## 知乎

### 标题

我开源了一个 213 因子的 PyTorch 量化框架，并在沪深 300 上做了含交易成本验证

### 导语

量化回测里，最容易被忽略的不是模型，而是换手和成本。我把一个 213 因子的
PyTorch 多因子研究框架开源后，用 AkShare 当前沪深 300 数据做了一次
2021—2024 日频验证。结果里最值得看的不是缓冲组合 22.20% 的年化收益，而是
朴素因子组合从 31.57% 毛年化掉到 15.80% 净年化的过程。

正文使用 [`article_zh_213_factor_csi300.md`](article_zh_213_factor_csi300.md)
全文。结尾仅保留：

> 想检查工程是否能在你的环境跑通：
> [打开零账号 Colab，跑一遍并提交结果](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)

## 掘金

### 标题

从 213 维 PyTorch 因子张量到含成本回测：一个可复现的 A 股研究工程

### 正文

这次分享重点不是策略收益，而是工程边界：

- 使用 mask 表达停牌、涨跌停和缺失单元；
- 将 213 个因子统一成 `[日期, 股票, 因子]` 张量；
- 用同一份配置串起模型、组合和回测；
- 自动生成 Markdown / JSON 报告；
- 对 0、7、15、30 bps 做成本压力测试。

公开沪深 300 验证中，朴素日频因子组合毛年化为 31.57%，但 7 bps 下净年化只剩
15.80%。加入透明的持仓缓冲规则后，换手从 0.3627 降到 0.1397，净年化为
22.20%。这说明工程优化的重点不是继续堆网络，而是先让因子边际穿过组合和成本层。

项目同时保留了 IC 加权失败、部分优化组合落后基线，以及 MLP 路径尚未形成有效
组合等负面结果。

想从代码路径而不是收益截图开始：
[打开零账号 Colab，跑一遍并提交结果](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)

## V2EX

### 标题

[分享创造] 开源一个 PyTorch 多因子量化研究栈，求真实机器复现反馈

### 正文

做了一个研究用途的 `ml-quant-trading`，目前包含 213 个因子、mask-aware
PyTorch 算子、A 股涨跌停/停牌处理、MLP/Transformer、Markowitz 和含成本回测。

最近补了 AkShare 当前沪深 300 的 2021—2024 公开验证。一个比较诚实的结果是：
朴素日频因子选择毛收益看起来不错，但 7 bps 成本下被高换手吃掉；加简单持仓
缓冲后才超过同区间等权。失败模型也保留在报告里。

我更想收集安装失败、CPU/GPU 时间和不同环境结果，不想讨论“能不能跟单”。

入口只有一个：
[零账号 Colab，跑完可直接生成提交报告](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)

## 聚宽

### 标题

公开一个 213 因子研究框架：沪深 300 日频验证中，换手如何吃掉毛因子收益

### 正文

这次验证用 AkShare 当前沪深 300 成分股、2021—2024 日线和完整 213 因子库。
重点比较同一因子分数在两种持仓规则下的结果：

| 规则 | 7 bps 净年化 | Sharpe | 换手率 |
|---|---:|---:|---:|
| 朴素日频选择 | 15.80% | 0.701 | 0.3627 |
| 缓冲日频选择 | 22.20% | 0.919 | 0.1397 |

朴素规则的毛年化其实达到 31.57%，但成本拖累为 49.16%。这说明讨论因子有效性时，
选股分数和持仓更新规则不能拆开看。

边界也很明确：使用当前成分股而非历史时点成分股；没有行业、规模、流动性和真实
执行元数据；结果不是论文精确复现。

如果愿意帮忙验证代码路径：
[打开零账号 Colab，跑一遍并提交结果](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)

## 米筐

### 标题

因子有毛收益之后：用换手缓冲让 213 因子组合通过成本压力测试

### 正文

我把多因子研究代码整理成了一个 PyTorch 工程，并重点测试组合层是否能保留
因子边际。当前沪深 300 日频公开数据上，朴素因子均值策略虽然有 31.57% 毛年化，
但高换手使 7 bps 下净年化降到 15.80%。

缓冲规则不改变每日评分，只放宽退出区间，减少刚跌出入选边界就被替换的交易。
同区间换手下降约 61%，7 bps 下净年化为 22.20%。在 15 和 30 bps 压力下结果
继续下降，因此报告展示的是成本压力曲线，而不是一条容易误解的净值营销图。

研究限制包括当前成分股偏差、公共数据修订以及缺少生产执行约束。欢迎复现后用
自己的成本和股票池挑战结果：
[打开零账号 Colab，跑一遍并提交结果](https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb)
