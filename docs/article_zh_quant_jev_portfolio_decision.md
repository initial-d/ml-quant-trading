# 量化 Jev 不一定要接入模型：把多因子 Pipeline 封装成 Portfolio Decision Surface

> 项目：`ml-quant-trading`  
> 主题：量化 Jev / System-One-style portfolio decision adapter  
> 边界：研究和工程验证，不是实盘交易系统，不构成投资建议。

**GitHub 源码与完整文档：**
<https://github.com/initial-d/ml-quant-trading>

如果把 Jev 或 System One 的核心理解为“给定结构化状态，回答类型化问题”，那么量化
场景里最自然的结合点不一定是把某个外部模型塞进因子 pipeline，而是把 pipeline
本身封装成一个可调用的判断面。

这也是 `ml-quant-trading` 新增 portfolio decision adapter 的动机。

传统量化研究代码通常停在这里：

```text
data -> factors -> model scores -> portfolio weights -> backtest
```

这条链路能回答“历史回测表现如何”，但不太容易回答另一个更贴近系统接口的问题：

```text
给定今天的 pipeline state，这个资产应该增持、持有、减持还是排除？
目标权重大概是多少？
它是否因为停牌、涨跌停、流动性或风险约束不能交易？
```

portfolio decision adapter 做的事情，就是把这些问题变成一个稳定的 typed decision
contract。

## 什么叫量化 Jev-style？

这里的 Jev-style 不是官方 Jev 集成，也不是声称模型已经获得某种认证。它指的是一种
接口形态：

```text
structured state + typed questions -> typed answers
```

放到量化里，就是：

```text
因子、mask、预测、持仓、流动性、风险、成本
  + 关于资产和组合的问题
  -> 结构化 portfolio intent
```

比如默认问题包括：

- `asset_action`：increase / hold / decrease / exclude；
- `target_weight`：研究目标权重；
- `trade_allowed`：是否通过可交易性、流动性和风险检查；
- `risk_bucket`：low / medium / high / blocked；
- `max_participation_rate`：研究模拟里的最大参与率。

这些输出不是券商订单，也不是实盘交易指令。它们是研究系统里的 portfolio intent，
用来让 pipeline 的最终判断可以被调用、保存、比较和评估。

## 为什么不是 broker / order / execution？

因为这个项目的定位是 reproducible quant research baseline，不是 live trading
system。

如果现在加入 broker、账户、订单、实盘服务，项目性质会立刻变复杂：

- 需要处理凭证、安全、权限和账户风险；
- 需要处理交易所规则、订单状态、撤单和部分成交；
- 需要真实执行成本、滑点、容量和风控监控；
- README 里的研究结果也更容易被误读成可部署 alpha。

所以 adapter 刻意停在 portfolio decision surface：

```text
判断意图：可以
执行下单：不做
账户接入：不做
实盘服务：不做
```

这反而让它更适合作为公开研究项目的接口。边界清楚，别人可以复现，也可以接到自己
的研究系统里评估。

## 三步跑通：build-state -> decide -> evaluate

先运行项目自带 demo：

```bash
python -m pip install mlquantx
mlquant demo
```

它会生成标准 pipeline artifacts，包括：

```text
artifacts/small/panel.pt
artifacts/small/predictions.pt
artifacts/small/weights.pt
artifacts/small/summary.json
```

然后把 artifacts 转成 decision state：

```bash
mlquant systemone build-state \
  --artifacts-dir artifacts/small \
  --as-of-index -2 \
  --output artifacts/systemone/decision_state.json
```

再生成 typed portfolio decisions：

```bash
mlquant systemone decide \
  --state artifacts/systemone/decision_state.json \
  --output artifacts/systemone/portfolio_decision.json
```

如果要做离线评估，可以重建一个带 realized forward return 的 state：

```bash
mlquant systemone build-state \
  --artifacts-dir artifacts/small \
  --as-of-index -2 \
  --include-realized \
  --output artifacts/systemone/decision_state_eval.json
```

然后评估已经生成的 decision：

```bash
mlquant systemone evaluate \
  --state artifacts/systemone/decision_state_eval.json \
  --decision artifacts/systemone/portfolio_decision.json \
  --output artifacts/systemone/portfolio_decision_eval.json
```

注意：`--include-realized` 只用于 evaluation，不能用于 decision。也就是说，决策时
只能看 as-of 当时的 pipeline state；评估时才使用下一期实际收益。

## state 里有什么？

一个 decision state 可以包含：

- `universe`：资产列表；
- `predictions.score`：模型或因子 pipeline 给出的预测分数；
- `predictions.rank`：横截面分位排名；
- `tradability`：停牌、涨停、跌停、可交易性；
- `liquidity`：成交额、20 日 ADV 等流动性信息；
- `portfolio.current_weight`：当前持仓权重；
- `portfolio.target_weight_baseline`：pipeline 原始目标权重；
- `risk.asset_volatility_20d`：资产短期波动；
- `cost_model.effective_cost_bps`：有效交易成本；
- `evidence`：数据来源、artifact 路径和研究边界。

从工程角度看，这比“输出一个自然语言解释”更有用，因为它能被机器读取、被测试、
被版本化，也能被外部 agent 或评测框架调用。

## evaluate 在评估什么？

当前 evaluator 是一个离线研究诊断，不是交易绩效证明。它会输出：

- `action_label_accuracy`：increase / hold / decrease / exclude 是否匹配下一期收益
  和可交易性形成的简单 hindsight label；
- `trade_allowed_state_consistency`：`trade_allowed` 是否与 state 中的可交易性、流动性
  和风险检查一致；
- `weight_delta_direction_accuracy`：目标权重变化方向是否与下一期收益方向一致；
- `target_weight_forward_return`：目标权重在下一期的 realized return；
- `target_minus_current_forward_return`：相对当前权重的增量收益；
- `one_period_turnover` 和 `estimated_one_period_cost`：一期间换手和估算成本；
- `hindsight_regret`：同等暴露下相对于 hindsight 最优资产的差距。

这些指标的意义是检查 typed decision surface 是否有可评估性。它不会证明策略能实盘，
但会让“这个 pipeline 如何做判断”变得可审计。

## 它和原来的 213 因子项目是什么关系？

原项目的主线仍然是：

```text
213 mask-aware factors
  -> ML baselines
  -> Markowitz / portfolio construction
  -> cost-aware backtest
  -> public-data validation
```

decision adapter 是在这条链路后面加的一层接口：

```text
pipeline artifacts -> decision state -> typed answers -> offline evaluation
```

它不替代因子研究，不替代回测，也不替代论文里的 mask-first contribution。它的价值是把
原本散落在报告、权重和预测文件里的判断逻辑，变成一个可以复用的 contract。

这对三个群体比较有用：

- 量化研究者：可以把策略判断拆成 typed questions，而不是只看最终净值；
- agent / eval 开发者：可以让 coding agent 或 quant agent 对同一 state 产生可比较的
  typed answers；
- 工程团队：可以在不接入 broker 的前提下，把研究 pipeline 包装成可调用的 decision
  interface。

## 什么情况下它才真正有价值？

如果只是写一个 `decide()`，价值有限。现在补上的关键是闭环：

```text
build-state：从真实 pipeline artifacts 生成状态
decide：输出类型化 portfolio intent
evaluate：用离线 realized returns 做诊断
```

这让“量化 Jev-style”不只是概念，而是可以跑的东西。别人可以换自己的 artifacts、
自己的股票池、自己的 cost assumption，然后比较同一套 typed questions 下的结果。

## 当前限制

仍然需要明确：

- 这是 deterministic / heuristic adapter，不是训练好的概率模型；
- 当前不发布 calibrated probabilities；
- evaluation 使用的是简单 hindsight label，不是严肃的因果检验；
- public-data 和 synthetic 结果不能被描述成可部署 alpha；
- 不提供 broker、order、account、execution、live service。

这些限制不是缺陷，而是公开研究项目应该保留的边界。

## 我希望它成为怎样的参考？

我不希望它只是“又一个量化回测仓库”。更理想的定位是：

> 一个可复现的多因子研究 baseline，同时提供一个量化 Jev-style typed portfolio
> decision surface，用于把 pipeline state 转换成可调用、可评估、可审计的组合判断。

如果你正在做量化 agent、System One/Jev-style 模型接口、或者希望把研究 pipeline 变成
结构化 decision API，可以直接从这里开始：

**GitHub：**
<https://github.com/initial-d/ml-quant-trading>

**完整 adapter 文档：**
<https://github.com/initial-d/ml-quant-trading/blob/main/docs/systemone_adapter.md>

**零账号 Colab：**
<https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/main/notebooks/quickstart_colab.ipynb>
