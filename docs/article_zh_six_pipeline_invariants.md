# 别先看夏普比率：我给 213 因子量化流水线加了 6 项工程审计

讨论量化回测时，人们通常先看年化收益、夏普比率和模型结构。但一条流水线
完全可能在画出漂亮曲线的同时，把更基础的问题做错：**哪些数据在当时可见、
哪些单元格有效、仓位究竟从哪一期开始获得收益。**

我给开源项目
[`ml-quant-trading`](https://github.com/initial-d/ml-quant-trading)
新增了一套确定性的技术审计。项目使用 PyTorch 计算 213 维因子，并贯穿模型、
组合构建和含交易成本回测。审计不试图证明策略存在 Alpha，而是检查在讨论收益
之前必须成立的 6 个工程不变量。

## 一条命令复现

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e '.[dev]'
python scripts/technical_pipeline_audit.py
```

整个审计只使用固定随机种子的合成数据，不需要行情账户，并同时输出 Markdown
和 JSON。维护者参考结果保存在
[`docs/audits/technical-audit-20260809`](audits/technical-audit-20260809/technical_audit.md)。

## 1. 因子目录必须有明确的形状契约

审计直接计算完整注册表，而不是相信配置文件里写着“213 因子”：

```python
factors, factor_mask, names = compute_legacy_set(panel)

assert factors.shape == (180, 40, 213)
assert len(names) == 213
assert torch.isfinite(factors).all()
```

参考运行得到有限值张量 `[180, 40, 213]`。这可以抓住因子未注册、维度顺序
错误，以及被后续聚合隐藏的 NaN/Inf 传播。

## 2. 合成数据生成必须逐位可复现

合成数据不是市场证据，但非常适合工程验证。同一个配置和随机种子生成的两个
面板必须完全一致：

```python
first = make_synthetic_panel(config)
second = make_synthetic_panel(config)

assert torch.equal(first.close, second.close)
assert torch.equal(first.volume, second.volume)
assert torch.equal(first.mask, second.mask)
```

这样贡献者无需行情账号就能建立稳定基线，再逐步切换到 yfinance、AkShare、
Baostock 或持牌数据。

## 3. 被屏蔽的数据不能污染有效输出

停牌、上市前区间、涨跌停和缺失值都不应该被当作普通的零。项目因此在
`[日期, 资产]` 张量旁传递布尔掩码。

审计采用“投毒测试”：把所有无效输入替换成数量级为 `1e9` 的极端值，然后
重新计算，并只比较两次运行中共同有效的单元格：

```python
poisoned_x = torch.where(mask, x, torch.full_like(x, 1e9))

clean, clean_mask = ts_mean(x, mask, 20)
poisoned, poisoned_mask = ts_mean(poisoned_x, mask, 20)
valid = clean_mask & poisoned_mask

assert (clean - poisoned).abs()[valid].max() <= 1e-6
```

目前检查了横截面排名、滚动均值、滚动排名、滚动相关性和指数加权均值五个
核心算子。投毒后，有效位置的最大漂移为 `0`。

这里必须明确：这是对底层张量算子的工程不变量检查，不等于对每一个因子的
经济含义做了独立验证。

## 4. 前向标签不能越过样本边界

一周期收益标签只有在 `t` 和 `t+1` 都可交易时才有效：

```python
target[t] = close[t + 1] / close[t] - 1
target_mask[t] = tradable[t] & tradable[t + 1]
```

审计独立重建预期掩码并要求完全相等，同时要求最后一行标签为零且被屏蔽，
因为样本中不存在它的 `t+1`。

## 5. 仓位必须滞后一期再获得收益

回测引擎采用明确约定：

```text
weights[t-1] earn returns[t]
```

审计使用手工构造的两资产路径。在零成本条件下，预期总收益序列为
`[0.00, 0.10, -0.20, 0.30]`，实际输出必须逐项一致。

这比只检查最终净值更有意义。如果少了这次滞后，模型可能用同一期收益选择
仓位，再让该仓位获得同一期收益，形成典型的前视偏差。

## 6. 交易成本必须按费率线性变化

审计构造一个不断在两只资产之间切换、资产收益为零的组合，分别使用 1 bps
和 10 bps 费率运行。累计成本之比必须恰好为 10：

```python
one = run_backtest(weights, returns, costs_bps=1).cost_drag_cumulative
ten = run_backtest(weights, returns, costs_bps=10).cost_drag_cumulative

assert np.isclose(ten / one, 10.0)
```

这同时澄清了指标单位：`cost_drag_cumulative` 表示全回测区间累计支付的成本，
不是一个年化收益指标。

## 当前参考结果

提交 `575a71c` 的维护者运行通过了全部检查：

| 工程不变量 | 结果 |
|---|---:|
| 213 因子张量形状与有限值 | PASS |
| 确定性合成数据生成 | PASS |
| 五个核心算子的掩码隔离 | PASS |
| 前向标签边界与端点掩码 | PASS |
| 滞后执行时序 | PASS |
| 交易成本线性 | PASS |

## 它没有证明什么

这套审计没有证明因子拥有样本外预测能力，也没有证明合成数据代表真实市场，
更没有证明系统可以直接用于实盘。后续仍然需要公开或持牌行情、滚动验证、
强基线、滑点与成交约束，以及独立复现。

它只把讨论提前了一步：在问模型能不能赚钱以前，先确认代码表达的时序、掩码
和成本，确实是我们以为的那个意思。

快速体验：

```bash
python -m pip install mlquantx
mlquant demo
```

源码与复现入口：
[github.com/initial-d/ml-quant-trading](https://github.com/initial-d/ml-quant-trading)。

