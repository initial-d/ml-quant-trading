# ml-quant-trading

Languages: [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

> 本文为中文摘要，完整内容请参阅 [English README](README.md)。

> **机器学习增强的多因子量化交易**——采用偏差修正的横截面投资组合优化方法。
>
> [arXiv:2507.07107](https://arxiv.org/abs/2507.07107) &nbsp;|&nbsp; Yimin Du，2025

## 项目概述

`ml-quant-trading` 是一个简洁、便于 fork 的端到端 A 股量化研究系统。仓库集成了张量因子引擎、213 维因子库、偏差修正、ML baseline、Markowitz 投资组合优化、向量化回测、synthetic 与公开数据演示，以及 CI、测试与 benchmark 工具，适合复现和扩展多因子量化研究。

## 研究与教学用途声明

> **仅用于研究和教学。** 本项目不构成金融或投资建议，也不是可直接用于实盘交易的生产系统。回测结果不代表真实交易表现，并会受到数据质量、交易成本、滑点及建模假设等因素影响。所有结果都应视为研究验证，而非已经证实的样本外收益或可部署 alpha。已知限制请参阅 [Reality Check](docs/reality_check.md)。

## 核心模块

| 模块 | 功能 |
|------|------|
| `features.tensor_factors` | 带 mask 的 GPU 向量化基础算子（`rank`、`corr`、`ewma`、`ts_*`） |
| `features.legacy_factors` | 204 个手工构建的 alpha 因子（参阅[因子手册](docs/factor_handbook.md)） |
| `features.alpha101` | Alpha101 风格的公式化因子 |
| `features.neutralize` | 横截面与行业中性化 |
| `features.bias` | 涨停、跌停与停牌偏差修正 |
| `training.augment` | GBM 数据增强 |
| `models.nets` | MLP / Transformer |
| `models.losses` | AdjMSE、IC、RankIC loss |
| `portfolio.markowitz` | 横截面 Markowitz 优化（收缩协方差、禁止卖空） |
| `backtest.engine` | 向量化回测，输出 Sharpe / IC / IR / DD 等指标 |

因子库共包含 **213 个因子**：9 个精选 Alpha101 公式和 204 个手工构建的 legacy 因子。完整说明请参阅[因子手册](docs/factor_handbook.md)。

## 数据源

| 数据源 | 市场 | 访问方式 | 说明 |
|--------|------|----------|------|
| [Baostock](http://baostock.com) | A 股 | 免费注册 | 项目支持的 A 股数据加载器，需要账号 |
| [yfinance](https://pypi.org/project/yfinance/) | 美股 / ETF | 公开访问，有速率限制 | 用于公开数据验证和跨市场示例 |
| Synthetic | 不适用 | 零配置 | 使用固定 seed 确定性生成的 GBM panel，用于 pipeline 冒烟测试 |

本仓库不重新分发市场数据。Baostock 与 yfinance 数据由加载脚本按需下载；Synthetic 数据则根据固定 seed 确定性生成。

## 快速开始

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
pip install -e .[dev]        # 如需 CUDA，请添加 ,gpu；如需 MOSEK solver，请添加 ,mosek

# 30 秒冒烟测试（Synthetic：200 只股票 × 500 天）
make paper CONFIG=configs/small.yaml
```

### 公开数据验证（可选）

轻量示例请打开 [`notebooks/public_factor_ic.ipynb`](notebooks/public_factor_ic.ipynb)。如需运行规模更大的 yfinance walk-forward 验证：

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --max-tickers 100
```

运行方式和输出说明请参阅 [Public-Data Validation](docs/public_data_validation.md)。这些结果仅用于验证诊断，不是交易建议。

## 关键文档

- [英文 README](README.md)
- [Reality Check and Validation Status](docs/reality_check.md)
- [Public-Data Validation](docs/public_data_validation.md)
- [Public-Data Mini Reproduction](docs/public_data_mini_reproduction.md)
- [Architecture Overview](docs/architecture.md)
- [Factor Handbook（因子手册）](docs/factor_handbook.md)
- [FAQ](docs/faq.md)
- [Contributing Guide](CONTRIBUTING.md)

## 引用

如果本项目对你的研究有帮助，请引用：

```bibtex
@article{du2025mlquant,
  title  = {Machine Learning Enhanced Multi-Factor Quantitative Trading:
            A Cross-Sectional Portfolio Optimization Approach with Bias Correction},
  author = {Du, Yimin},
  journal= {arXiv preprint arXiv:2507.07107},
  year   = {2025},
  url    = {https://arxiv.org/abs/2507.07107}
}
```

## License

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。
