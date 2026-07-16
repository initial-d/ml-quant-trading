# ml-quant-trading

Languages: [English](README.md) | [简体中文](README.zh-CN.md) | [繁體中文](README.zh-TW.md)

> 本文為中文摘要，完整內容請參閱 [English README](README.md)。

> **機器學習增強的多因子量化交易**——採用偏差修正的橫截面投資組合最佳化方法。
>
> [arXiv:2507.07107](https://arxiv.org/abs/2507.07107) &nbsp;|&nbsp; Yimin Du，2025

## 專案概述

`ml-quant-trading` 是一套簡潔、方便 fork 的端到端 A 股量化研究系統。程式碼庫整合了張量因子引擎、213 維因子庫、偏差修正、ML baseline、Markowitz 投資組合最佳化、向量化回測、synthetic 與公開資料示範，以及 CI、測試與 benchmark 工具，適合重現及延伸多因子量化研究。

## 研究與教學用途聲明

> **僅供研究與教學使用。** 本專案不構成金融或投資建議，也不是可直接用於實盤交易的正式交易系統。回測結果不代表真實交易表現，並會受到資料品質、交易成本、滑價及建模假設等因素影響。所有結果均應視為研究驗證，而非已證實的樣本外績效或可部署 alpha。已知限制請參閱 [Reality Check](docs/reality_check.md)。

## 核心模組

| 模組 | 功能 |
|------|------|
| `features.tensor_factors` | 帶有 mask 的 GPU 向量化基礎運算子（`rank`、`corr`、`ewma`、`ts_*`） |
| `features.legacy_factors` | 204 個手工建構的 alpha 因子（請參閱[因子手冊](docs/factor_handbook.md)） |
| `features.alpha101` | Alpha101 風格的公式化因子 |
| `features.neutralize` | 橫截面與產業中性化 |
| `features.bias` | 漲停、跌停與停牌偏差修正 |
| `training.augment` | GBM 資料增強 |
| `models.nets` | MLP / Transformer |
| `models.losses` | AdjMSE、IC、RankIC loss |
| `portfolio.markowitz` | 橫截面 Markowitz 最佳化（收縮共變異數、禁止放空） |
| `backtest.engine` | 向量化回測，輸出 Sharpe / IC / IR / DD 等指標 |

因子庫共包含 **213 個因子**：9 個精選 Alpha101 公式及 204 個手工建構的 legacy 因子。完整說明請參閱[因子手冊](docs/factor_handbook.md)。

## 資料來源

| 資料來源 | 市場 | 存取方式 | 說明 |
|----------|------|----------|------|
| [Baostock](http://baostock.com) | A 股 | 免費註冊 | 專案支援的 A 股資料載入器，需要帳號 |
| [yfinance](https://pypi.org/project/yfinance/) | 美股 / ETF | 公開存取，有速率限制 | 用於公開資料驗證與跨市場範例 |
| Synthetic | 不適用 | 零設定 | 以固定 seed 決定性產生的 GBM panel，用於 pipeline 冒煙測試 |

本程式碼庫不會重新散布市場資料。Baostock 與 yfinance 資料會透過載入腳本依需求下載；Synthetic 資料則根據固定 seed 決定性產生。

## 快速開始

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
pip install -e .[dev]        # 如需 CUDA，請加入 ,gpu；如需 MOSEK solver，請加入 ,mosek

# 30 秒冒煙測試（Synthetic：200 檔股票 × 500 天）
make paper CONFIG=configs/small.yaml
```

### 公開資料驗證（選用）

輕量範例請開啟 [`notebooks/public_factor_ic.ipynb`](notebooks/public_factor_ic.ipynb)。如需執行規模較大的 yfinance walk-forward 驗證：

```bash
python scripts/public_data_validation.py \
  --source yfinance \
  --preset us-large-100 \
  --max-tickers 100
```

執行方式與輸出說明請參閱 [Public-Data Validation](docs/public_data_validation.md)。這些結果僅用於驗證診斷，不是交易建議。

## 關鍵文件

- [英文 README](README.md)
- [Reality Check and Validation Status](docs/reality_check.md)
- [Public-Data Validation](docs/public_data_validation.md)
- [Public-Data Mini Reproduction](docs/public_data_mini_reproduction.md)
- [Architecture Overview](docs/architecture.md)
- [Factor Handbook（因子手冊）](docs/factor_handbook.md)
- [FAQ](docs/faq.md)
- [Contributing Guide](CONTRIBUTING.md)

## 引用

若本專案對你的研究有所幫助，請引用：

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

本專案採用 MIT License，詳見 [`LICENSE`](LICENSE)。
