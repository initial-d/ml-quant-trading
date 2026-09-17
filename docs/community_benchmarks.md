# Community benchmark reports

This repository gets the most value when users turn private experiments
into public, comparable evidence. If you run the pipeline on your own
hardware or market data, please share a short benchmark report in the
issue tracker.

Good reports help three groups:

* Researchers can compare factor-engine throughput, IC, Sharpe, turnover,
  and cost drag across machines and markets.
* Practitioners can decide whether the stack is worth adapting before
  committing private data or GPU time.
* Paper readers can trace claims back to commands, configs, and
  reproducible environment details.

For public submissions, use the
[reproducibility contract](reproducibility_contract.md) as the minimum evidence
standard. It separates engineering reproducibility from research robustness and
lists the fields needed for later audit.

## Minimal smoke report

Run the synthetic-data path first. It does not need proprietary data and
usually completes quickly enough to serve as an environment check.

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e .[dev]
make paper CONFIG=configs/small.yaml
```

Please include:

| Field | Example |
|---|---|
| Commit | `git rev-parse --short HEAD` |
| OS / Python | `Windows 11, Python 3.11` |
| CPU / GPU | `i7-12700K, RTX 4090` or `CPU only` |
| Config | `configs/small.yaml` |
| Runtime | `42 s end-to-end` |
| Key metrics | Sharpe, IC, turnover, max drawdown |
| Notes | Any solver, CUDA, or dependency changes |

For benchmark protocol v1 reports, also include the generated JSON artifact or
its checksum, the PyTorch thread settings, and any repeat-run instability.

## Public-data validation report

If you use public data, link the exact dataset source and keep the
transformation steps explicit. The most useful reports add one or more
of the following:

* a cross-market check outside A-shares;
* a cost or slippage stress table;
* bootstrap or confidence intervals for Sharpe / IC;
* a comparison between synthetic-data and real-data behavior;
* a failure case where the factor stack does not transfer.

Public-data reports should preserve the provider, retrieval timestamp, resolved
universe, date range, failed tickers, and a hash or fingerprint of the
normalized input panel when possible. Provider failures such as rate limits are
valid reports when they include enough detail for another user to diagnose the
same failure mode.

## How to cite the project

If this code, factor library, or benchmark protocol is useful in a paper,
report, blog post, or agent benchmark, cite:

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

For GitHub reports, also link the repository commit you ran so readers
can reproduce the exact code state.
