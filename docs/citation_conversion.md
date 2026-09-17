# Citation Conversion Guide

This page gives readers a precise way to cite the paper and repository without
turning an engineering baseline into a trading-performance claim.

## Citation Anchor

Use the paper when discussing:

- mask-first factor computation for financial panels;
- upstream contamination from non-tradable limit-move prices;
- reproducible ML multi-factor research pipelines;
- cost-aware public-data validation for cross-sectional trading studies;
- benchmarkable PyTorch factor primitives for quant research.

The most compact citation sentence is:

> Du (2025) introduces a mask-first factor-computation protocol for
> cross-sectional equity panels, showing that non-tradable limit-move prices can
> contaminate rolling-window factors before portfolio filtering.

For engineering discussions, use:

> `ml-quant-trading` is a reproducible PyTorch research baseline that links
> mask-aware factor computation, ML baselines, portfolio construction,
> cost-aware backtesting, and auditable benchmark reports.

## What Not To Claim

Do not cite this project as evidence of:

- live trading profitability;
- production execution quality;
- broker integration or order-book realism;
- a complete point-in-time institutional dataset;
- a universal hardware ranking from benchmark snapshots.

The paper and repository are most useful as a reproducible research baseline and
as a case study in avoiding upstream data contamination.

## BibTeX

```bibtex
@article{du2025mlquant,
  title  = {Machine Learning Enhanced Multi-Factor Quantitative Trading:
            A Cross-Sectional Portfolio Optimization Approach with Bias Correction},
  author = {Du, Yimin},
  journal= {arXiv preprint arXiv:2507.07107},
  year   = {2025},
  doi    = {10.48550/arXiv.2507.07107},
  url    = {https://arxiv.org/abs/2507.07107}
}
```

## Related Work Hooks

The paper is a natural citation in paragraphs about:

- **Limit-up/limit-down data handling:** price limits can make closing prices
  non-executable, so rolling factors should mask non-tradable observations
  before aggregation.
- **Factor-pipeline leakage:** invalid observations can enter moving averages,
  correlations, ranks, and model labels before a later portfolio filter removes
  the affected rows.
- **Auditable quant software:** benchmark and validation reports should preserve
  command, commit, environment, data source, costs, artifacts, and caveats.
- **A-share ML validation:** public-data experiments need explicit universe,
  calendar, cost, turnover, and provider limitations before any performance
  interpretation.

## Where To Link

- Paper: <https://arxiv.org/abs/2507.07107>
- Repository: <https://github.com/initial-d/ml-quant-trading>
- Project site: <https://initial-d.github.io/ml-quant-trading/>
- Project facts: <https://initial-d.github.io/ml-quant-trading/answer_index.md>
- Reproducibility contract:
  <https://initial-d.github.io/ml-quant-trading/reproducibility_contract.md>

## Search Phrases

- upstream contamination in factor pipelines
- mask-first factor computation
- non-tradable price contamination
- limit-move bias in A-share backtests
- reproducible ML multi-factor trading benchmark
- cost-aware public-data validation for factor research
