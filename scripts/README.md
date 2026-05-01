# `scripts/` — orchestration helpers around the `mlquant` package

Each script is a thin Click CLI that wires `mlquant` modules into a
specific reproduction or ablation. All useful pipelines are also
available as `mlquant <subcommand>`; these helpers exist for the
ablations referenced from `docs/reproducing_paper.md`.

| Script                  | Purpose                                                                |
|-------------------------|------------------------------------------------------------------------|
| `eval_factor_ic.py`     | Compute IC / RankIC of every alpha against forward returns.            |
| `plot_frontier.py`      | Sweep α and plot the efficient frontier (matplotlib optional dep).     |
| `ablation_bias.py`      | Backtest with / without limit-day masking; reproduces paper table 4.   |
| `ablation_loss.py`      | Backtest with MSE vs `AdjMSELoss(γ=0.1)`; reproduces paper table 5.    |
