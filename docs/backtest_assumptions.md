# Backtest Assumptions and Limitations

This document explains how to interpret backtest results produced by `ml-quant-trading`. It is intended to make research assumptions explicit before users compare Sharpe ratios, annual returns, drawdowns, or factor IC values.

The project is for research and engineering experimentation. It is not financial advice, investment advice, or a trading recommendation.

## What the Backtest Measures

The backtest engine is designed to evaluate cross-sectional prediction and portfolio construction workflows. It helps answer questions such as:

- Does a factor or model rank securities in a useful direction?
- How stable are IC, RankIC, turnover, drawdown, and risk-adjusted returns?
- How sensitive is the result to masks, bias correction, transaction costs, and portfolio constraints?
- Can the same pipeline run on synthetic data and user-provided real data?

It should not be interpreted as a live trading system without additional execution, risk, compliance, and data-quality work.

## Data Availability

The repository supports public or synthetic reproduction paths, but the paper's full real-market experiments depend on proprietary A-share data that cannot be redistributed here.

When reporting results, include:

- data source and license constraints
- date range
- stock universe definition
- adjustment method for prices and corporate actions
- delisting and suspension handling
- whether the universe is point-in-time or reconstructed after the fact

## Survivorship Bias

Survivorship bias can materially overstate backtest performance. A realistic equity universe should include delisted names, suspended names, and securities that were tradable at the time, not only securities that survived until the end of the sample.

If your data source does not provide a point-in-time universe, treat results as exploratory rather than production-grade.

## Lookahead and Data Leakage

Cross-sectional ML pipelines are especially vulnerable to leakage. Common risks include:

- using post-event membership lists for historical universes
- calculating features with future-adjusted data not available at the prediction time
- normalizing across a time range that includes validation or test data
- using future returns indirectly through target construction, masking, or filtering
- tuning hyperparameters repeatedly on the same test period

A publishable result should document the train, validation, and test split, plus any gap or purging rule used to prevent overlap.

### What `mlquant demo` does

The CLI pipeline splits by date. `train_frac` (default `0.6`) sets the first
out-of-sample date index; the model trains only on earlier dates, its
validation set is the most recent slice of complete training dates
(`val_split_mode: time`), and the backtest scores only the out-of-sample window.
`summary.json` records both windows under `evaluation`. The one-day label of the
last training date uses the close of the first out-of-sample date, which is
already known when the first out-of-sample signal is formed, so no purge gap is
needed for the default one-day horizon; longer horizons need one.

Set `train.purge_dates` (default `0`) for an explicit gap immediately before
the trainer's time-validation boundary. It counts complete observed signal dates
in `FactorDataset`, not calendar days, and keeps the latest validation dates fixed.
For an h-period label known at close of t+h, a gap of at least h-1 signal periods
keeps the last training endpoint at or before the first validation signal close;
use h if it must be strictly earlier. Sparse observations can make an observed-date
gap more conservative. Select the gap from the actual label horizon, data timing
and research protocol: the trainer does not infer that information. This gap
controls the internal training/validation split; the CLI's outer out-of-sample
boundary and separate walk-forward scripts need their own horizon-aware policy.
Nonzero gaps require `val_split_mode="time"` and at least one remaining training
date. `Trainer.split_info` and `save_dir/split.json` record all three partitions'
sample/date counts and numeric date-ID ranges so the chosen split can be inspected.

`Trainer` holds out whole latest observed dates for `val_split_mode="time"`;
`val_split` therefore denotes a fraction of dates, not individual stock-days.
The default `batch_mode="auto"` uses full daily cross-sections for `ICLoss` and
`RankICLoss`, with one optimizer step per date and equally weighted daily losses.
`batch_size` is ignored for daily batching: memory depends on the largest daily
cross-section. Soft RankIC uses `chunk_size=256` by default above that many
stocks, with N × chunk_size pairwise matrices and activation checkpointing during
training. `RankICLoss(chunk_size=None)` retains the dense reference. Pairwise work
remains quadratic and backward recomputation adds runtime; reduction rounding
can differ across blocks/devices. This does not bound model activations or total
training memory, and no GPU performance estimate is provided here.
Use `batch_mode="date"` for a custom daily objective. Ordinary regression uses
flat sample batches and sample-weighted epoch loss; it retains the final partial
batch. Explicit `val_split_mode="random"` splits whole dates in daily mode and
individual samples in flat mode. Random splits are reproducibility options,
not chronological out-of-sample evidence. Constant/singleton IC batches contribute
zero loss and zero gradients; they are retained, not counted as informative IC.

This is a single split, not a walk-forward evaluation. Demo metrics come from
synthetic data and are expected to hover around zero or below after costs.
`scripts/public_data_validation.py` runs rolling walk-forward splits.

## Limit-Move Masks

Without exchange-published `limit_up` / `limit_down` prices, limit days are
detected with a `|return| > limit_pct` proxy. A scalar `limit_pct` (default
`0.098`) only fits Main Board names. Set `limit_pct: board` in the config, or
pass `mlquant.features.board_limit_pct(stocks, dates)` to `limit_move_mask`,
to use ±20 % thresholds for STAR Market and ChiNext (ChiNext from 2020-08-24)
and ±30 % for the Beijing Stock Exchange. The ticker code cannot reveal ST
status (±5 %) or the unlimited first trading days after an IPO; supply real
limit prices when those cases matter.

## Transaction Costs

Backtest returns should be reported both before and after transaction costs. Cost assumptions should include commissions, taxes, fees, and any exchange-specific trading expenses that apply to the target market.

At minimum, report:

- cost model formula
- one-way or round-trip interpretation
- turnover definition
- whether costs are applied before or after portfolio normalization
- sensitivity to higher and lower cost assumptions

If cost assumptions are missing, returns are best interpreted as signal-quality diagnostics, not implementable strategy returns.

## Slippage and Market Impact

The default research path does not prove that trades can be executed at the modeled prices. Slippage and impact depend on liquidity, order size, participation rate, volatility, spread, queue priority, and market regime.

For realistic deployment, add assumptions for:

- execution price relative to open, close, VWAP, or next-bar price
- bid-ask spread
- market impact as a function of volume participation
- failed fills and partial fills
- trading halts and limit-up / limit-down constraints

Capacity analysis should be performed separately from factor-quality analysis.

## Limit-Up, Limit-Down, and Halt Handling

A-share research needs explicit handling for limit-up, limit-down, halt, and pre/post-listing states. These cases affect both feature computation and portfolio execution.

The project treats tradability masks as first-class data. Users should still verify that their data source represents limit and halt states consistently and that the mask matches the intended execution rule.

## Rebalancing and Holding Period

Performance can change sharply with rebalance frequency and holding period. When sharing results, include:

- rebalance frequency
- holding period
- portfolio turnover
- whether overlapping holdings are allowed
- whether weights are market-neutral, long-only, or constrained by sector or size

## Portfolio Constraints

Markowitz-style optimization is sensitive to expected returns, covariance estimates, risk aversion, and constraints. Report the solver, constraint set, covariance estimation method, and any no-short, leverage, or exposure constraints.

If a commercial solver such as MOSEK is used, include that fact. The default open-source solver path may produce slightly different numerical behavior.

## Minimum Reporting Checklist

When opening an issue, pull request, paper note, or benchmark result, include:

- commit SHA or release version
- config file
- random seed
- data source
- universe definition
- train / validation / test periods
- transaction-cost assumptions
- slippage or execution assumptions
- rebalance frequency and holding period
- key metrics: IC, RankIC, annual return, volatility, Sharpe, max drawdown, turnover, and cost drag

## Practical Interpretation

Use this project to study research pipelines, compare modeling choices, reproduce paper-style components, and build better quantitative research infrastructure. Treat any reported backtest as a hypothesis that still needs independent validation, execution modeling, robustness checks, and risk review before real capital is involved.
