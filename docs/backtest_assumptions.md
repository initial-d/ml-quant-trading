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
