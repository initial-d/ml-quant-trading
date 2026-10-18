# Predeclared revalidation scope — 2026-10-17

Recorded before external data requests or new backtest outcomes.
Base commit: `ccdcf16a7c0768dbb31fa22f91ae02b692f0fd9b`.

Primary question: does changing tomorrow's availability alter today's portfolio?
The two public validation scripts currently use the forward-label mask for
portfolio selection and MLP inference. Fix this causality issue first, retaining
forward-label filtering for historical training and IC estimates.

Public-data check: AkShare, CPU, forward-adjusted (`qfq`) daily OHLCV,
2021-01-01 to 2025-01-01. Probe `600519` and `000001` once each (loader retries=1).
If both succeed, expand to this fixed 20-name diagnostic universe:
`600519,000001,600036,601318,600900,601166,600030,000333,000651,000858,002415,002475,600276,601888,600887,601012,600309,601398,601288,601939`.
It is a deliberately chosen current large-cap sample, not point-in-time CSI 300
membership and not an exact rerun of the July 300-name panel. Do not cherry-pick
replacement symbols after seeing outcomes. If the probe fails, preserve the
failure and do not hammer upstream or silently switch providers.

Fixed run choices: 213 registered factors, daily cadence, top 20%, buffer exit
40%, walk-forward train/test/step=504/63/63, 1 epoch, batch 4096, hidden 32,
seed 7, covariance window 126, IC window 252, optimizer candidates 20,
weight cap 0.05, risk aversion 1, min raw coverage 90%; 5 bps costs + 2 bps
slippage, sensitivity 0/7/15/30 bps; 100 moving-block bootstrap samples with
20-day blocks. Costs are a research simplification, not execution estimates.

Keep the normalized public panel local and hash it; publish metadata, commands,
logs and derived summaries only. Document provider version, retrieval timestamp,
OS/Python/PyTorch/hardware, dependency freeze, CPU thread counts and warnings.
The original July panel is absent here, so a change versus July cannot be
attributed solely to a code fix. Preserve the old report and its numbers.

If public data is unavailable, deliver the causality fix and deterministic
regression evidence, plus the public-data blocker. Synthetic engineering
results do not replace public-market evidence or establish alpha.
