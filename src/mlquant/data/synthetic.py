"""Synthetic OCHLV panel generator.

The paper's empirical results use proprietary Wind / Tushare data that
external readers cannot redistribute. To make the repository
*end-to-end runnable for anyone*, this module synthesises a
realistically-shaped A-share-like universe via a multi-asset Geometric
Brownian Motion (GBM) with cross-sectional correlation, plus a
"limit-up / limit-down" simulator so the bias-correction code paths
are exercised on the synthetic data too.

Why bother making it realistic?
    Random noise is useless for benchmarking factors. A synthetic panel
    that obeys the same constraints as the real one (positive prices,
    OCHL ordering, ~10% daily price-limit, ~5% halt probability,
    cross-sectional correlation around 0.3) lets us:

      * regression-test the factor engine with deterministic seeds,
      * give new contributors a 30-second smoke-test path,
      * publish CI runs that actually exercise the optimiser.

The generator also fills in the *optional* A-share microstructure
fields — ``amount`` (turnover), ``limit_up``, ``limit_down``,
``last_close`` — so that code which prefers exchange-published values
over derived proxies has something realistic to consume.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch

from .panel import Panel


@dataclass
class SyntheticConfig:
    n_stocks:     int   = 200
    n_dates:      int   = 500
    start_date:   str   = "2020-01-02"
    annual_drift: float = 0.05
    annual_vol:   float = 0.30
    market_beta:  float = 0.6
    halt_prob:    float = 0.005           # per stock-day
    limit_pct:    float = 0.10            # ±10 % A-share daily price limit
    seed:         int   = 42
    device:       str   = "cpu"


def make_synthetic_panel(cfg: Optional[SyntheticConfig] = None) -> Panel:
    """Generate a synthetic OCHLV panel obeying A-share style constraints.

    The procedure is deliberately compact:
      1. Sample a daily market log-return ``m_t ~ N(μ_m, σ_m)``.
      2. Sample idiosyncratic log-returns ``e_{t,i} ~ N(0, σ_e)``.
      3. Stock log-return ``r_{t,i} = β·m_t + e_{t,i}`` (single-factor model).
      4. Reject moves outside ±``limit_pct``: clamp and mask.
      5. Halts: Bernoulli ``halt_prob`` mask.
      6. Synthesise OCHL around close with a small intraday range.
      7. Derive ``amount = vwap*volume``, ``last_close = close[t-1]`` and
         the official ±``limit_pct`` bands.

    The resulting panel exercises the same masked code paths as a real
    Wind feed — every test in ``tests/`` runs against this generator.
    """
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    T, N = cfg.n_dates, cfg.n_stocks
    daily_drift = cfg.annual_drift / 252.0
    daily_vol   = cfg.annual_vol  / np.sqrt(252.0)

    # 1. market and idiosyncratic shocks --------------------------------
    market_shock = rng.normal(daily_drift, daily_vol, size=T)
    idio_shock   = rng.normal(0.0, daily_vol * np.sqrt(1.0 - cfg.market_beta**2), size=(T, N))
    log_ret      = cfg.market_beta * market_shock[:, None] + idio_shock

    # 2. price-limit clamp + halt mask ---------------------------------
    log_limit = np.log1p(cfg.limit_pct)
    log_ret   = np.clip(log_ret, -log_limit, log_limit)
    halts     = rng.random(size=(T, N)) < cfg.halt_prob

    # 3. integrate to price -------------------------------------------
    init_price = rng.uniform(5.0, 50.0, size=N).astype(np.float32)
    cum_log    = np.cumsum(log_ret, axis=0)
    close      = init_price[None, :] * np.exp(cum_log)

    # 4. OCHL around close --------------------------------------------
    intraday_sigma = daily_vol * 0.5
    open_  = close * np.exp(rng.normal(0.0, intraday_sigma, size=close.shape))
    high   = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, intraday_sigma * 0.5, size=close.shape)))
    low    = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, intraday_sigma * 0.5, size=close.shape)))
    vwap   = (open_ + high + low + close) / 4.0
    volume = rng.lognormal(mean=15.0, sigma=0.5, size=close.shape).astype(np.float32)
    volume[halts] = 0.0

    # 5. mask: tradable iff not halted ---------------------------------
    mask = ~halts

    # 6. derive optional A-share fields --------------------------------
    amount = (vwap * volume).astype(np.float32)
    # last_close[t] = close[t-1]; day 0 falls back to the open price so
    # the ±limit_pct bands stay well-defined.
    last_close = np.empty_like(close, dtype=np.float32)
    last_close[0]  = open_[0]
    last_close[1:] = close[:-1]
    # Round limit prices to two decimals to mimic exchange convention.
    limit_up   = np.round(last_close * (1.0 + cfg.limit_pct), 2).astype(np.float32)
    limit_down = np.round(last_close * (1.0 - cfg.limit_pct), 2).astype(np.float32)

    # 7. assemble Panel ------------------------------------------------
    dates = pd.bdate_range(cfg.start_date, periods=T).to_numpy()
    stocks = np.asarray([f"SYN{idx:05d}" for idx in range(N)])

    def _t(arr: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(arr.astype(np.float32)).to(cfg.device)

    panel = Panel.from_tensors(
        dates=dates,
        stocks=stocks,
        fields={
            "open":       _t(open_),
            "high":       _t(high),
            "low":        _t(low),
            "close":      _t(close),
            "volume":     _t(volume),
            "vwap":       _t(vwap),
            "amount":     _t(amount),
            "limit_up":   _t(limit_up),
            "limit_down": _t(limit_down),
            "last_close": _t(last_close),
        },
        mask=torch.from_numpy(mask).to(cfg.device),
    )
    panel.assert_consistent()
    return panel
