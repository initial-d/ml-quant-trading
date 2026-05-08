# Factor catalogue

This page lists every factor shipped in the curated `mlquant.features`
library, with a one-line rationale and a pointer to its
implementation.

The full feature set comprises **213 factors**: **9 curated Alpha101
formulas** (`features.alpha101`) plus **204 hand-crafted legacy factors**
(`features.legacy_factors`) organised into nine thematic families. All
factors are mask-contract compliant and GPU-vectorised.

The legacy factors are ported from `legacy/features/Feature.py` into
modern tensor-based modules, auto-registered into `LEGACY_REGISTRY`
and accessible via `compute_legacy_set()`.

## Tensor primitives

All factors are built from these primitives in
``mlquant.features.tensor_factors``. Every primitive is mask-aware,
GPU-vectorised, and has a unit test cross-checking it against the
pandas reference.

| Primitive | Signature | Notes |
|-----------|-----------|-------|
| `cs_rank` | ``(x, mask) -> (rank, mask)`` | percentile rank in (0, 1], ties averaged |
| `cs_zscore` | ``(x, mask) -> (z, mask)`` | per-date standardisation |
| `ts_sum`, `ts_mean`, `ts_std` | ``(x, mask, w) -> ...`` | trailing window |
| `ts_min`, `ts_max`, `ts_rank` | ``(x, mask, w) -> ...`` | trailing window |
| `ts_corr`, `ts_cov` | ``(x, y, mask, w) -> ...`` | rolling pairwise |
| `ewma` | ``(x, mask, alpha) -> ...`` | float64 recurrence |
| `delay`, `delta` | ``(x, mask, k) -> ...`` | cross-time arithmetic |

## Curated alpha set

| Name | Idea | Module |
|------|------|--------|
| `alpha001` | momentum (rank of `ts_argmax(close, 5)`) | `alpha101.alpha_001` |
| `alpha002` | volume-vs-intraday-return reversal | `alpha101.alpha_002` |
| `alpha003` | open / volume rank divergence | `alpha101.alpha_003` |
| `alpha004` | low-quantile mean reversion | `alpha101.alpha_004` |
| `alpha006` | open / volume rolling correlation | `alpha101.alpha_006` |
| `alpha007` | 20-day deviation from mean, z-scored | `alpha101.alpha_007` |
| `alpha012` | sign(Δvol) × −Δclose | `alpha101.alpha_012` |
| `alpha053` | 9-day Δ(close-location) | `alpha101.alpha_053` |
| `alpha101` | intraday close location within range | `alpha101.alpha_101` |

## Legacy factor zoo (204 factors)

All factors from the original `Feature.py` are ported into modular
files under `src/mlquant/features/`:

| Module | Family | Count | Description |
|--------|--------|-------|-------------|
| `_factors_better.py` | `better_*` | 28 | Momentum/reversal variants |
| `_factors_best.py` | `best_*` | 21 | Close-location momentum |
| `_factors_old.py` | `old_*` | 50 | Classic alpha signals |
| `_factors_stock.py` | `stock_*` | 22 | Per-stock derived series |
| `_factors_extra.py` | `extra_*` | 14 | Turnover + amount features |
| `_factors_add.py` | `add_*` | 30 | Additional factor variants |
| `_factors_change.py` | `change_*` | 5 | Short-window velocity changes |
| `_factors_original.py` | `original_*` | 28 | Direct close/volume stats |
| `_factors_market.py` | `cs_rank_*` | 6 | Market breadth signals |

Usage:
```python
from mlquant.features import compute_legacy_set, LEGACY_REGISTRY

# Compute all 204 factors
factors, mask, names = compute_legacy_set(panel)

# Compute a subset
factors, mask, names = compute_legacy_set(panel, names=("best_001", "old_027"))
```

## Adding a new factor

1. Implement it on top of the tensor primitives — no Python loops
   over stocks or dates.
2. For Alpha101-style: decorate with ``@_register("alphaXXX")`` in
   ``mlquant.features.alpha101``.
3. For legacy-style: decorate with ``@register_legacy_factor("name")``
   in the appropriate ``_factors_*.py`` module.
4. Add a one-liner to this file describing the rationale.
5. Add a smoke test (both `test_alpha101.py` and `test_tensor_factors.py`
   iterate their registries, so registered factors are auto-tested for
   shape / finiteness; deeper tests welcome).
