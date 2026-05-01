# Factor catalogue

This page lists every factor shipped in the curated `mlquant.features`
library, with a one-line rationale and a pointer to its
implementation.

The full Alpha101++ extension (~600 factors) used in the paper's
ablations lives in [`legacy/`](../legacy/) for archival purposes; the
curated subset below is what the repository's tests pin and the
default config uses.

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

## Adding a new factor

1. Implement it on top of the tensor primitives — no Python loops
   over stocks or dates.
2. Decorate it with ``@_register("alphaXXX")`` in
   ``mlquant.features.alpha101``.
3. Add a one-liner to this file describing the rationale.
4. Add a smoke test (`tests/test_alpha101.py` already iterates the
   registry, so a registered factor is auto-tested for shape /
   finiteness; deeper tests welcome).
