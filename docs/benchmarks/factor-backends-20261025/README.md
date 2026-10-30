# Factor backend integration and numerical audit (2026-10-25)

Synthetic CPU engineering evidence only. No market data, backtest, costs,
predictive quality or investment claim. Review: [PR #81](https://github.com/initial-d/ml-quant-trading/pull/81).

The factor APIs and feature CLI now accept an explicit `rolling_backend="linear"`
for mean/std/cov/corr. The default remains `unfold`. Linear uses disjoint FP64
block prefixes/suffixes and combines central moments; it avoids expanded rolling
windows and subtracting cumulative variance statistics across the whole history.
Its arithmetic/storage are O(T*N), including padding bounded by another T*N.

## Full engine: a memory/CPU tradeoff

Both backends compute all 213 factors, cross-sectional z-scores and a contiguous
`[T,N,213]` output. Fresh subprocess per backend, 2 warmups, 5 measured repeats,
seed 42, PyTorch threads 1/1. Input creation, serialization and diagnostics are
outside timing; the process RSS snapshot precedes saving and diagnostics.

| Synthetic panel | Unfold median | Linear median | Unfold peak RSS | Linear peak RSS | RSS reduction |
|---|---:|---:|---:|---:|---:|
| 500 x 200 | 0.541 s | 0.816 s | 617.1 MB | 593.3 MB | 3.9% |
| 750 x 1000 | 4.706 s | 6.818 s | 2648.4 MB | 2100.4 MB | 20.7% |

**The full engine is slower with linear moments here (51% / 45% more elapsed time).**
Most existing formulas use short windows and unchanged ranks/extrema/EWMA; long-
window kernel gains do not establish a full-pipeline speedup. RSS includes Python,
tensors and cached allocations. It is not a precise count of kernel allocations.

All 213 individual factor masks agreed in both panels; outputs were finite and
joint masks matched the intersection of individual masks. Values are not
interchangeable: 32 / 41 factors had a maximum valid difference above 1e-5.
For example `old_071` nests time-series ranks around correlations of discrete
rank inputs. On 750 x 1000, its mean sampled Spearman agreement was 0.9770,
maximum percentile-rank shift 0.7125 and maximum z-scored difference 2.5568.
These are comparisons, not acceptance tolerances. Value/mask diagnostics cover
each factor's own valid cells; ranks sample 25 dates, including warmup rows.
The independent sampled FP64 rank-correlation oracle checks a moment used in
`old_063`; it is not an oracle for the entire nonlinear factor library.

## Long-window kernel result

The repaired implementation was rerun separately on the original finite normal
750 x 1000 panel, with 10% missing observations and the same threads/repeats.

| Kernel | Window | Unfold median | Linear median | Speedup | Unfold / linear peak RSS |
|---|---:|---:|---:|---:|---:|
| mean | 20 | 5.43 ms | 15.30 ms | 0.36x | 320.2 / 330.0 MB |
| mean | 60 | 15.22 ms | 12.94 ms | 1.18x | 427.6 / 331.2 MB |
| mean | 252 | 47.66 ms | 12.49 ms | 3.82x | 761.0 / 330.0 MB |
| corr | 20 | 28.97 ms | 32.54 ms | 0.89x | 513.9 / 417.8 MB |
| corr | 60 | 72.87 ms | 29.46 ms | 2.47x | 941.7 / 421.4 MB |
| corr | 252 | 223.67 ms | 30.08 ms | 7.44x | 2284.1 / 416.4 MB |

All kernel masks agreed; maximum valid mean/corr differences were 3.06e-5 /
3.58e-7. Sampled independent FP64 oracles and regression tests passed. The 252-
day correlation RSS reduction was 81.8%, for this kernel and environment only.

## Preserved numerical failure and repair

The pre-fix global-prefix implementation ran successfully as a process but
produced inaccurate variances after large past outliers. Its raw reports remain
in [pre-fix/500x200.json](pre-fix/500x200.json) and
[pre-fix/750x1000.json](pre-fix/750x1000.json), source commit
`02e703e649f00f152c214b29f2bfca26196595dd`. The 750 x 1000 `better_020` comparison
had a maximum z-scored difference of 4.0807. A first valid signal near 0.03
followed by windows near 1e-9 caused subtractive cancellation of central sums.

The new block method centres each prefix/suffix at an observation contained in
that piece and merges the two disjoint summaries. On the same panel, the maximum
`better_020` difference fell to 2.38e-6. Tests compare independent direct FP64
windows after multiple 0.02–0.04 outliers, holes and block boundaries. Extreme
within-window dynamic ranges can still need a direct FP64 reference; the method
is not exact arithmetic. Clamping correlations, finite filtering and FP64 tie
changes still explain material differences in nonlinear rank pipelines.

## Reproduction and raw evidence

Measured code commit: `2620ac1` (full SHA in JSON). Apple M5, macOS 26.6.2 arm64,
Python 3.12.14, PyTorch 2.14.1, NumPy 2.5.4, SciPy 1.18.1, CPU only. No CUDA
throughput measured; MPS lacks FP64 and is rejected. Full-engine input:
`SyntheticConfig` GBM, halt probability .005, limit mask .098 applied before
factors. Identical input hashes and exact commands/environment are in each JSON.

```bash
python scripts/benchmark_factor_backends.py --n-dates 500 --n-stocks 200 \
  --warmup 2 --repeat 5 --seed 42 --json-out artifacts/factor-backends-500x200.json
python scripts/benchmark_factor_backends.py --n-dates 750 --n-stocks 1000 \
  --warmup 2 --repeat 5 --seed 42 --json-out artifacts/factor-backends-750x1000.json
python scripts/benchmark_rolling_moments.py --n-dates 750 --n-stocks 1000 \
  --windows 20,60,252 --repeat 5 --warmup 2 --seed 42 --json-out artifacts/rolling-blocks.json
```

Raw full-engine [500 x 200](500x200.json) / [750 x 1000](750x1000.json),
run output [small](500x200.txt) / [large](750x1000.txt),
[kernel JSON](rolling-kernels.json) / [output](rolling-kernels.txt),
[figure](window-sweep.svg), [local checks](local-checks.json).
The earlier October 24 global-prefix kernel report remains a historical record;
its 17.57x result does not describe the repaired block implementation.
