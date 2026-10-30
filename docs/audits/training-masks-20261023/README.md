# Training-mask regression — 2026-10-23

Code commit: `5c6064cdb0d94423fdf5886cea4143ea33f483ee`. [Mask contract](../../mask_contract.md).
Raw checks: [pytest](pytest.txt), [environment](environment.json).

```bash
python -m pytest
python -m ruff check src tests
```

The clean Python 3.12.14 / PyTorch 2.14.1 environment on macOS
26.6.2, Apple M5 CPU passed 152 tests; Ruff passed. No GPU or external market data
was used. These are synthetic/unit-test engineering checks, with no trading
performance estimate and no applicable cost assumption.

The explicit forward-label mask preserves a valid observed return endpoint even
when next-day factor eligibility is false. Nonfinite training inputs are excluded.
The CLI uses this contract; callers omitting the new mask retain the old mask
intersection. Historical demo model metrics can therefore change. No archived
public/proprietary performance result is revalidated here.

The first run in an older Python 3.9.6 / PyTorch 2.8.0 environment failed three
existing tests: missing AkShare and SciPy numerical warnings promoted to errors.
[Full failure details](initial-environment-failures.txt) are preserved. A clean
`python -m pip install -e '.[dev]'` installation resolved those environment
failures without changing unrelated tests.
