# Forward-label endpoint regression — 2026-10-24

Code commit: `13f81da5f8ecc5dc6270a09c50188b6c54938b7a`. [Mask contract](../../mask_contract.md).

```bash
python -m pytest
python -m ruff check src tests
```

Result: 173 tests passed; Ruff passed. [Raw test output](pytest.txt).
Python 3.12.14, PyTorch 2.14.1, macOS-26.6.2-arm64-arm-64bit,
Apple M5, CPU only; no GPU use. Input source: deterministic synthetic and
unit-test fixtures, including NaN, infinite, nonpositive prices and overflow
controls. No external market data. Cost assumptions are not applicable to
label-construction regression; this is engineering evidence, not an alpha or
performance estimate.

Invalid endpoints no longer produce a true label mask or a clamped extreme
return. Masked NaN/Inf cells yield zero values with false masks. Genuine zero
returns remain valid. Multi-row horizons check endpoints only; they do not prove
intermediate-session coverage or order feasibility. Positive finite closes whose
computed float32 return overflows are also excluded. This can change training
sample counts on affected input data; no archived market result is revalidated.
