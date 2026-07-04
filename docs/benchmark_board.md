# Community Benchmark Board

This page tracks benchmark reports shared by users. Submit results with the
`Benchmark result` issue template.

## How to Submit

```bash
python -m pip install -e .[dev]
make benchmark
```

Then open a benchmark issue and paste the printed Markdown table.

For larger panels:

```bash
python scripts/benchmark_tensor_factors.py \
  --device auto \
  --n-dates 1500 \
  --n-stocks 3000 \
  --window 20 \
  --repeat 5 \
  --warmup 2
```

## Results

| Contributor | Commit | OS | Python | PyTorch | CPU | GPU | Command | Notes |
|---|---|---|---|---|---|---|---|---|
| Maintainer | pending | pending | pending | pending | pending | pending | `make benchmark` | First report needed |

## What Makes a Good Benchmark Report

- The command is copy-pasted exactly.
- The commit SHA is included.
- CPU and GPU names are included.
- CUDA availability is stated.
- The result table is pasted without editing numbers.
- Any unusual conditions are documented, such as shared GPU, thermal throttling, or low memory.

## Why This Helps

Benchmark reports help users answer practical questions:

- Does GPU help for my panel size?
- Which operations dominate runtime?
- Is performance sensitive to PyTorch or CUDA version?
- What hardware is enough for a student or research workflow?
