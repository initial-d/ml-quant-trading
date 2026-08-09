# v0.2.6 - PyPI-to-GitHub Contributor Path

`v0.2.6` keeps the one-command PyPI experience and adds a clear path from a
successful demo to the source code, reproducible reports, and contributions.

## Install and run

```bash
python -m pip install --upgrade mlquantx
mlquant demo
```

The PyPI distribution is `mlquantx`; the import package and CLI remain
`mlquant`.

## What changed

- Show the GitHub repository and reproduction-report form after all five demo
  stages finish successfully.
- Add traceable repository and report-form links to generated Markdown and JSON
  summaries.
- Put the source checkout beside the PyPI quick start across the main, Chinese,
  PyPI, and documentation landing pages.
- Add dedicated source and Discussions links to PyPI project metadata.
- Preserve the research boundary: the deterministic demo validates the
  engineering path, not live or out-of-sample profitability.

## Customize the pipeline

```bash
git clone https://github.com/initial-d/ml-quant-trading.git
cd ml-quant-trading
python -m pip install -e '.[dev]'
```

Use the source checkout to change factors, models, data sources, portfolio
constraints, or backtest assumptions.
