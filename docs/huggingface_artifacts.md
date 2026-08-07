# Hugging Face Artifact Export

The project can publish a small dataset and model checkpoint without
redistributing proprietary or public-provider market data. Both artifacts come
from the deterministic synthetic quick start.

## Artifact repositories

| Hugging Face repository | Contents | Intended use |
|---|---|---|
| `dddyym/ml-quant-trading-synthetic` | Viewer-ready compressed CSV, generator config, checksum manifest, dataset card | Installation, CI, teaching, and pipeline smoke tests |
| `dddyym/ml-quant-trading-synthetic-mlp` | PyTorch checkpoint, factor order, architecture config, metrics, model card | Checkpoint loading and inference smoke tests |

Neither artifact represents real instruments, proprietary data, deployable
alpha, or evidence of live performance.

## Build locally

Run the deterministic pipeline, then export the two repository directories:

```bash
mlquant demo
python scripts/export_huggingface_artifacts.py \
  --artifacts-dir artifacts/small \
  --config configs/small.yaml \
  --output-dir artifacts/huggingface \
  --namespace dddyym
```

The exporter writes:

```text
artifacts/huggingface/
├── bundle_manifest.json
├── ml-quant-trading-synthetic/
│   ├── README.md
│   ├── artifact_manifest.json
│   ├── source_config.yaml
│   └── data/synthetic_ohlcv.csv.gz
└── ml-quant-trading-synthetic-mlp/
    ├── README.md
    ├── artifact_manifest.json
    ├── config.json
    ├── feature_names.json
    ├── metrics.json
    ├── pytorch_model.bin
    └── source_config.yaml
```

The dataset gzip stream uses a fixed timestamp so the same panel produces the
same SHA-256 digest. Manifests record the source commit and file checksums.

## Upload after authentication

Install and authenticate the Hugging Face CLI locally. Never paste a full token
into an issue, PR, notebook, or chat transcript.

```bash
python -m pip install --upgrade huggingface_hub
hf auth login
```

Create and upload the dataset repository:

```bash
hf repo create dddyym/ml-quant-trading-synthetic --repo-type dataset --exist-ok
hf upload dddyym/ml-quant-trading-synthetic \
  artifacts/huggingface/ml-quant-trading-synthetic . \
  --repo-type dataset
```

Create and upload the model repository:

```bash
hf repo create dddyym/ml-quant-trading-synthetic-mlp --exist-ok
hf upload dddyym/ml-quant-trading-synthetic-mlp \
  artifacts/huggingface/ml-quant-trading-synthetic-mlp .
```

After upload:

1. Confirm the dataset viewer renders rows and column types.
2. Run the model-card loading snippet in a clean environment.
3. Link both repositories from the
   [Hugging Face paper page](https://huggingface.co/papers/2507.07107).
4. Update [Issue #1](https://github.com/initial-d/ml-quant-trading/issues/1)
   with the live URLs and close it only after both smoke checks pass.

## Safety boundary

The exporter refuses configs without a `synthetic` section. It accepts only the
local demo artifacts supplied on the command line; it does not contain any
market-data downloader or Hub credential handling.
