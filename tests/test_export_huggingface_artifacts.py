from __future__ import annotations

import csv
import gzip
import hashlib
import json
from pathlib import Path

import torch

from scripts.export_huggingface_artifacts import export_bundle


def _write_demo_artifacts(tmp_path: Path) -> tuple[Path, Path]:
    artifacts = tmp_path / "artifacts"
    (artifacts / "checkpoints").mkdir(parents=True)
    panel = {
        "dates": ["2026-01-01", "2026-01-02"],
        "stocks": ["SYN000", "SYN001"],
        "open": torch.tensor([[1.0, 2.0], [1.1, 2.1]]),
        "high": torch.tensor([[1.2, 2.2], [1.3, 2.3]]),
        "low": torch.tensor([[0.9, 1.9], [1.0, 2.0]]),
        "close": torch.tensor([[1.1, 2.1], [1.2, 2.2]]),
        "volume": torch.tensor([[100.0, 200.0], [110.0, 210.0]]),
        "vwap": torch.tensor([[1.05, 2.05], [1.15, 2.15]]),
        "mask": torch.tensor([[True, True], [True, False]]),
    }
    torch.save(panel, artifacts / "panel.pt")
    torch.save(
        {"factors": torch.zeros(2, 2, 2), "mask": panel["mask"], "names": ["f1", "f2"]},
        artifacts / "features.pt",
    )
    torch.save({"net.0.weight": torch.ones(2, 2)}, artifacts / "checkpoints" / "best.pt")
    (artifacts / "summary.json").write_text(
        json.dumps({"workflow": "synthetic", "metrics": {"sharpe": 0.0}}), encoding="utf-8"
    )
    config = tmp_path / "small.yaml"
    config.write_text("seed: 42\nhidden: 2\nsynthetic:\n  n_dates: 2\n  n_stocks: 2\n", encoding="utf-8")
    return artifacts, config


def test_export_bundle_is_viewer_ready_and_explicitly_synthetic(tmp_path: Path):
    artifacts, config = _write_demo_artifacts(tmp_path)
    output = tmp_path / "huggingface"

    manifest = export_bundle(
        artifacts_dir=artifacts,
        config_path=config,
        output_dir=output,
        namespace="example",
        source_commit="abc123",
    )

    dataset_dir = output / "ml-quant-trading-synthetic"
    model_dir = output / "ml-quant-trading-synthetic-mlp"
    data_path = dataset_dir / "data" / "synthetic_ohlcv.csv.gz"

    with gzip.open(data_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 4
    assert rows[-1]["tradable"] == "0"
    assert manifest["dataset_rows"] == 4
    assert "no real instruments" in (dataset_dir / "README.md").read_text(encoding="utf-8")
    assert "not as a market model" in (model_dir / "README.md").read_text(encoding="utf-8")
    assert json.loads((model_dir / "config.json").read_text())["in_dim"] == 2
    assert json.loads((model_dir / "feature_names.json").read_text()) == ["f1", "f2"]

    dataset_manifest = json.loads((dataset_dir / "artifact_manifest.json").read_text())
    assert dataset_manifest["data_sha256"] == hashlib.sha256(data_path.read_bytes()).hexdigest()


def test_export_bundle_refuses_non_synthetic_config(tmp_path: Path):
    artifacts, config = _write_demo_artifacts(tmp_path)
    config.write_text("seed: 42\n", encoding="utf-8")

    try:
        export_bundle(artifacts_dir=artifacts, config_path=config, output_dir=tmp_path / "out")
    except ValueError as exc:
        assert "synthetic" in str(exc)
    else:
        raise AssertionError("non-synthetic config should not be exportable")
