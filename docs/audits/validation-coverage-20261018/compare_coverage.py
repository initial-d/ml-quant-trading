"""Offline report comparison on one real factor-library synthetic input.

Run from the repository root with OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1.
No external data is requested. Synthetic outputs are engineering evidence only.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
import scripts.akshare_csi300_full_pipeline as current
from scripts.audit_validation_report import audit_report, findings_to_dicts

BASE = "a0b33491266d578d63c300ad3d79832d237c40db"
OUT = ROOT / "artifacts/revalidation-20261018/coverage"
OUT.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def archived_module(name, path):
    source = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


old_helpers = archived_module("coverage_base_helpers", "scripts/public_data_validation.py")
old = archived_module("coverage_base_full", "scripts/akshare_csi300_full_pipeline.py")
old._fit_predict_model = old_helpers._fit_predict_model
panel = make_synthetic_panel(SyntheticConfig(n_dates=120, n_stocks=20, halt_prob=0, seed=7))
panel.assert_consistent()
fingerprint = hashlib.sha256()
for name in ("open", "high", "low", "close", "volume", "vwap", "amount", "mask"):
    fingerprint.update(name.encode())
    fingerprint.update(getattr(panel, name).numpy().tobytes())

cfg = current.FullPipelineConfig(
    source="synthetic", preset="custom", tickers=(), start="2021-01-01", end="2022-01-01",
    max_tickers=20, device="cpu", costs_bps=5, slippage_bps=2, cost_grid_bps=(0, 7, 15, 30),
    train_window=60, test_window=20, step=20, top_quantile=0.2, seed=7, epochs=1,
    batch_size=128, hidden=16, factor_set="all", covariance_window=20,
    optimizer_candidates=10, optimizer_weight_cap=0.1, optimizer_risk_aversion=1,
    rebalance_step=1, factor_ic_window=30, min_tradable_ratio=0.9,
    bootstrap_samples=0, bootstrap_block_size=20, output_dir=OUT,
    command=(sys.executable, str(Path(__file__).relative_to(ROOT))),
)
books = {}
reports = {}
for label, module in (("base", old), ("current", current)):
    module.load_validation_panel = lambda config: panel
    writer = module._write_outputs

    def capture(config, data, names, rows, sensitivity, strategies, returns,
                _label=label, _writer=writer, **kwargs):
        books[_label] = {name: weights.copy() for name, weights in strategies.items()}
        _writer(config, data, names, rows, sensitivity, strategies, returns, **kwargs)

    module._write_outputs = capture
    module.run_full_pipeline(replace(cfg, output_dir=OUT / label))
    reports[label] = json.loads((OUT / label / "summary.json").read_text())

assert reports["base"]["results"] == reports["current"]["results"]
assert reports["base"]["cost_sensitivity"] == reports["current"]["cost_sensitivity"]
assert all(np.array_equal(books["base"][name], weights) for name, weights in books["current"].items())
diagnostics = reports["current"]["diagnostics"]
assert diagnostics["model"]["trained_splits"] == 0
assert diagnostics["strategies"]["mlp_top20_raw_daily"]["invested_dates"] == 0
comparison = {
    "category": "synthetic engineering evidence; no investment or predictive evidence",
    "base_commit": BASE,
    "current_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "python": platform.python_version(), "pytorch": torch.__version__, "platform": platform.platform(),
    "device": "cpu", "torch_threads": 1, "torch_interop_threads": 1,
    "synthetic_config": {"n_dates": 120, "n_stocks": 20, "halt_prob": 0, "seed": 7},
    "panel_sha256": fingerprint.hexdigest(),
    "panel_hash_method": "SHA-256 of field names and C-order bytes: open/high/low/close/volume/vwap/amount/mask",
    "weights_identical": True, "metrics_identical": True, "cost_sensitivity_identical": True,
    "costs_bps": 5, "slippage_bps": 2,
    "diagnostics": diagnostics,
    "audit": {label: findings_to_dicts(audit_report(report)) for label, report in reports.items()},
    "limitations": [
        "Short synthetic panel deliberately lacks enough joint-factor training samples; not a market experiment.",
        "Base runner/writer and model helper loaded from recorded Git commit; other unchanged helpers are shared.",
        "No bootstrap inference; full-panel metrics retain warm-up cash and differing exposure windows.",
    ],
}
(OUT / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
print(json.dumps({"identical_weights_and_metrics": True, "diagnostics": diagnostics}, indent=2))
