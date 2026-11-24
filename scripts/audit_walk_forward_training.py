"""Predeclared planted-signal controls; engineering evidence, not market alpha."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import time

import torch

from mlquant.backtest import daily_factor_ic
from mlquant.models.nets import MLPRegressor
from mlquant.training import TrainConfig, WalkForwardConfig, train_walk_forward


def quality(predictions, mask, labels, label_mask, start, stop):
    paired = mask[start:stop] & label_mask[start:stop]
    ic = daily_factor_ic(predictions[start:stop], labels[start:stop], mask[start:stop], label_mask[start:stop])
    mse = (predictions[start:stop][paired].double() - labels[start:stop][paired].double()).square().mean()
    return {"start": start, "stop": stop, "pairs": int(paired.sum()), "mse": float(mse),
            "mean_ic": float(ic.ic[ic.valid].mean()) if ic.valid.any() else None,
            "mean_rank_ic": float(ic.rank_ic[ic.valid].mean()) if ic.valid.any() else None,
            "valid_ic_dates": int(ic.valid.sum())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = []
    for case in ("stationary", "sign_switch", "no_signal"):
        for seed in (11, 29):
            generator = torch.Generator().manual_seed(seed)
            factors = torch.randn(180, 30, 2, generator=generator)
            noise = .005 * torch.randn(180, 30, generator=generator)
            coefficient = torch.full((180, 1), .02)
            if case == "sign_switch":
                coefficient[90:] *= -1
            elif case == "no_signal":
                coefficient.zero_()
            labels = coefficient * factors[..., 0] + noise
            mask = torch.ones_like(labels, dtype=torch.bool)
            eligible = mask.clone()
            eligible[-1] = False
            entry = {"case": case, "seed": seed, "switch_row": 90 if case == "sign_switch" else None,
                     "input_sha256": hashlib.sha256(factors.numpy().tobytes() + labels.numpy().tobytes()).hexdigest(),
                     "policies": []}
            for window in ("rolling", "expanding"):
                destination = args.output_dir / f"{case}-{seed}-{window}"
                config = WalkForwardConfig(train_dates=60, test_dates=30, window=window, seed=seed)
                training = TrainConfig(epochs=12, batch_size=512, lr=.003, weight_decay=.001,
                                       val_split=.25, save_dir=str(destination / "checkpoints"))
                started = time.perf_counter()
                result = train_walk_forward(factors, mask, labels, eligible, cfg=config, train_cfg=training,
                    model_factory=lambda: MLPRegressor(2, hidden=8), loss_factory=torch.nn.MSELoss)
                elapsed = time.perf_counter() - started
                model = MLPRegressor(2, hidden=8)
                model.load_state_dict(torch.load(result.report["folds"][0]["checkpoint"], weights_only=True))
                model.eval()
                with torch.no_grad():
                    # Same daily prediction batch policy as the refitting API.
                    fixed = torch.stack([model(row) for row in factors])
                first = result.report["oos_start"]
                outcomes = []
                for fold in result.report["folds"]:
                    start, stop = fold["test_start"], fold["test_end_exclusive"]
                    outcomes.append({"fold": fold["fold"],
                        "refit": quality(result.predictions, result.prediction_mask, labels, eligible, start, stop),
                        "frozen_first_model": quality(fixed, result.prediction_mask, labels, eligible, start, stop)})
                entry["policies"].append({"window": window, "elapsed_seconds": elapsed,
                    "refit": quality(result.predictions, result.prediction_mask, labels, eligible, first, 180),
                    "frozen_first_model": quality(fixed, result.prediction_mask, labels, eligible, first, 180),
                    "refit_prediction_sha256": hashlib.sha256(result.predictions.numpy().tobytes()).hexdigest(),
                    "fold_outcomes": outcomes,
                    "training_report": {**result.report, "training_config": {**result.report["training_config"],
                                                                             "save_dir": str(destination)}}})
                torch.save({"refit": result.predictions, "frozen_first_model": fixed,
                            "mask": result.prediction_mask, "labels": labels, "label_mask": eligible},
                           destination / "predictions.pt")
            cases.append(entry)
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    root = Path(__file__).resolve().parents[1]
    source_files = [Path(__file__), root / "src/mlquant/training/walk_forward.py",
                    root / "src/mlquant/training/trainer.py", root / "src/mlquant/training/dataset.py",
                    root / "src/mlquant/models/nets.py"]
    payload = {"category": "synthetic planted-process engineering control", "protocol": "walk-forward-training-audit-v1",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version, "pytorch": torch.__version__,
        "os": platform.platform(), "cpu": cpu, "gpu": "unused", "threads": 1, "interop_threads": 1,
        "data_source": "fixed-seed Gaussian features and explicit +/-0.02 signal coefficients with Gaussian noise",
        "cost_assumptions": "none; supervised forecast quality only, no trading or portfolio P&L",
        "shape": [180, 30, 2], "cases": cases,
        "implementation_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                                   for path in source_files},
        "limitations": ["Signals and regime switch are deliberately planted; these controls are not market-performance evidence.",
                        "No return, turnover, fee, investment or calibration claim is inferred from forecast MSE/IC.",
                        "Frozen control is exactly the first fold's best validation checkpoint; refits incur extra computation.",
                        "Windows, seeds, epochs and cases are declared in code, with no choice of a winning configuration.",
                        "Full inputs, predictions and checkpoints remain local; all configurations, folds and outcomes are reported."]}
    (args.output_dir / "report.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
