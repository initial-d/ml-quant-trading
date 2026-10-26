"""Screen all registered factors by daily IC/RankIC on a synthetic panel.

No external market-data fetch. This is a diagnostic, not predictive evidence.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform
import shlex
import subprocess
import sys

import click
import torch

from mlquant.backtest.factor_ic import daily_factor_ic
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features import iter_legacy_factors, limit_move_mask
from mlquant.utils.config import load_config
from mlquant.utils.seed import seed_everything


def close_to_close_labels(panel):
    """Adjacent session labels; never impute across an unavailable close."""
    close = panel.close
    eligible = panel.mask & torch.isfinite(close) & (close > 0)
    mask = torch.zeros_like(eligible)
    mask[:-1] = eligible[:-1] & eligible[1:]
    labels = torch.full_like(close, float("nan"))
    labels[:-1] = torch.where(mask[:-1], close[1:] / close[:-1] - 1.0, float("nan"))
    return labels, mask


def summarize(result):
    def metric(series):
        eligible = series[result.valid]
        if not eligible.numel():
            return {"mean": None, "std": None, "ic_ir": None}
        std = float(eligible.std(unbiased=True)) if eligible.numel() > 1 else None
        mean = float(eligible.mean())
        return {"mean": mean, "std": std, "ic_ir": mean / std if std and std > 0 else None}
    return {"ic": metric(result.ic), "rank_ic": metric(result.rank_ic),
            "valid_dates": int(result.valid.sum()), "total_dates": result.valid.numel(),
            "pair_counts": result.count.cpu().tolist(),
            "daily_ic": [float(v) if ok else None for v, ok in zip(result.ic, result.valid)],
            "daily_rank_ic": [float(v) if ok else None for v, ok in zip(result.rank_ic, result.valid)]}


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--factor", "names", multiple=True, help="Repeat for a subset; default all 213 factors.")
@click.option("--min-assets", default=3, type=click.IntRange(min=2))
@click.option("--rolling-backend", type=click.Choice(["unfold", "linear"]), default="unfold")
@click.option("--json-out", type=click.Path(path_type=Path))
def main(config_path, names, min_assets, rolling_backend, json_out):
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    syn = SyntheticConfig(**cfg.synthetic.to_dict()) if "synthetic" in cfg else SyntheticConfig()
    raw_panel = make_synthetic_panel(syn)
    targets, label_mask = close_to_close_labels(raw_panel)
    # Bias masking must precede every rolling factor; future label availability
    # only restricts retrospective evaluation pairs, never factor construction.
    panel = replace(raw_panel, mask=limit_move_mask(raw_panel, limit_pct=cfg.get("limit_pct", .098)))
    rows = []
    with torch.no_grad():
        for name, values, mask in iter_legacy_factors(panel, names=names or None,
                                                      rolling_backend=rolling_backend):
            row = {"name": name, **summarize(daily_factor_ic(values, targets, mask, label_mask,
                                                            min_assets=min_assets))}
            rows.append(row)
            ic, rank_ic = row["ic"]["mean"], row["rank_ic"]["mean"]
            click.echo(f"{name:<20} IC={ic!s:<22} RankIC={rank_ic!s:<22} dates={row['valid_dates']}")
            del values, mask
    if json_out:
        cpu = platform.processor() or platform.machine()
        if sys.platform == "darwin":
            cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        report = {"protocol": "daily-factor-ic-v1", "category": "synthetic engineering diagnostic",
                  "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
                  "pytorch": torch.__version__, "os": platform.platform(), "cpu": cpu,
                  "device": str(panel.close.device), "gpu": "not used" if panel.close.device.type == "cpu" else torch.cuda.get_device_name(),
                  "threads": 1, "interop_threads": 1, "data_source": "seeded SyntheticConfig GBM",
                  "config": cfg.to_dict(), "min_assets": min_assets, "rolling_backend": rolling_backend,
                  "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                  "input_sha256": {name: hashlib.sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
                                    for name, tensor in {"raw_close": raw_panel.close, "raw_mask": raw_panel.mask,
                                                         "factor_mask": panel.mask, "labels": targets,
                                                         "label_mask": label_mask}.items()},
                  "cost_assumptions": "No transactions or paid data; IC excludes execution costs.",
                  "limitations": ["Full-sample synthetic screening, not held-out factor selection or alpha evidence.",
                                  "No inference p-values; daily IC IR is unannualized and ignores temporal dependence.",
                                  "Adjacent available positive closes only; halt-crossing labels are not imputed.",
                                  "Constant or undersized cross-sections are missing dates, not zero IC."],
                  "factors": rows}
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
