"""Compute IC and RankIC for every alpha registered in the catalogue.

Usage:
    python scripts/eval_factor_ic.py --config configs/small.yaml
"""
from __future__ import annotations

import click
import numpy as np
import torch

from mlquant.backtest.metrics import information_coefficient, rank_information_coefficient
from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.features.alpha101 import ALPHA_REGISTRY
from mlquant.features.bias import limit_move_mask
from mlquant.utils.config import load_config
from mlquant.utils.seed import seed_everything


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def main(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    syn = SyntheticConfig(**cfg.synthetic.to_dict()) if "synthetic" in cfg else SyntheticConfig()
    panel = make_synthetic_panel(syn)
    bias = limit_move_mask(panel, limit_pct=cfg.get("limit_pct", 0.098))

    fwd = torch.zeros_like(panel.close)
    fwd[:-1] = panel.close[1:] / panel.close[:-1].clamp_min(1e-9) - 1.0

    valid = (panel.mask & bias)[:-1].cpu().numpy()
    fwd_np = fwd[:-1].cpu().numpy()

    rows = []
    for name, fn in ALPHA_REGISTRY.items():
        v, m = fn(panel)
        v_np = (v[:-1] * (panel.mask & m)[:-1].float()).cpu().numpy()
        m_np = (m[:-1].cpu().numpy() & valid)
        ic    = information_coefficient(v_np[m_np], fwd_np[m_np])
        rk_ic = rank_information_coefficient(v_np[m_np], fwd_np[m_np])
        rows.append((name, ic, rk_ic))

    click.echo(f"{'name':<10}{'IC':>10}{'RankIC':>10}")
    click.echo("-" * 30)
    for name, ic, rk in rows:
        click.echo(f"{name:<10}{ic:>10.4f}{rk:>10.4f}")
    click.echo("-" * 30)
    click.echo(f"mean IC : {np.mean([r[1] for r in rows]):>10.4f}")


if __name__ == "__main__":          # pragma: no cover
    main()
