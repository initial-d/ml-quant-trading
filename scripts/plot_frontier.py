"""Sweep α and print (or plot) the efficient frontier.

Usage:
    python scripts/plot_frontier.py --config configs/small.yaml
"""
from __future__ import annotations

import click
import numpy as np

from mlquant.data.synthetic import SyntheticConfig, make_synthetic_panel
from mlquant.portfolio.frontier import efficient_frontier
from mlquant.portfolio.markowitz import MarkowitzConfig
from mlquant.utils.config import load_config


@click.command()
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option("--lookback", default=120, type=int)
@click.option("--save",     default=None, type=click.Path())
def main(config_path: str, lookback: int, save: str | None) -> None:
    cfg = load_config(config_path)
    syn = SyntheticConfig(**cfg.synthetic.to_dict()) if "synthetic" in cfg else SyntheticConfig()
    panel = make_synthetic_panel(syn)

    rets_t = panel.returns.cpu().numpy()
    if rets_t.shape[0] <= lookback + 1:
        raise SystemExit("not enough dates for the requested look-back")

    history = rets_t[-lookback:]
    mu = history.mean(axis=0)

    points = efficient_frontier(mu, history,
                                risk_aversions=(0.25, 0.5, 1, 2, 5, 10, 25, 50, 100),
                                cfg=MarkowitzConfig(weight_cap=0.05))
    click.echo(f"{'α':>8}{'E[R]':>12}{'Var':>12}")
    for p in points:
        click.echo(f"{p.risk_aversion:>8.2f}{p.expected_return:>12.5f}{p.expected_variance:>12.6f}")

    if save:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise SystemExit("install matplotlib to use --save")
        xs = [p.expected_variance for p in points]
        ys = [p.expected_return  for p in points]
        plt.figure()
        plt.plot(xs, ys, marker="o")
        plt.xlabel("variance"); plt.ylabel("expected return"); plt.title("Efficient frontier")
        plt.tight_layout()
        plt.savefig(save, dpi=150)
        click.echo(f"wrote {save}")


if __name__ == "__main__":          # pragma: no cover
    main()
