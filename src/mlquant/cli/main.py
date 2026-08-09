"""Command-line entry point: ``mlquant <subcommand> --config configs/foo.yaml``.

Subcommands form a deterministic pipeline; each writes its outputs to a
sub-directory of ``artifacts/`` so downstream stages can be re-run
without recomputing upstream work.
"""
from __future__ import annotations

import json
import math
import pickle
from importlib.resources import as_file, files
from pathlib import Path

import click
import numpy as np
import torch

from ..backtest.engine import run_backtest
from ..data.synthetic import SyntheticConfig, make_synthetic_panel
from ..features.legacy_factors import compute_legacy_set
from ..features.alpha101 import compute_alpha_set  # backward compat
from ..features.bias import limit_move_mask
from ..models.losses import AdjMSELoss
from ..models.nets import MLPRegressor
from ..portfolio.markowitz import MarkowitzConfig, MarkowitzOptimizer
from ..training.dataset import FactorDataset
from ..training.trainer import TrainConfig, Trainer
from ..utils.config import load_config
from ..utils.seed import seed_everything


PROJECT_URL = "https://github.com/initial-d/ml-quant-trading"
REPRODUCTION_REPORT_URL = (
    f"{PROJECT_URL}/issues/new?template=reproduction_report.yml"
)


def _artifacts_dir(cfg) -> Path:
    p = Path(cfg.get("artifacts_dir", "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def _write_backtest_summary(
    output_dir: Path,
    metrics: dict,
    *,
    config_path: str,
    costs_bps: float,
) -> tuple[Path, Path]:
    """Write portable JSON and Markdown summaries for sharing and review."""
    clean_metrics = {
        key: value if not isinstance(value, float) or math.isfinite(value) else None
        for key, value in metrics.items()
    }
    payload = {
        "project": "ml-quant-trading",
        "repository": PROJECT_URL,
        "reproduction_report": REPRODUCTION_REPORT_URL,
        "workflow": "synthetic factor-to-backtest demo",
        "config": config_path,
        "costs_bps": float(costs_bps),
        "metrics": clean_metrics,
        "disclaimer": (
            "Synthetic research output only; not investment advice or evidence of live performance."
        ),
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    rows = []
    for key, value in clean_metrics.items():
        if value is None:
            display = "n/a"
        elif isinstance(value, float):
            display = f"{value:.6f}"
        else:
            display = str(value)
        rows.append(f"| `{key}` | {display} |")

    markdown = "\n".join(
        [
            "# mlquant demo result",
            "",
            f"- Config: `{config_path}`",
            f"- Transaction cost assumption: `{costs_bps:g} bps`",
            "- Workflow: synthetic data → 213 factors → MLP → Markowitz → backtest",
            f"- Source: [{PROJECT_URL}]({PROJECT_URL})",
            (
                "- Share a reproducible run: "
                f"[open the report form]({REPRODUCTION_REPORT_URL})"
            ),
            "",
            "| Metric | Value |",
            "|---|---:|",
            *rows,
            "",
            "> Synthetic research output only. This is not investment advice or evidence of",
            "> live or out-of-sample trading performance.",
            "",
        ]
    )
    markdown_path = output_dir / "summary.md"
    markdown_path.write_text(markdown)
    return markdown_path, json_path


# ---------------------------------------------------------------------------
# Click app
# ---------------------------------------------------------------------------
@click.group()
def cli() -> None:
    """ml-quant-trading command-line interface."""


@cli.command("demo")
@click.option(
    "--config",
    "config_path",
    default=None,
    show_default="packaged small config",
    type=click.Path(exists=True),
)
@click.pass_context
def cmd_demo(ctx: click.Context, config_path: str | None) -> None:
    """Run the complete synthetic factor-to-backtest pipeline."""
    if config_path is not None:
        _run_demo(ctx, config_path)
        return

    default_config = files("mlquant.configs").joinpath("small.yaml")
    with as_file(default_config) as default_path:
        click.echo("Using packaged default config (override with --config PATH).\n")
        _run_demo(ctx, str(default_path))


def _run_demo(ctx: click.Context, config_path: str) -> None:
    """Run all demo stages with an explicit, existing config path."""
    stages = (
        ("1/5 Generate deterministic synthetic data", cmd_gen_data),
        ("2/5 Compute the 213-factor tensor", cmd_features),
        ("3/5 Train the baseline model", cmd_train),
        ("4/5 Build portfolio weights", cmd_portfolio),
        ("5/5 Run the cost-aware backtest", cmd_backtest),
    )
    click.echo("mlquant demo — data → factors → model → portfolio → backtest\n")
    for label, command in stages:
        click.echo(f"[{label}]")
        ctx.invoke(command, config_path=config_path)
        click.echo()
    click.echo("Demo complete. Inspect the backtest summary above and artifacts/ for outputs.")
    click.echo("\nNext steps")
    click.echo(f"  Customize or inspect the source: {PROJECT_URL}")
    click.echo(f"  Share a reproducible run:       {REPRODUCTION_REPORT_URL}")
    click.echo("  If this saved you setup time, consider starring the repository.")


@cli.command("gen-data")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_gen_data(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    syn = SyntheticConfig(**cfg.synthetic.to_dict()) if "synthetic" in cfg else SyntheticConfig()
    panel = make_synthetic_panel(syn)
    out = _artifacts_dir(cfg) / "panel.pt"
    torch.save({
        "dates": panel.dates, "stocks": panel.stocks,
        "open":  panel.open,  "high":  panel.high,  "low":  panel.low,
        "close": panel.close, "volume": panel.volume, "vwap": panel.vwap,
        "mask":  panel.mask,
    }, out)
    click.echo(f"wrote {out}  ({panel.n_dates} dates × {panel.n_stocks} stocks)")


@cli.command("features")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_features(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel_blob = torch.load(art / "panel.pt", weights_only=False)
    from ..data.panel import Panel
    panel = Panel(
        dates=panel_blob["dates"], stocks=panel_blob["stocks"],
        open=panel_blob["open"], high=panel_blob["high"], low=panel_blob["low"],
        close=panel_blob["close"], volume=panel_blob["volume"], vwap=panel_blob["vwap"],
        mask=panel_blob["mask"],
    )
    bias_mask = limit_move_mask(panel, limit_pct=cfg.get("limit_pct", 0.098))
    factors, mask, names = compute_legacy_set(panel)
    mask = mask & bias_mask
    torch.save({"factors": factors, "mask": mask, "names": names}, art / "features.pt")
    click.echo(f"wrote features: {factors.shape}  alphas={names}")


@cli.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_train(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel_blob = torch.load(art / "panel.pt", weights_only=False)
    feat_blob  = torch.load(art / "features.pt", weights_only=False)

    close = panel_blob["close"]
    fwd_ret = torch.zeros_like(close)
    fwd_ret[:-1] = close[1:] / close[:-1].clamp_min(1e-9) - 1.0

    ds = FactorDataset(
        factors=feat_blob["factors"],
        mask=feat_blob["mask"],
        forward_returns=fwd_ret,
    )
    n_features = ds.features.shape[1]
    model = MLPRegressor(in_dim=n_features, hidden=cfg.get("hidden", 128))
    trainer = Trainer(model, AdjMSELoss(gamma=cfg.get("loss_gamma", 0.1)),
                      TrainConfig(**(cfg.train.to_dict() if "train" in cfg else {})))
    trainer.fit(ds)

    # Write predictions for every (t, n) cell that's tradable.
    with torch.no_grad():
        T, N, F = feat_blob["factors"].shape
        flat = feat_blob["factors"].reshape(-1, F)
        pred = model(flat.to(trainer.cfg.device)).cpu().reshape(T, N)
    pred = pred * feat_blob["mask"].float()
    torch.save({"pred": pred}, art / "predictions.pt")
    click.echo(f"wrote predictions: {pred.shape}")


@cli.command("portfolio")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_portfolio(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel_blob = torch.load(art / "panel.pt", weights_only=False)
    pred_blob  = torch.load(art / "predictions.pt", weights_only=False)

    close = panel_blob["close"].numpy()
    mask = panel_blob["mask"].numpy()
    pred = pred_blob["pred"].numpy()
    rets = np.zeros_like(close)
    rets[1:] = close[1:] / np.clip(close[:-1], 1e-9, None) - 1.0

    look_back = cfg.get("cov_lookback", 60)
    mk_cfg = MarkowitzConfig(**(cfg.portfolio.to_dict() if "portfolio" in cfg else {}))
    opt = MarkowitzOptimizer(mk_cfg)

    T, N = pred.shape
    weights = np.zeros_like(pred)
    for t in range(look_back, T - 1):
        tradable = mask[t]
        idx = np.where(tradable)[0]
        if idx.size < 5:
            continue
        mu = pred[t, idx].astype(np.float64)
        history = rets[t - look_back:t, idx]
        try:
            w = opt.solve(mu, history)
        except Exception as exc:                          # pragma: no cover
            click.echo(f"[warn] solver failed at t={t}: {exc}")
            continue
        weights[t, idx] = w
    torch.save({"weights": torch.from_numpy(weights)}, art / "weights.pt")
    click.echo(f"wrote weights: {weights.shape}")


@cli.command("backtest")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_backtest(config_path: str) -> None:
    cfg = load_config(config_path)
    art = _artifacts_dir(cfg)
    panel_blob = torch.load(art / "panel.pt", weights_only=False)
    weight_blob = torch.load(art / "weights.pt", weights_only=False)

    close = panel_blob["close"].numpy()
    rets = np.zeros_like(close)
    rets[1:] = close[1:] / np.clip(close[:-1], 1e-9, None) - 1.0

    weights = weight_blob["weights"].numpy()
    costs_bps = cfg.get("costs_bps", 5.0)
    res = run_backtest(weights, rets, costs_bps=costs_bps)
    out = art / "backtest.pkl"
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    markdown_path, json_path = _write_backtest_summary(
        art,
        res.metrics,
        config_path=config_path,
        costs_bps=costs_bps,
    )

    click.echo("\nBacktest summary")
    click.echo("-" * 40)
    for k, v in res.metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:<12s}  {v:>10.4f}")
        else:
            click.echo(f"  {k:<12s}  {v:>10}")
    click.echo(f"\nShareable reports: {markdown_path}  {json_path}")


def main() -> None:
    cli()


if __name__ == "__main__":              # pragma: no cover
    main()
