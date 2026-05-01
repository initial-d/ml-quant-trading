"""Command-line entry point: ``mlquant <subcommand> --config configs/foo.yaml``.

Subcommands form a deterministic pipeline; each writes its outputs to a
sub-directory of ``artifacts/`` so downstream stages can be re-run
without recomputing upstream work.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import click
import numpy as np
import torch

from ..backtest.engine import run_backtest
from ..data.synthetic import SyntheticConfig, make_synthetic_panel
from ..features.alpha101 import compute_alpha_set
from ..features.bias import limit_move_mask
from ..models.losses import AdjMSELoss
from ..models.nets import MLPRegressor
from ..portfolio.markowitz import MarkowitzConfig, MarkowitzOptimizer
from ..training.dataset import FactorDataset
from ..training.trainer import TrainConfig, Trainer
from ..utils.config import load_config
from ..utils.seed import seed_everything


def _artifacts_dir(cfg) -> Path:
    p = Path(cfg.get("artifacts_dir", "artifacts"))
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Click app
# ---------------------------------------------------------------------------
@click.group()
def cli() -> None:
    """ml-quant-trading command-line interface."""


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
    factors, mask, names = compute_alpha_set(panel)
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
    res = run_backtest(weights, rets, costs_bps=cfg.get("costs_bps", 5.0))
    out = art / "backtest.pkl"
    with open(out, "wb") as fh:
        pickle.dump(res, fh)

    click.echo("\nBacktest summary")
    click.echo("-" * 40)
    for k, v in res.metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:<12s}  {v:>10.4f}")
        else:
            click.echo(f"  {k:<12s}  {v:>10}")


def main() -> None:
    cli()


if __name__ == "__main__":              # pragma: no cover
    main()
