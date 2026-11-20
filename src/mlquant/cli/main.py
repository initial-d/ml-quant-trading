"""Command-line entry point: ``mlquant <subcommand> --config configs/foo.yaml``.

Subcommands form a deterministic pipeline; each writes its outputs to a
sub-directory of ``artifacts/`` so downstream stages can be re-run
without recomputing upstream work.
"""
from __future__ import annotations

import json
import gzip
import hashlib
import math
import pickle
import platform
import shlex
import subprocess
import sys
from dataclasses import replace
from importlib.resources import as_file, files
from pathlib import Path

import click
import numpy as np
import torch

from ..backtest.engine import run_backtest
from ..backtest.self_financing import run_self_financing_backtest
from ..backtest.ic_inference import infer_factor_ic_report
from ..backtest.factor_walk_forward import walk_forward_factor_ic
from ..data.synthetic import SyntheticConfig, make_synthetic_panel
from ..features.legacy_factors import compute_legacy_set
from ..features.bias import board_limit_pct, limit_move_mask
from ..features.rolling_backend import validate_rolling_backend
from ..features.target import target_01
from ..models.losses import AdjMSELoss
from ..models.nets import MLPRegressor
from ..portfolio.markowitz import MarkowitzConfig, MarkowitzOptimizer
from ..systemone import build_state_from_artifacts as systemone_build_state_from_artifacts
from ..systemone import decide as systemone_decide
from ..systemone import evaluate_decisions as systemone_evaluate_decisions
from ..systemone import load_questions as load_systemone_questions
from ..systemone import load_state as load_systemone_state
from ..systemone import save_json as save_systemone_json
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


_PANEL_FIELDS = ("open", "high", "low", "close", "volume", "vwap", "mask")
_OPTIONAL_PANEL_FIELDS = ("amount", "limit_up", "limit_down", "last_close")


def _load_panel(art: Path):
    """Rebuild the :class:`Panel` written by ``gen-data``, optional fields included."""
    from ..data.panel import Panel

    blob = torch.load(art / "panel.pt", weights_only=False)
    optional = {k: blob[k] for k in _OPTIONAL_PANEL_FIELDS if blob.get(k) is not None}
    return Panel(
        dates=blob["dates"], stocks=blob["stocks"],
        **{k: blob[k] for k in _PANEL_FIELDS},
        **optional,
    )


def _resolve_limit_pct(cfg, panel):
    """``limit_pct`` is a number, or ``"board"`` for per-board A-share thresholds."""
    limit_pct = cfg.get("limit_pct", 0.098)
    if isinstance(limit_pct, str):
        if limit_pct != "board":
            raise click.BadParameter(f"limit_pct must be a number or 'board', got {limit_pct!r}")
        return board_limit_pct(panel.stocks, panel.dates)
    return float(limit_pct)


def _oos_start(cfg, n_dates: int) -> int:
    """First out-of-sample date index; the model only trains on dates before it."""
    train_frac = float(cfg.get("train_frac", 0.6))
    if not 0.0 < train_frac < 1.0:
        raise click.BadParameter(f"train_frac must be in (0, 1), got {train_frac}")
    return max(1, min(n_dates - 2, int(n_dates * train_frac)))


def _date_str(value) -> str:
    return str(np.datetime_as_string(np.datetime64(value, "D")))


def _write_backtest_summary(
    output_dir: Path,
    metrics: dict,
    *,
    config_path: str,
    costs_bps: float,
    evaluation: dict | None = None,
    accounting: dict | None = None,
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
    if evaluation is not None:
        payload["evaluation"] = evaluation
    if accounting is not None:
        payload["accounting"] = accounting
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    window_lines = []
    if evaluation is not None:
        window_lines = [
            (
                f"- Train window: `{evaluation['train_start']}` → `{evaluation['train_end']}`;"
                f" out-of-sample window: `{evaluation['oos_start']}` → `{evaluation['oos_end']}`"
            ),
            "- Metrics cover the out-of-sample window only; the model never trained on it.",
        ]

    rows = []
    for key, value in clean_metrics.items():
        if value is None:
            display = "n/a"
        elif isinstance(value, float):
            display = f"{value:.6f}"
        else:
            display = str(value)
        rows.append(f"| `{key}` | {display} |")
    accounting_lines = [] if accounting is None else [
        f"- Accounting mode: `{accounting['mode']}`",
        "- " + accounting["description"],
    ]

    markdown = "\n".join(
        [
            "# mlquant demo result",
            "",
            f"- Config: `{config_path}`",
            f"- Transaction cost assumption: `{costs_bps:g} bps`",
            *window_lines,
            *accounting_lines,
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
    markdown_path.write_text(markdown, encoding="utf-8")
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
    blob = {"dates": panel.dates, "stocks": panel.stocks}
    for key in _PANEL_FIELDS + _OPTIONAL_PANEL_FIELDS:
        blob[key] = getattr(panel, key)
    torch.save(blob, out)
    click.echo(f"wrote {out}  ({panel.n_dates} dates × {panel.n_stocks} stocks)")


@cli.command("features")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
@click.option(
    "--rolling-backend", type=click.Choice(["unfold", "linear"]), default=None,
    help="Rolling moments: overrides config rolling_backend (default unfold). "
         "Linear uses FP64 block summaries on CPU/CUDA and can change factor ranks.",
)
def cmd_features(config_path: str, rolling_backend: str | None = None) -> None:
    cfg = load_config(config_path)
    try:
        backend = validate_rolling_backend(
            rolling_backend if rolling_backend is not None else cfg.get("rolling_backend", "unfold")
        )
    except ValueError as exc:
        raise click.BadParameter(str(exc), param_hint="rolling_backend") from exc
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel = _load_panel(art)
    bias_mask = limit_move_mask(panel, limit_pct=_resolve_limit_pct(cfg, panel))
    # Mask-first: factors read ``panel.mask``, so limit-move prices are
    # excluded from every rolling window, not only from the limit day's row.
    factors, mask, names = compute_legacy_set(
        replace(panel, mask=bias_mask), rolling_backend=backend,
    )
    mask = mask & bias_mask
    torch.save({"factors": factors, "mask": mask, "names": names,
                "rolling_backend": backend}, art / "features.pt")
    click.echo(f"wrote features: {factors.shape}  rolling_backend={backend}  alphas={names}")


@cli.command("train")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_train(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel = _load_panel(art)
    feat_blob  = torch.load(art / "features.pt", weights_only=False)
    fwd_ret, label_mask = target_01(panel)

    # Train only on dates before the out-of-sample window. Slicing to
    # ``oos + 1`` keeps the label of the last training day (close at ``oos``),
    # which is already known when the first out-of-sample signal is formed.
    T = fwd_ret.shape[0]
    oos = _oos_start(cfg, T)
    ds = FactorDataset(
        factors=feat_blob["factors"][: oos + 1],
        mask=feat_blob["mask"][: oos + 1],
        forward_returns=fwd_ret[: oos + 1],
        label_mask=label_mask[: oos + 1],
    )
    n_features = ds.features.shape[1]
    model = MLPRegressor(in_dim=n_features, hidden=cfg.get("hidden", 128))
    trainer = Trainer(model, AdjMSELoss(gamma=cfg.get("loss_gamma", 0.1)),
                      TrainConfig(**(cfg.train.to_dict() if "train" in cfg else {})))
    trainer.fit(ds)

    # Write predictions for every tradable out-of-sample (t, n) cell.
    with torch.no_grad():
        T, N, F = feat_blob["factors"].shape
        flat = feat_blob["factors"].reshape(-1, F)
        pred = model(flat.to(trainer.cfg.device)).cpu().reshape(T, N)
    pred = pred * feat_blob["mask"].float()
    pred[:oos] = 0.0
    torch.save({"pred": pred, "oos_start": oos}, art / "predictions.pt")
    click.echo(
        f"wrote predictions: {pred.shape}  (trained on dates [0, {oos}),"
        f" out-of-sample from {oos})"
    )


@cli.command("portfolio")
@click.option("--config", "config_path", required=True, type=click.Path(exists=True))
def cmd_portfolio(config_path: str) -> None:
    cfg = load_config(config_path)
    seed_everything(cfg.get("seed", 42))
    art = _artifacts_dir(cfg)
    panel = _load_panel(art)
    pred_blob  = torch.load(art / "predictions.pt", weights_only=False)

    mask = panel.mask.numpy()
    pred = pred_blob["pred"].numpy()
    # Halt-aware returns: loaders zero-fill halted prices.
    rets = panel.returns.numpy()

    look_back = cfg.get("cov_lookback", 60)
    mk_cfg = MarkowitzConfig(**(cfg.portfolio.to_dict() if "portfolio" in cfg else {}))
    opt = MarkowitzOptimizer(mk_cfg)

    T, N = pred.shape
    oos = int(pred_blob.get("oos_start", 0))
    # Fewer names than this cannot reach full investment under weight_cap.
    min_names = max(5, math.ceil((1.0 - mk_cfg.cash_weight) / mk_cfg.weight_cap - 1e-9))
    weights = np.zeros_like(pred)
    for t in range(max(look_back, oos), T - 1):
        tradable = mask[t]
        idx = np.where(tradable)[0]
        if idx.size < min_names:
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
@click.option("--accounting", default=None, type=click.Choice(["target_difference", "self_financing"]),
              help="Override backtest_accounting in config; default target_difference for archived compatibility.")
def cmd_backtest(config_path: str, accounting: str | None = None) -> None:
    cfg = load_config(config_path)
    art = _artifacts_dir(cfg)
    panel_blob = torch.load(art / "panel.pt", weights_only=False)
    weight_blob = torch.load(art / "weights.pt", weights_only=False)
    pred_path = art / "predictions.pt"
    oos = 0
    if pred_path.exists():
        oos = int(torch.load(pred_path, weights_only=False).get("oos_start", 0))

    # Halt-aware returns: loaders zero-fill halted prices.
    rets = _load_panel(art).returns.numpy()

    weights = weight_blob["weights"].numpy()
    costs_bps = cfg.get("costs_bps", 5.0)
    mode = accounting if accounting is not None else cfg.get("backtest_accounting", "target_difference")
    if mode not in ("target_difference", "self_financing"):
        raise click.BadParameter("backtest_accounting must be target_difference or self_financing")
    # Score only the out-of-sample window; the engine charges the entry
    # trade on its first row.
    operation = run_self_financing_backtest if mode == "self_financing" else run_backtest
    res = operation(weights[oos:], rets[oos:], costs_bps=costs_bps)
    accounting_metadata = {"mode": mode, "description": (
        "Marked holdings and residual cash; one-way fees on actual trades paid from NAV. "
        "Long-only, complete fills, zero cash interest; no execution/settlement/volume model."
        if mode == "self_financing" else
        "Approximation: fees on consecutive target differences, including entry; no drift-aware trade ledger."
    )}
    if mode == "self_financing":
        accounting_metadata.update(initial_equity=res.initial_equity, bankrupt_index=res.bankrupt_index,
                                   target_rounding_max_excess=res.target_rounding_max_excess,
                                   ledger="backtest.pkl", fee_basis="one-way actual traded notional")
    evaluation = None
    if oos > 0:
        dates = panel_blob["dates"]
        evaluation = {
            "train_start": _date_str(dates[0]),
            "train_end": _date_str(dates[oos - 1]),
            "oos_start": _date_str(dates[oos]),
            "oos_end": _date_str(dates[-1]),
            "n_train_dates": oos,
            "n_oos_dates": int(len(dates) - oos),
        }
    out = art / "backtest.pkl"
    with open(out, "wb") as fh:
        pickle.dump(res, fh)
    markdown_path, json_path = _write_backtest_summary(
        art,
        res.metrics,
        config_path=config_path,
        costs_bps=costs_bps,
        evaluation=evaluation,
        accounting=accounting_metadata,
    )

    click.echo("\nBacktest summary" + (" (out-of-sample window)" if evaluation else ""))
    click.echo("-" * 40)
    for k, v in res.metrics.items():
        if isinstance(v, float):
            click.echo(f"  {k:<12s}  {v:>10.4f}")
        else:
            click.echo(f"  {k:<12s}  {v:>10}")
    click.echo(f"\nShareable reports: {markdown_path}  {json_path}")


def _frozen_factor_report(input_report, json_out, operation, options, implementation_files):
    """Shared frozen-input and provenance contract for factor report commands."""
    if input_report.resolve() == json_out.resolve():
        raise click.BadParameter("output must not replace the frozen input report", param_hint="json-out")
    raw = input_report.read_bytes()
    try:
        payload = gzip.decompress(raw) if input_report.suffix == ".gz" else raw
        source = json.loads(payload)
        report = operation(source, **options)
    except (ValueError, TypeError, OSError, KeyError) as exc:
        raise click.BadParameter(str(exc), param_hint="input-report/inference options") from exc
    source_root = Path(__file__).resolve().parents[3]
    commit = None
    if (source_root / "pyproject.toml").is_file():
        try:
            commit = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"],
                                             text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    cpu = platform.processor() or platform.machine()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    elif Path("/proc/cpuinfo").is_file():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    fingerprints = {Path(path).name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for path in implementation_files}
    implementation_hash = (next(iter(fingerprints.values())) if len(fingerprints) == 1
                           else hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest())
    report.update(
        input_report=str(input_report), input_report_sha256=hashlib.sha256(raw).hexdigest(),
        input_payload_sha256=hashlib.sha256(payload).hexdigest(),
        source_metadata={key: value for key, value in source.items() if key != "factors"},
        environment={"commit": commit, "command": shlex.join([sys.executable, *sys.argv]),
                     "python": sys.version, "numpy": np.__version__, "pytorch": torch.__version__,
                     "os": platform.platform(), "cpu": cpu, "device": "cpu", "gpu": "unused",
                     "implementation_sha256": implementation_hash, "implementation_files": fingerprints,
                     "data_source": "frozen input report; source_metadata preserves original evidence category",
                     "cost_assumptions": "No transactions, market-data downloads, GPU or paid API; IC excludes execution costs"},
    )
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


@cli.command("factor-inference")
@click.option("--input-report", required=True,
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              help="Frozen daily-factor IC JSON or JSON.gz; no factor recomputation or data fetch.")
@click.option("--json-out", required=True, type=click.Path(path_type=Path))
@click.option("--max-lag", default=None, type=click.IntRange(min=0),
              help="Newey-West lag in original panel rows; default automatic, at least horizon-1.")
@click.option("--min-dates", default=20, show_default=True, type=click.IntRange(min=2))
@click.option("--confidence", default=.95, show_default=True, type=click.FloatRange(0, 1, min_open=True, max_open=True))
@click.option("--alpha", default=.05, show_default=True, type=click.FloatRange(0, 1, min_open=True, max_open=True))
@click.option("--method", default="by", show_default=True, type=click.Choice(["by", "bh", "holm"]))
def cmd_factor_inference(input_report, json_out, max_lag, min_dates, confidence, alpha, method):
    """HAC mean IC tests and joint IC/RankIC multiple-testing correction."""
    from ..backtest import ic_inference
    report = _frozen_factor_report(input_report, json_out, infer_factor_ic_report,
                                  dict(max_lag=max_lag, min_dates=min_dates, confidence=confidence,
                                       alpha=alpha, method=method), [ic_inference.__file__])
    family = report["multiple_testing"]
    click.echo(f"{family['family_size']} hypotheses; {family['testable']} testable; "
               f"p <= {alpha:g}: {family['unadjusted_rejections']}; "
               f"{method} adjusted: {family['adjusted_rejections']}")
    click.echo(f"wrote {json_out}")


@cli.command("factor-walk-forward")
@click.option("--input-report", required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("--json-out", required=True, type=click.Path(path_type=Path))
@click.option("--train-dates", default=200, show_default=True, type=click.IntRange(min=2))
@click.option("--test-dates", default=50, show_default=True, type=click.IntRange(min=1))
@click.option("--purge-dates", default=None, type=click.IntRange(min=1), help="Original-row gap; defaults to label_horizon.")
@click.option("--max-factors", default=10, show_default=True, type=click.IntRange(min=1))
@click.option("--metric", default="rank_ic", show_default=True, type=click.Choice(["ic", "rank_ic"]))
@click.option("--max-lag", default=None, type=click.IntRange(min=0))
@click.option("--min-dates", default=20, show_default=True, type=click.IntRange(min=2))
@click.option("--confidence", default=.95, show_default=True, type=click.FloatRange(0, 1, min_open=True, max_open=True))
@click.option("--alpha", default=.05, show_default=True, type=click.FloatRange(0, 1, min_open=True, max_open=True))
@click.option("--method", default="by", show_default=True, type=click.Choice(["by", "bh", "holm"]))
def cmd_factor_walk_forward(input_report, json_out, **options):
    """Purged trailing-factor screening and frozen-direction future IC checks."""
    from ..backtest import factor_walk_forward, ic_inference
    report = _frozen_factor_report(input_report, json_out, walk_forward_factor_ic, options,
                                  [factor_walk_forward.__file__, ic_inference.__file__])
    summary = report["summary"]
    click.echo(f"{summary['folds']} folds; {summary['folds_with_selection']} with selection; "
               f"{summary['test_rows_with_selection']}/{summary['test_rows']} test rows with selection")
    click.echo(f"wrote {json_out}")


@cli.group("systemone")
def cmd_systemone() -> None:
    """Typed decision adapter for research evidence artifacts."""


@cmd_systemone.command("build-state")
@click.option(
    "--artifacts-dir",
    "artifacts_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Directory containing panel.pt, predictions.pt, weights.pt, and optional summary.json.",
)
@click.option(
    "--as-of-index",
    default=-2,
    show_default=True,
    type=int,
    help="Date index used to build the state. Negative indices count from the end.",
)
@click.option(
    "--include-realized/--no-include-realized",
    default=False,
    show_default=True,
    help="Include next-period realized returns for offline evaluation only.",
)
@click.option(
    "--costs-bps",
    default=None,
    type=float,
    help="Override effective cost assumption in basis points.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path for the generated decision_state.json.",
)
def cmd_systemone_build_state(
    artifacts_dir: Path,
    as_of_index: int,
    include_realized: bool,
    costs_bps: float | None,
    output_path: Path,
) -> None:
    """Build a typed portfolio decision state from pipeline artifacts."""
    payload = systemone_build_state_from_artifacts(
        artifacts_dir,
        as_of_index=as_of_index,
        include_realized=include_realized,
        costs_bps=costs_bps,
    )
    save_systemone_json(payload, output_path)
    click.echo(f"wrote {output_path}")


@cmd_systemone.command("decide")
@click.option(
    "--state",
    "state_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to a validation summary.json or compatible state JSON.",
)
@click.option(
    "--questions",
    "questions_path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional JSON/YAML file with typed question definitions.",
)
@click.option(
    "--output",
    "output_path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path for the typed decision JSON.",
)
def cmd_systemone_decide(
    state_path: Path,
    questions_path: Path | None,
    output_path: Path | None,
) -> None:
    """Convert a research report into System-One-style typed decisions."""
    state = load_systemone_state(state_path)
    questions = (
        load_systemone_questions(questions_path)
        if questions_path is not None
        else None
    )
    payload = systemone_decide(state, questions=questions, source_path=state_path)
    if output_path is not None:
        save_systemone_json(payload, output_path)
        click.echo(f"wrote {output_path}")
    else:
        text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        click.echo(text, nl=False)


@cmd_systemone.command("evaluate")
@click.option(
    "--state",
    "state_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Decision state JSON with evaluation.forward_return included.",
)
@click.option(
    "--decision",
    "decision_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Typed decision JSON produced by `mlquant systemone decide`.",
)
@click.option(
    "--return-threshold",
    default=0.001,
    show_default=True,
    type=float,
    help="Forward-return threshold used to form hindsight action labels.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(path_type=Path),
    help="Path for the decision evaluation JSON.",
)
def cmd_systemone_evaluate(
    state_path: Path,
    decision_path: Path,
    return_threshold: float,
    output_path: Path,
) -> None:
    """Evaluate typed portfolio decisions with realized offline evidence."""
    state = load_systemone_state(state_path)
    decision = load_systemone_state(decision_path)
    payload = systemone_evaluate_decisions(
        state,
        decision,
        return_threshold=return_threshold,
    )
    save_systemone_json(payload, output_path)
    click.echo(f"wrote {output_path}")


def main() -> None:
    cli()


if __name__ == "__main__":              # pragma: no cover
    main()
