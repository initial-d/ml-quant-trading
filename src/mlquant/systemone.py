"""System-One-style portfolio decisions for research artifacts.

The adapter answers typed portfolio-intent questions from a structured state
that contains factors, predictions, tradability, liquidity, holdings, risk, and
cost assumptions. It deliberately stops at research portfolio intent; it does
not create broker orders or connect to live accounts.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:  # pragma: no cover - exercised in environments with PyYAML.
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


MODEL_ID = "mlquant-systemone-portfolio-decision"
MODEL_VERSION = "0.1.0-preview"
CONTRACT_VERSION = "system-one-compatible-v0"

DEFAULT_QUESTIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "asset_action",
        "type": "choice",
        "options": ["increase", "hold", "decrease", "exclude"],
    },
    {"id": "target_weight", "type": "score", "min": 0.0, "max": 1.0},
    {"id": "trade_allowed", "type": "noul"},
    {
        "id": "risk_bucket",
        "type": "choice",
        "options": ["low", "medium", "high", "blocked"],
    },
    {"id": "max_participation_rate", "type": "score", "min": 0.0, "max": 1.0},
)


def load_state(path: str | Path) -> dict[str, Any]:
    """Load a portfolio decision state JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Write a deterministic JSON payload."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output_path


def load_questions(path: str | Path | None) -> list[dict[str, Any]]:
    """Load optional typed question definitions from JSON or YAML."""
    if path is None:
        return [dict(item) for item in DEFAULT_QUESTIONS]
    question_path = Path(path)
    text = question_path.read_text(encoding="utf-8")
    if question_path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to load YAML question files")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if isinstance(payload, Mapping) and isinstance(payload.get("questions"), list):
        payload = payload["questions"]
    if not isinstance(payload, list):
        raise ValueError("question file must contain a list or a {questions: [...]} object")
    return [dict(item) for item in payload]


def decide(
    state: Mapping[str, Any],
    *,
    questions: Sequence[Mapping[str, Any]] | None = None,
    source_path: str | Path | None = None,
) -> dict[str, Any]:
    """Answer typed portfolio-intent questions for one state."""
    resolved_questions = list(questions) if questions is not None else _default_asset_questions(state)
    answers: dict[str, dict[str, Any]] = {}
    for question in resolved_questions:
        qid = str(question.get("id", ""))
        asset = str(question.get("asset", ""))
        key = f"{qid}:{asset}" if asset else qid
        answers[key] = _answer_question(question, state)
    return {
        "model": MODEL_ID,
        "model_version": MODEL_VERSION,
        "contract": CONTRACT_VERSION,
        "source": str(source_path) if source_path is not None else None,
        "as_of": state.get("as_of"),
        "answers": answers,
        "disclaimer": (
            "Research portfolio-intent decisions only; not investment advice, "
            "not broker instructions, and not live trading execution."
        ),
    }


def build_state_from_artifacts(
    artifacts_dir: str | Path,
    *,
    as_of_index: int = -2,
    include_realized: bool = False,
    costs_bps: float | None = None,
) -> dict[str, Any]:
    """Build a decision state from local research pipeline artifacts.

    The expected directory is the standard CLI artifact layout containing
    ``panel.pt``, ``predictions.pt`` and ``weights.pt``. Realized forward
    returns are included only when explicitly requested for offline evaluation.
    """
    import numpy as np
    import torch

    root = Path(artifacts_dir)
    panel_path = root / "panel.pt"
    pred_path = root / "predictions.pt"
    weights_path = root / "weights.pt"
    for required in (panel_path, pred_path, weights_path):
        if not required.exists():
            raise FileNotFoundError(f"missing required artifact: {required}")

    panel = torch.load(panel_path, map_location="cpu", weights_only=False)
    predictions = torch.load(pred_path, map_location="cpu", weights_only=False)
    weights_blob = torch.load(weights_path, map_location="cpu", weights_only=False)

    close = _to_numpy(panel["close"]).astype(float)
    volume = _to_numpy(panel["volume"]).astype(float)
    vwap = _to_numpy(panel.get("vwap", panel["close"])).astype(float)
    mask = _to_numpy(panel["mask"]).astype(bool)
    pred = _to_numpy(predictions.get("pred", predictions.get("predictions"))).astype(float)
    weights = _to_numpy(weights_blob["weights"]).astype(float)

    if close.shape != pred.shape or close.shape != weights.shape:
        raise ValueError(
            "panel close, predictions, and weights must share shape "
            f"(got {close.shape}, {pred.shape}, {weights.shape})"
        )

    idx = _resolve_index(as_of_index, close.shape[0])
    stocks = [str(item) for item in panel["stocks"]]
    dates = [str(item) for item in panel["dates"]]
    amount = np.where(np.isfinite(vwap * volume), vwap * volume, 0.0)
    returns = np.zeros_like(close, dtype=float)
    returns[1:] = close[1:] / np.clip(close[:-1], 1e-12, None) - 1.0

    score_row = pred[idx]
    rank_row = _percentile_ranks(score_row, mask[idx])
    current_idx = max(0, idx - 1)
    cost = _artifact_costs_bps(root, costs_bps)

    state: dict[str, Any] = {
        "as_of": dates[idx],
        "as_of_index": idx,
        "universe": stocks,
        "predictions": {
            "horizon": "1d",
            "score": _asset_map(stocks, score_row, mask=mask[idx], default=0.0),
            "rank": _asset_map(stocks, rank_row, mask=mask[idx], default=None),
        },
        "tradability": {
            asset: {
                "tradable": bool(mask[idx, j]),
                "halted": False,
                "limit_up": False,
                "limit_down": False,
            }
            for j, asset in enumerate(stocks)
        },
        "liquidity": {
            asset: {
                "amount": _finite_or_none(amount[idx, j]),
                "adv20": _finite_or_none(
                    np.nanmean(amount[max(0, idx - 19) : idx + 1, j])
                ),
            }
            for j, asset in enumerate(stocks)
        },
        "portfolio": {
            "current_weight": _asset_map(stocks, weights[current_idx], default=0.0),
            "target_weight_baseline": _asset_map(stocks, weights[idx], default=0.0),
        },
        "risk": {
            "asset_volatility_20d": {
                asset: _finite_or_none(
                    np.nanstd(returns[max(1, idx - 19) : idx + 1, j])
                )
                for j, asset in enumerate(stocks)
            },
            "max_position_weight": max(0.05, _finite_float(np.nanmax(weights[idx]), 0.0)),
        },
        "cost_model": {
            "effective_cost_bps": cost,
            "max_participation_rate": 0.10,
        },
        "evidence": {
            "data_source": "local_pipeline_artifacts",
            "source_artifacts": str(root),
            "claim_boundary": "research_decision_state_not_alpha_or_execution",
            "summary": _artifact_summary(root),
        },
    }

    if include_realized and idx + 1 < close.shape[0]:
        state["evaluation"] = {
            "realized_date": dates[idx + 1],
            "forward_return": _asset_map(stocks, returns[idx + 1], mask=mask[idx], default=0.0),
            "next_weight_baseline": _asset_map(stocks, weights[idx + 1], default=0.0),
            "note": "Forward returns are for offline evaluation only and must not be used at decision time.",
        }
    return _json_safe(state)


def evaluate_decisions(
    state: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    return_threshold: float = 0.001,
) -> dict[str, Any]:
    """Evaluate typed portfolio decisions against realized offline evidence.

    The evaluation is a research diagnostic. It measures action/weight alignment
    against realized forward returns and state consistency; it does not certify
    trading performance.
    """
    evaluation = _mapping(state.get("evaluation"))
    forward = _mapping(evaluation.get("forward_return"))
    if not forward:
        raise ValueError("state must include evaluation.forward_return for offline evaluation")

    universe = [str(asset) for asset in state.get("universe", [])]
    answers = _mapping(decision.get("answers"))
    current = _mapping(_mapping(state.get("portfolio")).get("current_weight"))
    costs_bps = _as_float(_mapping(state.get("cost_model")).get("effective_cost_bps"), 0.0) or 0.0

    per_asset: list[dict[str, Any]] = []
    target_weights: dict[str, float] = {}
    current_weights: dict[str, float] = {}
    allowed_assets: list[str] = []
    action_correct = 0
    action_total = 0
    allowed_consistent = 0
    allowed_total = 0
    delta_direction_correct = 0
    delta_direction_total = 0
    increase_returns: list[float] = []
    decrease_returns: list[float] = []

    for asset in universe:
        realized = _as_float(forward.get(asset), None)
        if realized is None:
            continue
        action = _answer_value(answers, "asset_action", asset, None)
        target = _as_float(_answer_value(answers, "target_weight", asset, 0.0), 0.0) or 0.0
        allowed = _answer_value(answers, "trade_allowed", asset, None)
        state_allowed = _state_allowed(state, asset)
        current_weight = _as_float(current.get(asset), 0.0) or 0.0
        target_weights[asset] = target
        current_weights[asset] = current_weight
        if allowed is True:
            allowed_assets.append(asset)

        ideal_action = _ideal_action(realized, state_allowed, threshold=return_threshold)
        if action is not None:
            action_total += 1
            action_correct += int(str(action) == ideal_action)
        if allowed is not None:
            allowed_total += 1
            allowed_consistent += int(bool(allowed) == state_allowed)

        delta = target - current_weight
        if abs(delta) > 1e-12 and abs(realized) > return_threshold:
            delta_direction_total += 1
            delta_direction_correct += int((delta > 0 and realized > 0) or (delta < 0 and realized < 0))
        if action == "increase":
            increase_returns.append(realized)
        if action in {"decrease", "exclude"}:
            decrease_returns.append(realized)

        per_asset.append(
            {
                "asset": asset,
                "action": action,
                "ideal_action_from_forward_return": ideal_action,
                "trade_allowed": allowed,
                "state_allowed": state_allowed,
                "current_weight": current_weight,
                "target_weight": target,
                "realized_forward_return": realized,
                "weight_delta": delta,
            }
        )

    target_return = sum(target_weights[a] * (_as_float(forward.get(a), 0.0) or 0.0) for a in target_weights)
    current_return = sum(current_weights[a] * (_as_float(forward.get(a), 0.0) or 0.0) for a in current_weights)
    eligible = allowed_assets or list(target_weights)
    equal_return = (
        sum((_as_float(forward.get(a), 0.0) or 0.0) for a in eligible) / len(eligible)
        if eligible
        else 0.0
    )
    gross_exposure = sum(abs(weight) for weight in target_weights.values())
    best_forward = max(((_as_float(forward.get(a), None), a) for a in eligible), default=(None, ""))[0]
    oracle_return = gross_exposure * best_forward if best_forward is not None else 0.0
    abs_delta_sum = sum(abs(target_weights.get(a, 0.0) - current_weights.get(a, 0.0)) for a in set(target_weights) | set(current_weights))
    turnover = 0.5 * abs_delta_sum
    estimated_cost = abs_delta_sum * costs_bps * 1e-4

    metrics = {
        "n_assets_evaluated": len(per_asset),
        "action_label_accuracy": _safe_ratio(action_correct, action_total),
        "trade_allowed_state_consistency": _safe_ratio(allowed_consistent, allowed_total),
        "weight_delta_direction_accuracy": _safe_ratio(delta_direction_correct, delta_direction_total),
        "target_weight_forward_return": target_return,
        "current_weight_forward_return": current_return,
        "equal_weight_allowed_forward_return": equal_return,
        "target_minus_current_forward_return": target_return - current_return,
        "target_minus_equal_weight_forward_return": target_return - equal_return,
        "one_period_turnover": turnover,
        "estimated_one_period_cost": estimated_cost,
        "target_weight_forward_return_after_cost": target_return - estimated_cost,
        "hindsight_same_exposure_best_asset_return": oracle_return,
        "hindsight_regret": oracle_return - target_return,
        "increase_minus_decrease_forward_return": (
            _mean(increase_returns) - _mean(decrease_returns)
            if increase_returns and decrease_returns
            else None
        ),
    }
    return {
        "model": "mlquant-systemone-decision-evaluator",
        "contract": "system-one-decision-evaluation-v0",
        "as_of": state.get("as_of"),
        "realized_date": evaluation.get("realized_date"),
        "metrics": _json_safe(metrics),
        "per_asset": _json_safe(per_asset),
        "disclaimer": (
            "Offline research diagnostic only; it uses realized forward returns "
            "and must not be used as a live trading signal."
        ),
    }


def _default_asset_questions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    universe = state.get("universe")
    if not isinstance(universe, list) or not universe:
        return [dict(item) for item in DEFAULT_QUESTIONS]
    questions: list[dict[str, Any]] = []
    for asset in universe:
        for template in DEFAULT_QUESTIONS:
            item = dict(template)
            item["asset"] = str(asset)
            questions.append(item)
    return questions


def _answer_question(question: Mapping[str, Any], state: Mapping[str, Any]) -> dict[str, Any]:
    qid = str(question.get("id", ""))
    qtype = str(question.get("type", "choice"))
    asset = str(question.get("asset", ""))
    signals = _asset_signals(state, asset)

    if qtype == "choice":
        options = [str(item) for item in question.get("options", [])]
        value, rationale = _choice_decision(qid, signals)
        if options and value not in options:
            value = options[-1]
            rationale = f"{rationale}; mapped to final option because resolved value was not listed"
        return _typed_answer("choice", value, rationale, asset=asset)
    if qtype == "score":
        value, rationale = _score_decision(qid, signals)
        lo = _as_float(question.get("min"), 0.0)
        hi = _as_float(question.get("max"), 1.0)
        value = max(lo, min(hi, value))
        answer = _typed_answer("score", value, rationale, asset=asset)
        answer["range"] = {"min": lo, "max": hi}
        return answer
    if qtype == "noul":
        value, rationale = _noul_decision(qid, signals)
        return _typed_answer("noul", value, rationale, asset=asset)
    return _typed_answer(qtype, None, f"unsupported question type for {qid}", asset=asset)


def _typed_answer(qtype: str, value: Any, rationale: str, *, asset: str) -> dict[str, Any]:
    return {
        "type": qtype,
        "asset": asset or None,
        "value": value,
        "confidence": None,
        "probabilities": None,
        "rationale": rationale,
    }


def _asset_signals(state: Mapping[str, Any], asset: str) -> dict[str, Any]:
    universe = [str(item) for item in state.get("universe", []) if item is not None]
    predictions = _mapping(state.get("predictions"))
    tradability = _mapping(state.get("tradability"))
    liquidity = _mapping(state.get("liquidity"))
    portfolio = _mapping(state.get("portfolio"))
    risk = _mapping(state.get("risk"))
    cost_model = _mapping(state.get("cost_model"))

    asset_tradability = _mapping(tradability.get(asset))
    asset_liquidity = _mapping(liquidity.get(asset))
    score = _asset_value(predictions.get("score"), asset, default=0.0)
    rank = _asset_value(predictions.get("rank"), asset, default=None)
    current_weight = _asset_value(portfolio.get("current_weight"), asset, default=0.0)
    proposed_weight = _asset_value(portfolio.get("target_weight_baseline"), asset, default=None)
    max_weight = _asset_value(risk.get("max_position_weight"), asset, default=None)
    if max_weight is None:
        max_weight = _as_float(risk.get("max_position_weight"), 0.05)
    asset_vol = _asset_value(risk.get("asset_volatility_20d"), asset, default=None)
    adv20 = _as_float(asset_liquidity.get("adv20"), None)
    amount = _as_float(asset_liquidity.get("amount"), None)
    base_participation = _as_float(cost_model.get("max_participation_rate"), 0.10)
    cost_bps = _as_float(cost_model.get("effective_cost_bps"), 0.0)

    tradable = bool(asset_tradability.get("tradable", True))
    blocked = (
        not asset
        or (universe and asset not in universe)
        or not tradable
        or bool(asset_tradability.get("halted", False))
        or bool(asset_tradability.get("limit_up", False))
        or bool(asset_tradability.get("limit_down", False))
    )
    missing_liquidity = adv20 is None and amount is None
    liquidity_scale = _liquidity_scale(adv20, amount)
    risk_bucket = _risk_bucket(
        blocked=blocked,
        missing_liquidity=missing_liquidity,
        asset_vol=asset_vol,
        cost_bps=cost_bps,
        liquidity_scale=liquidity_scale,
    )
    allowed = not blocked and risk_bucket != "blocked"
    target_weight = _target_weight(
        score=score,
        rank=rank,
        current_weight=current_weight,
        proposed_weight=proposed_weight,
        max_weight=max_weight,
        allowed=allowed,
        risk_bucket=risk_bucket,
        cost_bps=cost_bps,
    )
    action = _asset_action(current_weight=current_weight, target_weight=target_weight, allowed=allowed)
    participation = 0.0 if not allowed else round(base_participation * liquidity_scale, 6)

    return {
        "asset": asset,
        "score": score,
        "rank": rank,
        "current_weight": current_weight,
        "target_weight": target_weight,
        "max_weight": max_weight,
        "allowed": allowed,
        "blocked": blocked,
        "risk_bucket": risk_bucket,
        "max_participation_rate": participation,
        "tradability": dict(asset_tradability),
        "liquidity": dict(asset_liquidity),
        "cost_bps": cost_bps,
        "missing_liquidity": missing_liquidity,
        "action": action,
    }


def _choice_decision(qid: str, signals: Mapping[str, Any]) -> tuple[str, str]:
    if qid == "asset_action":
        return str(signals["action"]), _action_rationale(signals)
    if qid == "risk_bucket":
        return str(signals["risk_bucket"]), _risk_rationale(signals)
    return "unsupported_question", f"no resolver registered for {qid}"


def _score_decision(qid: str, signals: Mapping[str, Any]) -> tuple[float, str]:
    if qid == "target_weight":
        return float(signals["target_weight"]), _target_weight_rationale(signals)
    if qid == "max_participation_rate":
        return float(signals["max_participation_rate"]), _participation_rationale(signals)
    return 0.0, f"no resolver registered for {qid}"


def _noul_decision(qid: str, signals: Mapping[str, Any]) -> tuple[bool, str]:
    if qid == "trade_allowed":
        return bool(signals["allowed"]), _trade_allowed_rationale(signals)
    return False, f"no resolver registered for {qid}"


def _asset_action(*, current_weight: float, target_weight: float, allowed: bool) -> str:
    if not allowed:
        return "exclude"
    delta = target_weight - current_weight
    if delta > 0.0025:
        return "increase"
    if delta < -0.0025:
        return "decrease"
    return "hold"


def _target_weight(
    *,
    score: float,
    rank: float | None,
    current_weight: float,
    proposed_weight: float | None,
    max_weight: float,
    allowed: bool,
    risk_bucket: str,
    cost_bps: float,
) -> float:
    if not allowed:
        return 0.0
    if proposed_weight is not None:
        raw = proposed_weight
    elif rank is not None:
        raw = max_weight * max(0.0, min(1.0, rank))
    else:
        raw = max_weight if score > 0 else 0.0
    if risk_bucket == "high":
        raw *= 0.50
    elif risk_bucket == "medium":
        raw *= 0.75
    if cost_bps >= 20 and abs(raw - current_weight) < 0.01:
        raw = current_weight
    return round(max(0.0, min(max_weight, raw)), 6)


def _risk_bucket(
    *,
    blocked: bool,
    missing_liquidity: bool,
    asset_vol: float | None,
    cost_bps: float,
    liquidity_scale: float,
) -> str:
    if blocked or missing_liquidity or liquidity_scale <= 0:
        return "blocked"
    if asset_vol is not None and asset_vol >= 0.04:
        return "high"
    if cost_bps >= 20 or liquidity_scale < 0.5 or (asset_vol is not None and asset_vol >= 0.025):
        return "medium"
    return "low"


def _liquidity_scale(adv20: float | None, amount: float | None) -> float:
    liquidity = adv20 if adv20 is not None else amount
    if liquidity is None or liquidity <= 0:
        return 0.0
    if liquidity >= 1_000_000_000:
        return 1.0
    if liquidity >= 200_000_000:
        return 0.75
    if liquidity >= 50_000_000:
        return 0.50
    return 0.25


def _action_rationale(signals: Mapping[str, Any]) -> str:
    if signals["action"] == "exclude":
        return "asset is blocked by tradability, universe, liquidity, or risk checks"
    return (
        "target weight compared with current weight after prediction, risk, "
        "liquidity, and cost constraints"
    )


def _risk_rationale(signals: Mapping[str, Any]) -> str:
    if signals["risk_bucket"] == "blocked":
        return "asset is not eligible because tradability or liquidity checks failed"
    return "bucket derived from volatility, liquidity scale, and effective cost assumptions"


def _target_weight_rationale(signals: Mapping[str, Any]) -> str:
    return "research target weight capped by max position size and adjusted for risk and costs"


def _participation_rationale(signals: Mapping[str, Any]) -> str:
    return "maximum research participation rate scaled by available liquidity"


def _trade_allowed_rationale(signals: Mapping[str, Any]) -> str:
    return "true only when asset is tradable, liquid enough, and not risk-blocked"


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _asset_value(value: Any, asset: str, *, default: float | None) -> float | None:
    if isinstance(value, Mapping):
        return _as_float(value.get(asset), default)
    return _as_float(value, default)


def _as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        raise ValueError("artifact tensor is missing")
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_index(index: int, length: int) -> int:
    if length <= 0:
        raise ValueError("cannot resolve an index for an empty artifact")
    resolved = index if index >= 0 else length + index
    if resolved < 0 or resolved >= length:
        raise IndexError(f"as_of_index {index} is outside artifact length {length}")
    return int(resolved)


def _percentile_ranks(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    ranks = np.full(values.shape, np.nan, dtype=float)
    ok = valid & np.isfinite(values)
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return ranks
    order = idx[np.argsort(values[idx], kind="mergesort")]
    if order.size == 1:
        ranks[order[0]] = 1.0
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, order.size)
    return ranks


def _asset_map(
    assets: Sequence[str],
    values: Sequence[Any],
    *,
    mask: Sequence[bool] | None = None,
    default: float | None = None,
) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for idx, asset in enumerate(assets):
        if mask is not None and not bool(mask[idx]):
            out[str(asset)] = default
            continue
        out[str(asset)] = _finite_or_none(values[idx])
        if out[str(asset)] is None and default is not None:
            out[str(asset)] = default
    return out


def _finite_or_none(value: Any) -> float | None:
    out = _as_float(value, None)
    return out if out is not None else None


def _finite_float(value: Any, default: float) -> float:
    out = _as_float(value, default)
    return default if out is None else out


def _artifact_summary(root: Path) -> dict[str, Any]:
    summary_path = root / "summary.json"
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"summary_path": str(summary_path), "parse_error": True}
    return {
        "summary_path": str(summary_path),
        "workflow": payload.get("workflow"),
        "config": payload.get("config"),
        "metrics": payload.get("metrics", {}),
    }


def _artifact_costs_bps(root: Path, override: float | None) -> float:
    if override is not None:
        return float(override)
    summary_path = root / "summary.json"
    if not summary_path.exists():
        return 5.0
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return 5.0
    for key in ("costs_bps", "effective_costs_bps"):
        value = _as_float(payload.get(key), None)
        if value is not None:
            return value
    metadata = _mapping(payload.get("metadata"))
    value = _as_float(metadata.get("effective_costs_bps"), None)
    return 5.0 if value is None else value


def _answer_value(
    answers: Mapping[str, Any],
    qid: str,
    asset: str,
    default: Any,
) -> Any:
    answer = _mapping(answers.get(f"{qid}:{asset}"))
    return answer.get("value", default)


def _state_allowed(state: Mapping[str, Any], asset: str) -> bool:
    return bool(_asset_signals(state, asset)["allowed"])


def _ideal_action(realized_return: float, state_allowed: bool, *, threshold: float) -> str:
    if not state_allowed:
        return "exclude"
    if realized_return > threshold:
        return "increase"
    if realized_return < -threshold:
        return "decrease"
    return "hold"


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values))
