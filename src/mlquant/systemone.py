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
