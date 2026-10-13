import json
from pathlib import Path

import torch
from click.testing import CliRunner

from mlquant.cli.main import cli
from mlquant.systemone import build_state_from_artifacts, decide, evaluate_decisions, load_questions


def _portfolio_state():
    return {
        "as_of": "2024-12-31",
        "universe": ["600519", "000001", "300750"],
        "predictions": {
            "horizon": "1d",
            "score": {
                "600519": 0.018,
                "000001": -0.006,
                "300750": 0.012,
            },
            "rank": {
                "600519": 0.91,
                "000001": 0.23,
                "300750": 0.75,
            },
        },
        "tradability": {
            "600519": {
                "tradable": True,
                "halted": False,
                "limit_up": False,
                "limit_down": False,
            },
            "000001": {
                "tradable": True,
                "halted": False,
                "limit_up": False,
                "limit_down": False,
            },
            "300750": {
                "tradable": False,
                "halted": True,
                "limit_up": False,
                "limit_down": False,
            },
        },
        "liquidity": {
            "600519": {"amount": 1_523_000_000, "adv20": 1_340_000_000},
            "000001": {"amount": 210_000_000, "adv20": 190_000_000},
            "300750": {"amount": 900_000_000, "adv20": 850_000_000},
        },
        "portfolio": {
            "current_weight": {
                "600519": 0.02,
                "000001": 0.01,
                "300750": 0.03,
            },
            "target_weight_baseline": {
                "600519": 0.04,
                "000001": 0.00,
                "300750": 0.04,
            },
        },
        "risk": {
            "asset_volatility_20d": {
                "600519": 0.021,
                "000001": 0.018,
                "300750": 0.019,
            },
            "portfolio_volatility": 0.132,
            "max_position_weight": 0.05,
        },
        "cost_model": {
            "effective_cost_bps": 7,
            "max_participation_rate": 0.10,
        },
        "evidence": {
            "data_source": "akshare",
            "claim_boundary": "public_data_validation_not_alpha",
        },
    }


def _write_artifacts(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    dates = ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"]
    stocks = ["AAA", "BBB", "CCC"]
    close = torch.tensor(
        [
            [10.0, 20.0, 30.0],
            [11.0, 19.0, 30.0],
            [12.0, 18.0, 33.0],
            [13.0, 17.0, 30.0],
        ]
    )
    volume = torch.full_like(close, 1_000_000.0)
    mask = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, True, False],
            [True, True, True],
        ]
    )
    torch.save(
        {
            "dates": dates,
            "stocks": stocks,
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "volume": volume,
            "vwap": close,
            "mask": mask,
        },
        root / "panel.pt",
    )
    torch.save(
        {
            "pred": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.01, -0.02, 0.03],
                    [0.04, -0.01, 0.02],
                    [0.02, -0.03, 0.01],
                ]
            )
        },
        root / "predictions.pt",
    )
    torch.save(
        {
            "weights": torch.tensor(
                [
                    [0.0, 0.0, 0.0],
                    [0.2, 0.0, 0.1],
                    [0.3, 0.0, 0.0],
                    [0.1, 0.2, 0.0],
                ]
            )
        },
        root / "weights.pt",
    )
    (root / "summary.json").write_text(
        json.dumps(
            {
                "workflow": "synthetic factor-to-backtest demo",
                "costs_bps": 7,
                "metrics": {"sharpe": 0.5},
            }
        ),
        encoding="utf-8",
    )


def test_systemone_answers_asset_portfolio_questions():
    payload = decide(
        _portfolio_state(),
        questions=[
            {
                "id": "asset_action",
                "type": "choice",
                "asset": "600519",
                "options": ["increase", "hold", "decrease", "exclude"],
            },
            {"id": "target_weight", "type": "score", "asset": "600519", "min": 0, "max": 0.05},
            {"id": "trade_allowed", "type": "noul", "asset": "600519"},
            {
                "id": "risk_bucket",
                "type": "choice",
                "asset": "600519",
                "options": ["low", "medium", "high", "blocked"],
            },
            {
                "id": "max_participation_rate",
                "type": "score",
                "asset": "600519",
                "min": 0,
                "max": 0.10,
            },
        ],
        source_path="decision_state.json",
    )

    answers = payload["answers"]
    assert payload["model"] == "mlquant-systemone-portfolio-decision"
    assert answers["asset_action:600519"]["value"] == "increase"
    assert answers["target_weight:600519"]["value"] == 0.04
    assert answers["trade_allowed:600519"]["value"] is True
    assert answers["risk_bucket:600519"]["value"] == "low"
    assert answers["max_participation_rate:600519"]["value"] == 0.10
    assert answers["asset_action:600519"]["confidence"] is None
    assert answers["asset_action:600519"]["probabilities"] is None


def test_systemone_blocks_untradable_asset():
    payload = decide(
        _portfolio_state(),
        questions=[
            {
                "id": "asset_action",
                "type": "choice",
                "asset": "300750",
                "options": ["increase", "hold", "decrease", "exclude"],
            },
            {"id": "target_weight", "type": "score", "asset": "300750", "min": 0, "max": 0.05},
            {"id": "trade_allowed", "type": "noul", "asset": "300750"},
            {
                "id": "risk_bucket",
                "type": "choice",
                "asset": "300750",
                "options": ["low", "medium", "high", "blocked"],
            },
        ],
    )

    answers = payload["answers"]
    assert answers["asset_action:300750"]["value"] == "exclude"
    assert answers["target_weight:300750"]["value"] == 0.0
    assert answers["trade_allowed:300750"]["value"] is False
    assert answers["risk_bucket:300750"]["value"] == "blocked"


def test_systemone_default_questions_expand_over_universe():
    payload = decide(_portfolio_state())

    assert "asset_action:600519" in payload["answers"]
    assert "asset_action:000001" in payload["answers"]
    assert "asset_action:300750" in payload["answers"]
    assert "target_weight:600519" in payload["answers"]


def test_systemone_question_loader_accepts_questions_object(tmp_path: Path):
    questions = tmp_path / "questions.json"
    questions.write_text(
        json.dumps(
            {
                "questions": [
                    {
                        "id": "target_weight",
                        "type": "score",
                        "asset": "600519",
                        "min": 0,
                        "max": 0.05,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    loaded = load_questions(questions)

    assert loaded == [
        {"id": "target_weight", "type": "score", "asset": "600519", "min": 0, "max": 0.05}
    ]


def test_systemone_cli_writes_portfolio_decision_json(tmp_path: Path):
    state = tmp_path / "decision_state.json"
    state.write_text(json.dumps(_portfolio_state()), encoding="utf-8")
    output = tmp_path / "decision.json"

    result = CliRunner().invoke(
        cli,
        ["systemone", "decide", "--state", str(state), "--output", str(output)],
    )

    assert result.exit_code == 0
    assert "wrote" in result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["answers"]["asset_action:600519"]["value"] == "increase"
    assert payload["answers"]["trade_allowed:300750"]["type"] == "noul"


def test_systemone_build_state_from_artifacts_includes_offline_evidence(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)

    state = build_state_from_artifacts(artifacts, as_of_index=2, include_realized=True)

    assert state["as_of"] == "2024-01-03"
    assert state["universe"] == ["AAA", "BBB", "CCC"]
    assert state["predictions"]["score"]["AAA"] == 0.03999999910593033
    assert state["tradability"]["CCC"]["tradable"] is False
    assert state["portfolio"]["current_weight"]["AAA"] == 0.20000000298023224
    assert state["portfolio"]["target_weight_baseline"]["AAA"] == 0.30000001192092896
    assert state["cost_model"]["effective_cost_bps"] == 7.0
    assert state["evaluation"]["realized_date"] == "2024-01-04"
    assert state["evaluation"]["forward_return"]["AAA"] > 0


def test_systemone_evaluates_typed_decisions_with_realized_returns():
    state = _portfolio_state()
    state["evaluation"] = {
        "realized_date": "2025-01-02",
        "forward_return": {
            "600519": 0.02,
            "000001": -0.01,
            "300750": 0.03,
        },
    }
    decision = decide(state)

    report = evaluate_decisions(state, decision, return_threshold=0.001)

    assert report["contract"] == "system-one-decision-evaluation-v0"
    assert report["metrics"]["n_assets_evaluated"] == 3
    assert report["metrics"]["trade_allowed_state_consistency"] == 1.0
    assert report["metrics"]["action_label_accuracy"] == 1.0
    assert report["metrics"]["target_weight_forward_return"] > 0
    assert report["metrics"]["estimated_one_period_cost"] >= 0


def test_systemone_build_state_and_evaluate_cli(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    _write_artifacts(artifacts)
    state = tmp_path / "state.json"
    decision = tmp_path / "decision.json"
    evaluation = tmp_path / "evaluation.json"

    runner = CliRunner()
    build_result = runner.invoke(
        cli,
        [
            "systemone",
            "build-state",
            "--artifacts-dir",
            str(artifacts),
            "--as-of-index",
            "2",
            "--include-realized",
            "--output",
            str(state),
        ],
    )
    assert build_result.exit_code == 0

    decide_result = runner.invoke(
        cli,
        ["systemone", "decide", "--state", str(state), "--output", str(decision)],
    )
    assert decide_result.exit_code == 0

    evaluate_result = runner.invoke(
        cli,
        [
            "systemone",
            "evaluate",
            "--state",
            str(state),
            "--decision",
            str(decision),
            "--output",
            str(evaluation),
        ],
    )
    assert evaluate_result.exit_code == 0
    payload = json.loads(evaluation.read_text(encoding="utf-8"))
    assert payload["metrics"]["n_assets_evaluated"] == 3
