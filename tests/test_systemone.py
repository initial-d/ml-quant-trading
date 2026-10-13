import json
from pathlib import Path

from click.testing import CliRunner

from mlquant.cli.main import cli
from mlquant.systemone import decide, load_questions


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
