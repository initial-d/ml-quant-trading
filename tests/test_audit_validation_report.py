import json
from pathlib import Path

from scripts.audit_validation_report import audit_report, has_errors, markdown_report


def _valid_report():
    return {
        "metadata": {
            "command": "python scripts/public_data_validation.py --source synthetic",
            "source": "synthetic",
            "preset": "us-large-100",
            "python": "3.11.0",
            "platform": "test",
            "pytorch": "2.8.0",
            "costs_bps": 5.0,
            "slippage_bps": 2.0,
            "effective_costs_bps": 7.0,
            "panel": {
                "n_dates": 252,
                "n_stocks": 50,
                "tradable_ratio": 0.99,
                "stocks_with_no_data": [],
            },
        },
        "results": [
            {
                "strategy": "equal_weight",
                "ann_return": 0.1,
                "ann_vol": 0.2,
                "sharpe": 0.5,
                "max_dd": 0.1,
                "turnover": 0.2,
                "cost_drag": 0.01,
                "final_equity": 1.1,
            }
        ],
    }


def test_audit_validation_report_passes_clean_report(tmp_path: Path):
    report = _valid_report()
    findings = audit_report(report)

    assert not has_errors(findings)
    assert findings[0].severity == "pass"
    text = markdown_report(findings, source=tmp_path / "summary.json")
    assert "Validation Report Audit" in text


def test_audit_validation_report_flags_errors_and_warnings():
    report = _valid_report()
    report["metadata"]["effective_costs_bps"] = -1.0
    report["metadata"]["panel"]["tradable_ratio"] = 0.2
    report["results"][0]["final_equity"] = 0.0

    findings = audit_report(report)

    assert has_errors(findings)
    assert any(item.check == "effective_costs_bps" for item in findings)
    assert any(item.check == "panel.tradable_ratio" for item in findings)
    assert any(item.check == "equal_weight.final_equity" for item in findings)


def test_audit_validation_report_fixture_is_json_serializable():
    encoded = json.dumps(_valid_report())
    assert "equal_weight" in encoded
