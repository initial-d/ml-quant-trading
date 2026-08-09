from pathlib import Path

from scripts.technical_pipeline_audit import run_audit, write_report


def test_technical_pipeline_audit_passes_and_writes_reports(tmp_path: Path) -> None:
    report = run_audit(n_dates=80, n_stocks=12, seed=42)

    assert report["passed"] is True
    assert len(report["checks"]) == 6
    assert {check["name"] for check in report["checks"]} == {
        "factor_catalog",
        "deterministic_generation",
        "mask_isolation",
        "forward_label_boundary",
        "lagged_execution_alignment",
        "transaction_cost_linearity",
    }

    markdown_path, json_path = write_report(report, tmp_path)
    assert markdown_path.exists()
    assert json_path.exists()
    assert "Overall status: PASS" in markdown_path.read_text()
    assert '"passed": true' in json_path.read_text()
