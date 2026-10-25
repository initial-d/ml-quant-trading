import json
from pathlib import Path
import subprocess
import sys


def test_full_engine_subprocess_evidence_and_per_factor_masks(tmp_path):
    output = tmp_path / "report.json"
    script = Path(__file__).parents[1] / "scripts" / "benchmark_factor_backends.py"
    subprocess.run([sys.executable, str(script), "--n-dates", "100", "--n-stocks", "4",
                    "--repeat", "1", "--warmup", "0", "--json-out", str(output)],
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert not report["failures"]
    assert report["factor_count"] == 213
    assert report["commit"] and report["cpu"] and report["cost_assumptions"]
    assert len(report["cases"]) == 2
    assert len({case["input_sha256"] for case in report["cases"]}) == 1
    assert all(case["joint_mask_matches_factor_intersection"] for case in report["cases"])
    assert all(case["nonfinite_output_cells"] == 0 for case in report["cases"])
    assert all(case["shape"] == [100, 4, 213] for case in report["cases"])
    rows = report["comparison"]["per_factor"]
    assert len(rows) == 213
    assert {row["name"] for row in rows} == set(report["cases"][0]["names"])
    # Per-factor observations must survive even when the stricter joint mask
    # is false. Otherwise the long warmup conceals numerical/rank differences.
    joint_count = report["cases"][0]["valid_joint_cells"]
    assert any(row["common_valid_cells"] > joint_count for row in rows)
    assert all("mask_different_cells" in row and "mean_spearman" in row for row in rows)
