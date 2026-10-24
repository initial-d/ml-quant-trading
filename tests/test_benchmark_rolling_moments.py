import json
from pathlib import Path
import subprocess
import sys


def test_controlled_benchmark_subprocess_protocol(tmp_path):
    output = tmp_path / "report.json"
    script = Path(__file__).parents[1] / "scripts" / "benchmark_rolling_moments.py"
    subprocess.run([sys.executable, str(script), "--n-dates", "10", "--n-stocks", "3",
                    "--windows", "3", "--warmup", "0", "--repeat", "1",
                    "--json-out", str(output)], check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert len(report["cases"]) == 4 and not report["failures"]
    assert len({case["input_sha256"] for case in report["cases"]}) == 1
    assert all(check["passed"] for check in report["comparisons"])
    assert report["commit"] and report["cost_assumptions"] and report["cpu"]
    assert all(case["command"] and case["seconds"] for case in report["cases"])
