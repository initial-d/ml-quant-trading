import json
from pathlib import Path
import subprocess
import sys


def test_stack_and_stream_processes_preserve_all_factor_hashes(tmp_path):
    output = tmp_path / "stream.json"
    script = Path(__file__).parents[1] / "scripts/benchmark_factor_streaming.py"
    subprocess.run([sys.executable, str(script), "--n-dates", "100", "--n-stocks", "4",
                    "--warmup", "0", "--repeat", "1", "--json-out", str(output)],
                   check=True, capture_output=True, text=True)
    report = json.loads(output.read_text())
    assert not report["failures"]
    assert report["comparison"]["all_factor_bytes_and_joint_mask_equal"]
    assert len(report["cases"]) == 2
    assert all(len(case["factor_value_sha256"]) == 213 for case in report["cases"])
    assert report["commit"] and report["cpu"] and report["cost_assumptions"]
