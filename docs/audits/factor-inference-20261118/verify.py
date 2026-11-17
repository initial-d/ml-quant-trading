"""Verify retained evidence hashes and counts; no inference or data downloads."""
import gzip
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def main():
    base = Path(__file__).resolve().parent
    root = base.parents[2]
    summary = json.loads((base / "summary.json").read_text())
    source_raw = (root / summary["source_path"]).read_bytes()
    source_payload = gzip.decompress(source_raw)
    source = json.loads(source_payload)
    require(sha256(source_raw) == summary["source_sha256"], "source hash changed")
    names = [row["name"] for row in source["factors"]]
    require(len(names) == len(set(names)) == 213, "source family changed")
    reports = {}
    for case in summary["cases"]:
        payload = gzip.decompress((base / case["file"]).read_bytes())
        require(sha256(payload) == case["payload_sha256"], "inference hash changed")
        report = json.loads(payload)
        require(report["protocol"] == "factor-ic-inference-v1", "protocol changed")
        require(report["category"] == source["category"], "source category changed")
        require(report["input_report_sha256"] == sha256(source_raw), "source link changed")
        require(report["input_payload_sha256"] == sha256(source_payload), "source payload link changed")
        require(report["environment"]["commit"] == summary["commit"], "implementation commit changed")
        require([row["name"] for row in report["factors"]] == names, "factor order changed")
        cells = [row[metric] for row in report["factors"] for metric in ("ic", "rank_ic")]
        multiple = report["multiple_testing"]
        require(multiple == case["multiple_testing"], "summary decisions changed")
        require(len(cells) == multiple["family_size"] == 426, "joint family changed")
        require(sorted({cell["max_lag"] for cell in cells}) == case["lags"], "lags changed")
        alpha = multiple["alpha"]
        testable = sum(cell["p_value"] is not None for cell in cells)
        raw = sum(cell["p_value"] is not None and cell["p_value"] <= alpha for cell in cells)
        adjusted = sum(cell["adjusted_p_value"] is not None and cell["adjusted_p_value"] <= alpha
                       for cell in cells)
        require(testable == multiple["testable"], "testable count changed")
        require(raw == multiple["unadjusted_rejections"], "raw rejection count changed")
        require(adjusted == multiple["adjusted_rejections"], "adjusted rejection count changed")
        require(all(cell["reject"] == (cell["adjusted_p_value"] is not None
                                       and cell["adjusted_p_value"] <= alpha)
                    for cell in cells), "decision flags changed")
        reports[case["case"]] = (payload, report)
    oracle = json.loads((base / "oracle.json").read_text())
    auto_payload, auto = reports["auto"]
    require(oracle["status"] == "passed", "oracle did not pass")
    require(oracle["source_sha256"] == sha256(source_raw), "oracle source changed")
    require(oracle["inference_sha256"] == sha256(auto_payload), "oracle report changed")
    require(oracle["implementation_sha256"] == auto["environment"]["implementation_sha256"],
            "oracle implementation changed")
    require(oracle["tests"] == oracle["checked_hac_series"] == 426, "oracle coverage changed")
    require(oracle["multiple_testing"] == auto["multiple_testing"], "oracle family changed")
    print("Passed retained-evidence consistency: 3 reports, 426 tests each, full-family oracle.")
    print("This verifies saved evidence; it does not rerun statistical inference or market validation.")


if __name__ == "__main__":
    main()
