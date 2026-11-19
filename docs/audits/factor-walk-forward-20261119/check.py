"""Independent dense-Bartlett/Statsmodels correction and adversarial clock audit.

The Statsmodels dependency is audit-only. No factor recomputation or downloads.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
from pathlib import Path
import shlex
import sys

import numpy as np
import statsmodels
from statsmodels.stats.multitest import multipletests

from mlquant.backtest import walk_forward_factor_ic


def read(path):
    raw = path.read_bytes()
    payload = gzip.decompress(raw) if path.suffix == ".gz" else raw
    return json.loads(payload), raw, payload


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    source, raw, payload = read(args.source)
    result, _, result_payload = read(args.report)
    require(result["input_report_sha256"] == hashlib.sha256(raw).hexdigest(), "source changed")
    require(result["input_payload_sha256"] == hashlib.sha256(payload).hexdigest(), "source payload changed")
    cfg = result["config"]
    names = [row["name"] for row in source["factors"]]
    total = len(source["factors"][0]["daily_ic"])
    expected_starts = list(range(cfg["train_dates"] + cfg["purge_dates"], total, cfg["test_dates"]))
    require([fold["test_start"] for fold in result["folds"]] == expected_starts, "schedule changed")
    require(result["category"] == source["category"], "evidence category changed")
    errors = {"mean": [], "standard_error": [], "p_value": []}
    method = {"by": "fdr_by", "bh": "fdr_bh", "holm": "holm"}[cfg["method"]]
    mutation_checks = 0
    for fold in result["folds"]:
        a, b, s, e = (fold[k] for k in ("train_start", "train_end_exclusive", "test_start", "test_end_exclusive"))
        require(b - a == cfg["train_dates"] and s - b == cfg["purge_dates"], "training/gap changed")
        require(b - 1 + result["label_horizon"] < s, "training label overlaps test")
        require(e == min(s + cfg["test_dates"], total), "test endpoint changed")
        require([row["name"] for row in fold["training"]["factors"]] == names, "family changed")
        family_p, oracle_cells = [], {}
        for original, trained in zip(source["factors"], fold["training"]["factors"]):
            for metric in ("ic", "rank_ic"):
                values = np.asarray(original["daily_" + metric][a:b], dtype=float)
                observed = np.isfinite(values); n = int(observed.sum())
                cell = trained[metric]
                require(cell["valid_dates"] == n and cell["total_dates"] == b - a, "date grid compressed")
                if n < cfg["min_dates"] or np.all(values[observed] == values[observed][0]):
                    require(cell["p_value"] is None, "untestable series gained a p-value")
                    family_p.append(1.0)
                    oracle_cells[(original["name"], metric)] = None
                    continue
                lag = cfg["max_lag"]
                if lag is None:
                    lag = min(max(result["label_horizon"] - 1, int(4 * (n / 100) ** (2 / 9))), len(values) - 1)
                require(cell["max_lag"] == lag, "bandwidth changed")
                mean = float(values[observed].mean())
                u = np.where(observed, values - mean, 0.0)
                indexes = np.arange(len(values))
                kernel = np.maximum(1 - np.abs(indexes[:, None] - indexes[None, :]) / (lag + 1), 0)
                se = math.sqrt(float(u @ kernel @ u) / n**2 * n / (n - 1))
                p = math.erfc(abs(mean / se) / math.sqrt(2))
                for key, expected, actual in (("mean", mean, cell["mean"]),
                                             ("standard_error", se, cell["hac_standard_error"]),
                                             ("p_value", p, cell["p_value"])):
                    require(np.isclose(expected, actual, rtol=1e-10, atol=1e-13), "dense Bartlett mismatch: " + key)
                    errors[key].append(abs(expected - actual))
                family_p.append(p)
                oracle_cells[(original["name"], metric)] = (mean, mean / se)
        adjusted = multipletests(family_p, method=method)[1]
        candidates = []
        for index, name in enumerate(names):
            row = fold["training"]["factors"][index]
            for offset, metric in enumerate(("ic", "rank_ic")):
                if row[metric]["p_value"] is not None:
                    require(np.isclose(row[metric]["adjusted_p_value"], adjusted[2 * index + offset],
                                       rtol=1e-10, atol=1e-13), "Statsmodels correction mismatch")
            offset = 0 if cfg["metric"] == "ic" else 1
            oracle = oracle_cells[(name, cfg["metric"])]
            if oracle is not None and adjusted[2 * index + offset] <= cfg["alpha"]:
                mean, statistic = oracle
                candidates.append((adjusted[2 * index + offset], -abs(statistic), name, 1 if mean > 0 else -1))
        candidates.sort()
        expected = [(name, direction) for _, _, name, direction in candidates[:cfg["max_factors"]]]
        require([(d["name"], d["direction"]) for d in fold["selection"]] == expected, "selection oracle mismatch")
        require(fold["training"]["multiple_testing"]["family_size"] == 2 * len(names), "joint family lost tests")
        # Change each current fold's future, including the purged boundary.
        changed = copy.deepcopy(source)
        for row in changed["factors"]:
            for key in ("daily_ic", "daily_rank_ic"):
                row[key][b:] = [None if v is None else -.9 if v >= 0 else .9 for v in row[key][b:]]
        rerun = walk_forward_factor_ic(changed, **cfg)["folds"][fold["fold"]]
        # Match the retained JSON representation (e.g. interval tuple -> list).
        rerun = json.loads(json.dumps(rerun, allow_nan=False))
        require(rerun["training"] == fold["training"] and rerun["selection"] == fold["selection"],
                "future mutation changed past decisions")
        mutation_checks += 1
    control = None
    if source["category"] == "synthetic planted IC timing control":
        first = result["folds"][0]
        selected = {d["name"]: d for d in first["selection"]}
        outcomes = {d["name"]: d for d in first["test"]}
        require("future_winner" not in selected, "future-only winner selected early")
        require(selected["reversal"]["direction"] == 1 and outcomes["reversal"]["rank_ic"]["mean"] < -.2,
                "reversal was hidden by hindsight orientation")
        require(any("future_winner" in {d["name"] for d in f["selection"]} for f in result["folds"][1:]),
                "future winner never acquired past evidence")
        control = {"future_winner_excluded_initially": True, "later_past_evidence_can_select_it": True,
                   "reversal_first_test_signed_rank_ic": outcomes["reversal"]["rank_ic"]["mean"]}
    output = {"status": "passed", "category": source["category"], "source_sha256": hashlib.sha256(raw).hexdigest(),
              "report_payload_sha256": hashlib.sha256(result_payload).hexdigest(),
              "implementation_commit": result["environment"]["commit"],
              "implementation_sha256": result["environment"]["implementation_sha256"],
              "command": shlex.join([sys.executable, *sys.argv]), "python": sys.version,
              "numpy": np.__version__, "statsmodels": statsmodels.__version__,
              "environment": result["environment"], "summary": result["summary"],
              "fold_families": len(result["folds"]), "hypotheses_per_fold": 2 * len(names),
              "checked_hac_series": len(errors["mean"]),
              "maximum_absolute_errors": {key: max(values, default=0) for key, values in errors.items()},
              "future_and_purge_mutation_checks": mutation_checks, "planted_control": control,
              "limitations": ["Engineering timing/oracle checks, not market alpha or post-selection calibration.",
                              "Input causality and declared horizon are assumptions; multiple folds do not form a single corrected family."]}
    args.json_out.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: output[key] for key in ("status", "checked_hac_series", "future_and_purge_mutation_checks", "summary")}))


if __name__ == "__main__":
    main()
