"""Adversarial timing, reversal, missingness and frozen-selection controls."""
import copy
import gzip
import hashlib
import json

import numpy as np
import pytest
from click.testing import CliRunner

from mlquant.backtest import walk_forward_factor_ic
from mlquant.cli.main import cli


def source_report():
    noise = np.tile([-.03, .03], 70)
    paths = {"positive": .2 + noise, "negative": -.2 - noise,
             "reversal": np.r_[.4 + noise[:60], -.4 + noise[60:]],
             "future_winner": np.r_[np.tile([-.2, .2], 30), np.zeros(5), .6 + noise[65:]],
             "constant": np.full(140, .1), "missing": .2 + noise}
    paths["missing"][[10, 20, 83, 87, 120, 137]] = np.nan
    return {"category": "synthetic timing control", "protocol": "daily-factor-ic-v1",
            "commit": "test-source", "label_horizon": 5,
            "factors": [{"name": name, "daily_ic": [float(v) if np.isfinite(v) else None for v in values],
                         "daily_rank_ic": [float(v) if np.isfinite(v) else None for v in values]}
                        for name, values in paths.items()]}


def run(source, **kwargs):
    return walk_forward_factor_ic(source, train_dates=60, test_dates=20, **kwargs)


def first_decision(result):
    fold = result["folds"][0]
    return fold["training"], fold["selection"]


def test_future_mutations_change_outcomes_but_not_past_selection_or_direction():
    source = source_report()
    baseline = run(source)
    changed = copy.deepcopy(source)
    for row in changed["factors"]:
        for metric in ("daily_ic", "daily_rank_ic"):
            row[metric][65:] = [.8] * 75
    adversarial = run(changed)
    assert first_decision(adversarial) == first_decision(baseline)
    assert adversarial["folds"][0]["test"] != baseline["folds"][0]["test"]
    assert "future_winner" not in {d["name"] for d in baseline["folds"][0]["selection"]}
    assert any("future_winner" in {d["name"] for d in fold["selection"]} for fold in baseline["folds"][1:])


def test_purged_label_overlap_cannot_influence_current_selection():
    source = source_report()
    baseline = run(source)
    changed = copy.deepcopy(source)
    for row in changed["factors"]:
        for metric in ("daily_ic", "daily_rank_ic"):
            row[metric][60:65] = [-.99] * 5
    assert run(changed)["folds"][0] == baseline["folds"][0]
    for fold in baseline["folds"]:
        assert fold["last_training_label_row"] < fold["test_start"]
        assert fold["last_training_label_row"] == fold["train_end_exclusive"] - 1 + 5
        assert fold["train_end_exclusive"] - fold["train_start"] == 60
        assert fold["purge_end_exclusive"] - fold["purge_start"] == 5


def test_training_sign_does_not_flip_to_make_bad_holdout_look_good():
    result = run(source_report())
    fold = result["folds"][0]
    decisions = {d["name"]: d for d in fold["selection"]}
    outcomes = {d["name"]: d for d in fold["test"]}
    assert decisions["negative"]["direction"] == -1
    assert outcomes["negative"]["ic"]["mean"] == pytest.approx(.2)
    assert decisions["reversal"]["direction"] == 1
    assert outcomes["reversal"]["ic"]["mean"] == pytest.approx(-.4)
    assert outcomes["reversal"]["rank_ic"]["mean"] < 0


def test_windows_do_not_overlap_drop_tail_or_compress_missing_rows():
    source = source_report()
    before = copy.deepcopy(source)
    result = run(source)
    assert source == before
    rows = [i for f in result["folds"] for i in range(f["test_start"], f["test_end_exclusive"])]
    assert rows == list(range(65, 140)) and len(rows) == len(set(rows))
    assert result["folds"][-1]["partial_test"] is True
    assert result["folds"][-1]["test_end_exclusive"] - result["folds"][-1]["test_start"] == 15
    missing = next(r for r in result["oos_by_factor"] if r["name"] == "missing")
    assert missing["ic"]["daily"][:65] == [None] * 65
    assert missing["ic"]["daily"][83] is None
    assert 83 in missing["selected_rows"]  # Selected even though its future IC is missing.
    assert missing["ic"]["daily"][84] == pytest.approx(.17)
    assert result["summary"]["test_rows"] == 75
    training_missing = next(r for r in result["folds"][0]["training"]["factors"] if r["name"] == "missing")
    assert training_missing["ic"]["valid_dates"] == 58
    assert training_missing["ic"]["total_dates"] == 60


def test_no_future_eligibility_filter_and_observed_coverage_is_separate():
    source = source_report()
    row = source["factors"][0]
    row["daily_ic"][65:] = [None] * 75
    row["daily_rank_ic"][65:] = [None] * 75
    source["factors"] = [row]
    result = run(source)
    fold = result["folds"][0]
    assert fold["selection"][0]["name"] == "positive"
    assert fold["test"][0]["ic"]["valid_dates"] == 0
    assert fold["test"][0]["ic"]["mean"] is None
    assert fold["observed_test_rows"] == result["summary"]["observed_test_rows"] == 0
    assert result["summary"]["test_rows_with_selection"] > 0


def test_full_joint_family_includes_constant_tests_and_never_falls_back():
    result = run(source_report())
    family = result["folds"][0]["training"]["multiple_testing"]
    assert family["family_size"] == 12 and family["testable"] == 10
    assert "constant" not in {r["name"] for r in result["oos_by_factor"]}
    source = source_report()
    source["factors"] = [source["factors"][4]]
    empty = run(source)
    assert all(f["status"] == "no_selection" for f in empty["folds"])
    assert empty["oos_by_factor"] == []
    assert empty["summary"]["test_rows_with_selection"] == 0


def test_tied_selection_is_name_deterministic_capped_and_input_order_independent():
    source = source_report()
    a = copy.deepcopy(source["factors"][0]); a["name"] = "a"
    z = copy.deepcopy(a); z["name"] = "z"
    source["factors"] = [z, a]
    first = run(source, max_factors=1)
    source["factors"].reverse()
    second = run(source, max_factors=1)
    assert [f["selection"] for f in first["folds"]] == [f["selection"] for f in second["folds"]]
    assert all([d["name"] for d in f["selection"]] == ["a"] for f in first["folds"])


def test_chosen_metric_only_and_extra_purge():
    source = source_report()
    source["factors"] = source["factors"][:1]
    source["factors"][0]["daily_rank_ic"] = np.tile([-.2, .2], 70).tolist()
    rank = run(source, purge_dates=10)
    ic = run(source, metric="ic", purge_dates=10)
    assert rank["folds"][0]["selection"] == []
    assert ic["folds"][0]["selection"][0]["name"] == "positive"
    assert ic["folds"][0]["test_start"] == 70


@pytest.mark.parametrize("kwargs", [{"purge_dates": 4}, {"purge_dates": True}, {"purge_dates": 5.5},
                                    {"max_factors": 0}, {"max_factors": True}, {"metric": "returns"},
                                    {"min_dates": 61}, {"min_dates": 1}, {"max_lag": 3}, {"max_lag": 60},
                                    {"confidence": 1}, {"alpha": 0}, {"method": "none"}])
def test_invalid_protocol_is_rejected(kwargs):
    with pytest.raises(ValueError):
        run(source_report(), **kwargs)


@pytest.mark.parametrize("kwargs", [{"train_dates": True}, {"test_dates": 0}, {"train_dates": 5},
                                    {"train_dates": 135}, {"train_dates": 20, "test_dates": True}])
def test_invalid_window_is_rejected(kwargs):
    with pytest.raises(ValueError):
        walk_forward_factor_ic(source_report(), **kwargs)


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_hashes_freezes_input_and_serializes_full_training_decisions(tmp_path, compressed):
    source = tmp_path / ("source.json.gz" if compressed else "source.json")
    payload = json.dumps(source_report()).encode()
    raw = gzip.compress(payload) if compressed else payload
    source.write_bytes(raw)
    output = tmp_path / "new" / "result.json"
    args = ["factor-walk-forward", "--input-report", str(source), "--json-out", str(output),
            "--train-dates", "60", "--test-dates", "20"]
    command = CliRunner().invoke(cli, args)
    assert command.exit_code == 0, command.output
    result = json.loads(output.read_text())
    assert result["protocol"] == "factor-ic-walk-forward-v1"
    assert result["input_report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["input_payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["source_metadata"]["commit"] == "test-source"
    assert set(result["environment"]["implementation_files"]) == {"factor_walk_forward.py", "ic_inference.py"}
    assert result["environment"]["gpu"] == "unused"
    assert len(result["folds"][0]["training"]["factors"]) == 6
    args[args.index(str(output))] = str(source)
    overwrite = CliRunner().invoke(cli, args)
    assert overwrite.exit_code != 0 and source.read_bytes() == raw
