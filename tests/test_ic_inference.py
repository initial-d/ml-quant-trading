"""Independent covariance-matrix and hand-calculated multiple-testing controls."""
import gzip
import json
import math

import numpy as np
import pytest
from click.testing import CliRunner
from scipy.stats import norm

from mlquant.backtest.ic_inference import (
    adjust_pvalues, ic_mean_inference, infer_factor_ic_report,
)
from mlquant.cli.main import cli


def matrix_oracle(values, lag):
    values = np.asarray(values, dtype=float)
    observed = np.isfinite(values)
    mean = values[observed].mean()
    influence = np.where(observed, values - mean, 0.)
    index = np.arange(len(values))
    kernel = np.maximum(1 - np.abs(index[:, None] - index[None, :]) / (lag + 1), 0)
    n = observed.sum()
    sandwich = np.sum(influence[:, None] * kernel * influence[None, :], dtype=np.float64)
    se = math.sqrt(float(sandwich) / n**2 * n / (n - 1))
    return mean, se


@pytest.mark.parametrize("lag", [0, 1, 7, 19])
@pytest.mark.parametrize("missing", [False, True])
def test_hac_matches_independent_dense_bartlett_matrix_and_normal_reference(lag, missing):
    values = np.random.default_rng(84).uniform(-.25, .35, size=80)
    if missing:
        values[::4] = np.nan
        values[31:38] = np.nan
    mean, se = matrix_oracle(values, lag)
    result = ic_mean_inference(values, max_lag=lag)
    assert result.status == "ok"
    assert result.mean == pytest.approx(mean)
    assert result.hac_standard_error == pytest.approx(se, rel=1e-12)
    assert result.hac_t == pytest.approx(mean / se, rel=1e-12)
    assert result.p_value == pytest.approx(2 * norm.sf(abs(mean / se)), rel=1e-12)
    np.testing.assert_allclose(result.confidence_interval, mean + np.array([-1, 1]) * norm.ppf(.975) * se)
    observed = np.isfinite(values)
    assert result.lag_pair_counts == [observed.sum()] + [np.count_nonzero(observed[k:] & observed[:-k]) for k in range(1, lag + 1)]
    if lag == 0:
        assert result.hac_standard_error == pytest.approx(np.nanstd(values, ddof=1) / np.sqrt(observed.sum()))


def test_missing_dates_do_not_become_adjacent_or_zero_ic():
    values = np.tile([-.2, None, .4, None, .1], 10)
    result = ic_mean_inference(values, max_lag=1)
    collapsed = ic_mean_inference(np.asarray(values, dtype=float)[np.isfinite(np.asarray(values, dtype=float))], max_lag=1)
    assert result.mean == pytest.approx(.1)
    assert result.total_dates == 50 and result.valid_dates == 30
    assert result.lag_pair_counts == [30, 9]
    assert result.hac_standard_error != pytest.approx(collapsed.hac_standard_error)
    mean, se = matrix_oracle(values, 1)
    assert result.mean == pytest.approx(mean)
    assert result.hac_standard_error == pytest.approx(se)


def test_serial_dependence_changes_inference_without_changing_mean_or_ic_ir():
    values = np.repeat(np.random.default_rng(7).uniform(-.2, .2, 60), 10) + .01
    result = ic_mean_inference(values, max_lag=10)
    iid = ic_mean_inference(values, max_lag=0)
    assert result.mean == iid.mean and result.ic_ir == iid.ic_ir
    assert result.hac_standard_error > 2 * result.iid_standard_error
    assert result.p_value > iid.p_value
    # Bartlett covariance remains nonnegative for strong negative dependence.
    alternating = ic_mean_inference(np.tile([-.1, .1], 100), max_lag=19)
    assert 0 < alternating.hac_standard_error < alternating.iid_standard_error


@pytest.mark.parametrize("values,status", [([], "insufficient_dates"), ([None] * 50, "insufficient_dates"),
                                          ([.1] * 10, "insufficient_dates"), ([.1] * 50, "constant_series")])
def test_short_and_constant_series_have_no_inferential_p_value(values, status):
    result = ic_mean_inference(values)
    assert result.status == status
    assert result.p_value is None and result.hac_t is None and result.confidence_interval is None


def test_overlap_bandwidth_is_explicit_and_cannot_be_silently_too_short():
    values = np.linspace(-.3, .4, 30)
    assert ic_mean_inference(values, horizon=10).max_lag == 9
    with pytest.raises(ValueError, match="horizon-1"):
        ic_mean_inference(values, horizon=5, max_lag=3)
    with pytest.raises(ValueError, match="original date grid"):
        ic_mean_inference(values, max_lag=30)


@pytest.mark.parametrize("kwargs", [{"horizon": True}, {"horizon": 0}, {"min_dates": 1},
                                    {"max_lag": True}, {"max_lag": -1}, {"confidence": 1}, {"confidence": float("nan")}])
def test_invalid_inference_options(kwargs):
    with pytest.raises(ValueError):
        ic_mean_inference(np.linspace(-.2, .2, 30), **kwargs)


@pytest.mark.parametrize("values", [[float("inf")], [-1.1], [[.1, .2]]])
def test_invalid_ic_series(values):
    with pytest.raises(ValueError):
        ic_mean_inference(values)


@pytest.mark.parametrize("method,expected", [
    ("bh", [.025, .05, .05, .01, np.nan]),
    ("by", np.array([.025, .05, .05, .01, np.nan]) * sum(1 / k for k in range(1, 6))),
    ("holm", [.04, .09, .09, .01, np.nan]),
])
def test_adjustments_match_hand_computation_and_keep_untestable_tests_in_family(method, expected):
    values = np.array([.01, .04, .03, .002, np.nan])
    np.testing.assert_allclose(adjust_pvalues(values, method=method), expected)
    permutation = [4, 1, 3, 2, 0]
    np.testing.assert_allclose(adjust_pvalues(values[permutation], method=method), np.asarray(expected)[permutation])
    # An untestable factor still belongs to the originally declared search.
    assert adjust_pvalues([.01, None], method="bh")[0] == pytest.approx(.02)
    assert adjust_pvalues([.01], method="bh")[0] == pytest.approx(.01)


@pytest.mark.parametrize("method", ["by", "bh", "holm"])
def test_adjustments_cover_empty_tied_zero_and_unit_probabilities(method):
    assert adjust_pvalues([], method=method).size == 0
    np.testing.assert_equal(adjust_pvalues([0, 0, 1, None], method=method), [0, 0, 1, np.nan])


@pytest.mark.parametrize("values", [[-1], [1.01], [float("inf")], [[.1]]])
def test_invalid_p_values(values):
    with pytest.raises(ValueError):
        adjust_pvalues(values)


def source_report():
    values = np.linspace(-.1, .3, 40).tolist()
    return {"protocol": "daily-factor-ic-v1", "category": "synthetic engineering diagnostic",
            "commit": "frozen-source", "factors": [
                {"name": "candidate", "daily_ic": values, "daily_rank_ic": values,
                 "total_dates": 40, "valid_dates": 40},
                {"name": "constant", "daily_ic": [.1] * 40, "daily_rank_ic": [.1] * 40},
            ]}


def test_report_uses_joint_metric_family_and_preserves_source_category():
    report = infer_factor_ic_report(source_report(), max_lag=0)
    assert report["multiple_testing"]["family_size"] == 4
    assert report["multiple_testing"]["testable"] == 2
    assert report["category"] == "synthetic engineering diagnostic"
    assert report["source_commit"] == "frozen-source"
    constant = report["factors"][1]
    assert constant["ic"]["adjusted_p_value"] is None and not constant["ic"]["reject"]
    raw = report["factors"][0]["ic"]["p_value"]
    expected = adjust_pvalues([raw, raw, None, None])[0]
    assert report["factors"][0]["ic"]["adjusted_p_value"] == pytest.approx(expected)
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("mutation", ["duplicate", "length", "mask", "count", "missing", "object"])
def test_report_rejects_inconsistent_families(mutation):
    report = source_report()
    if mutation == "duplicate":
        report["factors"][1]["name"] = "candidate"
    elif mutation == "length":
        report["factors"][1]["daily_ic"].pop()
    elif mutation == "mask":
        report["factors"][1]["daily_ic"][0] = None
    elif mutation == "count":
        report["factors"][0]["valid_dates"] = 39
    elif mutation == "missing":
        del report["factors"][1]["daily_rank_ic"]
    else:
        report["factors"][1] = "bad"
    with pytest.raises(ValueError):
        infer_factor_ic_report(report)


@pytest.mark.parametrize("compressed", [False, True])
def test_cli_reuses_frozen_reports_and_writes_inspectable_inference(tmp_path, compressed):
    import hashlib
    source = tmp_path / ("source.json.gz" if compressed else "source.json")
    payload = json.dumps(source_report()).encode()
    raw = gzip.compress(payload) if compressed else payload
    source.write_bytes(raw)
    output = tmp_path / "inference.json"
    result = CliRunner().invoke(cli, ["factor-inference", "--input-report", str(source),
        "--json-out", str(output), "--max-lag", "3"])
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text())
    assert report["input_report_sha256"] == hashlib.sha256(raw).hexdigest()
    assert report["input_payload_sha256"] == hashlib.sha256(payload).hexdigest()
    assert report["environment"]["gpu"] == "unused"
    assert report["source_metadata"]["category"] == "synthetic engineering diagnostic"
    assert report["multiple_testing"]["family_size"] == 4
    same = CliRunner().invoke(cli, ["factor-inference", "--input-report", str(source), "--json-out", str(source)])
    assert same.exit_code != 0 and source.read_bytes() == raw
