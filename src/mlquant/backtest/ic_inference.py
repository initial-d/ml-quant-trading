"""Calendar-aware HAC inference and explicit multiple-testing families for IC."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from statistics import NormalDist

import numpy as np


@dataclass(frozen=True)
class ICMeanInference:
    mean: float | None
    std: float | None
    ic_ir: float | None
    valid_dates: int
    total_dates: int
    max_lag: int
    lag_pair_counts: list[int]
    status: str
    iid_standard_error: float | None = None
    hac_standard_error: float | None = None
    hac_t: float | None = None
    p_value: float | None = None
    confidence_interval: tuple[float, float] | None = None


def ic_mean_inference(
    daily_ic, *, max_lag: int | None = None, horizon: int = 1,
    min_dates: int = 20, confidence: float = .95,
) -> ICMeanInference:
    """Test a mean daily IC against zero using a Bartlett/Newey-West sandwich.

    Input is one IC per original panel row; NaN/None are missing observations.
    Missing positions stay on that grid: their *centred influence* is zero,
    while the mean and denominator use observed dates only. Missing IC values
    are never interpreted as zero IC or compressed into adjacent observations.

    The mean variance is u' K u / n**2, with K[i,j] equal to
    max(1-|i-j|/(L+1), 0), and finite-sample multiplier n/(n-1).
    Two-sided p-values/intervals use an asymptotic normal reference. Automatic
    L=floor(4*(n/100)**(2/9)), at least horizon-1, capped at grid length-1.
    Explicit L must cover horizon-1 and fit the original grid. This bandwidth
    rule does not guarantee removal of all longer-range serial dependence.
    Constant/short series get a status and no inferential p-value.
    """
    if type(horizon) is not int or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    if type(min_dates) is not int or min_dates < 2:
        raise ValueError("min_dates must be an integer >= 2")
    if isinstance(confidence, bool) or not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must be finite and in (0, 1)")
    if max_lag is not None and (type(max_lag) is not int or max_lag < horizon - 1):
        raise ValueError("max_lag must be a nonnegative integer covering horizon-1")
    values = np.asarray(daily_ic, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("daily_ic must be a one-dimensional original-date series")
    if np.isinf(values).any():
        raise ValueError("infinite IC is invalid; use NaN/None for missing dates")
    observed = np.isfinite(values)
    eligible = values[observed]
    if np.any(np.abs(eligible) > 1 + 1e-12):
        raise ValueError("finite IC observations must lie in [-1, 1]")
    total, n = len(values), len(eligible)
    if max_lag is not None and total and max_lag >= total:
        raise ValueError("max_lag must be smaller than the original date grid")
    if max_lag is None:
        lags = min(max(horizon - 1, int(4 * (n / 100) ** (2 / 9))), max(total - 1, 0))
    else:
        lags = max_lag if total else 0
    pairs = [n] + [int(np.count_nonzero(observed[lag:] & observed[:-lag]))
                   for lag in range(1, lags + 1)]
    constant = bool(n and np.all(eligible == eligible[0]))
    mean = float(eligible[0] if constant else eligible.mean()) if n else None
    deviations = eligible - mean if n else eligible
    scale = float(np.max(np.abs(deviations))) if n else 0.0
    std = (scale * math.sqrt(float(np.dot(deviations / scale, deviations / scale)) / (n - 1))
           if n > 1 and scale else 0.0 if n > 1 else None)
    base = dict(mean=mean, std=std, ic_ir=mean / std if std else None,
                valid_dates=n, total_dates=total, max_lag=lags, lag_pair_counts=pairs)
    if n < min_dates or horizon >= total:
        return ICMeanInference(**base, status="insufficient_dates")
    if not scale:
        return ICMeanInference(**base, status="constant_series")
    influence = np.zeros(total, dtype=np.float64)
    influence[observed] = deviations / scale
    # The full moving-sum norm is exactly the PSD Bartlett quadratic form.
    # It avoids cancellation between positive diagonal and negative lag terms.
    moving = np.convolve(influence, np.ones(lags + 1), mode="full")
    se = math.sqrt(float(np.dot(moving, moving)) / (lags + 1) * n / (n - 1)) * scale / n
    if not math.isfinite(se) or se <= 0:
        return ICMeanInference(**base, status="zero_hac_variance")
    statistic = mean / se
    critical = NormalDist().inv_cdf((1 + confidence) / 2)
    return ICMeanInference(**base, status="ok", iid_standard_error=std / math.sqrt(n),
                           hac_standard_error=se, hac_t=statistic,
                           p_value=math.erfc(abs(statistic) / math.sqrt(2)),
                           confidence_interval=(mean - critical * se, mean + critical * se))


def adjust_pvalues(p_values, *, method: str = "by") -> np.ndarray:
    """Adjusted p-values over every supplied hypothesis, including missing ones.

    BY (default) controls FDR under arbitrary cross-test dependence, conditional
    on valid marginal p-values. BH needs independence/positive dependence;
    Holm controls family-wise error under arbitrary dependence. HAC p-values
    remain asymptotic approximations. None/NaN tests contribute p=1 to the
    family size and retain NaN adjusted output; dropping them must not reduce
    the declared family. Corrections do not cover unreported experiments.
    """
    if method not in ("by", "bh", "holm"):
        raise ValueError("method must be by, bh or holm")
    values = np.asarray(p_values, dtype=np.float64)
    if values.ndim != 1 or np.isinf(values).any():
        raise ValueError("p_values must be one-dimensional and contain no infinity")
    valid = np.isfinite(values)
    if np.any((values[valid] < 0) | (values[valid] > 1)):
        raise ValueError("finite p-values must lie in [0, 1]")
    count = len(values)
    if not count:
        return values.copy()
    effective = np.where(valid, values, 1.0)
    order = np.argsort(effective, kind="stable")
    sorted_p = effective[order]
    if method == "holm":
        corrected = np.maximum.accumulate(sorted_p * np.arange(count, 0, -1))
    else:
        multiplier = math.fsum(1 / k for k in range(1, count + 1)) if method == "by" else 1.0
        scaled = sorted_p * (count * multiplier / np.arange(1, count + 1))
        corrected = np.minimum.accumulate(scaled[::-1])[::-1]
    output = np.empty_like(values)
    output[order] = np.clip(corrected, 0.0, 1.0)
    output[~valid] = np.nan
    return output


def _validated_daily_report(report: dict):
    """Validate frozen series without computing any full-sample statistics."""
    if not isinstance(report, dict):
        raise ValueError("report must be an object")
    factors = report.get("factors")
    if not isinstance(factors, list) or not factors:
        raise ValueError("report must contain a nonempty factors list")
    horizon = report.get("label_horizon", 1)
    if type(horizon) is not int or horizon < 1:
        raise ValueError("label_horizon must be a positive integer")
    names, series, length = set(), [], None
    for factor in factors:
        if not isinstance(factor, dict):
            raise ValueError("each factor must be an object")
        name = factor.get("name")
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("factor names must be unique nonempty strings")
        names.add(name)
        pair = []
        for metric in ("ic", "rank_ic"):
            key = "daily_" + metric
            if key not in factor:
                raise ValueError("each factor must contain daily_ic and daily_rank_ic")
            values = np.asarray(factor[key], dtype=np.float64)
            if values.ndim != 1:
                raise ValueError("daily_ic must be a one-dimensional original-date series")
            if np.isinf(values).any() or np.any(np.abs(values[np.isfinite(values)]) > 1 + 1e-12):
                raise ValueError("finite IC observations must lie in [-1, 1]; infinity is invalid")
            if length is None:
                length = len(values)
            if len(values) != length or factor.get("total_dates", length) != length:
                raise ValueError("all factor series must share the original date grid")
            if factor.get("valid_dates", int(np.isfinite(values).sum())) != int(np.isfinite(values).sum()):
                raise ValueError("valid_dates disagrees with the retained daily series")
            pair.append(values)
        if not np.array_equal(np.isfinite(pair[0]), np.isfinite(pair[1])):
            raise ValueError("daily IC and RankIC must share their reported eligible dates")
        series.append((name, *pair))
    return series, horizon, length


def infer_factor_ic_report(
    report: dict, *, max_lag: int | None = None, min_dates: int = 20,
    confidence: float = .95, alpha: float = .05, method: str = "by",
) -> dict:
    """Infer a frozen daily-IC report, with one joint IC + RankIC test family.

    Factor names must be unique and all series use the same original panel-row
    grid. Expected input: factors[{name, daily_ic, daily_rank_ic}], with optional
    total_dates/valid_dates consistency checks and label_horizon (default 1).
    All supplied factors and both metrics enter correction, including untestable
    series. This reports full-sample inference, not held-out factor selection.
    """
    if isinstance(alpha, bool) or not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must be finite and in (0, 1)")
    series, horizon, length = _validated_daily_report(report)
    rows, tests = [], []
    for name, ic, rank_ic in series:
        row = {"name": name}
        for metric, values in (("ic", ic), ("rank_ic", rank_ic)):
            result = ic_mean_inference(values, max_lag=max_lag, horizon=horizon,
                                       min_dates=min_dates, confidence=confidence)
            row[metric] = asdict(result)
            tests.append(row[metric])
        rows.append(row)
    adjusted = adjust_pvalues([test["p_value"] for test in tests], method=method)
    for test, value in zip(tests, adjusted):
        test["adjusted_p_value"] = float(value) if np.isfinite(value) else None
        test["reject"] = bool(np.isfinite(value) and value <= alpha)
    return {
        "protocol": "factor-ic-inference-v1", "category": report.get("category", "unspecified source diagnostic"),
        "source_protocol": report.get("protocol"), "source_commit": report.get("commit"),
        "label_horizon": horizon, "total_dates": length,
        "inference": {"estimator": "Bartlett/Newey-West mean sandwich, n/(n-1) correction",
                      "reference": "two-sided asymptotic normal", "confidence": confidence,
                      "interval_scope": "pointwise, not simultaneous across the family",
                      "max_lag_requested": max_lag, "min_dates": min_dates,
                      "date_clock": "original panel row positions; missing observations remain gaps"},
        "multiple_testing": {"method": method, "alpha": alpha, "family_size": len(tests),
                             "testable": sum(t["p_value"] is not None for t in tests),
                             "unadjusted_rejections": sum(t["p_value"] is not None and t["p_value"] <= alpha for t in tests),
                             "adjusted_rejections": sum(t["reject"] for t in tests),
                             "family": "all supplied factors x (IC, RankIC), including untestable series"},
        "limitations": ["Asymptotic inference; bandwidth and sample size affect calibration.",
                        "BY permits arbitrary cross-factor dependence conditional on valid marginal p-values; BH needs independence/positive dependence.",
                        "Missing-date inference is conditional on the observed mask; informative missingness is not repaired.",
                        "Correction covers this declared family, not unreported trials or subsequent factor tuning.",
                        "Full-sample inference is not held-out selection, transaction-cost evidence or deployable alpha."],
        "factors": rows,
    }
