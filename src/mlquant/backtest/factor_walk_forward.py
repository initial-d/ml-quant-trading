"""Purged rolling factor selection with frozen, training-only decisions."""
from __future__ import annotations

import numpy as np

from .ic_inference import _validated_daily_report, infer_factor_ic_report


def _describe(values):
    """Descriptive oriented IC; deliberately no post-selection p-values."""
    observed = values[np.isfinite(values)]
    constant = bool(len(observed) and np.all(observed == observed[0]))
    mean = float(observed[0] if constant else observed.mean()) if len(observed) else None
    deviations = observed - mean if len(observed) else observed
    scale = float(np.max(np.abs(deviations))) if len(observed) else 0.0
    std = (scale * float(np.sqrt(np.dot(deviations / scale, deviations / scale) / (len(observed) - 1)))
           if len(observed) > 1 and scale else 0.0 if len(observed) > 1 else None)
    return {"mean": mean, "std": std, "ic_ir": mean / std if std else None,
            "valid_dates": len(observed), "total_dates": len(values),
            "daily": [float(value) if np.isfinite(value) else None for value in values]}


def walk_forward_factor_ic(
    report: dict, *, train_dates: int = 200, test_dates: int = 50,
    purge_dates: int | None = None, max_factors: int = 10, metric: str = "rank_ic",
    max_lag: int | None = None, min_dates: int = 20, confidence: float = .95,
    alpha: float = .05, method: str = "by",
) -> dict:
    """Screen only trailing IC histories and evaluate disjoint future windows.

    A signal on original row t has its label ending at t+label_horizon. For
    test start s, training ends *exclusively* at s-purge_dates. The default gap
    is label_horizon, so every training label ends strictly before s. Gaps are
    original panel rows, not compressed valid dates. Each rolling train window
    contains train_dates rows; each test window contains test_dates rows except
    the retained final partial window.

    Every training fold corrects the entire supplied IC + RankIC family. The
    chosen metric must reject after correction. Rank by adjusted p, descending
    absolute HAC statistic, then name; cap at max_factors. Freeze direction from
    training mean. No fallback chooses an insignificant factor. OOS results
    retain signed per-factor IC and missing dates, not portfolio returns or the
    IC of a factor combination. Per-fold BY does not control FDR across time,
    parameter tuning, or a retrospectively chosen input family.
    """
    for name, value in (("train_dates", train_dates), ("test_dates", test_dates), ("max_factors", max_factors)):
        if type(value) is not int or value < 1:
            raise ValueError(name + " must be a positive integer")
    if metric not in ("ic", "rank_ic"):
        raise ValueError("metric must be ic or rank_ic")
    series, horizon, total = _validated_daily_report(report)
    gap = horizon if purge_dates is None else purge_dates
    if type(gap) is not int or gap < horizon:
        raise ValueError("purge_dates must be an integer at least label_horizon")
    if type(min_dates) is not int or min_dates < 2 or train_dates < min_dates or train_dates <= horizon:
        raise ValueError("train_dates must cover min_dates >= 2 and exceed label_horizon")
    first_test = train_dates + gap
    if first_test >= total:
        raise ValueError("no test rows remain after training and purge")
    by_name = {name: (ic, rank_ic) for name, ic, rank_ic in series}
    tracks, folds = {}, []
    for start in range(first_test, total, test_dates):
        stop = min(start + test_dates, total)
        train_stop, train_start = start - gap, start - gap - train_dates
        training_source = {"category": report.get("category", "unspecified source diagnostic"),
                           "protocol": report.get("protocol"), "commit": report.get("commit"),
                           "label_horizon": horizon,
                           "factors": [{"name": name, "daily_ic": ic[train_start:train_stop],
                                        "daily_rank_ic": rank_ic[train_start:train_stop]}
                                       for name, ic, rank_ic in series]}
        training = infer_factor_ic_report(training_source, max_lag=max_lag, min_dates=min_dates,
                                          confidence=confidence, alpha=alpha, method=method)
        candidates = [row for row in training["factors"] if row[metric]["reject"]]
        candidates.sort(key=lambda row: (row[metric]["adjusted_p_value"], -abs(row[metric]["hac_t"]), row["name"]))
        # Freeze these before looking at any held-out values.
        decisions = [{"name": row["name"], "direction": 1 if row[metric]["mean"] > 0 else -1,
                      "training_mean": row[metric]["mean"], "adjusted_p_value": row[metric]["adjusted_p_value"]}
                     for row in candidates[:max_factors]]
        outcomes = []
        for decision in decisions:
            name, direction = decision["name"], decision["direction"]
            if name not in tracks:
                tracks[name] = {"selected_rows": [], "ic": np.full(total, np.nan), "rank_ic": np.full(total, np.nan)}
            track = tracks[name]
            track["selected_rows"].extend(range(start, stop))
            outcome = {"name": name}
            for key, values in zip(("ic", "rank_ic"), by_name[name]):
                signed = direction * values[start:stop]
                track[key][start:stop] = signed
                outcome[key] = _describe(signed)
            outcomes.append(outcome)
        observed_rows = sum(any(outcome["ic"]["daily"][i] is not None for outcome in outcomes)
                            for i in range(stop - start))
        folds.append({"fold": len(folds), "train_start": train_start, "train_end_exclusive": train_stop,
                      "purge_start": train_stop, "purge_end_exclusive": start,
                      "last_training_label_row": train_stop - 1 + horizon,
                      "test_start": start, "test_end_exclusive": stop, "partial_test": stop - start < test_dates,
                      "status": "selected" if decisions else "no_selection",
                      "observed_test_rows": observed_rows,
                      "training": training, "selection": decisions, "test": outcomes})
    oos = [{"name": name, "selected_rows": track["selected_rows"],
            "ic": _describe(track["ic"]), "rank_ic": _describe(track["rank_ic"])}
           for name, track in sorted(tracks.items())]
    return {
        "protocol": "factor-ic-walk-forward-v1", "category": report.get("category", "unspecified source diagnostic"),
        "source_protocol": report.get("protocol"), "source_commit": report.get("commit"),
        "total_dates": total, "label_horizon": horizon,
        "config": {"train_dates": train_dates, "test_dates": test_dates, "purge_dates": gap,
                   "metric": metric, "max_factors": max_factors, "max_lag": max_lag,
                   "min_dates": min_dates, "confidence": confidence, "alpha": alpha, "method": method},
        "selection_rule": "training-only corrected metric p <= alpha; adjusted p, descending |HAC t|, name; training mean direction",
        "clock": "original panel row positions, preserved missing-date gaps; label at t ends at t+label_horizon",
        "summary": {"folds": len(folds), "folds_with_selection": sum(bool(f["selection"]) for f in folds),
                    "test_rows": total - first_test,
                    "test_rows_with_selection": sum(f["test_end_exclusive"] - f["test_start"] for f in folds if f["selection"]),
                    "observed_test_rows": sum(f["observed_test_rows"] for f in folds),
                    "distinct_selected_factors": len(tracks)},
        "folds": folds, "oos_by_factor": oos,
        "limitations": ["Input factors, universe and preprocessing must themselves be causal and fixed independently of held-out outcomes.",
                        "Training labels end strictly before test start under the declared horizon; misdeclared label timing is not repaired.",
                        "HAC inference is asymptotic; each fold corrects its joint IC/RankIC family, not the entire sequence of folds or unreported tuning.",
                        "OOS series are descriptive per-factor signed IC with selection-conditioned missing dates; no post-selection significance guarantee.",
                        "Mean per-factor IC is not the IC of a combined score, portfolio P&L, cost evidence or deployable alpha."],
    }
