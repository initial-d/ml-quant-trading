"""Plot an existing CPU rolling-moments report; requires optional matplotlib."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    report = json.loads(args.report.read_text())
    if report["failures"] or not all(c["passed"] for c in report["comparisons"]):
        raise ValueError("report has failed cases; inspect raw evidence before plotting")
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), layout="constrained")
    for col, kind in enumerate(("mean", "corr")):
        for method, color in (("unfold", "#c45c42"), ("linear", "#237fa0")):
            cases = sorted((c for c in report["cases"] if c["kernel"] == f"{kind}-{method}"),
                           key=lambda c: c["window"])
            windows = [c["window"] for c in cases]
            axes[0, col].plot(windows, [c["median_seconds"] * 1000 for c in cases],
                              "o-", color=color, label=method)
            if any(c["peak_process_rss_bytes"] is None for c in cases):
                raise ValueError("RSS unavailable on this platform")
            axes[1, col].plot(windows, [c["peak_process_rss_bytes"] / 1e6 for c in cases],
                              "o-", color=color, label=method)
        axes[0, col].set_title(f"Trailing {kind}: warm median time")
        axes[0, col].set_ylabel("Milliseconds")
        axes[1, col].set_title(f"Trailing {kind}: whole-process peak RSS")
        axes[1, col].set_ylabel("MB (decimal)")
        for row in (0, 1):
            axes[row, col].set_xlabel("Window (dates)")
            axes[row, col].set_xticks(windows)
            axes[row, col].set_ylim(bottom=0)
            axes[row, col].grid(alpha=.2)
            axes[row, col].legend()
    cfg = report["settings"]
    fig.suptitle(f"{report['cpu']} CPU / {cfg['n_dates']} x {cfg['n_stocks']} synthetic panel / 1 thread\n"
                 "Fresh subprocess per case; RSS includes interpreter and allocator caching", fontsize=11)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    fig.savefig(args.output.with_suffix(".svg"))
    plt.close(fig)


if __name__ == "__main__":
    main()
