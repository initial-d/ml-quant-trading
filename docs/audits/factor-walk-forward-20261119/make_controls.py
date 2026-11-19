"""Generate planted IC timing controls, not market observations or alpha."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    seed = 20261119
    noise = np.random.default_rng(seed).normal(0, .035, 305)
    zero_mean = noise[:100] - noise[:100].mean()
    paths = {"stable": .15 + noise,
             "reversal": np.r_[.35 + noise[:100], -.35 + noise[100:]],
             "future_winner": np.r_[zero_mean, noise[100:105], .4 + noise[105:]]}
    report = {"protocol": "daily-factor-ic-v1", "category": "synthetic planted IC timing control",
              "label_horizon": 5, "seed": seed,
              "data_source": "Constructed IC sequences, not correlations computed from market data",
              "factors": [{"name": name, "daily_ic": values.tolist(), "daily_rank_ic": values.tolist()}
                          for name, values in paths.items()]}
    args.json_out.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
