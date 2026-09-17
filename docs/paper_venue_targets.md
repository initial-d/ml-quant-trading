# Paper Venue and Cross-List Targets

This page records a restrained submission plan for turning the arXiv paper and
repository into citeable academic surface area.

## Current Position

- The paper already exists on arXiv as `2507.07107`.
- The current useful citation angle is narrower than the full repository:
  **mask-first factor computation to avoid upstream contamination from
  non-tradable limit-move prices**.
- The repository supports that angle with inspectable code, tests, public-data
  validation docs, benchmark reports, and a reproducibility contract.

## arXiv Cross-List Plan

Do not force another arXiv cross-list only for visibility. A cross-list should
be used only if the paper is directly useful to that archive's readers and the
paper body supports the category.

Current recommendation:

| Category | Recommendation | Reason |
|---|---|---|
| `q-fin.PM` | Keep as primary | The work is fundamentally portfolio and factor research. |
| `cs.CE` | Keep as existing cross-list | The repository emphasizes computational engineering and reproducibility. |
| `cs.LG` | Defer | Add only after a methods-focused revision with clearer ML benchmarks, ablations, and comparison against ML baselines. |
| `stat.ML` | Defer | Add only if the paper becomes more statistical-methods oriented. |

If a future v3 is prepared for `cs.LG` or `stat.ML`, add:

- model comparison table with fixed public-data splits;
- ablations for mask-first versus post-filtered pipelines;
- explicit data-leakage and non-tradability definitions;
- numerical tolerance and reproducibility protocol;
- a smaller canonical benchmark that readers can run in under ten minutes.

## Formal Venue Targets

Treat these as ordered targets, not places to spam.

| Target | Fit | Recommended submission form |
|---|---|---|
| ACM ICAIF | Best fit for AI in finance, factor models, trading, validation, and reproducible financial ML. | Full paper or workshop paper after tightening the empirical story. |
| ICAIF workshops | Strong fit if the main-track window is closed or the paper is better framed as reproducibility / model-risk tooling. | Short paper or poster on mask-first factor pipelines. |
| NeurIPS workshops | Good fit only for workshops on financial ML, time series, data-centric ML, evaluations, or agents. | Workshop paper focused on contamination, ablation, and benchmark design. |
| AAAI / IAAI workshops | Possible fit for applied AI, AI in finance, or AI systems with strong reproducibility evidence. | Workshop paper or applied track note. |
| Journal of Financial Data Science / Quantitative Finance style outlets | Longer-term fit after stronger independent validation and clearer point-in-time data handling. | Revised article with conservative claims. |

## One-Page Workshop Pitch

**Title:** Mask-First Factor Pipelines for Limit-Move Equity Markets

**Thesis:** In markets with limit-up/limit-down rules, non-tradable closing
prices can contaminate rolling factor features before a later portfolio filter
removes non-executable rows. A mask-first factor pipeline applies tradability
masks before moving averages, correlations, ranks, labels, and backtests,
turning an easy-to-miss implementation detail into an auditable research
protocol.

**Evidence to emphasize:**

- minimal synthetic example showing post-filtering is too late;
- A-share public-data validation path with explicit caveats;
- factor primitive tests for mask-aware rolling semantics;
- reproducibility contract for command, environment, data fingerprints,
  artifacts, checksums, and tolerances;
- code release with issue templates for external reports.

**What to avoid:** Do not lead with Sharpe. Lead with measurement validity,
data hygiene, and reproducibility.

## Revision Checklist Before Submission

- Add a small table comparing post-filter and mask-first factor calculations.
- Move the limit-move contamination example into the introduction.
- Put live-trading and investment-advice disclaimers near performance tables.
- Add a short reproducibility appendix linking to the repository's contract.
- Cite the exact code release or commit used for the paper experiments.
- Include public-data limitations and provider revision risk.
