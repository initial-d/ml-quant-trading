# SSRN Submission Packet

This is a copy-ready packet for submitting the paper to SSRN without changing
the technical claims. The positioning is narrower than the repository tagline:
make the paper easy to discover for researchers interested in factor-pipeline
validity, A-share price limits, and reproducible financial ML.

## Recommended SSRN Discovery Title

Use this if SSRN allows the display metadata to emphasize the revised v2
contribution:

Mask-First Factor Pipelines for Limit-Move Markets: Machine Learning Enhanced
Multi-Factor Quantitative Trading with Bias Correction

## Strict Title Match

Machine Learning Enhanced Multi-Factor Quantitative Trading: A Cross-Sectional
Portfolio Optimization Approach with Bias Correction

Use this if SSRN requires the metadata title to match the uploaded PDF exactly.

## Short Running Title

Mask-First Factor Pipelines for ML Quant Trading

## Abstract

Rolling-window factor pipelines for equity markets can suffer from upstream
data contamination when observations later deemed non-tradable are allowed to
enter factor calculations, labels, and model inputs before portfolio filtering.
This issue is especially visible in Chinese A-share research, where limit-up and
limit-down rules can make closing prices non-executable while still leaving them
in raw panels.

This paper presents a reproducible PyTorch research stack for cross-sectional
multi-factor trading experiments with a mask-first dataflow. The implementation
combines 213 factor dimensions, mask-aware tensor primitives, simple
machine-learning baselines, portfolio optimization, and cost-aware vectorized
backtesting. The core methodological point is not a live trading claim; it is
that tradability masks should be applied before rolling factor aggregation so
that non-executable prices do not silently contaminate downstream signals.

The accompanying open-source repository provides synthetic no-account smoke
tests, public-data validation workflows, benchmark reports, and issue templates
for reporting successful, failed, and caveated runs. Public-data results are
documented with transaction costs, turnover, provider limitations, and
reproducibility metadata. The project is intended as a research-engineering
baseline for studying factor computation, validation hygiene, and auditable ML
quant workflows.

## Keywords

quantitative finance; factor investing; machine learning; PyTorch;
cross-sectional portfolio optimization; A-share market; limit-up limit-down;
tradability mask; backtesting; reproducible research

## Suggested JEL Codes

- C45: Neural Networks and Related Topics
- C58: Financial Econometrics
- G11: Portfolio Choice; Investment Decisions
- G12: Asset Pricing; Trading Volume; Bond Interest Rates
- G17: Financial Forecasting and Simulation

## Research Networks To Try

Prioritize SSRN networks and topic areas around:

- Financial Economics Network;
- Economics Research Network;
- econometrics;
- asset pricing;
- portfolio management;
- market microstructure;
- machine learning and AI in finance.

## Cover Note

This submission is an arXiv working paper with an accompanying MIT-licensed
research implementation. The main contribution is a mask-first factor-computing
protocol for avoiding upstream contamination from non-tradable limit-move prices
in cross-sectional equity panels. The empirical material should be read as
research validation and software evidence, not investment advice or a live
trading-performance claim.

## Workshop Abstract Variant

Many workshops want a shorter, sharper abstract. Use this version when the
submission page asks for 150-250 words:

In cross-sectional equity research, a common implementation pattern computes
rolling-window factors on raw prices and removes non-tradable rows only later,
during label construction or portfolio formation. In limit-move markets such as
Chinese A-shares, this post-filtering order can be too late: non-executable
closing prices have already entered moving averages, correlations, ranks, and
model inputs. This paper frames the issue as upstream contamination and presents
a mask-first factor pipeline that applies tradability masks before factor
aggregation. The accompanying PyTorch implementation provides 213 mask-aware
factor dimensions, public and synthetic data paths, cost-aware backtesting,
benchmark reports, and reproducibility templates. The goal is not to claim live
trading profitability, but to make a subtle data-hygiene problem visible and
auditable for ML-based factor research.

## Links To Include

- arXiv: <https://arxiv.org/abs/2507.07107>
- Code: <https://github.com/initial-d/ml-quant-trading>
- Project site: <https://initial-d.github.io/ml-quant-trading/>
- Citation guide:
  <https://initial-d.github.io/ml-quant-trading/citation_conversion.md>

## Pre-Submission Checklist

- Use the latest arXiv PDF, not an outdated local PDF.
- Keep the author name, title, and abstract consistent with arXiv unless SSRN
  requires separate display metadata.
- Add the repository URL in the paper metadata and abstract note.
- Avoid language implying live deployability, investment advice, or production
  readiness.
- After SSRN posts the paper, add the SSRN URL to `docs/citation_conversion.md`,
  `docs/answer_index.md`, `docs/llms.txt`, and the project README.
