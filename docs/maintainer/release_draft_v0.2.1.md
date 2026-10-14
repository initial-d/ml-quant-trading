# v0.2.1 - Validation Entrypoints and Outreach Follow-through

`v0.2.1` is a patch release for the public validation funnel. It fixes a Colab
entrypoint, records the current visibility surface, and points new contributors
toward reproducible benchmark and public-data reports.

This is not a new alpha claim and not a production-trading release. It is a
small maintenance release after the `v0.2.0` public validation workflow.

## Highlights

- Fixed the Baostock Colab demo bootstrap so fresh Colab runtimes clone
  `https://github.com/initial-d/ml-quant-trading.git`.
- Added the Colab Baostock demo to the tracked live entry points.
- Added a follow-up validation digest for the current reproduction surface.
- Recorded the verified Hugging Face paper entry and the current public
  discovery channels.
- Consolidated outreach follow-up into Discussions #13 so benchmark and
  reproduction notes have one landing place.

## Validation Position

The useful contribution in this release is operational clarity:

- public-data failures should be reported as blockers;
- synthetic runs remain plumbing checks, not signal evidence;
- Colab and Baostock are intended to make the A-share route easier to try;
- external discussions are used to collect validation patterns, not to claim
  that the project has deployable alpha.

See [`docs/validation_digest_20260727.md`](https://github.com/initial-d/ml-quant-trading/blob/v0.2.1/docs/validation_digest_20260727.md)
for the follow-up validation and visibility summary.

## Useful Entry Points

- Start here: [`docs/start_here.md`](https://github.com/initial-d/ml-quant-trading/blob/v0.2.1/docs/start_here.md)
- Public-data validation: [`docs/public_data_validation.md`](https://github.com/initial-d/ml-quant-trading/blob/v0.2.1/docs/public_data_validation.md)
- Validation digest: [`docs/validation_digest_20260727.md`](https://github.com/initial-d/ml-quant-trading/blob/v0.2.1/docs/validation_digest_20260727.md)
- Colab Baostock demo: <https://colab.research.google.com/github/initial-d/ml-quant-trading/blob/v0.2.1/demo_baostock.ipynb>
- Benchmark discussion: <https://github.com/initial-d/ml-quant-trading/discussions/13>
- Pairing issue: <https://github.com/initial-d/ml-quant-trading/issues/22>

## Compatibility Notes

- Package version is now `0.2.1`.
- Python and PyTorch requirements are unchanged from `v0.2.0`.
- Existing validation commands and report formats remain compatible.
