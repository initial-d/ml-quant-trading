# Community Guide

The project grows best when people can contribute small, credible improvements.

## Contribution Lanes

### Benchmark Reports

Run the benchmark script and submit hardware/runtime details. This is the easiest way to help
without changing code.

### Paired Contributions

Open a pairing request when a contribution would benefit from two people, such
as one person running a public-data validation and another turning the result
into a reviewed report. Keep the task small, include reproduction commands, and
only use co-author trailers for people who materially contributed to the merged
PR.

### Public-Data Examples

Add a notebook or docs page that uses a public data source. Keep the universe small, state
caveats clearly, and avoid performance hype.

For validation reports, run `scripts/audit_validation_report.py` before posting.
This keeps community results comparable and catches missing metadata, weak data
coverage, or malformed metric rows before they become maintainer work.

### Factor Documentation

Improve explanations for factor families, assumptions, masks, and edge cases.

### Tests and Reproducibility

Add tests for tensor operations, neutralization, bias correction, or backtesting edge cases.

### Research Notes

Document ablations, failed experiments, or limitations. Honest negative results are useful.

## Maintainer Response Rules

- Reply to reproducibility reports first.
- Turn repeated questions into docs.
- Thank benchmark contributors by name in release notes.
- Keep issues small enough for first-time contributors.
- Prefer clear caveats over inflated claims.

## Contributor Operations

The project should stay easy to contribute to without requiring proprietary
tools or private maintainer context.

Current decision:

- Use GitHub issues plus `docs/roadmap.md` as the public contributor board.
- Keep labels focused on contribution type and review priority.
- Link each community request, benchmark report, or reproduction PR back to a
  small issue when practical.
- Defer a formal GitHub Project board until there are enough simultaneous
  contributor tasks to justify the extra maintenance.
- Defer automated PR review services until repeated review bottlenecks appear.

Review automation can help with style, missing tests, and documentation drift,
but it should not generate noisy comments or gate small documentation
contributions. CI, reproducible commands, and maintainer review remain the
baseline workflow.

## Labels to Use

- `good first issue`: small, self-contained contribution.
- `benchmark`: CPU/GPU benchmark report.
- `case study`: public-data or reproduction example.
- `documentation`: docs, tutorials, explanations.
- `research`: factor, modeling, portfolio, or evaluation ideas.
- `release`: launch and packaging work.

## Healthy Growth Signals

- People can run the quick start.
- Users submit benchmark results.
- External users ask reproducibility questions.
- Small docs PRs arrive.
- Public-data examples improve.
- The README stays honest and current.
