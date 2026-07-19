# Security Policy

## Supported Versions

Security fixes are made against the latest release and the `main` branch. Older
releases may not receive backports while the project remains in the `0.x`
research phase.

| Version | Supported |
| --- | --- |
| Latest release | Yes |
| `main` | Yes |
| Older releases | No |

## Reporting a Vulnerability

Please do not open a public issue for a suspected vulnerability. Use GitHub's
[private vulnerability reporting](https://github.com/initial-d/ml-quant-trading/security/advisories/new)
instead.

Include the affected version or commit, reproduction steps, the expected impact,
and any suggested mitigation. Remove API keys, broker credentials, licensed
market data, and other sensitive material from the report.

The maintainer will acknowledge a report as soon as practical, validate the
finding, and coordinate a fix and disclosure timeline when the issue is
confirmed. This volunteer research project cannot promise a fixed response time,
but good-faith reports will receive attribution unless the reporter prefers to
remain anonymous.

## Scope

Reports about the maintained `src/mlquant` package, command-line interface,
packaging, and repository automation are in scope. The code under `legacy/` is
provided for research reference and is not actively supported; security-relevant
legacy findings are still welcome when they affect a documented workflow.

This policy is for software security issues. Questions about model quality,
factor validity, backtest assumptions, or financial risk should use a normal
issue or discussion and must not be treated as investment advice.
