# v0.2.5 - Portable Packaging and Reproducibility Artifacts

`v0.2.5` is the first PyPI release of the project, distributed as `mlquantx`
while the import package and CLI remain `mlquant`. It removes a subtle packaging
trap: the command-line entry point was installable, but its default demo config
previously lived only in a repository checkout. The deterministic config now
ships inside the wheel, so a fresh environment can run the advertised demo from
any working directory.

## Highlights

- Install directly with `python -m pip install mlquantx`.
- Run `mlquant demo` from any directory without cloning the repository.
- Keep explicit `mlquant demo --config PATH` behavior for custom research runs.
- Include the deterministic small config in the wheel and test it against the
  repository copy.
- Include the post-v0.2.4 Hugging Face dataset/model export and documentation.
- Modernize license metadata for current PyPI build tooling.

## Verification

- Built both wheel and source distribution from a clean build environment.
- Checked both distributions with `twine check --strict`.
- Verified the wheel contains `mlquant/configs/small.yaml`.
- Installed the wheel in an isolated environment outside the repository.
- Passed CI on Python 3.9, 3.10, and 3.11, including the full CLI smoke test.

## Research Boundary

The bundled demo uses deterministic synthetic data. It verifies installation
and the factor-to-backtest engineering path; it is not evidence of live or
out-of-sample trading performance and is not investment advice.
