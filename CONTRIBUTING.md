# Contributing

Thanks for helping. Bug reports, small fixes and new examples are all welcome.

## Set up

```bash
git clone https://github.com/prashant7799/segmentscan
cd segmentscan
uv venv .venv --python 3.11
uv pip install -p .venv -e ".[dev]"
```

(`python -m venv .venv` and `pip install -e ".[dev]"` work too.)

## Before you open a pull request

```bash
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Both must pass. The full suite takes one to two minutes because of the FDR simulations.

## Ground rules

- Statistical correctness is the point of this package. A change to the tests, the split or the
  p-value adjustment needs a test that shows it is still right, ideally a seeded simulation.
- Tests and examples use synthetic data or public datasets with a clear licence. Do not commit
  real or private data. The `data/` folder is ignored on purpose.
- Tests must not use the network.
- Prefer plain, readable code over clever code. Short functions, clear names, a comment where the
  reason is not obvious.
- Add a line to `CHANGELOG.md` for anything a user would notice.
