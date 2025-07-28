#!/bin/sh

# From the root directory, execute `./scripts/run_linters_and_checks.sh` to
# run linters. Execute `./scripts/run_linters_and_checks.sh --checks`
# to run linters and checks.
main() {
  # As a default, run linters only. Add option to run checks
  case $1 in
      -c|--checks) checks=true;
  esac
  # Run linters
  uv run --frozen ruff format

  # Optionally run style checks, docstring coverage, and test coverage.
  # The results of the test coverage will additionally be saved to htmlcov.
  if [ $checks ]
  then
    uv run --frozen ruff check
    uv run --frozen mypy
    uv run --frozen interrogate -v
    uv run --frozen codespell --check-filenames
    uv run --frozen pytest --cov aind_zarr_utils
  fi
}

main "$@"
