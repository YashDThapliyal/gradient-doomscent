# PROGRESS

## Final report

Not written yet. v0 is in progress.

## Status

- Current step: 2 (install ViZDoom and Sample Factory).

## Gates

### Step 1: repo scaffold: PASS (2026-10-14)

- Command: `uv run ruff check . && uv run ruff format --check . && uv run pytest -q`
- Output: `All checks passed!`, `5 files already formatted`, `4 passed in 0.62s`.
- Files: `pyproject.toml`, `uv.lock`, `doomscent/config.py`, `tests/test_config.py`, `.github/workflows/ci.yml`, `.pre-commit-config.yaml`.

## Decisions

- Package name is `doomscent` (not `doomrl`), per the project brief. W&B project name is `gradient-doomscent`.
- The package lives at the repo root (`doomscent/`) as SPEC.md's repo structure shows, using uv_build with `module-root = ""`.
- `spec.md` and `claude.md` were provided in lowercase. macOS is case-insensitive, so they resolve as `SPEC.md` and `CLAUDE.md`; left untouched.

## Deviations from spec

None yet.

## Open issues

None yet.
