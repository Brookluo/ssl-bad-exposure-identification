# Refactoring Phase 2 — Package, Types, Tests, Error Handling, Config

**Date**: 2026-05-03
**Status**: approved
**Scope**: `src/`, `tests/`, root config files

## Goal

Make the research codebase installable, type-safe, tested, and config-driven without disrupting existing workflows.

## Non-goals

- Rewriting notebooks
- Fixing `finetune_model.py` (incomplete, out of scope)
- Modifying `distributed.py` beyond error handling (vendored from Meta)
- Adding CI/CD (no NERSC environment available)

## Phase 1 — Packaging + Type Hints (zero risk)

### pyproject.toml

Single file at repo root. Package name: `ssl-bad-exposure-identification`. 

- `requires-python >= 3.10` (for `Path | str` union syntax)
- Core dependencies from `requirements.txt`
- Optional: `[viz]` for matplotlib/scipy, `[dev]` for pytest/pytest-cov
- `[project.scripts]` entry point: `master-eval = "src.master_eval:main"`
- `pip install -e .` makes `from src.decam_info import ...` work everywhere

### Type hints

Add to public function signatures only. Files: `read_image.py`, `decam_info.py`, `decam_dataset.py`, `decam_focalplane.py`, `master_eval.py`, `parallel_eval.py`, `inference.py`, `util.py`.

Skip: `distributed.py` (vendored), `finetune_model.py` (incomplete), `decam_postage_stamps.py` (Phase 4), function internals.

Use `from __future__ import annotations` for forward refs. All union types use `|` syntax (Python 3.10+).

## Phase 2 — Smoke Tests (low risk)

Test pure-logic, no-GPU, no-data modules. Create `tests/` at repo root.

| Test file | Covers | Deps |
|-----------|--------|------|
| `test_decam_info.py` (~12) | CCD mappings, `decode_reason`, `decode_vi_source`, `decode_ml_label`, `is_miss_2ccd` | None |
| `test_read_image.py` (~4) | Round-trip: synthetic NPZ write → `read_img_npz` | numpy |
| `test_inference.py` (~3) | Synthetic HDF5 → `read_embeds` | h5py |
| `test_util.py` (~4) | `get_info_from_html` on minimal HTML, `make_webpage` | None |
| `test_parallel_eval.py` (~3) | `split_dset` CSV splitting | pandas |
| `conftest.py` | Shared fixtures: temp dir, sample CSV, sample NPZ, sample HDF5 | — |

Skipped: anything needing torch, GPU, real FITS files, matplotlib, SLURM, or multi-process.

Invocation: `pip install -e ".[dev]" && pytest tests/ -v`

## Phase 3 — Error Handling (low risk)

Guard at system boundaries only. No internal invariant wrapping.

| File | Change |
|------|--------|
| `master_eval.py` | Check subprocess exit codes; raise on non-zero |
| `read_image.py` | Raise `FileNotFoundError` with path on missing file |
| `parallel_eval.py` | Warn + CPU fallback when `torch.cuda.device_count() == 0` |
| `decam_dataset.py` | Add `from astropy.io import fits` (pre-existing missing import) |
| `distributed.py` | Change bare `except: pass` to `except Exception: pass` in `try_barrier` |
| `util.py` | Return `[]` on malformed HTML instead of crashing |

## Phase 4 — Config Module (medium risk)

### New file: `src/config.py`

`PipelineConfig` and `VisualizationConfig` dataclasses loaded from YAML + env vars:

- Priority: CLI args > env vars > YAML config > defaults
- `--config` flag on `master_eval.py` and `parallel_eval.py` (optional; CLI-only still works)
- `decam_postage_stamps.py` functions accept `config` param instead of reading `os.environ`

### Config YAML

Expand `src/config.yaml` to cover all tunable parameters from all phases.

## Backward compatibility

- `bin/master_eval` and `bin/parallel_eval` unchanged
- All existing CLI args preserved
- `--config` is additive and optional
- Notebooks unaffected (use `pip install -e .` for imports)

## Verification

- `pytest tests/ -v` passes
- `python -m py_compile` on all `src/*.py` passes
- `pip install -e .` succeeds
- Type checker (mypy/pyright) shows no new errors on changed files
