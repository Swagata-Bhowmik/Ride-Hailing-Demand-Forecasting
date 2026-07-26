"""Notebook execution smoke test (Requirement 8.6).

Requirement 8.6 states: *THE Notebook SHALL run end to end without errors from a
clean state.* This test enforces that guarantee in CI by executing
``notebook/demand_forecasting.ipynb`` top-to-bottom with a fresh kernel and
asserting that no cell raises.

Keeping the run light
---------------------
The notebook is written with two switches so it stays cheap to execute in CI:

* ``REAL_DATA_AVAILABLE`` — automatically ``False`` when the ~1 GB
  ``data/fhvhv_2026-04.parquet`` file is absent (which it is in CI / a clean
  checkout, since ``data/`` is git-ignored). On the ``False`` path the notebook
  runs on a tiny, clearly-labelled synthetic demo instead of the real feed.
* ``RUN_HEAVY`` — hard-coded ``False`` in the notebook, gating Prophet / VAR /
  LSTM-GRU and any full real-data training.

So with no real data present and ``RUN_HEAVY`` off, the whole notebook executes
on the fast demo path. This test deliberately does **not** download data or flip
those switches; it verifies the exact "clean state" a fresh clone would run in.

If ``nbconvert`` / ``nbformat`` (or a usable Jupyter kernel) are not installed in
the sandbox, the test skips gracefully rather than failing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Skip gracefully if the notebook execution stack is not installed.
nbformat = pytest.importorskip("nbformat", reason="nbformat is required to run the notebook smoke test")
nbconvert_preprocessors = pytest.importorskip(
    "nbconvert.preprocessors", reason="nbconvert is required to run the notebook smoke test"
)

ExecutePreprocessor = nbconvert_preprocessors.ExecutePreprocessor

# Repository root is the parent of this tests/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_PATH = REPO_ROOT / "notebook" / "demand_forecasting.ipynb"

# The notebook runs on the light demo path when the real (git-ignored) data is
# absent. A generous timeout keeps CI robust on slow shared runners while the
# demo path itself is fast.
CELL_TIMEOUT_SECONDS = 600


def _real_data_present() -> bool:
    """Whether the heavy real-data file is present (it should not be in CI)."""
    return (REPO_ROOT / "data" / "fhvhv_2026-04.parquet").exists()


def test_notebook_file_exists() -> None:
    """The notebook deliverable must exist to be smoke-tested."""
    assert NOTEBOOK_PATH.is_file(), f"Notebook not found at {NOTEBOOK_PATH}"


def test_notebook_runs_top_to_bottom_without_errors() -> None:
    """Execute the notebook end-to-end on the light/demo path (R8.6).

    Runs every cell with a fresh kernel from a clean state and asserts none
    raises. Execution happens with the repository root as the working directory
    so the notebook's ``REPO_ROOT = Path.cwd()`` bootstrap can import ``src/``.
    """
    if _real_data_present():
        pytest.skip(
            "Real FHVHV parquet is present; this smoke test targets the light "
            "demo path. Run the full notebook manually for the heavy path."
        )

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)

    preprocessor = ExecutePreprocessor(
        timeout=CELL_TIMEOUT_SECONDS,
        kernel_name="python3",
        allow_errors=False,  # any cell raising fails the test
    )

    try:
        # resources.metadata.path sets the kernel's working directory so that
        # `Path.cwd()` inside the notebook resolves to the repo root.
        preprocessor.preprocess(
            notebook,
            resources={"metadata": {"path": str(REPO_ROOT)}},
        )
    except Exception as exc:  # noqa: BLE001 - surface any execution failure
        # Missing/mismatched Jupyter kernel is an environment issue, not a
        # notebook defect: skip rather than fail so CI without a kernel is green.
        message = str(exc).lower()
        if "no such kernel" in message or "kernelspec" in message or "kernel" in message and "not" in message:
            pytest.skip(f"No usable Jupyter kernel available in this environment: {exc}")
        raise AssertionError(f"Notebook failed to execute end-to-end: {exc}") from exc

    # Sanity check: the notebook actually contains executable cells.
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells, "Notebook has no code cells to execute"
