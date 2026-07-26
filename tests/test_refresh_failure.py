"""Forced-failure automation test for the scheduled auto-refresh (Requirement 10.4).

Design reference:
- Testing Strategy -> Integration tests: "A forced-failure automation run records
  the failure in logs (R10.4)."
- Error Handling -> "Automated run fails" row: the failure is preserved in the run
  logs and the process exits non-zero.
- Components and Interfaces -> Automation (`scripts/refresh_pipeline.py`).

The auto-refresh entry point ``scripts/refresh_pipeline.py`` guarantees that a
failed run is *observable*: it exits with code ``1`` and logs a ``STATUS=FAILURE``
line (plus a ``Refresh FAILED`` message) so the failure is preserved in the
GitHub Actions run logs (Requirement 10.4). A forced failure is triggered here by
pointing ``--data`` at a nonexistent Parquet path, which makes ``load_parquet``
raise before any modeling.

These are **example / integration** tests (not property tests): they drive the
real ``main(argv)`` entry point and capture its logging output with ``caplog``.
No real ~1 GB NYC TLC data is touched - the failure path never reads real data,
and the success path uses the script's own tiny synthetic fixture.

Requirements: 10.4.
"""

from __future__ import annotations

import logging

from scripts.refresh_pipeline import main


def test_forced_failure_records_failure_in_logs(caplog):
    """A run against a nonexistent data path exits 1 and logs the failure (R10.4)."""
    caplog.set_level(logging.INFO, logger="refresh_pipeline")

    exit_code = main(["--data", "data/does_not_exist.parquet"])

    # The process must exit non-zero so GitHub Actions marks the job as failed.
    assert exit_code == 1

    # The failure must be preserved in the run logs (Requirement 10.4): both the
    # machine-readable status line and a human-readable FAILED message appear.
    assert "STATUS=FAILURE" in caplog.text
    assert "FAILED" in caplog.text

    # STATUS=SUCCESS must NOT be logged on a failed run.
    assert "STATUS=SUCCESS" not in caplog.text


def test_synthetic_success_path_returns_zero(caplog):
    """The forced-synthetic path completes and logs SUCCESS with exit code 0 (R10.1)."""
    caplog.set_level(logging.INFO, logger="refresh_pipeline")

    exit_code = main(["--synthetic"])

    assert exit_code == 0
    assert "STATUS=SUCCESS" in caplog.text
    assert "STATUS=FAILURE" not in caplog.text
