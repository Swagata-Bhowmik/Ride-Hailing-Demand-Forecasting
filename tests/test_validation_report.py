"""Unit tests for report aggregation and multi-file orchestration (Task 2.6).

Covers the two pieces of the Data_Validator that sit above the individual
profilers:

- ``build_validation_report`` — aggregates schema, nulls, date range, duplicate
  count, and domain violations into a single ``ValidationReport`` (Requirement
  1.8), and ``load_parquet`` failing loudly with a path-naming error when the
  file is missing (Requirement 1.8).
- ``validate_month_frames`` / ``validate_month_files`` — multi-file orchestration
  that applies the profiling criteria 2-6 to *each* supplied month, returning one
  ``ValidationReport`` per label/path in order (Requirement 1.7).

These are example-based unit tests on small synthetic DataFrames, per the design's
Testing Strategy (property tests cover profiling accuracy and violation flagging;
aggregation and orchestration are covered by examples here).
"""

from __future__ import annotations

from collections import OrderedDict

import pandas as pd
import pytest

from src.validation import (
    DomainViolation,
    NullStat,
    SchemaReport,
    ValidationReport,
    build_validation_report,
    load_parquet,
    validate_month_files,
    validate_month_frames,
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


def _april_frame() -> pd.DataFrame:
    """A small synthetic April-2026 FHVHV-like frame with known issues.

    Constructed so every field of the ValidationReport has a non-trivial, exactly
    predictable value:
      - 5 rows, 3 columns
      - one null in ``base_passenger_fare``
      - one duplicate row (row 4 duplicates row 0 exactly)
      - one pickup outside the stated month (2026-03) -> 1 month violation
      - one negative fare -> 1 negative_base_passenger_fare violation
    """
    return pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                [
                    "2026-04-01 08:00:00",  # 0
                    "2026-04-02 09:30:00",  # 1
                    "2026-03-31 23:00:00",  # 2 -> outside stated month (March)
                    "2026-04-03 10:15:00",  # 3
                    "2026-04-01 08:00:00",  # 4 -> duplicate of row 0
                ]
            ),
            "base_passenger_fare": [10.0, 20.0, 30.0, -5.0, 10.0],
            # row 3 negative; but row indices differ, keep null elsewhere:
            "trip_miles": [1.0, None, 3.0, 4.0, 1.0],
        }
    )


# --------------------------------------------------------------------------- #
# Requirement 1.8 - build_validation_report aggregates all findings
# --------------------------------------------------------------------------- #


class TestBuildValidationReportStructure:
    """R1.8: one ValidationReport bundles schema, nulls, range, dups, violations."""

    def test_returns_validation_report_with_all_fields_populated(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))

        assert isinstance(report, ValidationReport)
        # Every field is present and of the expected type.
        assert isinstance(report.schema, SchemaReport)
        assert isinstance(report.nulls, dict)
        assert isinstance(report.duplicate_count, int)
        assert isinstance(report.domain_violations, list)

    def test_schema_reports_row_count_and_columns(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))

        assert report.schema.row_count == 5
        assert report.schema.column_names == [
            "pickup_datetime",
            "base_passenger_fare",
            "trip_miles",
        ]
        # dtypes maps every column to a string dtype.
        assert set(report.schema.dtypes) == set(df.columns)
        assert all(isinstance(v, str) for v in report.schema.dtypes.values())

    def test_nulls_match_ground_truth(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))

        assert set(report.nulls) == set(df.columns)
        assert isinstance(report.nulls["trip_miles"], NullStat)
        # Exactly one null in trip_miles (1/5 = 20%), none elsewhere.
        assert report.nulls["trip_miles"].count == 1
        assert report.nulls["trip_miles"].percentage == pytest.approx(20.0)
        assert report.nulls["pickup_datetime"].count == 0
        assert report.nulls["base_passenger_fare"].count == 0

    def test_date_range_is_min_max_tuple(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))

        assert report.date_range is not None
        lo, hi = report.date_range
        assert lo == pd.Timestamp("2026-03-31 23:00:00")
        assert hi == pd.Timestamp("2026-04-03 10:15:00")

    def test_duplicate_count_matches_injected(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))
        # Exactly one duplicate row (row 4 duplicates row 0).
        assert report.duplicate_count == 1

    def test_domain_violations_capture_month_and_negative(self) -> None:
        df = _april_frame()
        report = build_validation_report(df, stated_month=(2026, 4))

        types = {v.violation_type for v in report.domain_violations}
        assert "pickup_outside_stated_month" in types
        assert "negative_base_passenger_fare" in types

        by_type = {v.violation_type: v for v in report.domain_violations}
        assert by_type["pickup_outside_stated_month"].count == 1
        assert by_type["negative_base_passenger_fare"].count == 1
        # Each violation carries a concrete example record.
        for violation in report.domain_violations:
            assert isinstance(violation, DomainViolation)
            assert violation.example is not None

    def test_date_range_none_when_no_timestamp_column(self) -> None:
        # A frame without the pickup column still yields a report (range=None).
        df = pd.DataFrame({"base_passenger_fare": [1.0, 2.0, 3.0]})
        report = build_validation_report(df)

        assert report.date_range is None
        assert report.schema.row_count == 3
        assert report.duplicate_count == 0

    def test_empty_frame_produces_report_without_error(self) -> None:
        df = pd.DataFrame({"pickup_datetime": pd.to_datetime([]), "fare": []})
        report = build_validation_report(df)

        assert report.schema.row_count == 0
        assert report.date_range is None
        assert report.duplicate_count == 0
        assert report.domain_violations == []


# --------------------------------------------------------------------------- #
# Requirement 1.8 - load_parquet fails loudly, naming the path
# --------------------------------------------------------------------------- #


class TestLoadParquetErrors:
    """R1.8: a missing file raises a clear error naming the path."""

    def test_missing_path_raises_file_not_found_naming_the_path(self) -> None:
        missing = "data/definitely_not_here_2099-01.parquet"
        with pytest.raises(FileNotFoundError) as exc_info:
            load_parquet(missing)
        # The error message names the offending path so validation halts clearly.
        assert missing in str(exc_info.value)

    def test_unreadable_file_raises_value_error_naming_the_path(self, tmp_path) -> None:
        # A non-parquet file that exists but cannot be read as Parquet.
        bad = tmp_path / "not_really_parquet.parquet"
        bad.write_text("this is not a parquet file")
        with pytest.raises((ValueError, FileNotFoundError)) as exc_info:
            load_parquet(str(bad))
        assert str(bad) in str(exc_info.value)


# --------------------------------------------------------------------------- #
# Requirement 1.7 - multi-file orchestration applies criteria 2-6 per file
# --------------------------------------------------------------------------- #


def _month_frames() -> "OrderedDict[str, pd.DataFrame]":
    """Two distinct month frames with different, individually-known findings."""
    april = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                ["2026-04-01", "2026-04-02", "2026-04-02"]  # last is a duplicate
            ),
            "base_passenger_fare": [10.0, 20.0, 20.0],
        }
    )
    may = pd.DataFrame(
        {
            "pickup_datetime": pd.to_datetime(
                ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-04"]
            ),
            "base_passenger_fare": [5.0, -1.0, 7.0, 8.0],  # one negative
        }
    )
    return OrderedDict([("2026-04", april), ("2026-05", may)])


class TestValidateMonthFrames:
    """R1.7: profiling criteria 2-6 applied to EACH supplied month frame."""

    def test_returns_one_report_per_label_in_order(self) -> None:
        frames = _month_frames()
        reports = validate_month_frames(
            frames,
            stated_months={"2026-04": (2026, 4), "2026-05": (2026, 5)},
        )

        assert isinstance(reports, OrderedDict)
        assert list(reports.keys()) == ["2026-04", "2026-05"]
        assert all(isinstance(r, ValidationReport) for r in reports.values())

    def test_each_report_reflects_its_own_frame(self) -> None:
        frames = _month_frames()
        reports = validate_month_frames(
            frames,
            stated_months={"2026-04": (2026, 4), "2026-05": (2026, 5)},
        )

        april_report = reports["2026-04"]
        may_report = reports["2026-05"]

        # Criterion 2 (schema): row counts differ per file.
        assert april_report.schema.row_count == 3
        assert may_report.schema.row_count == 4

        # Criterion 4 (date range): each file's own min/max.
        assert april_report.date_range == (
            pd.Timestamp("2026-04-01"),
            pd.Timestamp("2026-04-02"),
        )
        assert may_report.date_range == (
            pd.Timestamp("2026-05-01"),
            pd.Timestamp("2026-05-04"),
        )

        # Criterion 5 (duplicates): only April has a duplicate.
        assert april_report.duplicate_count == 1
        assert may_report.duplicate_count == 0

        # Criterion 6 (violations): only May has a negative fare.
        may_types = {v.violation_type for v in may_report.domain_violations}
        assert "negative_base_passenger_fare" in may_types
        april_negatives = [
            v for v in april_report.domain_violations
            if v.violation_type.startswith("negative_")
        ]
        assert april_negatives == []

    def test_empty_mapping_returns_empty_ordereddict(self) -> None:
        reports = validate_month_frames(OrderedDict())
        assert isinstance(reports, OrderedDict)
        assert len(reports) == 0


class TestValidateMonthFiles:
    """R1.7 + R1.8: each path is loaded via load_parquet and profiled once."""

    def test_one_report_per_parquet_file(self, tmp_path) -> None:
        frames = _month_frames()
        paths = []
        for label, df in frames.items():
            path = tmp_path / f"fhvhv_{label}.parquet"
            df.to_parquet(path, engine="pyarrow")
            paths.append(str(path))

        stated_months = {
            paths[0]: (2026, 4),
            paths[1]: (2026, 5),
        }
        reports = validate_month_files(paths, stated_months=stated_months)

        assert isinstance(reports, OrderedDict)
        assert list(reports.keys()) == paths
        assert all(isinstance(r, ValidationReport) for r in reports.values())
        # Findings survive the round-trip through parquet.
        assert reports[paths[0]].schema.row_count == 3
        assert reports[paths[1]].schema.row_count == 4

    def test_missing_file_in_batch_raises_naming_the_path(self, tmp_path) -> None:
        good = tmp_path / "fhvhv_2026-04.parquet"
        _month_frames()["2026-04"].to_parquet(good, engine="pyarrow")
        missing = str(tmp_path / "fhvhv_2026-05.parquet")  # never written

        with pytest.raises(FileNotFoundError) as exc_info:
            validate_month_files([str(good), missing])
        assert missing in str(exc_info.value)
