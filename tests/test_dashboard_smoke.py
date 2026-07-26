"""Smoke test for the Streamlit storytelling dashboard (`dashboard/app.py`).

Design reference:
- Testing Strategy -> "Smoke and execution tests": *Dashboard app imports and
  renders its sections; upload of a valid fixture returns results (R9.1-9.4)*.
- Components and Interfaces -> Dashboard (`dashboard/app.py`).
- Requirements 9.1 (sidebar navigation), 9.2 (the required story sections),
  9.3 (model comparison + forecast visualizations), 9.4 (upload-and-analyze
  returns results for conforming data).

This is a smoke test, not a property test: it confirms the app wires together and
its pieces run end-to-end on a small fixture, rather than asserting a universal
property.

Streamlit is a heavy, display-oriented dependency that may not be installed in a
minimal CI/sandbox environment. Rather than skip the whole test when it is
missing, we install a **lightweight in-process ``streamlit`` stub** into
``sys.modules`` *before* importing ``dashboard.app``. The stub records the UI
calls the app makes (``st.header``, ``st.pyplot``, ``st.success`` ...) so the test
can assert that each section actually rendered output and that the upload flow
reached its success path. When a real Streamlit *is* installed, it is used as-is
and no stub is inserted.

The stub touches nothing in ``src/`` or ``dashboard/upload_validation`` - those
run for real - so the smoke test exercises the genuine analysis code paths
(``src.eda``, ``src.evaluation``, ``src.business``, ``validate_upload``) behind a
fake presentation layer.
"""

from __future__ import annotations

import io
import sys
from datetime import date, timedelta

# Force a headless matplotlib backend before any src.eda import pulls in pyplot,
# so the chart-producing sections run without a display.
import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

# --------------------------------------------------------------------------- #
# Lightweight Streamlit stub (only installed when Streamlit is not available)
# --------------------------------------------------------------------------- #


class _CtxStub:
    """Stand-in for objects used as context managers (e.g. ``st.expander``)."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _ColumnStub:
    """A single column returned by ``st.columns`` - every method is a no-op."""

    def __getattr__(self, name):
        def _noop(*args, **kwargs):
            return None

        return _noop


class _SidebarStub:
    """Stand-in for ``st.sidebar`` with the handful of methods the app uses."""

    def __init__(self, parent: "_FakeStreamlit") -> None:
        self._parent = parent

    def title(self, *a, **k):
        self._parent.record("sidebar.title")

    def caption(self, *a, **k):
        self._parent.record("sidebar.caption")

    def markdown(self, *a, **k):
        self._parent.record("sidebar.markdown")

    def radio(self, label, options, *a, **k):
        self._parent.record("sidebar.radio")
        return list(options)[0]


class _FakeStreamlit:
    """A minimal, call-recording stand-in for the Streamlit module.

    Any attribute that is not explicitly defined (``header``, ``markdown``,
    ``subheader``, ``pyplot``, ``dataframe``, ``info``, ``success`` ...) resolves
    to a recorder that appends the call name to :attr:`calls`, so tests can assert
    that a section produced output without needing a real Streamlit runtime.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.sidebar = _SidebarStub(self)
        self.file_uploader_return = None

    # -- helpers ---------------------------------------------------------------
    def record(self, name: str) -> None:
        self.calls.append(name)

    def reset(self) -> None:
        self.calls.clear()
        self.file_uploader_return = None

    # -- explicitly-shaped members the app relies on ---------------------------
    def set_page_config(self, *a, **k):
        self.record("set_page_config")

    def columns(self, spec, *a, **k):
        n = spec if isinstance(spec, int) else len(spec)
        self.record("columns")
        return [_ColumnStub() for _ in range(n)]

    def expander(self, *a, **k):
        self.record("expander")
        return _CtxStub()

    def file_uploader(self, *a, **k):
        self.record("file_uploader")
        return self.file_uploader_return

    # -- everything else becomes a recording no-op -----------------------------
    def __getattr__(self, name):
        # Only reached for attributes not set in __init__ / not defined above.
        # Dunder attributes (e.g. ``__file__``, ``__path__``) must behave as if
        # absent: module-introspection tools (such as Hypothesis's constant
        # scanner, which reads every ``sys.modules`` entry's ``__file__``) expect
        # these to be strings/None, not callables. Returning a recorder here would
        # break them, so raise AttributeError like a normal module would.
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)

        def _recorder(*args, **kwargs):
            self.calls.append(name)
            return None

        return _recorder


def _install_streamlit_stub() -> _FakeStreamlit:
    """Return the streamlit-like object the app will import.

    Uses the real Streamlit if it is importable; otherwise installs the fake into
    ``sys.modules['streamlit']`` so ``import streamlit`` inside the app resolves
    to it. Must run before ``dashboard.app`` is imported.
    """
    try:  # Prefer the real thing when it is available.
        import streamlit as real_streamlit  # noqa: F401

        return real_streamlit  # type: ignore[return-value]
    except Exception:  # pragma: no cover - depends on the environment
        fake = _FakeStreamlit()
        sys.modules["streamlit"] = fake
        return fake


# Install the stub (if needed) and import the app under test exactly once.
_ST = _install_streamlit_stub()

from dashboard.app import SECTIONS  # noqa: E402
import dashboard.app as app  # noqa: E402
from src.config import default_scope  # noqa: E402
from src.preparation import DEMAND_COLUMN, PERIOD_COLUMN, REGION_COLUMN  # noqa: E402
from dashboard.upload_validation import validate_upload  # noqa: E402


# The nine required story sections (R9.2) plus the upload-and-analyze mode (R9.4),
# mapped from the acceptance-criteria concept to the app's actual SECTIONS label.
REQUIRED_SECTIONS = {
    "business problem": "Business problem",
    "data source & limitations": "Data source & limitations",
    "EDA findings": "EDA findings",
    "data preparation": "Data preparation",
    "tools & technology": "Tools & technology",
    "models & method": "Models & method",
    "results": "Results",
    "business insights": "Business insights",
    "generalisation to India": "Generalisation to India",
    "upload & analyze": "Upload & analyze",
}


@pytest.fixture(autouse=True)
def _reset_streamlit_stub():
    """Clear recorded calls before each test when using the fake Streamlit."""
    if isinstance(_ST, _FakeStreamlit):
        _ST.reset()
    yield


def _valid_fixture() -> pd.DataFrame:
    """Build a small, fully-conforming long-format DemandSeries fixture.

    Two regions over 21 contiguous daily periods, so the EDA charts (seasonal
    decomposition needs >= 2 seasonal cycles at period=7) have enough history.
    """
    start = date(2026, 1, 1)
    periods = [pd.Timestamp(start + timedelta(days=i)) for i in range(21)]
    rows = []
    for region, base in (("Manhattan", 500), ("Brooklyn", 300)):
        for i, period in enumerate(periods):
            rows.append(
                {
                    PERIOD_COLUMN: period,
                    REGION_COLUMN: region,
                    DEMAND_COLUMN: base + i,
                }
            )
    return pd.DataFrame(rows, columns=[PERIOD_COLUMN, REGION_COLUMN, DEMAND_COLUMN])


# --------------------------------------------------------------------------- #
# R9.1 / R9.2 - the app imports and exposes every required story section
# --------------------------------------------------------------------------- #


def test_app_imports_and_exposes_navigation():
    """The dashboard module imports and wires sidebar navigation (R9.1)."""
    assert hasattr(app, "SECTIONS"), "app must expose a SECTIONS navigation map"
    assert hasattr(app, "main") and callable(app.main), "app must expose a main() entry point"
    assert isinstance(SECTIONS, dict) and len(SECTIONS) >= len(REQUIRED_SECTIONS)


def test_all_required_sections_present():
    """SECTIONS covers every required story section + upload mode (R9.2, R9.4)."""
    for concept, label in REQUIRED_SECTIONS.items():
        assert label in SECTIONS, f"missing required section for '{concept}': '{label}'"
        assert callable(SECTIONS[label]), f"section '{label}' must map to a render function"


# --------------------------------------------------------------------------- #
# R9.2 / R9.3 - each section renders without error and produces output
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("label", list(REQUIRED_SECTIONS.values()))
def test_section_renders(label):
    """Each story section runs end-to-end and emits UI output (R9.2, R9.3).

    Uses the real ``src`` analysis functions (EDA charts, evaluation comparison
    table + forecast plot, business impact) behind the presentation stub, so this
    also exercises the model-comparison and forecast visualizations (R9.3).
    """
    scope = default_scope()
    render = SECTIONS[label]

    # Must not raise.
    render(scope)

    # When using the fake Streamlit, assert the section actually rendered output
    # (every section starts with a header). With a real Streamlit this is a no-op.
    if isinstance(_ST, _FakeStreamlit):
        assert "header" in _ST.calls, f"section '{label}' produced no header output"


# --------------------------------------------------------------------------- #
# R9.4 - a valid uploaded fixture passes validation and the flow returns results
# --------------------------------------------------------------------------- #


def test_valid_fixture_passes_upload_validation():
    """A conforming long-format fixture passes upload validation (R9.4)."""
    result = validate_upload(_valid_fixture())
    assert result.ok, f"valid fixture should pass validation, got: {result.error}"


def test_upload_and_analyze_returns_results_for_valid_fixture():
    """Uploading a valid fixture drives the analyze flow to its success path (R9.4).

    Feeds a CSV of the valid fixture through ``st.file_uploader`` (via the stub)
    and runs ``render_upload_analyze``. A conforming upload must reach
    ``st.success`` and render the analysis (dataframe + demand chart), i.e. return
    results rather than an error.
    """
    if not isinstance(_ST, _FakeStreamlit):
        pytest.skip("upload-flow assertions require the recording Streamlit stub")

    fixture = _valid_fixture()

    # Present the fixture as an uploaded CSV file (a buffer with a .name attr, as
    # Streamlit's UploadedFile provides).
    buffer = io.BytesIO(fixture.to_csv(index=False).encode("utf-8"))
    buffer.name = "fixture.csv"
    _ST.file_uploader_return = buffer

    app.render_upload_analyze(default_scope())

    assert "success" in _ST.calls, "conforming upload should reach the success path"
    assert "error" not in _ST.calls, "conforming upload must not render an error"
    # Results were shown: the analysis renders the series table and a chart.
    assert "dataframe" in _ST.calls, "conforming upload should display the data table"
    assert "pyplot" in _ST.calls, "conforming upload should render the demand chart"
