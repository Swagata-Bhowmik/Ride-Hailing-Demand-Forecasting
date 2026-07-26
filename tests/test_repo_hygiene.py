"""Repository hygiene checks (Requirement 12).

These example-based tests assert that the repository's documentation and ignore
rules meet the project's hygiene requirements:

- R12.1: the README documents the data source, the results approach, and a
  forward-looking roadmap.
- R12.3 / R12.4: the `.gitignore` excludes the real data directory, virtual
  environments, secrets/credentials, and the personal project brief file so that
  none of them are ever committed.

Matching is deliberately flexible (case-insensitive keyword checks, and either
`data/` or `data/*` accepted) so the tests validate intent rather than exact
wording.
"""

from __future__ import annotations

from pathlib import Path

import pytest

# Repository root is the parent of the tests/ directory.
REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
GITIGNORE_PATH = REPO_ROOT / ".gitignore"
BRIEF_FILENAME = "PROJECT2_BRIEF_Ride_Hailing_Forecasting.md"


@pytest.fixture(scope="module")
def readme_text() -> str:
    assert README_PATH.exists(), f"README.md not found at {README_PATH}"
    return README_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gitignore_text() -> str:
    assert GITIGNORE_PATH.exists(), f".gitignore not found at {GITIGNORE_PATH}"
    return GITIGNORE_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def gitignore_entries(gitignore_text: str) -> list[str]:
    """Return non-comment, non-blank .gitignore lines, stripped."""
    entries = []
    for raw in gitignore_text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        entries.append(line)
    return entries


# --------------------------------------------------------------------------- #
# R12.1 - README documents data source, results, and roadmap
# --------------------------------------------------------------------------- #


class TestReadmeContent:
    """R12.1: README covers the data source, results, and roadmap."""

    def test_mentions_data_source_nyc_tlc(self, readme_text: str) -> None:
        lowered = readme_text.lower()
        assert "nyc tlc" in lowered, "README should name the NYC TLC data source"
        # A general "data source" framing should also be present.
        assert "data source" in lowered

    def test_mentions_results(self, readme_text: str) -> None:
        assert "results" in readme_text.lower(), "README should describe results"

    def test_mentions_roadmap(self, readme_text: str) -> None:
        assert "roadmap" in readme_text.lower(), "README should include a roadmap"


# --------------------------------------------------------------------------- #
# R12.3 / R12.4 - .gitignore excludes data, venvs, secrets, and the brief file
# --------------------------------------------------------------------------- #


class TestGitignoreExclusions:
    """R12.3/R12.4: sensitive/large artifacts are excluded from version control."""

    def test_excludes_data_directory(self, gitignore_entries: list[str]) -> None:
        # Accept either `data/` or `data/*` (both keep data contents out of git).
        assert any(
            entry in {"data/", "data/*"} or entry.rstrip("*").rstrip("/") == "data"
            for entry in gitignore_entries
        ), "gitignore should exclude the data/ directory"

    def test_excludes_virtual_environments(self, gitignore_entries: list[str]) -> None:
        venv_markers = {".venv/", "venv/", "env/", "ENV/", ".venv", "venv"}
        assert any(entry in venv_markers for entry in gitignore_entries), (
            "gitignore should exclude virtual environment directories"
        )

    def test_excludes_secrets(self, gitignore_entries: list[str]) -> None:
        lowered = [e.lower() for e in gitignore_entries]
        secret_markers = (".env", "*.pem", "*.key", "secrets.", ".streamlit/secrets.toml")
        assert any(
            any(marker in entry for marker in secret_markers) for entry in lowered
        ), "gitignore should exclude secrets/credentials"

    def test_excludes_project_brief(self, gitignore_entries: list[str]) -> None:
        assert BRIEF_FILENAME in gitignore_entries, (
            f"gitignore should exclude the project brief file {BRIEF_FILENAME}"
        )
