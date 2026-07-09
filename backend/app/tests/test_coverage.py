# medical-triage-agent-ai-poc/backend/app/tests/test_coverage.py

"""
Coverage validation test.

This test verifies that the project reaches the
minimum expected coverage threshold configured
for CI/CD pipelines.
"""

from __future__ import annotations

import json
from pathlib import Path


MINIMUM_COVERAGE = 80.0


def test_coverage_threshold():
    """
    Validate pytest-cov generated coverage report.

    Expected command:

    pytest \
        --cov=app \
        --cov-report=json:coverage.json

    """

    coverage_file = Path("coverage.json")

    assert coverage_file.exists(), (
        "coverage.json not found. " "Run pytest with --cov-report=json:coverage.json"
    )

    report = json.loads(coverage_file.read_text(encoding="utf-8"))

    total_coverage = report["totals"]["percent_covered"]

    assert total_coverage >= MINIMUM_COVERAGE, (
        f"Coverage too low: "
        f"{total_coverage:.2f}% "
        f"(required: {MINIMUM_COVERAGE:.2f}%)"
    )


def test_coverage_report_structure():
    """
    Verify coverage report structure.
    """

    coverage_file = Path("coverage.json")

    assert coverage_file.exists()

    report = json.loads(coverage_file.read_text(encoding="utf-8"))

    assert "totals" in report
    assert "files" in report

    assert "percent_covered" in report["totals"]


def test_critical_modules_present():
    """
    Verify critical modules are included
    in coverage reporting.
    """

    coverage_file = Path("coverage.json")

    report = json.loads(coverage_file.read_text(encoding="utf-8"))

    files = report.get("files", {})

    expected_modules = [
        "app/api",
        "app/llm",
        "app/anonymization",
        "app/monitoring",
    ]

    indexed_paths = " ".join(files.keys()).replace("\\", "/")

    for module in expected_modules:
        assert module in indexed_paths, f"{module} missing from coverage report"
