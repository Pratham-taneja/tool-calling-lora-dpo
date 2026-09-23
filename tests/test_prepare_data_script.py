from __future__ import annotations

import pytest

from scripts.prepare_data import percentile, percentile_summary


def test_percentile_summary_reports_required_percentiles() -> None:
    summary = percentile_summary([1, 2, 3, 4, 5])

    assert summary == {
        "p50": 3,
        "p90": 5,
        "p95": 5,
        "p99": 5,
        "max": 5,
    }


def test_percentile_validates_percent() -> None:
    with pytest.raises(ValueError, match="between 0 and 100"):
        percentile([1, 2, 3], 101)


def test_percentile_summary_rejects_empty_values() -> None:
    with pytest.raises(ValueError, match="empty dataset"):
        percentile_summary([])
