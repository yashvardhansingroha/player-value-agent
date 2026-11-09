"""Forecast scorer: interval width reported as an explicit MEAN and an explicit MEDIAN (the registered metric is the mean). Hand-built inputs where they differ."""
from pathlib import Path

import pandas as pd
import pytest

from src.forecast import scoring


def frame():
    # widths (hi - lo) / forecast: 0.2, 0.1, 3.0 -> mean 1.1, median 0.2. The fourth row has NO interval and must not enter any interval metric.
    return pd.DataFrame({
        "actual": [100.0, 98.0, 400.0, 50.0], "point_forecast": [100.0, 100.0, 100.0, 50.0],
        "interval_status": ["available", "available", "available", "unavailable"],
        "interval_lo": [90.0, 95.0, 0.0, None], "interval_hi": [110.0, 105.0, 300.0, None]})


def test_mean_and_median_widths_are_reported_separately_and_differ_when_the_data_are_skewed():
    m = scoring.metrics(frame())
    assert m["interval_width_over_forecast_mean"] == pytest.approx(1.1) and m["interval_width_over_forecast_median"] == pytest.approx(0.2)
    assert m["interval_width_over_forecast_mean"] != pytest.approx(m["interval_width_over_forecast_median"])
    assert m["n_with_interval"] == 3 and m["coverage"] == pytest.approx(2 / 3)  # the interval-less row is outside both coverage and width


def test_the_legacy_unsuffixed_key_is_documented_as_the_median_and_kept_for_the_phase_2_report():
    m = scoring.metrics(frame())
    assert m["interval_width_over_forecast"] == m["interval_width_over_forecast_median"] != pytest.approx(m["interval_width_over_forecast_mean"])
    assert "LEGACY" in scoring.__doc__ and "MEDIAN" in scoring.__doc__
    phase2 = (Path(__file__).resolve().parents[1] / "scripts" / "phase2_report.py").read_text(encoding="utf-8")
    assert "median width / forecast" in phase2  # the one consumer of the legacy key labels it as the median


def test_a_single_interval_has_equal_mean_and_median_and_no_interval_gives_no_width():
    one = frame().iloc[:1]
    m = scoring.metrics(one)
    assert m["interval_width_over_forecast_mean"] == pytest.approx(0.2) == pytest.approx(m["interval_width_over_forecast_median"])
    none = scoring.metrics(frame().iloc[3:])
    assert none["n_with_interval"] == 0 and "interval_width_over_forecast_mean" not in none and "coverage" not in none
