"""Day index <-> calendar date helpers. Day 0 is `start_date`; dates are ISO text 'YYYY-MM-DD'
(ISO text compares correctly as text, which the SQL relies on)."""
from datetime import date, timedelta

import numpy as np


def day_to_date(start_date: str, day: int) -> str:
    return (date.fromisoformat(start_date) + timedelta(days=int(day))).isoformat()


def date_to_day(start_date: str, iso_date: str) -> int:
    return (date.fromisoformat(iso_date) - date.fromisoformat(start_date)).days


def dates_for_days(start_date: str, days: np.ndarray) -> np.ndarray:
    """Vectorised day indexes -> ISO date strings."""
    start = np.datetime64(start_date, "D")
    return (start + np.asarray(days).astype("timedelta64[D]")).astype(str)
