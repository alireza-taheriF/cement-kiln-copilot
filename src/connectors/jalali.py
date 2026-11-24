"""Jalali (Solar Hijri) dates and Persian digits in plant exports.

Iranian lab sheets and shift logs are usually kept in Jalali dates with
Persian or Arabic-Indic digits (``۱۴۰۵/۰۹/۰۳ ۰۸:۳۰``). This module converts
them to Gregorian timestamps without an external dependency.

The conversion is the standard arithmetic algorithm (as used by
``jdatetime`` and Iranian banking software), exact for 1178-1633 SH.
"""

from __future__ import annotations

import re
from datetime import date, datetime

_PERSIAN = "۰۱۲۳۴۵۶۷۸۹"
_ARABIC = "٠١٢٣٤٥٦٧٨٩"
_DIGITS = {ord(c): str(i) for i, c in enumerate(_PERSIAN)}
_DIGITS.update({ord(c): str(i) for i, c in enumerate(_ARABIC)})

_JALALI_RE = re.compile(
    r"^\s*(?P<y>1[2-4]\d\d)[/\-.](?P<m>\d{1,2})[/\-.](?P<d>\d{1,2})"
    r"(?:[ T]+(?P<H>\d{1,2}):(?P<M>\d{2})(?::(?P<S>\d{2}))?)?\s*$"
)


def normalize_digits(text: str) -> str:
    """Replace Persian and Arabic-Indic digits with ASCII digits."""
    return str(text).translate(_DIGITS)


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> date:
    """Convert a Jalali date to a Gregorian :class:`datetime.date`."""
    if not (1 <= jm <= 12 and 1 <= jd <= 31):
        raise ValueError(f"invalid Jalali date {jy}/{jm}/{jd}.")
    jy += 1595
    days = -355668 + 365 * jy + (jy // 33) * 8 + ((jy % 33) + 3) // 4 + jd
    days += (jm - 1) * 31 if jm < 7 else (jm - 7) * 30 + 186
    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1
    leap = (gy % 4 == 0 and gy % 100 != 0) or gy % 400 == 0
    month_days = [0, 31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    for gm in range(1, 13):
        if gd <= month_days[gm]:
            break
        gd -= month_days[gm]
    return date(gy, gm, gd)


def parse_jalali_datetime(text: str) -> datetime:
    """Parse ``1405/09/03 08:30`` (any digit script) to a naive Gregorian datetime."""
    match = _JALALI_RE.match(normalize_digits(text))
    if not match:
        raise ValueError(f"not a Jalali date-time: {text!r}.")
    g = jalali_to_gregorian(int(match["y"]), int(match["m"]), int(match["d"]))
    return datetime(
        g.year,
        g.month,
        g.day,
        int(match["H"] or 0),
        int(match["M"] or 0),
        int(match["S"] or 0),
    )


def looks_jalali(text: str) -> bool:
    """Whether ``text`` is a date in the Jalali year range (1200-1499)."""
    return bool(_JALALI_RE.match(normalize_digits(text)))
