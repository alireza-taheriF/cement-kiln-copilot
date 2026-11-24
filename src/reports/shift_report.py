"""Shift handover report (Persian first), with an optional local LLM rewrite.

The report is built in two layers:

1. :func:`shift_facts` computes every number from the data: production,
   specific heat, fuel used, mill energy, lab results, alarm summary, fuel
   transfers, equipment warnings, copilot recommendations. This is the
   source of truth.
2. :func:`render_fa` / :func:`render_en` turn the facts into a fixed
   template. The template is always correct and needs no model.

:func:`polish_with_llm` can ask a *local* LLM (any OpenAI-compatible server
on the plant network: Ollama, vLLM, llama.cpp) to rewrite the template in
more natural prose. The rewrite is accepted only if every number in it
appears in the facts (:func:`numbers_consistent`); otherwise the template
is returned. A language model may rephrase the shift, never invent it.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from src.connectors.jalali import normalize_digits
from src.domain import tags as T
from src.domain.kpis import FuelProperties, add_derived_kpis
from src.reports.alarms import AlarmRule, derive_alarms, summarize

logger = logging.getLogger(__name__)

#: Default 8-hour shifts (start hour, name).
SHIFTS = ((7, "صبح / morning"), (15, "عصر / evening"), (23, "شب / night"))

_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def to_persian_digits(text: str) -> str:
    return text.translate(_PERSIAN_DIGITS)


def shift_window(ts: pd.Timestamp, shifts: Sequence[tuple[int, str]] = SHIFTS) -> tuple[pd.Timestamp, pd.Timestamp, str]:
    """Start, end and name of the shift that contains ``ts``."""
    ts = pd.Timestamp(ts)
    starts = sorted(shifts)
    day = ts.normalize()
    candidates = []
    for offset in (-1, 0):
        for hour, name in starts:
            candidates.append((day + pd.Timedelta(days=offset, hours=hour), name))
    candidates.sort()
    start, name = max((c for c in candidates if c[0] <= ts), key=lambda c: c[0])
    idx = [h for h, _ in starts].index(start.hour)
    next_hour = starts[(idx + 1) % len(starts)][0]
    end = start.normalize() + pd.Timedelta(hours=next_hour)
    if end <= start:
        end += pd.Timedelta(days=1)
    return start, end, name


def _round(value: Optional[float], digits: int = 1) -> Optional[float]:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def shift_facts(
    wide: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    lab: Optional[pd.DataFrame] = None,
    alarm_rules: Optional[List[AlarmRule]] = None,
    fuels: Optional[FuelProperties] = None,
    mill_ids: Sequence[int] = (1, 2),
    findings: Optional[Iterable[Any]] = None,
    recommendations: Optional[Dict[str, Any]] = None,
    shift_name: str = "",
) -> Dict[str, Any]:
    """All numbers of the shift, computed from data (no text)."""
    fuels = fuels or FuelProperties()
    if T.SPECIFIC_HEAT_KCAL_KG not in wide:
        wide = add_derived_kpis(wide, fuels, mill_ids)
    window = wide.loc[start:end]
    if len(window) and window.index[-1] == pd.Timestamp(end):
        window = window.iloc[:-1]
    if window.empty:
        raise ValueError("no data in the shift window.")
    step_h = float(pd.Series(window.index).diff().median() / pd.Timedelta(hours=1)) if len(window) > 1 else 1.0
    hours = len(window) * step_h

    def total(tag: str, factor: float = 1.0) -> Optional[float]:
        return float(window[tag].sum() * step_h * factor) if tag in window else None

    def mean(tag: str) -> Optional[float]:
        return float(window[tag].mean()) if tag in window else None

    facts: Dict[str, Any] = {
        "shift": shift_name,
        "start": pd.Timestamp(start).isoformat(),
        "end": pd.Timestamp(end).isoformat(),
        "hours": _round(hours),
        "clinker_t": _round(total(T.CLINKER_TPH), 0),
        "specific_heat_kcal_kg": _round(mean(T.SPECIFIC_HEAT_KCAL_KG)),
        "gas_nm3": _round(total(T.GAS_FLOW_NM3H), 0),
        "mazut_l": _round(
            total(T.MAZUT_FLOW_KGH, 1.0 / fuels.mazut_density_kg_per_l), 0
        ),
        "gas_share_pct": _round(100 * mean(T.GAS_HEAT_SHARE)) if T.GAS_HEAT_SHARE in window else None,
        "bz_temp_mean_c": _round(mean(T.BZ_TEMP_C)),
        "bz_temp_min_c": _round(float(window[T.BZ_TEMP_C].min())) if T.BZ_TEMP_C in window else None,
    }
    mills = {}
    for n in mill_ids:
        kw, tph = T.mill_tag(T.MILL_KW_TEMPLATE, n), T.mill_tag(T.MILL_TPH_TEMPLATE, n)
        if kw in window and tph in window:
            kwh = total(kw) or 0.0
            tonnes = total(tph) or 0.0
            run_h = float((window[kw] > 100).sum() * step_h)
            mills[f"mill_{n}"] = {
                "cement_t": _round(tonnes, 0),
                "kwh_per_t": _round(kwh / tonnes) if tonnes > 0 else None,
                "running_h": _round(run_h),
            }
    facts["mills"] = mills

    if lab is not None and not lab.empty:
        stamps = pd.to_datetime(lab["timestamp"])
        fl = lab[(lab["parameter"] == T.FREE_LIME_PCT) & (stamps >= start) & (stamps < end)]
        if not fl.empty:
            facts["free_lime"] = {
                "n_samples": int(len(fl)),
                "mean_pct": _round(fl["value"].mean(), 2),
                "max_pct": _round(fl["value"].max(), 2),
            }
    if alarm_rules:
        events = derive_alarms(window, alarm_rules)
        summary = summarize(events, hours)
        facts["alarms"] = {
            "n_events": summary.n_events,
            "bad_actors": summary.bad_actors[:3],
            "chattering": summary.chattering,
            "minutes_in_alarm": {
                k: _round(v["minutes_in_alarm"], 0) for k, v in summary.per_tag.items()
            },
        }
    if T.GAS_HEAT_SHARE in window:
        share = window[T.GAS_HEAT_SHARE]
        facts["fuel_transfer"] = bool((share.max() - share.min()) > 0.5)
    if findings is not None:
        facts["equipment"] = [
            {"severity": f.severity, "message_fa": f.message_fa, "message": f.message}
            for f in findings
            if f.severity != "ok"
        ]
    if recommendations is not None:
        facts["recommendations"] = recommendations
    return facts


def _fmt(value: Any, unit: str = "") -> str:
    if value is None:
        return "—"
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, (int, float)):
        text = f"{value:,}" if abs(value) >= 10000 else f"{value}"
    else:
        text = str(value)
    return f"{text} {unit}".strip()


def render_fa(facts: Dict[str, Any], persian_digits: bool = True) -> str:
    """Persian shift report (Markdown) from :func:`shift_facts`."""
    lines = [
        f"# گزارش شیفت {facts.get('shift', '')}".strip(),
        f"از {facts['start'].replace('T', ' ')} تا {facts['end'].replace('T', ' ')}",
        "",
        "## کوره",
        f"- تولید کلینکر: {_fmt(facts.get('clinker_t'), 'تن')}",
        f"- مصرف حرارت ویژه: {_fmt(facts.get('specific_heat_kcal_kg'), 'کیلوکالری بر کیلوگرم')}",
        f"- گاز مصرفی: {_fmt(facts.get('gas_nm3'), 'مترمکعب')} | مازوت: {_fmt(facts.get('mazut_l'), 'لیتر')}"
        f" | سهم گاز: {_fmt(facts.get('gas_share_pct'), '٪')}",
        f"- دمای زون پخت: میانگین {_fmt(facts.get('bz_temp_mean_c'), '°C')}، "
        f"کمینه {_fmt(facts.get('bz_temp_min_c'), '°C')}",
    ]
    if facts.get("fuel_transfer"):
        lines.append("- در این شیفت جابه‌جایی سوخت گاز/مازوت انجام شد.")
    if "free_lime" in facts:
        fl = facts["free_lime"]
        lines.append(
            f"- آهک آزاد (آزمایشگاه): میانگین {_fmt(fl['mean_pct'], '٪')}، "
            f"بیشینه {_fmt(fl['max_pct'], '٪')} از {fl['n_samples']} نمونه"
        )
    if facts.get("mills"):
        lines += ["", "## آسیاب‌ها"]
        for name, m in facts["mills"].items():
            lines.append(
                f"- {name.replace('mill_', 'آسیاب ')}: {_fmt(m['cement_t'], 'تن')}، "
                f"{_fmt(m['kwh_per_t'], 'کیلووات‌ساعت بر تن')}، کارکرد {_fmt(m['running_h'], 'ساعت')}"
            )
    if "alarms" in facts:
        a = facts["alarms"]
        lines += ["", "## آلارم‌ها", f"- تعداد رویداد: {a['n_events']}"]
        if a["bad_actors"]:
            lines.append(f"- پرتکرارترین: {'، '.join(a['bad_actors'])}")
        if a["chattering"]:
            lines.append(f"- آلارم‌های پرنوسان (chattering): {'، '.join(a['chattering'])}")
    if facts.get("equipment"):
        lines += ["", "## هشدار تجهیزات"]
        for item in facts["equipment"]:
            label = "آلارم" if item["severity"] == "alarm" else "هشدار"
            lines.append(f"- [{label}] {item['message_fa']}")
    if facts.get("recommendations"):
        r = facts["recommendations"]
        lines += [
            "",
            "## توصیه‌های دستیار",
            f"- تعداد توصیه: {_fmt(r.get('n_recommendations'))} | پذیرفته: {_fmt(r.get('accepted'))}"
            f" | ردشده: {_fmt(r.get('rejected'))}",
        ]
    lines += ["", "_این گزارش به‌صورت خودکار از داده‌های فرایند ساخته شده است._"]
    text = "\n".join(lines)
    return to_persian_digits(text) if persian_digits else text


def render_en(facts: Dict[str, Any]) -> str:
    """English shift report (Markdown) from :func:`shift_facts`."""
    lines = [
        f"# Shift report {facts.get('shift', '')}".strip(),
        f"{facts['start']} to {facts['end']}",
        "",
        "## Kiln",
        f"- Clinker: {_fmt(facts.get('clinker_t'), 't')}",
        f"- Specific heat: {_fmt(facts.get('specific_heat_kcal_kg'), 'kcal/kg')}",
        f"- Gas: {_fmt(facts.get('gas_nm3'), 'Nm3')} | Mazut: {_fmt(facts.get('mazut_l'), 'L')}"
        f" | Gas share: {_fmt(facts.get('gas_share_pct'), '%')}",
        f"- Burning zone: mean {_fmt(facts.get('bz_temp_mean_c'), '°C')}, min {_fmt(facts.get('bz_temp_min_c'), '°C')}",
    ]
    if "free_lime" in facts:
        fl = facts["free_lime"]
        lines.append(f"- Free lime (lab): mean {fl['mean_pct']} %, max {fl['max_pct']} % ({fl['n_samples']} samples)")
    for name, m in facts.get("mills", {}).items():
        lines.append(f"- {name}: {_fmt(m['cement_t'], 't')}, {_fmt(m['kwh_per_t'], 'kWh/t')}, {_fmt(m['running_h'], 'h')} running")
    if "alarms" in facts:
        lines.append(f"- Alarms: {facts['alarms']['n_events']} events; bad actors: {', '.join(facts['alarms']['bad_actors']) or '—'}")
    for item in facts.get("equipment", []):
        lines.append(f"- [{item['severity']}] {item['message']}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Optional LLM polish
# --------------------------------------------------------------------------- #
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[str]:
    clean = normalize_digits(text).replace("٫", ".").replace("٬", ",")
    return {n.replace(",", "") for n in _NUMBER_RE.findall(clean)}


def numbers_consistent(candidate: str, reference: str) -> bool:
    """Every number in ``candidate`` also appears in ``reference``."""
    return _numbers(candidate) <= _numbers(reference)


@dataclass
class LLMConfig:
    """An OpenAI-compatible chat endpoint on the plant network."""

    base_url: str  # e.g. http://localhost:11434/v1 for Ollama
    model: str
    api_key: str = "not-needed"
    timeout_s: float = 60.0


def polish_with_llm(report_md: str, facts: Dict[str, Any], config: LLMConfig) -> tuple[str, bool]:
    """Return ``(text, used_llm)``; falls back to ``report_md`` on any problem."""
    try:
        import httpx
    except ImportError:  # pragma: no cover
        return report_md, False
    prompt = (
        "این گزارش شیفت کارخانهٔ سیمان را به فارسی روان و کوتاه بازنویسی کن. "
        "هیچ عددی را تغییر نده، عدد جدید اضافه نکن و چیزی حدس نزن. "
        "فقط از اعداد همین گزارش استفاده کن.\n\n" + report_md
    )
    try:
        response = httpx.post(
            f"{config.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json={
                "model": config.model,
                "temperature": 0.2,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=config.timeout_s,
        )
        response.raise_for_status()
        text = response.json()["choices"][0]["message"]["content"].strip()
    except Exception as exc:  # network, schema, server errors: keep the template
        logger.warning("LLM polish failed, using the template report: %s", exc)
        return report_md, False
    reference = report_md + "\n" + json.dumps(facts, ensure_ascii=False)
    if not text or not numbers_consistent(text, reference):
        logger.warning("LLM output changed or invented numbers; using the template report.")
        return report_md, False
    return text, True
