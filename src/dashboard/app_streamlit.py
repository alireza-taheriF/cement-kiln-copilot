"""Streamlit operator dashboard for cement-kiln-copilot.

An end-to-end advisory demo that talks to the FastAPI backend using the
current endpoint contracts:

* ``GET  /health``                 - liveness probe
* ``GET  /``                       - service/version banner
* ``POST /v1/predict/energy-kpi``  - KPI forecasting
* ``POST /v1/recommend/setpoints`` - model-based setpoint recommendations
* ``GET  /v1/monitor/drift``       - feature z-scores versus training stats

Run with::

    streamlit run src/dashboard/app_streamlit.py

The dashboard is a thin, advisory-only UI: it never closes the control loop,
it only proposes setpoint adjustments for operators to consider.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
DEFAULT_API_BASE_URL = "http://localhost:8000"
PREDICT_PATH = "/v1/predict/energy-kpi"
RECOMMEND_PATH = "/v1/recommend/setpoints"
DRIFT_PATH = "/v1/monitor/drift"
REQUEST_TIMEOUT_SECONDS = 15

# Tags the synthetic generator understands, with realistic baselines.
SAMPLE_TAG_BASELINES: Dict[str, Tuple[float, float, float]] = {
    # tag: (baseline, oscillation_amplitude, noise_scale)
    "KILN_ZONE1_TEMP": (1380.0, 8.0, 1.5),
    "KILN_ZONE2_TEMP": (1320.0, 6.0, 1.2),
    "KILN_FUEL_FLOW": (42.0, 1.5, 0.3),
}
# Include every tag the demo model was trained on, so a sample payload always
# carries all required base tags for prediction/recommendation.
DEFAULT_SAMPLE_TAGS: List[str] = [
    "KILN_ZONE1_TEMP",
    "KILN_ZONE2_TEMP",
    "KILN_FUEL_FLOW",
]
DEFAULT_ADJUSTABLE_TAGS: List[str] = ["KILN_FUEL_FLOW"]
DEFAULT_STEP_FRACTIONS: List[float] = [-0.10, -0.05, 0.0, 0.05, 0.10]
SAMPLE_SEED = 42


# --------------------------------------------------------------------------- #
# Internationalization (i18n)
# --------------------------------------------------------------------------- #
# Persian is the default; English can be selected from the sidebar.
LANGUAGES: Dict[str, str] = {"fa": "فارسی", "en": "English"}
DEFAULT_LANG = "fa"

TRANSLATIONS: Dict[str, Dict[str, str]] = {
    "fa": {
        "lang_label": "🌐 زبان / Language",
        "app_title": "🏭 دستیار کورهٔ سیمان",
        "app_caption": "سامانهٔ پشتیبان تصمیم (حلقه‌باز و مشورتی) برای بهینه‌سازی کوره و آسیاب سیمان.",
        "checklist_title": "چک‌لیست دمو",
        "checklist_body": (
            "۱. در نوار کناری **وضعیت بک‌اند** را بررسی کنید.\n"
            "۲. **بارگذاری سناریوی دمو (همهٔ تگ‌ها)** را بزنید.\n"
            "۳. **اجرای پیش‌بینی KPI** را بزنید تا هدف پیش‌بینی شود.\n"
            "۴. یک **مقدار هدف** تعیین و **تگ‌های قابل تنظیم** را انتخاب کنید.\n"
            "۵. **اجرای توصیه** را بزنید و بهترین گزینه و امتیاز را ببینید.\n"
            "۶. پنل **پایش رانش** را ببینید: z-score ویژگی‌ها نسبت به آموزش.\n\n"
            "_فقط مشورتی: سامانه ست‌پوینت پیشنهاد می‌دهد و تجهیزات را فرمان نمی‌دهد._"
        ),
        "backend_connection": "اتصال بک‌اند",
        "api_base_url": "آدرس پایهٔ API",
        "api_base_url_help": "ریشهٔ سرور FastAPI.",
        "check_status": "بررسی وضعیت بک‌اند",
        "health": "سلامت",
        "root": "ریشه",
        "ok": "سالم",
        "service_banner": "سرویس: {service} (نسخهٔ {version})",
        "sample_data": "دادهٔ نمونه",
        "rows": "تعداد ردیف",
        "start_date": "تاریخ شروع",
        "tags": "تگ‌ها",
        "use_sample": "ساخت دادهٔ نمونه",
        "reset_payload": "پاک‌کردن داده",
        "select_one_tag": "حداقل یک تگ انتخاب کنید.",
        "load_demo": "بارگذاری سناریوی دمو (همهٔ تگ‌ها)",
        "input_data": "۱. دادهٔ ورودی",
        "input_mode": "حالت ورودی",
        "mode_synthetic": "نمونهٔ ساختگی",
        "mode_manual": "JSON دستی",
        "generate_hint": "برای ساخت داده، در نوار کناری **ساخت دادهٔ نمونه** را بزنید.",
        "rows_tags_caption": "{n} ردیف · تگ‌ها: {tags}",
        "request_payload_json": "بدنهٔ درخواست (JSON)",
        "payload_valid": "بدنهٔ JSON معتبر است.",
        "kpi_prediction": "۲. پیش‌بینی KPI",
        "run_prediction": "اجرای پیش‌بینی KPI",
        "prediction_failed": "پیش‌بینی ناموفق بود.",
        "m_target": "هدف",
        "m_backend": "موتور مدل",
        "m_rows": "ردیف‌ها",
        "m_model_version": "نسخهٔ مدل",
        "m_mean": "میانگین",
        "m_min": "کمینه",
        "m_max": "بیشینه",
        "no_predictions": "پاسخ هیچ پیش‌بینی‌ای نداشت.",
        "raw_prediction": "پاسخ خام پیش‌بینی",
        "setpoint_rec": "۳. توصیهٔ ست‌پوینت",
        "desired_target": "مقدار هدف",
        "top_k": "تعداد برتر (Top K)",
        "adjustable_tags": "تگ‌های قابل تنظیم",
        "step_fractions": "گام‌ها (با کاما جدا کنید)",
        "absolute_bounds": "کران‌های مطلق (اختیاری، JSON: تگ → [کمینه، بیشینه])",
        "run_recommendation": "اجرای توصیه",
        "select_one_adjustable": "حداقل یک تگ قابل تنظیم انتخاب کنید.",
        "recommendation_failed": "توصیه ناموفق بود.",
        "m_desired_target": "مقدار هدف",
        "m_baseline_mean": "میانگین پایه",
        "m_candidates": "گزینه‌ها",
        "backend_caption": "موتور مدل: {x}",
        "model_version_caption": "نسخهٔ مدل: {x}",
        "no_recommendations": "هیچ توصیه‌ای برنگشت.",
        "best_candidate": "بهترین گزینه (امتیاز {score}): {setpoints}",
        "score_caption": (
            "هرچه مقدار پیش‌بینی‌شدهٔ KPI به هدف نزدیک‌تر باشد، امتیاز کمتر "
            "است. ردیف بالا بهترین گزینه است."
        ),
        "raw_recommendation": "پاسخ خام توصیه",
        "drift_title": "پایش رانش",
        "drift_max_z": "بیشینه |z|",
        "drift_caption": "{n} پیش‌بینی اخیر در برابر آمار آموزش. فقط مشورتی.",
        "drift_hint": "برای دیدن z-score، وضعیت بک‌اند را بررسی کنید.",
        "drift_unavailable": "وضعیت رانش در دسترس نیست.",
        "drift_empty": "هنوز پیش‌بینی‌ای برای مقایسه ثبت نشده است.",
        "drift_features": "z-score ویژگی‌ها",
        "drift_inline": "رانش: بیشینه |z| = {z} روی {n} پیش‌بینی اخیر.",
    },
    "en": {
        "lang_label": "🌐 Language / زبان",
        "app_title": "🏭 Cement Kiln Copilot",
        "app_caption": "Advisory (open-loop) decision support for cement kiln & mill optimization.",
        "checklist_title": "Demo checklist",
        "checklist_body": (
            "1. **Check backend status** in the sidebar.\n"
            "2. **Load demo scenario (all tags)** in the sidebar.\n"
            "3. **Run KPI Prediction** to forecast the target.\n"
            "4. Set a **desired target** and pick **adjustable tags**.\n"
            "5. **Run Recommendation** and read the best candidate + score.\n"
            "6. Read the **drift status** panel: feature z-scores versus training.\n\n"
            "_Advisory only: the system suggests setpoints; it does not actuate "
            "equipment._"
        ),
        "backend_connection": "Backend connection",
        "api_base_url": "API base URL",
        "api_base_url_help": "FastAPI server root.",
        "check_status": "Check backend status",
        "health": "Health",
        "root": "Root",
        "ok": "OK",
        "service_banner": "Service: {service} (v{version})",
        "sample_data": "Sample data",
        "rows": "Rows",
        "start_date": "Start date",
        "tags": "Tags",
        "use_sample": "Use sample payload",
        "reset_payload": "Reset payload",
        "select_one_tag": "Select at least one tag.",
        "load_demo": "Load demo scenario (all tags)",
        "input_data": "1. Input data",
        "input_mode": "Input mode",
        "mode_synthetic": "Synthetic sample",
        "mode_manual": "Manual JSON",
        "generate_hint": "Use **Use sample payload** in the sidebar to generate data.",
        "rows_tags_caption": "{n} rows · tags: {tags}",
        "request_payload_json": "Request payload (JSON)",
        "payload_valid": "Payload is valid JSON.",
        "kpi_prediction": "2. KPI prediction",
        "run_prediction": "Run KPI Prediction",
        "prediction_failed": "Prediction failed.",
        "m_target": "Target",
        "m_backend": "Backend",
        "m_rows": "Rows",
        "m_model_version": "Model version",
        "m_mean": "Mean",
        "m_min": "Min",
        "m_max": "Max",
        "no_predictions": "The response contained no predictions.",
        "raw_prediction": "Raw prediction response",
        "setpoint_rec": "3. Setpoint recommendations",
        "desired_target": "Desired target",
        "top_k": "Top K",
        "adjustable_tags": "Adjustable tags",
        "step_fractions": "Step fractions (comma-separated)",
        "absolute_bounds": "Absolute bounds (optional JSON: tag -> [min, max])",
        "run_recommendation": "Run Recommendation",
        "select_one_adjustable": "Select at least one adjustable tag.",
        "recommendation_failed": "Recommendation failed.",
        "m_desired_target": "Desired target",
        "m_baseline_mean": "Baseline mean",
        "m_candidates": "Candidates",
        "backend_caption": "Backend: {x}",
        "model_version_caption": "Model version: {x}",
        "no_recommendations": "No recommendations were returned.",
        "best_candidate": "Best candidate (score {score}): {setpoints}",
        "score_caption": (
            "Scores are lower when the predicted KPI is closer to the desired "
            "target. The top row is the best candidate."
        ),
        "raw_recommendation": "Raw recommendation response",
        "drift_title": "Drift status",
        "drift_max_z": "Max |z|",
        "drift_caption": "Last {n} predictions vs training feature stats. Advisory only.",
        "drift_hint": "Check backend status to load feature z-scores.",
        "drift_unavailable": "Drift status is unavailable.",
        "drift_empty": "No predictions have been logged yet.",
        "drift_features": "Per-feature z-scores",
        "drift_inline": "Drift: max |z| = {z} over the last {n} predictions.",
    },
}


def t(key: str, **kwargs: Any) -> str:
    """Translate ``key`` for the active language, with optional formatting."""
    lang = st.session_state.get("lang", DEFAULT_LANG)
    table = TRANSLATIONS.get(lang, TRANSLATIONS[DEFAULT_LANG])
    text = table.get(key) or TRANSLATIONS["en"].get(key, key)
    return text.format(**kwargs) if kwargs else text


def _inject_rtl() -> None:
    """Apply right-to-left layout for Persian (keeps code/JSON left-aligned)."""
    st.markdown(
        """
        <style>
        .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
        [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
            direction: rtl; text-align: right;
        }
        pre, code, .stCode, [data-testid="stJson"] {
            direction: ltr; text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _select_language() -> str:
    """Render the language selector at the top of the sidebar; return the code."""
    codes = list(LANGUAGES.keys())
    current = st.session_state.get("lang", DEFAULT_LANG)
    return st.sidebar.selectbox(
        TRANSLATIONS.get(current, TRANSLATIONS[DEFAULT_LANG])["lang_label"],
        options=codes,
        index=codes.index(current) if current in codes else 0,
        format_func=lambda c: LANGUAGES[c],
        key="lang",
    )


# --------------------------------------------------------------------------- #
# Typed API helpers
# --------------------------------------------------------------------------- #
@dataclass
class ApiResult:
    """Result of an HTTP call to the backend.

    Attributes
    ----------
    ok:
        Whether the request succeeded with a 2xx status.
    status_code:
        HTTP status code, or ``None`` if the request never completed.
    data:
        Parsed JSON body on success (or best-effort on error).
    error:
        Human-readable error message when ``ok`` is ``False``.
    """

    ok: bool
    status_code: Optional[int] = None
    data: Any = None
    error: Optional[str] = None


def get_api_base_url() -> str:
    """Return the API base URL from the environment or the default."""
    return os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def _extract_error_detail(response: requests.Response) -> str:
    """Build a readable error message from a non-2xx response."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}: {response.text[:300]}"

    detail = body.get("detail") if isinstance(body, dict) else None
    if detail is None:
        return f"HTTP {response.status_code}: {json.dumps(body)[:300]}"
    if isinstance(detail, str):
        return f"HTTP {response.status_code}: {detail}"
    # FastAPI/Pydantic validation errors come back as a list of dicts.
    return f"HTTP {response.status_code}: {json.dumps(detail)[:500]}"


def _request(
    method: str,
    url: str,
    payload: Optional[Dict[str, Any]] = None,
) -> ApiResult:
    """Perform an HTTP request and normalize the outcome into an ApiResult."""
    try:
        response = requests.request(
            method, url, json=payload, timeout=REQUEST_TIMEOUT_SECONDS
        )
    except requests.RequestException as exc:  # network/timeout/connection
        return ApiResult(ok=False, status_code=None, error=f"Request failed: {exc}")

    if not response.ok:
        return ApiResult(
            ok=False,
            status_code=response.status_code,
            error=_extract_error_detail(response),
        )

    try:
        data = response.json()
    except ValueError:
        return ApiResult(
            ok=False,
            status_code=response.status_code,
            error="Backend returned a non-JSON response.",
        )
    return ApiResult(ok=True, status_code=response.status_code, data=data)


def call_health(base_url: str) -> ApiResult:
    """Call ``GET /health``."""
    return _request("GET", f"{base_url}/health")


def call_root(base_url: str) -> ApiResult:
    """Call ``GET /`` for the service banner."""
    return _request("GET", f"{base_url}/")


def call_prediction_api(base_url: str, payload: Dict[str, Any]) -> ApiResult:
    """Call ``POST /v1/predict/energy-kpi`` with a prediction payload."""
    body = {"timestamps": payload["timestamps"], "signals": payload["signals"]}
    return _request("POST", f"{base_url}{PREDICT_PATH}", payload=body)


def call_recommendation_api(base_url: str, payload: Dict[str, Any]) -> ApiResult:
    """Call ``POST /v1/recommend/setpoints`` with a recommendation payload."""
    return _request("POST", f"{base_url}{RECOMMEND_PATH}", payload=payload)


def call_drift(base_url: str, n: int = 20) -> ApiResult:
    """Call ``GET /v1/monitor/drift`` for the feature z-score panel."""
    return _request("GET", f"{base_url}{DRIFT_PATH}?n={n}")


# --------------------------------------------------------------------------- #
# Payload construction / parsing
# --------------------------------------------------------------------------- #
def build_sample_payload(
    n_rows: int,
    start_timestamp: datetime,
    tags: List[str],
    adjustable_tags: List[str],
) -> Dict[str, Any]:
    """Deterministically generate a synthetic request payload.

    Parameters
    ----------
    n_rows:
        Number of timestamps/samples to generate.
    start_timestamp:
        First timestamp; samples are spaced one minute apart.
    tags:
        Signal tags to synthesize (must be known to the generator).
    adjustable_tags:
        Tags whose latest value seeds ``current_setpoints``.

    Returns
    -------
    dict
        Payload with ``timestamps``, ``signals`` and ``current_setpoints``.
    """
    rng = np.random.default_rng(SAMPLE_SEED)
    index = pd.date_range(start_timestamp, periods=n_rows, freq="min")
    timestamps = [ts.isoformat() for ts in index]

    signals: Dict[str, List[float]] = {}
    steps = np.arange(n_rows)
    for tag in tags:
        baseline, amplitude, noise = SAMPLE_TAG_BASELINES.get(tag, (1.0, 0.1, 0.05))
        series = (
            baseline
            + amplitude * np.sin(steps / 6.0)
            + rng.normal(scale=noise, size=n_rows)
        )
        signals[tag] = [round(float(v), 4) for v in series]

    # Seed setpoints from the latest observed value of each adjustable tag.
    current_setpoints: Dict[str, float] = {}
    for tag in adjustable_tags:
        if tag in signals and signals[tag]:
            current_setpoints[tag] = signals[tag][-1]

    return {
        "timestamps": timestamps,
        "signals": signals,
        "current_setpoints": current_setpoints,
    }


def parse_json_payload(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse and minimally validate a JSON payload from the manual editor.

    Returns
    -------
    tuple
        ``(payload, None)`` on success or ``(None, error_message)`` on failure.
    """
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid JSON: {exc}"

    if not isinstance(parsed, dict):
        return None, "Payload must be a JSON object."

    missing = [k for k in ("timestamps", "signals") if k not in parsed]
    if missing:
        return None, f"Payload is missing required key(s): {missing}."
    if not isinstance(parsed["signals"], dict) or not parsed["signals"]:
        return None, "'signals' must be a non-empty object."
    return parsed, None


def parse_step_fractions(text: str) -> Tuple[List[float], Optional[str]]:
    """Parse a comma-separated list of step fractions."""
    try:
        fractions = [float(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        return [], f"Invalid step fractions: {exc}"
    if not fractions:
        return [], "Provide at least one step fraction."
    return fractions, None


def parse_absolute_bounds(
    text: str,
) -> Tuple[Optional[Dict[str, List[float]]], Optional[str]]:
    """Parse an optional absolute-bounds JSON object (or empty -> None)."""
    text = text.strip()
    if not text:
        return None, None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, f"Invalid bounds JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, "Bounds must be a JSON object of tag -> [min, max]."
    for tag, bound in parsed.items():
        if not isinstance(bound, (list, tuple)) or len(bound) != 2:
            return None, f"Bound for {tag!r} must be a 2-element [min, max]."
        if bound[0] > bound[1]:
            return None, f"Bound for {tag!r} has min > max."
    return parsed, None


# --------------------------------------------------------------------------- #
# Result renderers
# --------------------------------------------------------------------------- #
def render_prediction_results(data: Dict[str, Any]) -> None:
    """Render a successful prediction response."""
    predictions = [float(p) for p in data.get("predictions", [])]

    meta_cols = st.columns(4)
    meta_cols[0].metric(t("m_target"), str(data.get("target_name", "—")))
    meta_cols[1].metric(t("m_backend"), str(data.get("model_backend", "—")))
    meta_cols[2].metric(t("m_rows"), str(data.get("n_rows", len(predictions))))
    meta_cols[3].metric(t("m_model_version"), str(data.get("model_version", "—")))

    if not predictions:
        st.warning(t("no_predictions"))
        return

    series = pd.Series(predictions, name="prediction")
    stat_cols = st.columns(3)
    stat_cols[0].metric(t("m_mean"), f"{series.mean():.3f}")
    stat_cols[1].metric(t("m_min"), f"{series.min():.3f}")
    stat_cols[2].metric(t("m_max"), f"{series.max():.3f}")

    st.line_chart(pd.DataFrame({"prediction": series}))
    with st.expander(t("raw_prediction")):
        st.json(data)


def render_recommendation_results(data: Dict[str, Any]) -> None:
    """Render a successful recommendation response."""
    meta_cols = st.columns(4)
    meta_cols[0].metric(t("m_target"), str(data.get("target_name", "—")))
    meta_cols[1].metric(
        t("m_desired_target"), f"{float(data.get('desired_target', 0.0)):.3f}"
    )
    meta_cols[2].metric(
        t("m_baseline_mean"), f"{float(data.get('baseline_prediction_mean', 0.0)):.3f}"
    )
    meta_cols[3].metric(t("m_candidates"), str(data.get("n_candidates_evaluated", "—")))

    info_cols = st.columns(2)
    info_cols[0].caption(t("backend_caption", x=data.get("model_backend", "—")))
    info_cols[1].caption(t("model_version_caption", x=data.get("model_version", "—")))

    recommendations = data.get("recommendations", [])
    if not recommendations:
        st.warning(t("no_recommendations"))
        return

    rows: List[Dict[str, Any]] = []
    for rank, item in enumerate(recommendations, start=1):
        row: Dict[str, Any] = {
            "rank": rank,
            "score": float(item.get("score", float("nan"))),
            "predicted_mean": float(item.get("predicted_mean", float("nan"))),
            "predicted_min": float(item.get("predicted_min", float("nan"))),
            "predicted_max": float(item.get("predicted_max", float("nan"))),
        }
        for tag, value in item.get("candidate_setpoints", {}).items():
            row[f"setpoint::{tag}"] = float(value)
        rows.append(row)

    table = pd.DataFrame(rows).set_index("rank")

    best = recommendations[0]
    best_setpoints = ", ".join(
        f"{tag}={float(val):.3f}"
        for tag, val in best.get("candidate_setpoints", {}).items()
    )
    st.success(
        t(
            "best_candidate",
            score=f"{float(best.get('score', 0.0)):.4f}",
            setpoints=best_setpoints,
        )
    )

    st.dataframe(table, use_container_width=True)
    st.bar_chart(table[["score"]])
    st.caption(t("score_caption"))
    with st.expander(t("raw_recommendation")):
        st.json(data)


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
def _render_drift_panel() -> None:
    """Small advisory panel: recent prediction inputs versus training stats."""
    st.subheader(t("drift_title"))
    result: Optional[ApiResult] = st.session_state.get("drift_result")
    if result is None:
        st.caption(t("drift_hint"))
        return
    if not result.ok or not isinstance(result.data, dict):
        st.warning(result.error or t("drift_unavailable"))
        return

    data = result.data
    n_rows = int(data.get("n") or 0)
    max_abs = data.get("max_abs_z")
    if isinstance(max_abs, (int, float)):
        st.metric(t("drift_max_z"), f"{float(max_abs):.3f}")
    st.caption(t("drift_caption", n=n_rows))
    if n_rows == 0:
        st.caption(t("drift_empty"))
        return

    features = data.get("features") or {}
    if not isinstance(features, dict) or not features:
        return
    ranked = sorted(features.items(), key=lambda item: abs(float(item[1])), reverse=True)[:8]
    with st.expander(t("drift_features")):
        st.dataframe(
            pd.DataFrame(
                [{"feature": name, "z": float(score)} for name, score in ranked]
            ),
            use_container_width=True,
        )


def _render_status_badge(label: str, result: ApiResult) -> None:
    """Render a success/failure badge for a backend check."""
    if result.ok:
        st.success(f"{label}: {t('ok')}")
    else:
        st.error(f"{label}: {result.error}")


def _render_sidebar() -> str:
    """Render the sidebar (connection, status, sample controls); return base URL."""
    with st.sidebar:
        st.divider()
        st.header(t("backend_connection"))
        base_url = st.text_input(
            t("api_base_url"),
            value=get_api_base_url(),
            help=t("api_base_url_help"),
        ).rstrip("/")

        if st.button(t("check_status"), use_container_width=True):
            st.session_state["health_result"] = call_health(base_url)
            st.session_state["root_result"] = call_root(base_url)
            st.session_state["drift_result"] = call_drift(base_url)

        health_result: Optional[ApiResult] = st.session_state.get("health_result")
        root_result: Optional[ApiResult] = st.session_state.get("root_result")
        if health_result is not None:
            _render_status_badge(t("health"), health_result)
        if root_result is not None:
            if root_result.ok and isinstance(root_result.data, dict):
                st.info(
                    t(
                        "service_banner",
                        service=root_result.data.get("service", "—"),
                        version=root_result.data.get("version", "—"),
                    )
                )
            else:
                _render_status_badge(t("root"), root_result)

        st.divider()
        _render_drift_panel()
        st.divider()
        st.header(t("sample_data"))
        n_rows = int(st.number_input(t("rows"), min_value=2, max_value=2000, value=60))
        start_date = st.date_input(t("start_date"), value=datetime(2026, 1, 1).date())
        tags = st.multiselect(
            t("tags"),
            options=list(SAMPLE_TAG_BASELINES.keys()),
            default=DEFAULT_SAMPLE_TAGS,
        )

        col_a, col_b = st.columns(2)
        if col_a.button(t("use_sample"), use_container_width=True):
            if not tags:
                st.warning(t("select_one_tag"))
            else:
                adjustable_default = [t_ for t_ in DEFAULT_ADJUSTABLE_TAGS if t_ in tags]
                st.session_state["payload"] = build_sample_payload(
                    n_rows,
                    datetime.combine(start_date, datetime.min.time()),
                    tags,
                    adjustable_default or tags[:1],
                )
        if col_b.button(t("reset_payload"), use_container_width=True):
            st.session_state.pop("payload", None)

        if st.button(t("load_demo"), use_container_width=True):
            st.session_state["payload"] = build_sample_payload(
                60,
                datetime(2026, 1, 1),
                list(SAMPLE_TAG_BASELINES.keys()),
                DEFAULT_ADJUSTABLE_TAGS,
            )

    return base_url


# --------------------------------------------------------------------------- #
# Input section
# --------------------------------------------------------------------------- #
def _render_input_section() -> Optional[Dict[str, Any]]:
    """Render the input builder and return the active payload (or ``None``)."""
    st.subheader(t("input_data"))
    mode = st.radio(
        t("input_mode"),
        options=["synthetic", "manual"],
        format_func=lambda k: t(f"mode_{k}"),
        horizontal=True,
    )

    payload: Optional[Dict[str, Any]] = st.session_state.get("payload")

    if mode == "synthetic":
        if payload is None:
            st.info(t("generate_hint"))
            return None
        n_rows = len(payload.get("timestamps", []))
        tags = list(payload.get("signals", {}).keys())
        st.caption(t("rows_tags_caption", n=n_rows, tags=", ".join(tags) or "—"))
        preview = pd.DataFrame(payload.get("signals", {}))
        preview.insert(0, "timestamp", payload.get("timestamps", []))
        st.dataframe(preview.head(10), use_container_width=True)
        return payload

    # Manual JSON editor.
    default_text = json.dumps(
        payload
        or {
            "timestamps": [],
            "signals": {},
            "current_setpoints": {},
        },
        indent=2,
    )
    text = st.text_area(t("request_payload_json"), value=default_text, height=280)
    parsed, error = parse_json_payload(text)
    if error:
        st.error(error)
        return None
    st.session_state["payload"] = parsed
    st.success(t("payload_valid"))
    return parsed


# --------------------------------------------------------------------------- #
# Prediction / recommendation panels
# --------------------------------------------------------------------------- #
def _render_prediction_panel(base_url: str, payload: Dict[str, Any]) -> None:
    """Render the prediction panel."""
    st.subheader(t("kpi_prediction"))
    if st.button(t("run_prediction"), type="primary"):
        result = call_prediction_api(base_url, payload)
        if result.ok and isinstance(result.data, dict):
            render_prediction_results(result.data)
            drift = call_drift(base_url)
            st.session_state["drift_result"] = drift
            if drift.ok and isinstance(drift.data, dict):
                max_abs = drift.data.get("max_abs_z")
                if isinstance(max_abs, (int, float)):
                    st.caption(
                        t(
                            "drift_inline",
                            z=f"{float(max_abs):.3f}",
                            n=int(drift.data.get("n") or 0),
                        )
                    )
        else:
            st.error(result.error or t("prediction_failed"))


def _render_recommendation_panel(base_url: str, payload: Dict[str, Any]) -> None:
    """Render the recommendation panel and its controls."""
    st.subheader(t("setpoint_rec"))

    available_tags = list(payload.get("signals", {}).keys())
    current_setpoints: Dict[str, float] = dict(payload.get("current_setpoints", {}))

    col1, col2, col3 = st.columns(3)
    desired_target = float(
        col1.number_input(t("desired_target"), value=1380.0, step=1.0)
    )
    top_k = int(col2.number_input(t("top_k"), min_value=1, max_value=50, value=3))
    default_adjustable = [
        tag for tag in DEFAULT_ADJUSTABLE_TAGS if tag in available_tags
    ]
    adjustable_tags = col3.multiselect(
        t("adjustable_tags"),
        options=available_tags,
        default=default_adjustable or available_tags[:1],
    )

    step_text = st.text_input(
        t("step_fractions"),
        value=",".join(str(f) for f in DEFAULT_STEP_FRACTIONS),
    )
    bounds_text = st.text_area(
        t("absolute_bounds"),
        value="",
        height=80,
    )

    if st.button(t("run_recommendation"), type="primary"):
        step_fractions, step_err = parse_step_fractions(step_text)
        if step_err:
            st.error(step_err)
            return
        bounds, bounds_err = parse_absolute_bounds(bounds_text)
        if bounds_err:
            st.error(bounds_err)
            return
        if not adjustable_tags:
            st.error(t("select_one_adjustable"))
            return

        # Ensure adjustable tags have a current setpoint (seed from latest value).
        for tag in adjustable_tags:
            if tag not in current_setpoints:
                values = payload.get("signals", {}).get(tag, [])
                if values:
                    current_setpoints[tag] = float(values[-1])

        request_payload: Dict[str, Any] = {
            "timestamps": payload["timestamps"],
            "signals": payload["signals"],
            "current_setpoints": current_setpoints,
            "desired_target": desired_target,
            "adjustable_tags": adjustable_tags,
            "step_fractions": step_fractions,
            "top_k": top_k,
        }
        if bounds is not None:
            request_payload["absolute_bounds"] = bounds

        result = call_recommendation_api(base_url, request_payload)
        if result.ok and isinstance(result.data, dict):
            render_recommendation_results(result.data)
        else:
            st.error(result.error or t("recommendation_failed"))


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    """Render the full dashboard."""
    st.set_page_config(
        page_title="Cement Kiln Copilot", page_icon="🏭", layout="wide"
    )

    lang = _select_language()
    if lang == "fa":
        _inject_rtl()

    st.title(t("app_title"))
    st.caption(t("app_caption"))

    with st.expander(t("checklist_title"), expanded=False):
        st.markdown(t("checklist_body"))

    base_url = _render_sidebar()
    payload = _render_input_section()

    if payload is None:
        st.stop()

    left, right = st.columns(2)
    with left:
        _render_prediction_panel(base_url, payload)
    with right:
        _render_recommendation_panel(base_url, payload)


if __name__ == "__main__":
    main()
