"""Streamlit operator dashboard for cement-kiln-copilot (Persian first).

A thin client of the FastAPI backend. Tabs:

* **Kiln advisor** - ``POST /v1/advise/kiln`` + operator feedback
* **Mill schedule** - ``POST /v1/energy/mill-schedule``
* **Free lime** - ``POST /v1/quality/free-lime``
* **Equipment** - ``POST /v1/maintenance/condition``
* **Shift report** - ``POST /v1/reports/shift``
* **Model lab** - ``POST /v1/predict/energy-kpi``, ``POST /v1/recommend/setpoints``,
  ``GET /v1/monitor/drift``

Without plant data every tab can load a window of the simulated demo plant
(``GET /v1/demo/window``).

Run with::

    streamlit run src/dashboard/app_streamlit.py

The dashboard is a thin, advisory-only UI: it never closes the control loop,
it only proposes setpoint adjustments for operators to consider.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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

DEMO_PATH = "/v1/demo/window"
ADVISE_PATH = "/v1/advise/kiln"
FEEDBACK_PATH = "/v1/advise/feedback"
REASONS_PATH = "/v1/advise/reasons"
STATS_PATH = "/v1/advise/stats"
MILL_PATH = "/v1/energy/mill-schedule"
FREE_LIME_PATH = "/v1/quality/free-lime"
CONDITION_PATH = "/v1/maintenance/condition"
SHIFT_PATH = "/v1/reports/shift"
LONG_TIMEOUT_SECONDS = 60

DEFAULT_ADJUSTABLE_TAGS: List[str] = ["ID_FAN_PCT"]
DEFAULT_STEP_FRACTIONS: List[float] = [-0.03, -0.015, 0.0, 0.015, 0.03]


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
            "۲. در زبانهٔ **دستیار کوره** دادهٔ دمو را بارگذاری و **دریافت توصیه** را بزنید.\n"
            "۳. توصیه، بازهٔ اطمینان پیش‌بینی‌ها و دلیل آن را بخوانید و بازخورد ثبت کنید.\n"
            "۴. در **زمان‌بندی آسیاب** صرفه‌جویی برق نسبت به روش فعلی را ببینید.\n"
            "۵. **آهک آزاد**، **تجهیزات** و **گزارش شیفت** را هم امتحان کنید.\n\n"
            "_فقط مشورتی: سامانه پیشنهاد می‌دهد و هیچ فرمانی به DCS نمی‌فرستد._"
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
        "demo_end_row": "ردیف پایانی دادهٔ دمو",
        "start_date": "تاریخ شروع",
        "tags": "تگ‌ها",
        "use_sample": "ساخت دادهٔ نمونه",
        "reset_payload": "پاک‌کردن داده",
        "select_one_tag": "حداقل یک تگ انتخاب کنید.",
        "load_demo": "بارگذاری دادهٔ کارخانهٔ شبیه‌سازی‌شده",
        "input_data": "۱. دادهٔ ورودی",
        "input_mode": "حالت ورودی",
        "mode_synthetic": "کارخانهٔ شبیه‌سازی‌شده",
        "mode_manual": "JSON دستی",
        "generate_hint": "برای شروع، **بارگذاری دادهٔ کارخانهٔ شبیه‌سازی‌شده** را بزنید.",
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
            "2. In **Kiln advisor**, load the demo data and press **Get advice**.\n"
            "3. Read the recommendation, its forecast intervals and the reasons; send feedback.\n"
            "4. In **Mill schedule**, see the power saving against today's rule.\n"
            "5. Try **Free lime**, **Equipment** and **Shift report** too.\n\n"
            "_Advisory only: the system suggests; it never writes to the DCS._"
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
        "demo_end_row": "Demo end row",
        "start_date": "Start date",
        "tags": "Tags",
        "use_sample": "Use sample payload",
        "reset_payload": "Reset payload",
        "select_one_tag": "Select at least one tag.",
        "load_demo": "Load simulated plant data",
        "input_data": "1. Input data",
        "input_mode": "Input mode",
        "mode_synthetic": "Simulated plant",
        "mode_manual": "Manual JSON",
        "generate_hint": "Press **Load simulated plant data** to start.",
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


# Strings of the operational tabs (merged into TRANSLATIONS below).
_TAB_STRINGS: Dict[str, Dict[str, str]] = {
    "fa": {
        "tab_advisor": "دستیار کوره",
        "tab_mills": "زمان‌بندی آسیاب",
        "tab_quality": "آهک آزاد",
        "tab_equipment": "تجهیزات",
        "tab_shift": "گزارش شیفت",
        "tab_lab": "آزمایشگاه مدل",
        "simulated_note": "دادهٔ دمو از کارخانهٔ شبیه‌سازی‌شده است؛ برای تصمیم مهندسی استفاده نکنید.",
        "run_advice": "دریافت توصیه",
        "status": "وضعیت",
        "status_recommend": "توصیهٔ تغییر",
        "status_keep_current": "تنظیمات فعلی را نگه دارید",
        "status_restore_limits": "بازگرداندن به محدودهٔ مجاز",
        "status_no_safe_candidate": "گزینهٔ ایمن وجود ندارد",
        "lever": "اهرم",
        "current": "فعلی",
        "proposed": "پیشنهادی",
        "improvement": "بهبود (kcal/kg)",
        "outcomes": "پیش‌بینی متغیرهای حفاظتی",
        "savings_day": "صرفه‌جویی روزانه",
        "why": "چرا؟",
        "feedback": "بازخورد اپراتور",
        "operator": "نام اپراتور",
        "decision": "تصمیم",
        "accepted": "پذیرفتم",
        "rejected": "رد کردم",
        "modified": "با تغییر اجرا کردم",
        "reason": "دلیل",
        "send_feedback": "ثبت بازخورد",
        "feedback_saved": "بازخورد ثبت شد.",
        "acceptance": "نرخ پذیرش",
        "run_schedule": "محاسبهٔ برنامهٔ بهینه",
        "hours": "افق (ساعت)",
        "grid_price": "قیمت برق سهمیه (تومان/kWh)",
        "peak_price": "قیمت برق ساعت اوج (تومان/kWh)",
        "free_price": "قیمت برق آزاد (تومان/kWh)",
        "no_quota_hours": "ساعت‌های بدون سهمیه (مثلاً 19-23)",
        "saving": "صرفه‌جویی نسبت به روش فعلی",
        "free_avoided": "برق آزاد اجتناب‌شده (kWh)",
        "run_free_lime": "تخمین آهک آزاد",
        "free_lime_now": "آهک آزاد تخمینی",
        "lab_due": "نتیجهٔ آزمایشگاه این کلینکر",
        "run_condition": "بررسی وضعیت تجهیزات",
        "run_shift": "ساخت گزارش شیفت",
        "n_alarm": "آلارم",
        "n_warning": "هشدار",
    },
    "en": {
        "tab_advisor": "Kiln advisor",
        "tab_mills": "Mill schedule",
        "tab_quality": "Free lime",
        "tab_equipment": "Equipment",
        "tab_shift": "Shift report",
        "tab_lab": "Model lab",
        "simulated_note": "Demo data comes from the simulated plant; never use it for engineering decisions.",
        "run_advice": "Get advice",
        "status": "Status",
        "status_recommend": "Change recommended",
        "status_keep_current": "Keep current settings",
        "status_restore_limits": "Restore limits",
        "status_no_safe_candidate": "No safe candidate",
        "lever": "Lever",
        "current": "Current",
        "proposed": "Proposed",
        "improvement": "Improvement (kcal/kg)",
        "outcomes": "Guarded variables (forecast)",
        "savings_day": "Savings per day",
        "why": "Why?",
        "feedback": "Operator feedback",
        "operator": "Operator",
        "decision": "Decision",
        "accepted": "Accepted",
        "rejected": "Rejected",
        "modified": "Applied with changes",
        "reason": "Reason",
        "send_feedback": "Send feedback",
        "feedback_saved": "Feedback saved.",
        "acceptance": "Acceptance rate",
        "run_schedule": "Optimize schedule",
        "hours": "Horizon (h)",
        "grid_price": "Quota power price (per kWh)",
        "peak_price": "Peak-hour price (per kWh)",
        "free_price": "Free-market price (per kWh)",
        "no_quota_hours": "Hours without quota (e.g. 19-23)",
        "saving": "Saving vs today's rule",
        "free_avoided": "Free-market kWh avoided",
        "run_free_lime": "Estimate free lime",
        "free_lime_now": "Estimated free lime",
        "lab_due": "Lab result for this clinker due",
        "run_condition": "Check equipment",
        "run_shift": "Build shift report",
        "n_alarm": "Alarms",
        "n_warning": "Warnings",
    },
}
for _lang, _strings in _TAB_STRINGS.items():
    TRANSLATIONS[_lang].update(_strings)


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
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> ApiResult:
    """Perform an HTTP request and normalize the outcome into an ApiResult."""
    try:
        response = requests.request(method, url, json=payload, timeout=timeout)
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
def call_demo_window(base_url: str, rows: int, end_row: int, step: int = 1) -> ApiResult:
    """Call ``GET /v1/demo/window`` (simulated demo plant)."""
    return _request(
        "GET", f"{base_url}{DEMO_PATH}?rows={rows}&end_row={end_row}&step={step}"
    )


def window_only(payload: Dict[str, Any]) -> Dict[str, Any]:
    """The ``{timestamps, signals}`` part of a demo payload."""
    return {"timestamps": payload["timestamps"], "signals": payload["signals"]}


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
        cols = st.columns(3)
        end_row = int(cols[0].number_input(t("demo_end_row"), 60, 4032, 3450, step=12))
        rows = int(cols[1].number_input(t("rows"), 20, 600, 60, step=10))
        if cols[2].button(t("load_demo"), use_container_width=True):
            result = call_demo_window(st.session_state["base_url"], rows, end_row)
            if result.ok:
                data = result.data
                data["current_setpoints"] = {
                    tag: data["signals"][tag][-1]
                    for tag in DEFAULT_ADJUSTABLE_TAGS
                    if tag in data["signals"]
                }
                st.session_state["payload"] = data
                payload = data
            else:
                st.error(result.error)
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
        col1.number_input(t("desired_target"), value=1460.0, step=1.0)
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
# Operational tabs
# --------------------------------------------------------------------------- #
def _demo_controls(key: str, rows: int, end_row: int, step: int = 1) -> Optional[Dict[str, Any]]:
    """Load a demo window into ``st.session_state[key]`` and return it."""
    cols = st.columns(2)
    end = int(cols[0].number_input(t("demo_end_row"), 60, 4032, end_row, step=12, key=f"{key}_end"))
    if cols[1].button(t("load_demo"), key=f"{key}_load", use_container_width=True):
        result = call_demo_window(st.session_state["base_url"], rows, end, step)
        if result.ok:
            st.session_state[key] = result.data
        else:
            st.error(result.error)
    data = st.session_state.get(key)
    if data and data.get("simulated"):
        st.caption(t("simulated_note"))
    return data


def _render_advisor_tab(base_url: str) -> None:
    data = _demo_controls("advisor_window", rows=60, end_row=3450)
    if data is None:
        st.info(t("generate_hint"))
        return
    if st.button(t("run_advice"), type="primary"):
        result = _request("POST", f"{base_url}{ADVISE_PATH}", window_only(data), LONG_TIMEOUT_SECONDS)
        if result.ok:
            st.session_state["advice"] = result.data
        else:
            st.error(result.error)
    advice = st.session_state.get("advice")
    if not advice:
        return
    lang = st.session_state.get("lang", DEFAULT_LANG)
    status = advice["status"]
    show = {"recommend": st.success, "keep_current": st.info}.get(status, st.warning)
    show(f"{t('status')}: {t('status_' + status)}")
    rows = [
        {
            t("lever"): lever,
            t("current"): round(value, 2),
            t("proposed"): round((advice.get("proposed") or {}).get(lever, value), 2),
        }
        for lever, value in advice["current"].items()
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    cols = st.columns(3)
    cols[0].metric(t("improvement"), f"{advice.get('improvement') or 0:.1f}")
    savings = advice.get("savings") or {}
    if savings:
        cols[1].metric(t("savings_day"), f"{savings.get('heat_gcal_per_day', 0):.1f} Gcal")
        cols[2].metric("Mazut", f"{savings.get('mazut_l_per_day', 0):,.0f} L")
    outcomes = advice.get("proposed_outcomes") or advice.get("baseline_outcomes") or []
    if outcomes:
        st.subheader(t("outcomes"))
        st.dataframe(
            pd.DataFrame(outcomes)[["tag", "point", "lower", "upper", "margin", "ok"]].round(1),
            use_container_width=True,
            hide_index=True,
        )
    with st.expander(t("why"), expanded=True):
        for line in advice.get("explanation_fa" if lang == "fa" else "explanation", []):
            st.markdown(f"- {line}")

    if status in ("recommend", "restore_limits"):
        st.subheader(t("feedback"))
        reasons = _request("GET", f"{base_url}{REASONS_PATH}")
        reason_codes = list(reasons.data) if reasons.ok else ["other"]
        with st.form("feedback_form"):
            operator = st.text_input(t("operator"))
            decision = st.radio(
                t("decision"), ["accepted", "rejected", "modified"], format_func=t, horizontal=True
            )
            reason = st.selectbox(
                t("reason"),
                [None] + reason_codes,
                format_func=lambda c: "—" if c is None else reasons.data[c][lang] if reasons.ok else c,
            )
            if st.form_submit_button(t("send_feedback")):
                body = {
                    "advice_id": advice["advice_id"],
                    "decision": decision,
                    "operator": operator or "unknown",
                    "reason": reason,
                }
                sent = _request("POST", f"{base_url}{FEEDBACK_PATH}", body)
                st.success(t("feedback_saved")) if sent.ok else st.error(sent.error)
        stats = _request("GET", f"{base_url}{STATS_PATH}")
        if stats.ok and stats.data.get("acceptance_rate") is not None:
            st.metric(t("acceptance"), f"{100 * stats.data['acceptance_rate']:.0f}%")


def _parse_hours(text: str) -> List[int]:
    text = text.strip()
    if not text:
        return []
    lo, _, hi = text.partition("-")
    return list(range(int(lo), int(hi or lo)))


def _render_mill_tab(base_url: str) -> None:
    cols = st.columns(4)
    horizon = int(cols[0].number_input(t("hours"), 12, 168, 48, step=12))
    grid = float(cols[1].number_input(t("grid_price"), 0.0, value=700.0))
    peak = float(cols[2].number_input(t("peak_price"), 0.0, value=1500.0))
    free = float(cols[3].number_input(t("free_price"), 0.0, value=4000.0))
    blocked = _parse_hours(st.text_input(t("no_quota_hours"), "19-23"))
    if not st.button(t("run_schedule"), type="primary"):
        return
    hours = [h % 24 for h in range(horizon)]
    payload = {
        "mills": [
            {"name": "Mill 1", "rated_kw": 4500, "rated_tph": 105, "start_cost": 200000},
            {"name": "Mill 2", "rated_kw": 4600, "rated_tph": 96, "start_cost": 200000},
        ],
        "slot_hours": 1,
        "quota_kw": [0 if h in blocked else 9500 for h in hours],
        "grid_price": [peak if 17 <= h < 23 else grid for h in hours],
        "free_market_price": [free] * horizon,
        "free_market_cap_kw": [4600] * horizon,
        "dispatch_tph": [170 if 7 <= h < 19 else 40 for h in hours],
        "silo_initial_t": 6000,
        "silo_min_t": 2000,
        "silo_max_t": 12000,
        "baseline_start_below": [0.55, 0.45],
        "baseline_stop_above": [0.7, 0.65],
    }
    result = _request("POST", f"{base_url}{MILL_PATH}", payload, LONG_TIMEOUT_SECONDS)
    if not result.ok:
        st.error(result.error)
        return
    comparison = result.data["comparison"]
    cols = st.columns(2)
    cols[0].metric(t("saving"), f"{comparison['cost_saving_pct']:.1f}%")
    cols[1].metric(t("free_avoided"), f"{comparison['free_market_kwh_avoided']:,.0f}")
    slots = pd.DataFrame(result.data["optimized"]["slots"])
    st.bar_chart(slots[["Mill 1", "Mill 2"]])
    st.line_chart(slots[["silo_t"]])


def _render_quality_tab(base_url: str) -> None:
    data = _demo_controls("quality_window", rows=72, end_row=3600)
    if data is None or not st.button(t("run_free_lime"), type="primary"):
        return
    result = _request("POST", f"{base_url}{FREE_LIME_PATH}", window_only(data), LONG_TIMEOUT_SECONDS)
    if not result.ok:
        st.error(result.error)
        return
    body = result.data
    cols = st.columns(2)
    cols[0].metric(
        t("free_lime_now"),
        f"{body['free_lime_pct']:.2f} %",
        help=f"90%: {body['lower']:.2f}–{body['upper']:.2f}",
    )
    cols[1].metric(t("lab_due"), body["lab_due_at"].replace("T", " ")[:16])
    series = pd.DataFrame(body["series"]).set_index("timestamp")
    st.line_chart(series)


def _render_equipment_tab(base_url: str) -> None:
    data = _demo_controls("equipment_window", rows=1344, end_row=4032, step=3)
    if data is None or not st.button(t("run_condition"), type="primary"):
        return
    body = {"window": window_only(data), "reference_end": data["timestamps"][len(data["timestamps"]) // 2]}
    result = _request("POST", f"{base_url}{CONDITION_PATH}", body, LONG_TIMEOUT_SECONDS)
    if not result.ok:
        st.error(result.error)
        return
    cols = st.columns(2)
    cols[0].metric(t("n_alarm"), result.data["n_alarm"])
    cols[1].metric(t("n_warning"), result.data["n_warning"])
    lang = st.session_state.get("lang", DEFAULT_LANG)
    for finding in result.data["findings"]:
        if finding["severity"] == "ok":
            continue
        text = finding["message_fa"] if lang == "fa" else finding["message"]
        (st.error if finding["severity"] == "alarm" else st.warning)(text)


def _render_shift_tab(base_url: str) -> None:
    data = _demo_controls("shift_window", rows=120, end_row=1600)
    if data is None or not st.button(t("run_shift"), type="primary"):
        return
    body = {"window": window_only(data), "lab": data.get("lab"), "at": data["timestamps"][-30]}
    result = _request("POST", f"{base_url}{SHIFT_PATH}", body, LONG_TIMEOUT_SECONDS)
    if not result.ok:
        st.error(result.error)
        return
    lang = st.session_state.get("lang", DEFAULT_LANG)
    st.markdown(result.data["report_fa" if lang == "fa" else "report_en"])


def _render_model_lab_tab(base_url: str) -> None:
    payload = _render_input_section()
    if payload is None:
        return
    left, right = st.columns(2)
    with left:
        _render_prediction_panel(base_url, payload)
    with right:
        _render_recommendation_panel(base_url, payload)


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
    st.session_state["base_url"] = base_url

    tabs = st.tabs(
        [t("tab_advisor"), t("tab_mills"), t("tab_quality"), t("tab_equipment"), t("tab_shift"), t("tab_lab")]
    )
    renderers = (
        _render_advisor_tab,
        _render_mill_tab,
        _render_quality_tab,
        _render_equipment_tab,
        _render_shift_tab,
        _render_model_lab_tab,
    )
    for tab, render in zip(tabs, renderers):
        with tab:
            render(base_url)


if __name__ == "__main__":
    main()
