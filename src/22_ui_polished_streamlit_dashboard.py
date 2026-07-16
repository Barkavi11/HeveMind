from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

import altair as alt
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="HeveMind",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# APPLICATION CONFIGURATION
# ============================================================
API_BASE_URL = os.getenv(
    "HEVEMIND_API_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

API_KEY = os.getenv(
    "HEVEMIND_API_KEY",
    "",
).strip()

REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "HEVEMIND_DASHBOARD_TIMEOUT",
        "15",
    )
)

DEFAULT_PAGE_SIZE = 100

DECISION_ORDER = [
    "Insufficient Evidence",
    "High Risk",
    "Engineering Review",
    "Low Risk",
]

DECISION_CLASS = {
    "Insufficient Evidence": "decision-insufficient",
    "High Risk": "decision-high",
    "Engineering Review": "decision-review",
    "Low Risk": "decision-low",
}

DECISION_SCORE = {
    "Insufficient Evidence": 4,
    "High Risk": 3,
    "Engineering Review": 2,
    "Low Risk": 1,
}

DECISION_COLORS = {
    "Low Risk": "#1b7f5b",
    "Engineering Review": "#d89a1d",
    "High Risk": "#b43b3b",
    "Insufficient Evidence": "#6f4f9c",
}


# ============================================================
# STYLING
# ============================================================
CUSTOM_CSS = """
<style>
    :root {
        --navy-950: #071426;
        --navy-900: #0b1f36;
        --navy-800: #123252;
        --navy-700: #17456f;
        --blue-600: #1f6aa5;
        --blue-500: #2c7fbe;
        --blue-100: #dcecf7;
        --slate-900: #172230;
        --slate-700: #405267;
        --slate-500: #708196;
        --slate-300: #c8d2dc;
        --slate-200: #dce3e9;
        --slate-100: #edf2f6;
        --white: #ffffff;
        --green: #1b7f5b;
        --green-bg: #e6f4ed;
        --amber: #a96b00;
        --amber-bg: #fff3d9;
        --red: #b43b3b;
        --red-bg: #fdeaea;
        --purple: #6f4f9c;
        --purple-bg: #f0eafb;
    }

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f5f8fb 0%,
                #edf3f7 100%
            );
        color: var(--slate-900);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                var(--navy-950) 0%,
                var(--navy-900) 100%
            );
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    [data-testid="stSidebar"] * {
        color: #f4f8fb;
    }

    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stRadio label {
        color: #dbe8f2 !important;
        font-weight: 600;
    }

    [data-testid="stSidebar"] input {
        color: var(--slate-900) !important;
    }

    [data-testid="stSidebar"] [data-baseweb="select"] * {
        color: var(--slate-900) !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.3rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: var(--navy-950);
        letter-spacing: -0.02em;
    }

    .app-header {
        background:
            linear-gradient(
                110deg,
                var(--navy-950) 0%,
                var(--navy-800) 68%,
                var(--blue-600) 100%
            );
        border-radius: 18px;
        padding: 1.45rem 1.65rem;
        margin-bottom: 1.15rem;
        box-shadow: 0 14px 34px rgba(7, 20, 38, 0.16);
    }

    .app-header-title {
        color: var(--white);
        font-size: 2rem;
        font-weight: 760;
        line-height: 1.15;
        margin: 0;
    }

    .app-header-subtitle {
        color: #d8e8f4;
        font-size: 0.98rem;
        margin-top: 0.45rem;
        margin-bottom: 0;
    }

    .sidebar-brand {
        padding: 0.75rem 0.25rem 1.1rem 0.25rem;
    }

    .sidebar-brand-title {
        color: #ffffff;
        font-size: 1.55rem;
        font-weight: 760;
        line-height: 1.1;
    }

    .sidebar-brand-subtitle {
        color: #a9c4d8;
        font-size: 0.82rem;
        margin-top: 0.35rem;
        line-height: 1.35;
    }

    .status-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.65rem;
        margin-bottom: 1rem;
    }

    .status-chip {
        border-radius: 999px;
        padding: 0.38rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 650;
        border: 1px solid var(--slate-200);
        background: var(--white);
        color: var(--slate-700);
    }

    .status-chip-good {
        color: var(--green);
        border-color: #b9ddcd;
        background: var(--green-bg);
    }

    .metric-card {
        min-height: 150px;
        height: 100%;
        transition: transform 0.16s ease, box-shadow 0.16s ease;
        background: rgba(255, 255, 255, 0.96);
        border: 1px solid var(--slate-200);
        border-radius: 15px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(21, 48, 75, 0.07);
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(21, 48, 75, 0.11);
    }

    .metric-label {
        color: var(--slate-500);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.035em;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--navy-950);
        font-size: 1.62rem;
        overflow-wrap: anywhere;
        font-weight: 760;
        line-height: 1.15;
        margin-top: 0.42rem;
    }

    .metric-note {
        color: var(--slate-700);
        font-size: 0.8rem;
        margin-top: 0.42rem;
        line-height: 1.35;
    }

    .panel {
        display: none;
    }

    .panel-title {
        color: var(--navy-950);
        font-size: 1.06rem;
        font-weight: 730;
        margin-bottom: 0.4rem;
    }

    .panel-subtitle {
        color: var(--slate-500);
        font-size: 0.84rem;
        margin-bottom: 0.85rem;
        line-height: 1.4;
    }

    .decision-banner {
        border-radius: 15px;
        padding: 1rem 1.15rem;
        margin-bottom: 1rem;
        border: 1px solid;
    }

    .decision-banner-title {
        font-size: 1.2rem;
        font-weight: 760;
        margin: 0;
    }

    .decision-banner-note {
        font-size: 0.86rem;
        margin-top: 0.32rem;
        line-height: 1.4;
    }

    .decision-low {
        background: var(--green-bg);
        border-color: #b6dbc9;
        color: var(--green);
    }

    .decision-review {
        background: var(--amber-bg);
        border-color: #efd49b;
        color: var(--amber);
    }

    .decision-high {
        background: var(--red-bg);
        border-color: #efbcbc;
        color: var(--red);
    }

    .decision-insufficient {
        background: var(--purple-bg);
        border-color: #d4c2ee;
        color: var(--purple);
    }

    .evidence-note {
        border-left: 4px solid var(--blue-600);
        background: #eff7fc;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: var(--slate-700);
        font-size: 0.9rem;
        line-height: 1.55;
        margin-bottom: 0.8rem;
    }

    .warning-note {
        border-left: 4px solid var(--amber);
        background: var(--amber-bg);
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #6b4c11;
        font-size: 0.87rem;
        line-height: 1.5;
    }

    .rank-row {
        background: #f8fafc;
        border: 1px solid var(--slate-200);
        border-radius: 11px;
        padding: 0.72rem 0.82rem;
        margin-bottom: 0.55rem;
    }

    .rank-title {
        color: var(--navy-950);
        font-weight: 720;
    }

    .rank-detail {
        color: var(--slate-700);
        font-size: 0.82rem;
        margin-top: 0.25rem;
        line-height: 1.38;
    }

    .small-muted {
        color: var(--slate-500);
        font-size: 0.79rem;
        line-height: 1.4;
    }

    .divider {
        height: 1px;
        background: var(--slate-200);
        margin: 0.8rem 0 1rem 0;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--slate-200);
        border-radius: 12px;
        overflow: hidden;
        background: var(--white);
        box-shadow: 0 6px 18px rgba(21, 48, 75, 0.05);
    }

    [data-testid="stSelectbox"] > div > div,
    [data-testid="stTextInput"] input,
    [data-testid="stNumberInput"] input {
        background: #ffffff !important;
        color: var(--slate-900) !important;
        border-color: var(--slate-200) !important;
    }

    [data-testid="stSlider"] [role="slider"] {
        background: var(--blue-600) !important;
    }

    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #edf2f6;
    }

    ::-webkit-scrollbar-thumb {
        background: #aebdca;
        border-radius: 10px;
        border: 2px solid #edf2f6;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #879aaa;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.45rem;
        border-bottom: 1px solid var(--slate-200);
    }

    .stTabs [data-baseweb="tab"] {
        background: var(--white);
        border: 1px solid var(--slate-200);
        border-radius: 9px 9px 0 0;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .stTabs [aria-selected="true"] {
        background: #e8f2f9 !important;
        color: var(--navy-950) !important;
        border-color: #b8d2e5 !important;
        border-bottom: 3px solid var(--blue-600) !important;
    }

    .stTabs [aria-selected="true"] p {
        color: var(--navy-950) !important;
        font-weight: 700 !important;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
    }

    footer {
        visibility: hidden;
    }
</style>
"""

st.markdown(
    CUSTOM_CSS,
    unsafe_allow_html=True,
)


# ============================================================
# API CLIENT
# ============================================================
def api_headers() -> dict[str, str]:
    headers = {
        "Accept": "application/json",
    }

    if API_KEY:
        headers[
            "X-API-Key"
        ] = API_KEY

    return headers


class HeveMindAPIError(RuntimeError):
    pass


def api_get(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> Any:
    url = (
        f"{API_BASE_URL}{endpoint}"
    )

    try:
        response = requests.get(
            url,
            headers=api_headers(),
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )

    except requests.RequestException as error:
        raise HeveMindAPIError(
            f"Unable to connect to the HeveMind API at "
            f"{API_BASE_URL}. Confirm Script 19 is running. "
            f"Technical detail: {error}"
        ) from error

    try:
        payload = response.json()

    except ValueError:
        payload = {
            "error": {
                "message": response.text,
            }
        }

    if not response.ok:
        error_payload = (
            payload.get(
                "error",
                payload,
            )
            if isinstance(
                payload,
                dict,
            )
            else payload
        )

        if isinstance(
            error_payload,
            dict,
        ):
            message = error_payload.get(
                "message",
                str(
                    error_payload
                ),
            )

        else:
            message = str(
                error_payload
            )

        raise HeveMindAPIError(
            f"API request failed with HTTP "
            f"{response.status_code}: {message}"
        )

    return payload


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_health() -> dict[str, Any]:
    return api_get(
        "/health"
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_readiness() -> dict[str, Any]:
    return api_get(
        "/ready"
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_metadata() -> dict[str, Any]:
    return api_get(
        "/metadata"
    )


@st.cache_data(
    ttl=30,
    show_spinner=False,
)
def get_decision_summary() -> dict[str, Any]:
    return api_get(
        "/wafers/summary/decisions"
    )


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_wafers(
    decision: str | None = None,
    search: str | None = None,
    sort_by: str = "risk",
    sort_order: str = "desc",
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "page": 1,
        "page_size": DEFAULT_PAGE_SIZE,
        "sort_by": sort_by,
        "sort_order": sort_order,
    }

    if decision:
        params[
            "decision"
        ] = decision

    if search:
        params[
            "search"
        ] = search

    first_page = api_get(
        "/wafers",
        params=params,
    )

    items = list(
        first_page.get(
            "items",
            [],
        )
    )

    pagination = first_page.get(
        "pagination",
        {},
    )

    total_pages = int(
        pagination.get(
            "total_pages",
            1,
        )
        or 1
    )

    for page in range(
        2,
        total_pages + 1,
    ):
        page_params = dict(
            params
        )

        page_params[
            "page"
        ] = page

        next_page = api_get(
            "/wafers",
            params=page_params,
        )

        items.extend(
            next_page.get(
                "items",
                [],
            )
        )

    return {
        "items": items,
        "total_records": len(
            items
        ),
    }


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_wafer_report(
    wafer_id: str,
) -> dict[str, Any]:
    return api_get(
        f"/wafers/{quote(wafer_id, safe='')}"
    )


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_explanation(
    wafer_id: str,
) -> dict[str, Any]:
    return api_get(
        f"/wafers/{quote(wafer_id, safe='')}/explanation"
    )


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_history(
    wafer_id: str,
) -> dict[str, Any]:
    return api_get(
        f"/wafers/{quote(wafer_id, safe='')}/history"
    )


@st.cache_data(
    ttl=20,
    show_spinner=False,
)
def get_investigation(
    wafer_id: str,
) -> dict[str, Any]:
    return api_get(
        f"/wafers/{quote(wafer_id, safe='')}/investigation"
    )


# ============================================================
# DISPLAY UTILITIES
# ============================================================
def format_percentage(
    value: Any,
    digits: int = 1,
) -> str:
    try:
        numeric = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return "Unavailable"

    return (
        f"{numeric:.{digits}%}"
    )


def display_value(
    value: Any,
    fallback: str = "Unavailable",
) -> str:
    if value is None:
        return fallback

    text = str(
        value
    ).strip()

    if text.lower() in {
        "",
        "nan",
        "none",
    }:
        return fallback

    return text



def format_sensor_name(
    value: Any,
) -> str:
    text = display_value(
        value
    )

    if text.startswith(
        "sensor_"
    ):
        return (
            "Sensor "
            + text.removeprefix(
                "sensor_"
            )
        )

    return text



def render_metric_card(
    label: str,
    value: str,
    note: str,
) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_panel_header(
    title: str,
    subtitle: str,
) -> None:
    st.markdown(
        f"""
        <div class="panel-title">{title}</div>
        <div class="panel-subtitle">{subtitle}</div>
        """,
        unsafe_allow_html=True,
    )


def render_decision_banner(
    decision: str,
    evidence_agreement: str,
) -> None:
    css_class = DECISION_CLASS.get(
        decision,
        "decision-review",
    )

    st.markdown(
        f"""
        <div class="decision-banner {css_class}">
            <div class="decision-banner-title">
                {display_value(decision)}
            </div>
            <div class="decision-banner-note">
                Evidence status:
                {display_value(evidence_agreement)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def records_to_frame(
    records: list[dict[str, Any]],
) -> pd.DataFrame:
    if not records:
        return pd.DataFrame()

    return pd.DataFrame(
        records
    )


def build_decision_frame(
    payload: dict[str, Any],
) -> pd.DataFrame:
    rows = payload.get(
        "decisions",
        [],
    )

    dataframe = pd.DataFrame(
        rows
    )

    if dataframe.empty:
        return dataframe

    dataframe[
        "decision"
    ] = pd.Categorical(
        dataframe[
            "decision"
        ],
        categories=DECISION_ORDER,
        ordered=True,
    )

    return dataframe.sort_values(
        "decision"
    ).reset_index(
        drop=True
    )


def render_api_failure(
    error: Exception,
) -> None:
    st.error(
        str(
            error
        )
    )

    st.markdown(
        """
        Confirm the following:

        1. Script 19 is running at `http://127.0.0.1:8000`.
        2. The dashboard uses the same API key as the API, when enabled.
        3. Scripts 12 to 18 have generated their required outputs.
        """
    )


def build_priority_score(
    row: pd.Series,
) -> float:
    decision_score = float(
        DECISION_SCORE.get(
            str(
                row.get(
                    "uncertainty_adjusted_decision"
                )
            ),
            0,
        )
    )

    probability = float(
        row.get(
            "calibrated_failure_probability",
            0.0,
        )
        or 0.0
    )

    uncertainty = float(
        row.get(
            "combined_uncertainty",
            0.0,
        )
        or 0.0
    )

    confidence = float(
        row.get(
            "prediction_confidence",
            0.0,
        )
        or 0.0
    )

    return (
        0.55
        * decision_score
        + 0.20
        * probability
        + 0.20
        * uncertainty
        + 0.05
        * (
            1.0
            - confidence
        )
    )


# ============================================================
# SIDEBAR AND INITIAL DATA
# ============================================================
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">HeveMind</div>
        <div class="sidebar-brand-subtitle">
            Explainable semiconductor decision support
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    health_payload = get_health()
    readiness_payload = get_readiness()
    metadata_payload = get_metadata()
    decision_payload = get_decision_summary()
    all_wafers_payload = get_wafers(
        sort_by="risk",
        sort_order="desc",
    )

except HeveMindAPIError as error:
    st.markdown(
        """
        <div class="app-header">
            <div class="app-header-title">HeveMind</div>
            <div class="app-header-subtitle">
                Semiconductor decision-support dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_api_failure(
        error
    )

    st.stop()


api_healthy = (
    health_payload.get(
        "status"
    )
    == "healthy"
)

api_ready = bool(
    readiness_payload.get(
        "ready",
        False,
    )
)

available_reports = int(
    health_payload.get(
        "available_wafer_reports",
        0,
    )
)

all_wafers_df = records_to_frame(
    all_wafers_payload.get(
        "items",
        [],
    )
)

if not all_wafers_df.empty:
    all_wafers_df[
        "dashboard_triage_score"
    ] = all_wafers_df.apply(
        build_priority_score,
        axis=1,
    )

st.sidebar.markdown(
    f"""
    <div class="status-strip">
        <span class="status-chip status-chip-good">
            API {"Healthy" if api_healthy else "Degraded"}
        </span>
        <span class="status-chip">
            {available_reports} reports
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

navigation = st.sidebar.radio(
    "Workspace",
    options=[
        "Executive Overview",
        "Wafer Review",
        "Engineering Queue",
        "Sensor Intelligence",
        "System Governance",
    ],
)

st.sidebar.markdown(
    '<div class="divider"></div>',
    unsafe_allow_html=True,
)

st.sidebar.caption(
    "Validated held-out reports only. "
    "Raw-record production inference is not enabled."
)


# ============================================================
# APPLICATION HEADER
# ============================================================
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-title">
            HeveMind Semiconductor Decision-Support Platform
        </div>
        <div class="app-header-subtitle">
            Calibrated risk, uncertainty, historical evidence,
            explainability and sensor investigation priorities
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="status-strip">
        <span class="status-chip status-chip-good">
            Service operational
        </span>
        <span class="status-chip">
            API version {display_value(metadata_payload.get("api_version"))}
        </span>
        <span class="status-chip">
            Report store {"ready" if api_ready else "not ready"}
        </span>
        <span class="status-chip">
            Raw inference disabled
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXECUTIVE OVERVIEW
# ============================================================
if navigation == "Executive Overview":
    st.subheader(
        "Validated Cohort Overview"
    )

    st.caption(
        "Executive view of the 314 held-out SECOM records currently "
        "served by the deployment API."
    )

    decision_df = build_decision_frame(
        decision_payload
    )

    total_records = int(
        decision_payload.get(
            "total_records",
            available_reports,
        )
    )

    high_risk_records = int(
        decision_df.loc[
            decision_df[
                "decision"
            ]
            == "High Risk",
            "records",
        ].sum()
    ) if not decision_df.empty else 0

    review_records = int(
        decision_df.loc[
            decision_df[
                "decision"
            ]
            == "Engineering Review",
            "records",
        ].sum()
    ) if not decision_df.empty else 0

    insufficient_records = int(
        decision_df.loc[
            decision_df[
                "decision"
            ]
            == "Insufficient Evidence",
            "records",
        ].sum()
    ) if not decision_df.empty else 0

    average_probability = (
        float(
            all_wafers_df[
                "calibrated_failure_probability"
            ].mean()
        )
        if not all_wafers_df.empty
        else 0.0
    )

    average_confidence = (
        float(
            all_wafers_df[
                "prediction_confidence"
            ].mean()
        )
        if not all_wafers_df.empty
        else 0.0
    )

    average_uncertainty = (
        float(
            all_wafers_df[
                "combined_uncertainty"
            ].mean()
        )
        if not all_wafers_df.empty
        else 0.0
    )

    metric_row_one = st.columns(
        4
    )

    with metric_row_one[0]:
        render_metric_card(
            "Validated reports",
            str(
                total_records
            ),
            "Held-out records exposed through the deployment API.",
        )

    with metric_row_one[1]:
        render_metric_card(
            "High-risk queue",
            str(
                high_risk_records
            ),
            "Records prioritised for engineering inspection.",
        )

    with metric_row_one[2]:
        render_metric_card(
            "Engineering review",
            str(
                review_records
            ),
            "Records requiring structured human review.",
        )

    with metric_row_one[3]:
        render_metric_card(
            "Insufficient evidence",
            str(
                insufficient_records
            ),
            "Records for which the system safely abstained.",
        )

    st.write("")

    metric_row_two = st.columns(
        3
    )

    with metric_row_two[0]:
        render_metric_card(
            "Mean failure risk",
            format_percentage(
                average_probability,
                1,
            ),
            "Average calibrated probability across validated reports.",
        )

    with metric_row_two[1]:
        render_metric_card(
            "Mean prediction confidence",
            format_percentage(
                average_confidence,
                1,
            ),
            "Uncertainty-derived confidence; this is not model accuracy.",
        )

    with metric_row_two[2]:
        render_metric_card(
            "Mean uncertainty",
            format_percentage(
                average_uncertainty,
                1,
            ),
            "Average combined epistemic, data and distributional uncertainty.",
        )

    st.write("")

    left_column, middle_column, right_column = st.columns(
        [
            0.9,
            1.2,
            0.9,
        ]
    )

    with left_column:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Risk Distribution",
            "Operational share of the retained four-level decision policy.",
        )

        if not decision_df.empty:
            donut_df = decision_df[
                [
                    "decision",
                    "records",
                ]
            ].copy()

            donut_chart = (
                alt.Chart(
                    donut_df
                )
                .mark_arc(
                    innerRadius=70,
                    outerRadius=120,
                )
                .encode(
                    theta=alt.Theta(
                        "records:Q"
                    ),
                    color=alt.Color(
                        "decision:N",
                        sort=DECISION_ORDER,
                        scale=alt.Scale(
                            domain=list(
                                DECISION_COLORS.keys()
                            ),
                            range=list(
                                DECISION_COLORS.values()
                            ),
                        ),
                        legend=alt.Legend(
                            title=None,
                            orient="bottom",
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "decision:N",
                            title="Decision",
                        ),
                        alt.Tooltip(
                            "records:Q",
                            title="Records",
                        ),
                    ],
                )
                .properties(
                    height=340,
                    background="#ffffff",
                )
            )

            st.altair_chart(
                donut_chart,
                use_container_width=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with middle_column:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Risk-Confidence Map",
            "Each point represents a wafer. Upper-right points combine higher failure probability and higher confidence.",
        )

        if not all_wafers_df.empty:
            scatter_chart = (
                alt.Chart(
                    all_wafers_df
                )
                .mark_circle(
                    size=72,
                    opacity=0.65,
                )
                .encode(
                    x=alt.X(
                        "calibrated_failure_probability:Q",
                        title="Calibrated failure probability",
                        scale=alt.Scale(
                            domain=[
                                0,
                                max(
                                    0.2,
                                    float(
                                        all_wafers_df[
                                            "calibrated_failure_probability"
                                        ].max()
                                    ),
                                ),
                            ]
                        ),
                    ),
                    y=alt.Y(
                        "prediction_confidence:Q",
                        title="Prediction confidence",
                        scale=alt.Scale(
                            domain=[
                                0,
                                1,
                            ]
                        ),
                    ),
                    color=alt.Color(
                        "uncertainty_adjusted_decision:N",
                        title="Decision",
                        scale=alt.Scale(
                            domain=list(
                                DECISION_COLORS.keys()
                            ),
                            range=list(
                                DECISION_COLORS.values()
                            ),
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "wafer_id:N",
                            title="Wafer",
                        ),
                        alt.Tooltip(
                            "calibrated_failure_probability:Q",
                            title="Failure probability",
                            format=".2%",
                        ),
                        alt.Tooltip(
                            "prediction_confidence:Q",
                            title="Confidence",
                            format=".2%",
                        ),
                        alt.Tooltip(
                            "combined_uncertainty:Q",
                            title="Uncertainty",
                            format=".2%",
                        ),
                        alt.Tooltip(
                            "uncertainty_adjusted_decision:N",
                            title="Decision",
                        ),
                    ],
                )
                .properties(
                    height=340,
                    background="#ffffff",
                )
            )

            st.altair_chart(
                scatter_chart,
                use_container_width=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Data Familiarity",
            "Distribution of records by OOD-derived familiarity category.",
        )

        if (
            not all_wafers_df.empty
            and "data_familiarity_band"
            in all_wafers_df.columns
        ):
            familiarity_df = (
                all_wafers_df[
                    "data_familiarity_band"
                ]
                .fillna(
                    "Unavailable"
                )
                .value_counts()
                .rename_axis(
                    "data_familiarity_band"
                )
                .reset_index(
                    name="records"
                )
            )

            familiarity_chart = (
                alt.Chart(
                    familiarity_df
                )
                .mark_bar(
                    cornerRadiusTopRight=5,
                    cornerRadiusBottomRight=5,
                )
                .encode(
                    x=alt.X(
                        "records:Q",
                        title="Records",
                    ),
                    y=alt.Y(
                        "data_familiarity_band:N",
                        title=None,
                        sort="-x",
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "data_familiarity_band:N",
                            title="Familiarity",
                        ),
                        alt.Tooltip(
                            "records:Q",
                            title="Records",
                        ),
                    ],
                )
                .properties(
                    height=340,
                    background="#ffffff",
                )
            )

            st.altair_chart(
                familiarity_chart,
                use_container_width=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    lower_left, lower_right = st.columns(
        [
            1.05,
            0.95,
        ]
    )

    with lower_left:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Probability and Uncertainty Distribution",
            "Distributional view of model output and reliability.",
        )

        if not all_wafers_df.empty:
            distribution_tabs = st.tabs(
                [
                    "Failure probability",
                    "Confidence",
                    "Uncertainty",
                ]
            )

            with distribution_tabs[0]:
                probability_histogram = (
                    alt.Chart(
                        all_wafers_df
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "calibrated_failure_probability:Q",
                            bin=alt.Bin(
                                maxbins=20
                            ),
                            title="Failure probability",
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Records",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "count():Q",
                                title="Records",
                            ),
                        ],
                    )
                    .properties(
                        height=300,
                    )
                )

                st.altair_chart(
                    probability_histogram,
                    use_container_width=True,
                )

            with distribution_tabs[1]:
                confidence_histogram = (
                    alt.Chart(
                        all_wafers_df
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "prediction_confidence:Q",
                            bin=alt.Bin(
                                maxbins=20
                            ),
                            title="Prediction confidence",
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Records",
                        ),
                    )
                    .properties(
                        height=300,
                    )
                )

                st.altair_chart(
                    confidence_histogram,
                    use_container_width=True,
                )

            with distribution_tabs[2]:
                uncertainty_histogram = (
                    alt.Chart(
                        all_wafers_df
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "combined_uncertainty:Q",
                            bin=alt.Bin(
                                maxbins=20
                            ),
                            title="Combined uncertainty",
                        ),
                        y=alt.Y(
                            "count():Q",
                            title="Records",
                        ),
                    )
                    .properties(
                        height=300,
                    )
                )

                st.altair_chart(
                    uncertainty_histogram,
                    use_container_width=True,
                )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with lower_right:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Top Engineering Priorities",
            "Display-only triage order using the retained decision, probability and uncertainty.",
        )

        if not all_wafers_df.empty:
            top_queue = (
                all_wafers_df.loc[
                    all_wafers_df[
                        "uncertainty_adjusted_decision"
                    ]
                    .isin(
                        [
                            "Insufficient Evidence",
                            "High Risk",
                            "Engineering Review",
                        ]
                    )
                ]
                .sort_values(
                    by="dashboard_triage_score",
                    ascending=False,
                )
                .head(
                    12
                )
                .copy()
            )

            display_queue = top_queue[
                [
                    "wafer_id",
                    "uncertainty_adjusted_decision",
                    "calibrated_failure_probability",
                    "prediction_confidence",
                    "combined_uncertainty",
                    "priority_1_sensor",
                ]
            ].copy()

            display_queue[
                "calibrated_failure_probability"
            ] = display_queue[
                "calibrated_failure_probability"
            ].map(
                lambda value: format_percentage(
                    value,
                    1,
                )
            )

            display_queue[
                "prediction_confidence"
            ] = display_queue[
                "prediction_confidence"
            ].map(
                lambda value: format_percentage(
                    value,
                    1,
                )
            )

            display_queue[
                "combined_uncertainty"
            ] = display_queue[
                "combined_uncertainty"
            ].map(
                lambda value: format_percentage(
                    value,
                    1,
                )
            )

            display_queue[
                "priority_1_sensor"
            ] = display_queue[
                "priority_1_sensor"
            ].map(
                format_sensor_name
            )

            display_queue.columns = [
                "Wafer",
                "Decision",
                "Failure risk",
                "Confidence",
                "Uncertainty",
                "Priority sensor",
            ]

            st.dataframe(
                display_queue,
                use_container_width=True,
                hide_index=True,
                height=385,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="warning-note">
            This dashboard supports engineering prioritisation. It does not
            authorise release, rework, scrap or maintenance actions without
            human review and local operational procedures.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# WAFER REVIEW
# ============================================================
elif navigation == "Wafer Review":
    st.subheader(
        "Wafer-Level Engineering Review"
    )

    st.caption(
        "Inspect one validated wafer across prediction, uncertainty, "
        "historical evidence and sensor investigation layers."
    )

    filter_columns = st.columns(
        [
            0.7,
            1.3,
        ]
    )

    with filter_columns[0]:
        decision_filter = st.selectbox(
            "Decision filter",
            options=[
                "All decisions",
                *DECISION_ORDER,
            ],
        )

    with filter_columns[1]:
        search_text = st.text_input(
            "Search wafer ID",
            placeholder="Enter part of a wafer identifier",
        )

    selected_decision = (
        None
        if decision_filter
        == "All decisions"
        else decision_filter
    )

    try:
        wafer_payload = get_wafers(
            decision=selected_decision,
            search=(
                search_text.strip()
                if search_text.strip()
                else None
            ),
            sort_by="risk",
            sort_order="desc",
        )

    except HeveMindAPIError as error:
        render_api_failure(
            error
        )
        st.stop()

    wafer_items = wafer_payload.get(
        "items",
        [],
    )

    if not wafer_items:
        st.warning(
            "No wafer records match the selected filters."
        )
        st.stop()

    wafer_ids = [
        str(
            item.get(
                "wafer_id"
            )
        )
        for item in wafer_items
    ]

    selected_wafer_id = st.selectbox(
        "Validated wafer report",
        options=wafer_ids,
        key="selected_wafer_id",
    )

    try:
        report = get_wafer_report(
            selected_wafer_id
        )

        explanation = get_explanation(
            selected_wafer_id
        )

        history = get_history(
            selected_wafer_id
        )

        investigation = get_investigation(
            selected_wafer_id
        )

    except HeveMindAPIError as error:
        render_api_failure(
            error
        )
        st.stop()

    primary = report.get(
        "primary_prediction",
        {},
    )

    reliability = report.get(
        "data_reliability",
        {},
    )

    evidence = report.get(
        "evidence_assessment",
        {},
    )

    decision = display_value(
        primary.get(
            "operational_decision"
        )
    )

    evidence_agreement = display_value(
        evidence.get(
            "evidence_agreement"
        )
    )

    render_decision_banner(
        decision,
        evidence_agreement,
    )

    metric_columns = st.columns(
        5
    )

    with metric_columns[0]:
        render_metric_card(
            "Failure probability",
            format_percentage(
                primary.get(
                    "calibrated_failure_probability"
                ),
                1,
            ),
            "Beta-calibrated primary model estimate.",
        )

    with metric_columns[1]:
        render_metric_card(
            "Prediction confidence",
            format_percentage(
                primary.get(
                    "prediction_confidence"
                ),
                1,
            ),
            "Confidence derived from the uncertainty engine.",
        )

    with metric_columns[2]:
        render_metric_card(
            "Combined uncertainty",
            format_percentage(
                primary.get(
                    "combined_uncertainty"
                ),
                1,
            ),
            "Higher values indicate lower certainty.",
        )

    with metric_columns[3]:
        render_metric_card(
            "Data familiarity",
            display_value(
                reliability.get(
                    "data_familiarity_band"
                )
            ),
            "Engineer-friendly interpretation of OOD evidence.",
        )

    with metric_columns[4]:
        render_metric_card(
            "Priority sensor",
            format_sensor_name(
                investigation.get(
                    "priority_1_sensor"
                )
            ),
            "First statistical measurement recommended for review.",
        )

    st.write("")

    tabs = st.tabs(
        [
            "Decision Evidence",
            "Sensor Attribution",
            "Historical Cases",
            "Investigation Priorities",
            "Technical Record",
        ]
    )

    with tabs[0]:
        left_column, right_column = st.columns(
            [
                1.15,
                0.85,
            ]
        )

        with left_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Engineering Recommendation",
                "Recommendation generated from the retained decision and transparent evidence rules.",
            )

            st.markdown(
                f"""
                <div class="evidence-note">
                    {display_value(evidence.get("recommended_action"))}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="small-muted">
                    {display_value(evidence.get("engineering_evidence_narrative"))}
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Reliability Context",
                "Supporting evidence that qualifies how much trust to place in the decision.",
            )

            reliability_table = pd.DataFrame(
                [
                    {
                        "Measure": "Model evidence",
                        "Value": display_value(
                            evidence.get(
                                "model_evidence_strength"
                            )
                        ),
                    },
                    {
                        "Measure": "Prediction reliability",
                        "Value": display_value(
                            evidence.get(
                                "prediction_reliability"
                            )
                        ),
                    },
                    {
                        "Measure": "Data confidence",
                        "Value": format_percentage(
                            reliability.get(
                                "data_confidence"
                            ),
                            1,
                        ),
                    },
                    {
                        "Measure": "Missing sensor rate",
                        "Value": format_percentage(
                            reliability.get(
                                "missing_sensor_rate"
                            ),
                            1,
                        ),
                    },
                    {
                        "Measure": "Abstention reason",
                        "Value": display_value(
                            reliability.get(
                                "abstention_reason"
                            ),
                            "Not applicable",
                        ),
                    },
                ]
            )

            st.dataframe(
                reliability_table,
                use_container_width=True,
                hide_index=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="warning-note">
                Evidence aggregation does not create a second failure
                probability and does not override the retained decision.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[1]:
        shap_records = explanation.get(
            "local_shap_contributions",
            [],
        )

        shap_df = records_to_frame(
            shap_records
        )

        left_column, right_column = st.columns(
            [
                1.05,
                0.95,
            ]
        )

        with left_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Local SHAP Attribution",
                "Features that contributed most strongly to this model output.",
            )

            if shap_df.empty:
                st.info(
                    "No local SHAP rows are available for this wafer."
                )

            else:
                preferred_columns = [
                    "rank",
                    "feature",
                    "feature_value",
                    "shap_contribution",
                    "absolute_shap_contribution",
                    "contribution_direction",
                ]

                shap_display = shap_df[
                    [
                        column
                        for column in preferred_columns
                        if column in shap_df.columns
                    ]
                ].copy()

                shap_display = shap_display.rename(
                    columns={
                        "rank": "Rank",
                        "feature": "Sensor",
                        "feature_value": "Observed value",
                        "shap_contribution": "SHAP contribution",
                        "absolute_shap_contribution": "Absolute contribution",
                        "contribution_direction": "Direction",
                    }
                )

                st.dataframe(
                    shap_display,
                    use_container_width=True,
                    hide_index=True,
                    height=400,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Contribution Profile",
                "Positive values increase modelled risk; negative values reduce it.",
            )

            if (
                not shap_df.empty
                and "feature" in shap_df.columns
                and "shap_contribution" in shap_df.columns
            ):
                shap_chart_df = shap_df.copy()

                shap_chart_df[
                    "direction"
                ] = shap_chart_df[
                    "shap_contribution"
                ].map(
                    lambda value: (
                        "Increases risk"
                        if float(
                            value
                        )
                        > 0
                        else "Reduces risk"
                    )
                )

                shap_chart = (
                    alt.Chart(
                        shap_chart_df
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "shap_contribution:Q",
                            title="SHAP contribution",
                        ),
                        y=alt.Y(
                            "feature:N",
                            sort="-x",
                            title=None,
                        ),
                        color=alt.Color(
                            "direction:N",
                            title=None,
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "feature:N",
                                title="Sensor",
                            ),
                            alt.Tooltip(
                                "feature_value:Q",
                                title="Observed value",
                            ),
                            alt.Tooltip(
                                "shap_contribution:Q",
                                title="Contribution",
                                format=".5f",
                            ),
                        ],
                    )
                    .properties(
                        height=400,
                    )
                )

                st.altair_chart(
                    shap_chart,
                    use_container_width=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="warning-note">
                {display_value(explanation.get("interpretation_limit"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[2]:
        metric_columns = st.columns(
            4
        )

        with metric_columns[0]:
            render_metric_card(
                "Failed neighbours",
                str(
                    history.get(
                        "failed_neighbour_count",
                        "Unavailable",
                    )
                ),
                "Observed failures among the ten nearest development records.",
            )

        with metric_columns[1]:
            render_metric_card(
                "Passed neighbours",
                str(
                    history.get(
                        "passed_neighbour_count",
                        "Unavailable",
                    )
                ),
                "Observed passes among retrieved development records.",
            )

        with metric_columns[2]:
            render_metric_card(
                "Weighted failure evidence",
                format_percentage(
                    history.get(
                        "historical_weighted_failure_rate"
                    ),
                    1,
                ),
                "Similarity-weighted historical outcome rate.",
            )

        with metric_columns[3]:
            render_metric_card(
                "Mean similarity",
                format_percentage(
                    history.get(
                        "mean_similarity"
                    ),
                    1,
                ),
                "Analytical proximity, not a probability.",
            )

        st.write("")

        neighbour_df = records_to_frame(
            history.get(
                "nearest_historical_records",
                [],
            )
        )

        left_column, right_column = st.columns(
            [
                1.05,
                0.95,
            ]
        )

        with left_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Nearest Historical Records",
                "Case-based evidence retrieved exclusively from development records.",
            )

            if neighbour_df.empty:
                st.info(
                    "No historical neighbour records are available."
                )

            else:
                selected_columns = [
                    "neighbour_rank",
                    "historical_wafer_id",
                    "historical_status",
                    "similarity_score",
                    "distance",
                ]

                neighbour_display = neighbour_df[
                    [
                        column
                        for column in selected_columns
                        if column in neighbour_df.columns
                    ]
                ].copy()

                neighbour_display = neighbour_display.rename(
                    columns={
                        "neighbour_rank": "Rank",
                        "historical_wafer_id": "Historical wafer",
                        "historical_status": "Outcome",
                        "similarity_score": "Similarity",
                        "distance": "Distance",
                    }
                )

                st.dataframe(
                    neighbour_display,
                    use_container_width=True,
                    hide_index=True,
                    height=395,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Neighbour Outcome Profile",
                "Outcome composition of the retrieved historical records.",
            )

            if (
                not neighbour_df.empty
                and "historical_status"
                in neighbour_df.columns
            ):
                outcome_df = (
                    neighbour_df[
                        "historical_status"
                    ]
                    .fillna(
                        "Unknown"
                    )
                    .value_counts()
                    .rename_axis(
                        "historical_status"
                    )
                    .reset_index(
                        name="records"
                    )
                )

                outcome_chart = (
                    alt.Chart(
                        outcome_df
                    )
                    .mark_bar()
                    .encode(
                        x=alt.X(
                            "records:Q",
                            title="Records",
                        ),
                        y=alt.Y(
                            "historical_status:N",
                            sort="-x",
                            title=None,
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "historical_status:N",
                                title="Outcome",
                            ),
                            alt.Tooltip(
                                "records:Q",
                                title="Records",
                            ),
                        ],
                    )
                    .properties(
                        height=395,
                    )
                )

                st.altair_chart(
                    outcome_chart,
                    use_container_width=True,
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            f"""
            <div class="warning-note">
                {display_value(history.get("interpretation_limit"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tabs[3]:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Ranked Sensor Investigation Priorities",
            "Statistical measurements to inspect first for this wafer.",
        )

        priority_records = investigation.get(
            "ranked_sensor_priorities",
            [],
        )

        priority_df = records_to_frame(
            priority_records
        )

        if priority_df.empty:
            st.info(
                "No sensor investigation priorities are available."
            )

        else:
            left_column, right_column = st.columns(
                [
                    1.05,
                    0.95,
                ]
            )

            with left_column:
                for record in priority_records:
                    rank = display_value(
                        record.get(
                            "investigation_rank"
                        )
                    )

                    feature = display_value(
                        record.get(
                            "feature"
                        )
                    )

                    level = display_value(
                        record.get(
                            "priority_level"
                        )
                    )

                    score = format_percentage(
                        record.get(
                            "investigation_priority_score"
                        ),
                        1,
                    )

                    deviation = display_value(
                        record.get(
                            "deviation_level"
                        )
                    )

                    reason = display_value(
                        record.get(
                            "investigation_reason"
                        )
                    )

                    st.markdown(
                        f"""
                        <div class="rank-row">
                            <div class="rank-title">
                                Rank {rank}: {feature}
                            </div>
                            <div class="rank-detail">
                                Priority level: {level}<br>
                                Rank-aggregation score: {score}<br>
                                Measurement status: {deviation}<br>
                                {reason}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with right_column:
                if (
                    "feature" in priority_df.columns
                    and "investigation_priority_score"
                    in priority_df.columns
                ):
                    priority_chart = (
                        alt.Chart(
                            priority_df
                        )
                        .mark_bar()
                        .encode(
                            x=alt.X(
                                "investigation_priority_score:Q",
                                title="Priority score",
                            ),
                            y=alt.Y(
                                "feature:N",
                                sort="-x",
                                title=None,
                            ),
                            color=alt.Color(
                                "priority_level:N",
                                title="Priority level",
                            ),
                            tooltip=[
                                alt.Tooltip(
                                    "feature:N",
                                    title="Sensor",
                                ),
                                alt.Tooltip(
                                    "investigation_priority_score:Q",
                                    title="Priority score",
                                    format=".2%",
                                ),
                                alt.Tooltip(
                                    "deviation_level:N",
                                    title="Deviation",
                                ),
                            ],
                        )
                        .properties(
                            height=500,
                        )
                    )

                    st.altair_chart(
                        priority_chart,
                        use_container_width=True,
                    )

        st.markdown(
            f"""
            <div class="warning-note">
                {display_value(investigation.get("interpretation_limit"))}
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with tabs[4]:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Complete API Record",
            "Structured report returned by the HeveMind FastAPI service.",
        )

        st.json(
            report,
            expanded=False,
        )

        report_json = json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
        )

        st.download_button(
            label="Download engineering report",
            data=report_json,
            file_name=(
                f"{selected_wafer_id}_hevemind_report.json"
            ),
            mime="application/json",
            use_container_width=False,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )


# ============================================================
# ENGINEERING QUEUE
# ============================================================
elif navigation == "Engineering Queue":
    st.subheader(
        "Engineering Action Queue"
    )

    st.caption(
        "Prioritised operational queue based on the retained decision, "
        "calibrated probability and uncertainty."
    )

    queue_filter_columns = st.columns(
        4
    )

    with queue_filter_columns[0]:
        queue_decision = st.selectbox(
            "Queue category",
            options=[
                "All actionable records",
                "Insufficient Evidence",
                "High Risk",
                "Engineering Review",
                "Low Risk",
            ],
        )

    with queue_filter_columns[1]:
        queue_sort = st.selectbox(
            "Sort queue by",
            options=[
                "Dashboard triage order",
                "Failure probability",
                "Combined uncertainty",
                "Prediction confidence",
            ],
        )

    with queue_filter_columns[2]:
        minimum_probability = st.slider(
            "Minimum failure probability",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.01,
        )

    with queue_filter_columns[3]:
        queue_search = st.text_input(
            "Filter wafer ID",
            key="queue_search",
        )

    queue_df = all_wafers_df.copy()

    if queue_decision == "All actionable records":
        queue_df = queue_df.loc[
            queue_df[
                "uncertainty_adjusted_decision"
            ]
            .isin(
                [
                    "Insufficient Evidence",
                    "High Risk",
                    "Engineering Review",
                ]
            )
        ].copy()

    else:
        queue_df = queue_df.loc[
            queue_df[
                "uncertainty_adjusted_decision"
            ]
            == queue_decision
        ].copy()

    queue_df = queue_df.loc[
        queue_df[
            "calibrated_failure_probability"
        ]
        >= minimum_probability
    ].copy()

    if queue_search.strip():
        queue_df = queue_df.loc[
            queue_df[
                "wafer_id"
            ]
            .astype(str)
            .str.contains(
                queue_search.strip(),
                case=False,
                regex=False,
            )
        ].copy()

    sort_mapping = {
        "Dashboard triage order": (
            "dashboard_triage_score"
        ),
        "Failure probability": (
            "calibrated_failure_probability"
        ),
        "Combined uncertainty": (
            "combined_uncertainty"
        ),
        "Prediction confidence": (
            "prediction_confidence"
        ),
    }

    ascending = (
        queue_sort
        == "Prediction confidence"
    )

    queue_df = queue_df.sort_values(
        by=sort_mapping[
            queue_sort
        ],
        ascending=ascending,
    ).reset_index(
        drop=True
    )

    summary_columns = st.columns(
        4
    )

    with summary_columns[0]:
        render_metric_card(
            "Queue records",
            str(
                len(
                    queue_df
                )
            ),
            "Records matching the active filters.",
        )

    with summary_columns[1]:
        render_metric_card(
            "Mean failure risk",
            format_percentage(
                queue_df[
                    "calibrated_failure_probability"
                ].mean()
                if not queue_df.empty
                else 0.0,
                1,
            ),
            "Average calibrated probability in the queue.",
        )

    with summary_columns[2]:
        render_metric_card(
            "Mean uncertainty",
            format_percentage(
                queue_df[
                    "combined_uncertainty"
                ].mean()
                if not queue_df.empty
                else 0.0,
                1,
            ),
            "Average combined uncertainty in the queue.",
        )

    with summary_columns[3]:
        render_metric_card(
            "Low-confidence records",
            str(
                int(
                    (
                        queue_df[
                            "prediction_confidence"
                        ]
                        < 0.25
                    ).sum()
                )
                if not queue_df.empty
                else 0
            ),
            "Records with prediction confidence below 25%.",
        )

    st.write("")

    if queue_df.empty:
        st.info(
            "No records are present in the selected queue."
        )

    else:
        queue_display = queue_df[
            [
                "wafer_id",
                "uncertainty_adjusted_decision",
                "calibrated_failure_probability",
                "prediction_confidence",
                "combined_uncertainty",
                "data_familiarity_band",
                "evidence_agreement",
                "priority_1_sensor",
            ]
        ].copy()

        for column in [
            "calibrated_failure_probability",
            "prediction_confidence",
            "combined_uncertainty",
        ]:
            queue_display[
                column
            ] = queue_display[
                column
            ].map(
                lambda value: format_percentage(
                    value,
                    1,
                )
            )

        queue_display[
            "priority_1_sensor"
        ] = queue_display[
            "priority_1_sensor"
        ].map(
            format_sensor_name
        )

        queue_display = queue_display.rename(
            columns={
                "wafer_id": "Wafer ID",
                "uncertainty_adjusted_decision": "Decision",
                "calibrated_failure_probability": "Failure probability",
                "prediction_confidence": "Confidence",
                "combined_uncertainty": "Uncertainty",
                "data_familiarity_band": "Data familiarity",
                "evidence_agreement": "Evidence status",
                "priority_1_sensor": "Priority sensor",
            }
        )

        st.dataframe(
            queue_display,
            use_container_width=True,
            hide_index=True,
            height=560,
        )

        csv_data = queue_display.to_csv(
            index=False
        ).encode(
            "utf-8"
        )

        st.download_button(
            label="Download current queue",
            data=csv_data,
            file_name="hevemind_engineering_queue.csv",
            mime="text/csv",
        )

    st.markdown(
        """
        <div class="warning-note">
            Queue ordering is a transparent dashboard convenience.
            It does not replace the validated operational decision.
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# SENSOR INTELLIGENCE
# ============================================================
elif navigation == "Sensor Intelligence":
    st.subheader(
        "Sensor Investigation Intelligence"
    )

    st.caption(
        "Cohort-level view of sensors most frequently prioritised for "
        "statistical investigation."
    )

    if all_wafers_df.empty:
        st.info(
            "No sensor-priority data are available."
        )

    else:
        sensor_frequency_df = (
            all_wafers_df[
                "priority_1_sensor"
            ]
            .fillna(
                "Unavailable"
            )
            .value_counts()
            .rename_axis(
                "sensor"
            )
            .reset_index(
                name="priority_1_count"
            )
        )

        sensor_frequency_df[
            "share"
        ] = (
            sensor_frequency_df[
                "priority_1_count"
            ]
            / len(
                all_wafers_df
            )
        )

        top_sensor = (
            sensor_frequency_df.iloc[
                0
            ][
                "sensor"
            ]
            if not sensor_frequency_df.empty
            else "Unavailable"
        )

        top_sensor_count = (
            int(
                sensor_frequency_df.iloc[
                    0
                ][
                    "priority_1_count"
                ]
            )
            if not sensor_frequency_df.empty
            else 0
        )

        metric_columns = st.columns(
            4
        )

        with metric_columns[0]:
            render_metric_card(
                "Distinct priority sensors",
                str(
                    sensor_frequency_df[
                        "sensor"
                    ].nunique()
                ),
                "Distinct sensors appearing as priority one.",
            )

        with metric_columns[1]:
            render_metric_card(
                "Most frequent sensor",
                str(
                    top_sensor
                ),
                "Sensor most often ranked first.",
            )

        with metric_columns[2]:
            render_metric_card(
                "Priority-one occurrences",
                str(
                    top_sensor_count
                ),
                "Number of wafers for which the top sensor ranked first.",
            )

        with metric_columns[3]:
            render_metric_card(
                "Coverage share",
                format_percentage(
                    top_sensor_count
                    / len(
                        all_wafers_df
                    ),
                    1,
                ),
                "Share of wafers for which the top sensor ranked first.",
            )

        st.write("")

        left_column, right_column = st.columns(
            [
                1.05,
                0.95,
            ]
        )

        with left_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Top Priority-One Sensors",
                "How often each sensor was ranked first across the cohort.",
            )

            chart_df = sensor_frequency_df.head(
                20
            )

            sensor_chart = (
                alt.Chart(
                    chart_df
                )
                .mark_bar()
                .encode(
                    x=alt.X(
                        "priority_1_count:Q",
                        title="Priority-one count",
                    ),
                    y=alt.Y(
                        "sensor:N",
                        sort="-x",
                        title=None,
                    ),
                    tooltip=[
                        alt.Tooltip(
                            "sensor:N",
                            title="Sensor",
                        ),
                        alt.Tooltip(
                            "priority_1_count:Q",
                            title="Count",
                        ),
                        alt.Tooltip(
                            "share:Q",
                            title="Share",
                            format=".1%",
                        ),
                    ],
                )
                .properties(
                    height=560,
                )
            )

            st.altair_chart(
                sensor_chart,
                use_container_width=True,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        with right_column:
            st.markdown(
                '<div class="panel">',
                unsafe_allow_html=True,
            )

            render_panel_header(
                "Sensor Priority by Decision",
                "Decision-group breakdown for the most common priority-one sensors.",
            )

            cross_tab = pd.crosstab(
                all_wafers_df[
                    "priority_1_sensor"
                ],
                all_wafers_df[
                    "uncertainty_adjusted_decision"
                ],
            )

            cross_tab[
                "Total"
            ] = cross_tab.sum(
                axis=1
            )

            cross_tab = (
                cross_tab
                .sort_values(
                    by="Total",
                    ascending=False,
                )
                .head(
                    20
                )
                .reset_index()
            )

            st.dataframe(
                cross_tab,
                use_container_width=True,
                hide_index=True,
                height=560,
            )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )

        st.markdown(
            """
            <div class="warning-note">
                Frequent appearance reflects model influence and investigation
                priority. It does not prove that the sensor is a physical cause
                of semiconductor failure.
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# SYSTEM GOVERNANCE
# ============================================================
elif navigation == "System Governance":
    st.subheader(
        "System Governance and Methodological Controls"
    )

    st.caption(
        "Architecture, readiness checks and explicit interpretation limits."
    )

    readiness_checks = readiness_payload.get(
        "checks",
        [],
    )

    check_df = pd.DataFrame(
        readiness_checks
    )

    metric_columns = st.columns(
        3
    )

    with metric_columns[0]:
        render_metric_card(
            "API health",
            display_value(
                health_payload.get(
                    "status"
                )
            ).title(),
            "Runtime health reported by Script 19.",
        )

    with metric_columns[1]:
        render_metric_card(
            "Readiness",
            (
                "Ready"
                if readiness_payload.get(
                    "ready"
                )
                else "Not ready"
            ),
            "Backend, report store and wafer index checks.",
        )

    with metric_columns[2]:
        render_metric_card(
            "Available reports",
            str(
                available_reports
            ),
            "Validated wafer reports exposed by the API.",
        )

    st.write("")

    left_column, right_column = st.columns(
        [
            0.9,
            1.1,
        ]
    )

    with left_column:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Runtime Readiness Checks",
            "Checks returned directly by the API readiness endpoint.",
        )

        if not check_df.empty:
            st.dataframe(
                check_df,
                use_container_width=True,
                hide_index=True,
            )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    with right_column:
        st.markdown(
            '<div class="panel">',
            unsafe_allow_html=True,
        )

        render_panel_header(
            "Validated Architecture",
            "Retained analytical and operational components.",
        )

        architecture = metadata_payload.get(
            "architecture",
            {},
        )

        architecture_rows = []

        for key, value in architecture.items():
            architecture_rows.append(
                {
                    "Component": key.replace(
                        "_",
                        " ",
                    ).title(),
                    "Configuration": (
                        ", ".join(
                            value
                        )
                        if isinstance(
                            value,
                            list,
                        )
                        else value
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                architecture_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    render_panel_header(
        "Decision-Support Architecture",
        "End-to-end information flow from validated model output to engineering action.",
    )

    architecture_flow = pd.DataFrame(
        [
            {
                "Stage": 1,
                "Layer": "Primary Prediction",
                "Purpose": (
                    "Beta-calibrated failure probability"
                ),
            },
            {
                "Stage": 2,
                "Layer": "Uncertainty Engine",
                "Purpose": (
                    "Confidence, interval and abstention support"
                ),
            },
            {
                "Stage": 3,
                "Layer": "Explainability",
                "Purpose": (
                    "Local and global SHAP attribution"
                ),
            },
            {
                "Stage": 4,
                "Layer": "Historical Similarity",
                "Purpose": (
                    "Nearest development-record evidence"
                ),
            },
            {
                "Stage": 5,
                "Layer": "Evidence Aggregation",
                "Purpose": (
                    "Engineer-facing evidence agreement"
                ),
            },
            {
                "Stage": 6,
                "Layer": "Sensor Investigation",
                "Purpose": (
                    "Prioritised statistical inspection list"
                ),
            },
            {
                "Stage": 7,
                "Layer": "FastAPI and Dashboard",
                "Purpose": (
                    "Deployment and human interaction"
                ),
            },
        ]
    )

    st.dataframe(
        architecture_flow,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    render_panel_header(
        "Methodological Limits",
        "Statements that must remain visible in research and operational use.",
    )

    limits = metadata_payload.get(
        "methodological_limits",
        [],
    )

    for limit in limits:
        st.markdown(
            f"""
            <div class="warning-note" style="margin-bottom:0.55rem;">
                {display_value(limit)}
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True,
    )

    render_panel_header(
        "API Connection",
        "Runtime configuration used by this dashboard.",
    )

    connection_df = pd.DataFrame(
        [
            {
                "Setting": "API base URL",
                "Value": API_BASE_URL,
            },
            {
                "Setting": "Authentication",
                "Value": (
                    "API key enabled"
                    if API_KEY
                    else "Local mode without API key"
                ),
            },
            {
                "Setting": "Request timeout",
                "Value": (
                    f"{REQUEST_TIMEOUT_SECONDS:.1f} seconds"
                ),
            },
            {
                "Setting": "API version",
                "Value": display_value(
                    metadata_payload.get(
                        "api_version"
                    )
                ),
            },
        ]
    )

    st.dataframe(
        connection_df,
        use_container_width=True,
        hide_index=True,
    )

    if st.button(
        "Refresh API data",
    ):
        st.cache_data.clear()
        st.rerun()

    st.markdown(
        "</div>",
        unsafe_allow_html=True,
    )
