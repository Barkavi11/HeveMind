from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"
LOG_DIR = ROOT_DIR / "logs"
EXPORT_DIR = LOG_DIR / "exports"

AUDIT_ENGINE_PATH = (
    SRC_DIR
    / "24_audit_activity_tracking.py"
)

AUDIT_DATABASE_PATH = (
    LOG_DIR
    / "hevemind_audit.db"
)


# ============================================================
# GOVERNANCE CONFIGURATION
# ============================================================
APP_TITLE = (
    "HeveMind Governance and Security Operations"
)

APP_VERSION = os.getenv(
    "HEVEMIND_GOVERNANCE_VERSION",
    "1.0.0",
).strip()

STANDALONE_ADMIN_MODE = (
    os.getenv(
        "HEVEMIND_GOVERNANCE_STANDALONE_ADMIN_MODE",
        "false",
    )
    .strip()
    .lower()
    in {
        "1",
        "true",
        "yes",
        "on",
    }
)

DEFAULT_SUMMARY_DAYS = int(
    os.getenv(
        "HEVEMIND_GOVERNANCE_SUMMARY_DAYS",
        "30",
    )
)

DEFAULT_MAXIMUM_ROWS = int(
    os.getenv(
        "HEVEMIND_GOVERNANCE_MAXIMUM_ROWS",
        "1000",
    )
)

ALLOWED_ADMIN_ROLES = {
    "admin",
    "administrator",
}


# ============================================================
# DYNAMIC AUDIT ENGINE IMPORT
# ============================================================
def load_audit_engine() -> Any:
    if not AUDIT_ENGINE_PATH.exists():
        raise FileNotFoundError(
            f"Audit engine not found: {AUDIT_ENGINE_PATH}"
        )

    module_name = (
        "hevemind_audit_activity_tracking"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            AUDIT_ENGINE_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Unable to load the audit engine module."
        )

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[
        module_name
    ] = module

    specification.loader.exec_module(
        module
    )

    return module


AUDIT_ENGINE = load_audit_engine()


# ============================================================
# GENERIC UTILITIES
# ============================================================
def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    return utc_now().isoformat()


def safe_numeric(
    value: Any,
    fallback: float = 0.0,
) -> float:
    try:
        numeric_value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):
        return fallback

    if pd.isna(
        numeric_value
    ):
        return fallback

    return numeric_value


def safe_integer(
    value: Any,
    fallback: int = 0,
) -> int:
    return int(
        round(
            safe_numeric(
                value,
                float(
                    fallback
                ),
            )
        )
    )


def safe_text(
    value: Any,
    fallback: str = "Unavailable",
) -> str:
    if value is None:
        return fallback

    try:
        if pd.isna(
            value
        ):
            return fallback

    except (
        TypeError,
        ValueError,
    ):
        pass

    text = str(
        value
    ).strip()

    return text or fallback


def format_duration(
    seconds: Any,
) -> str:
    numeric_seconds = max(
        safe_numeric(
            seconds,
            0.0,
        ),
        0.0,
    )

    if numeric_seconds < 60:
        return (
            f"{numeric_seconds:.0f} sec"
        )

    minutes = (
        numeric_seconds
        / 60.0
    )

    if minutes < 60:
        return (
            f"{minutes:.1f} min"
        )

    hours = (
        minutes
        / 60.0
    )

    return (
        f"{hours:.1f} hr"
    )


def format_filesize(
    byte_count: int,
) -> str:
    size = float(
        byte_count
    )

    units = [
        "B",
        "KB",
        "MB",
        "GB",
    ]

    for unit in units:
        if size < 1024.0:
            return (
                f"{size:.1f} {unit}"
            )

        size /= 1024.0

    return (
        f"{size:.1f} TB"
    )


def ensure_export_directory() -> None:
    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def start_of_day_utc(
    days_ago: int = 0,
) -> datetime:
    reference = (
        utc_now()
        - timedelta(
            days=days_ago
        )
    )

    return reference.replace(
        hour=0,
        minute=0,
        second=0,
        microsecond=0,
    )


# ============================================================
# DATABASE HEALTH
# ============================================================
@dataclass(frozen=True)
class DatabaseHealth:
    database_exists: bool
    database_size_bytes: int
    integrity_ok: bool
    integrity_message: str
    event_rows: int
    session_rows: int
    active_sessions: int
    oldest_event_utc: str | None
    newest_event_utc: str | None


def read_database_health() -> DatabaseHealth:
    if not AUDIT_DATABASE_PATH.exists():
        return DatabaseHealth(
            database_exists=False,
            database_size_bytes=0,
            integrity_ok=False,
            integrity_message=(
                "Audit database does not exist."
            ),
            event_rows=0,
            session_rows=0,
            active_sessions=0,
            oldest_event_utc=None,
            newest_event_utc=None,
        )

    database_size_bytes = (
        AUDIT_DATABASE_PATH.stat().st_size
    )

    with sqlite3.connect(
        AUDIT_DATABASE_PATH
    ) as connection:
        connection.row_factory = (
            sqlite3.Row
        )

        integrity_row = connection.execute(
            "PRAGMA integrity_check"
        ).fetchone()

        integrity_message = (
            str(
                integrity_row[
                    0
                ]
            )
            if integrity_row is not None
            else "Unknown"
        )

        integrity_ok = (
            integrity_message.lower()
            == "ok"
        )

        event_row = connection.execute(
            """
            SELECT
                COUNT(*) AS records,
                MIN(timestamp_utc) AS oldest_event_utc,
                MAX(timestamp_utc) AS newest_event_utc
            FROM audit_events
            """
        ).fetchone()

        session_row = connection.execute(
            """
            SELECT
                COUNT(*) AS records,
                SUM(
                    CASE
                        WHEN ended_utc IS NULL
                        THEN 1
                        ELSE 0
                    END
                ) AS active_sessions
            FROM audit_sessions
            """
        ).fetchone()

    return DatabaseHealth(
        database_exists=True,
        database_size_bytes=(
            database_size_bytes
        ),
        integrity_ok=integrity_ok,
        integrity_message=(
            integrity_message
        ),
        event_rows=safe_integer(
            event_row[
                "records"
            ]
            if event_row is not None
            else 0
        ),
        session_rows=safe_integer(
            session_row[
                "records"
            ]
            if session_row is not None
            else 0
        ),
        active_sessions=safe_integer(
            session_row[
                "active_sessions"
            ]
            if session_row is not None
            else 0
        ),
        oldest_event_utc=(
            event_row[
                "oldest_event_utc"
            ]
            if event_row is not None
            else None
        ),
        newest_event_utc=(
            event_row[
                "newest_event_utc"
            ]
            if event_row is not None
            else None
        ),
    )


# ============================================================
# ANALYTICAL TABLE BUILDERS
# ============================================================
def parse_details_column(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    output = dataframe.copy()

    if (
        output.empty
        or "details_json"
        not in output.columns
    ):
        return output

    def parse_value(
        raw_value: Any,
    ) -> dict[str, Any]:
        if not isinstance(
            raw_value,
            str,
        ):
            return {}

        try:
            parsed = json.loads(
                raw_value
            )

        except json.JSONDecodeError:
            return {}

        return (
            parsed
            if isinstance(
                parsed,
                dict,
            )
            else {}
        )

    output[
        "details"
    ] = output[
        "details_json"
    ].map(
        parse_value
    )

    return output


def build_login_timeline(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "successful_logins",
                "failed_or_denied",
            ]
        )

    working = events.copy()

    working[
        "timestamp"
    ] = pd.to_datetime(
        working[
            "timestamp_utc"
        ],
        errors="coerce",
        utc=True,
    )

    working = working.loc[
        working[
            "timestamp"
        ].notna()
    ].copy()

    if working.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "successful_logins",
                "failed_or_denied",
            ]
        )

    working[
        "date"
    ] = working[
        "timestamp"
    ].dt.date.astype(
        str
    )

    authentication_events = working.loc[
        working[
            "category"
        ]
        == "authentication"
    ].copy()

    if authentication_events.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "successful_logins",
                "failed_or_denied",
            ]
        )

    successful = (
        authentication_events.loc[
            authentication_events[
                "outcome"
            ]
            == "success"
        ]
        .groupby(
            "date"
        )
        .size()
        .rename(
            "successful_logins"
        )
    )

    failed = (
        authentication_events.loc[
            authentication_events[
                "outcome"
            ].isin(
                [
                    "failure",
                    "denied",
                ]
            )
        ]
        .groupby(
            "date"
        )
        .size()
        .rename(
            "failed_or_denied"
        )
    )

    timeline = pd.concat(
        [
            successful,
            failed,
        ],
        axis=1,
    ).fillna(
        0
    )

    return (
        timeline.reset_index()
        .sort_values(
            "date"
        )
    )


def build_failed_authentication_table(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    filtered = events.loc[
        (
            events[
                "category"
            ]
            == "authentication"
        )
        & (
            events[
                "outcome"
            ].isin(
                [
                    "failure",
                    "denied",
                ]
            )
        )
    ].copy()

    if filtered.empty:
        return filtered

    columns = [
        "timestamp_utc",
        "user_email",
        "event_type",
        "outcome",
        "client_ip",
        "message",
        "source_component",
    ]

    available_columns = [
        column
        for column in columns
        if column
        in filtered.columns
    ]

    return filtered[
        available_columns
    ].sort_values(
        "timestamp_utc",
        ascending=False,
    )


def build_export_history(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()

    parsed = parse_details_column(
        events
    )

    exports = parsed.loc[
        parsed[
            "category"
        ]
        == "export"
    ].copy()

    if exports.empty:
        return exports

    exports[
        "export_format"
    ] = exports[
        "details"
    ].map(
        lambda value: safe_text(
            value.get(
                "export_format"
            ),
            "Unknown",
        )
    )

    exports[
        "record_count"
    ] = exports[
        "details"
    ].map(
        lambda value: value.get(
            "record_count"
        )
    )

    exports[
        "filename"
    ] = exports[
        "details"
    ].map(
        lambda value: safe_text(
            value.get(
                "filename"
            ),
            "Unavailable",
        )
    )

    return exports[
        [
            "timestamp_utc",
            "user_email",
            "role",
            "resource_type",
            "resource_id",
            "export_format",
            "record_count",
            "filename",
        ]
    ].sort_values(
        "timestamp_utc",
        ascending=False,
    )


def build_wafer_access_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "wafer_id",
                "views",
                "unique_users",
                "last_access_utc",
            ]
        )

    wafer_events = events.loc[
        (
            events[
                "category"
            ]
            == "wafer_access"
        )
        & (
            events[
                "resource_id"
            ].notna()
        )
    ].copy()

    if wafer_events.empty:
        return pd.DataFrame(
            columns=[
                "wafer_id",
                "views",
                "unique_users",
                "last_access_utc",
            ]
        )

    summary = (
        wafer_events.groupby(
            "resource_id"
        )
        .agg(
            views=(
                "event_id",
                "count",
            ),
            unique_users=(
                "user_email",
                pd.Series.nunique,
            ),
            last_access_utc=(
                "timestamp_utc",
                "max",
            ),
        )
        .reset_index()
        .rename(
            columns={
                "resource_id": (
                    "wafer_id"
                )
            }
        )
        .sort_values(
            [
                "views",
                "last_access_utc",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    return summary


def build_user_activity_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "user_email",
                "events",
                "wafer_views",
                "searches",
                "exports",
                "failed_or_denied",
                "last_activity_utc",
            ]
        )

    working = events.copy()

    working[
        "user_email"
    ] = working[
        "user_email"
    ].fillna(
        "Unauthenticated or System"
    )

    summary = (
        working.groupby(
            "user_email"
        )
        .agg(
            events=(
                "event_id",
                "count",
            ),
            wafer_views=(
                "category",
                lambda series: int(
                    (
                        series
                        == "wafer_access"
                    ).sum()
                ),
            ),
            searches=(
                "category",
                lambda series: int(
                    (
                        series
                        == "search"
                    ).sum()
                ),
            ),
            exports=(
                "category",
                lambda series: int(
                    (
                        series
                        == "export"
                    ).sum()
                ),
            ),
            failed_or_denied=(
                "outcome",
                lambda series: int(
                    series.isin(
                        [
                            "failure",
                            "denied",
                        ]
                    ).sum()
                ),
            ),
            last_activity_utc=(
                "timestamp_utc",
                "max",
            ),
        )
        .reset_index()
        .sort_values(
            "events",
            ascending=False,
        )
    )

    return summary


def build_session_analytics(
    sessions: pd.DataFrame,
) -> pd.DataFrame:
    if sessions.empty:
        return pd.DataFrame()

    output = sessions.copy()

    output[
        "duration_minutes"
    ] = (
        pd.to_numeric(
            output[
                "duration_seconds"
            ],
            errors="coerce",
        )
        / 60.0
    )

    output[
        "session_status"
    ] = output[
        "ended_utc"
    ].map(
        lambda value: (
            "Active"
            if pd.isna(
                value
            )
            or value in (
                None,
                "",
            )
            else "Ended"
        )
    )

    columns = [
        "session_id",
        "user_email",
        "display_name",
        "role",
        "auth_source",
        "started_utc",
        "last_activity_utc",
        "ended_utc",
        "duration_minutes",
        "session_status",
        "termination_reason",
        "client_ip",
        "source_component",
    ]

    available_columns = [
        column
        for column in columns
        if column
        in output.columns
    ]

    return output[
        available_columns
    ]


# ============================================================
# EXPORT SUPPORT
# ============================================================
def create_governance_export(
    *,
    events: pd.DataFrame,
    sessions: pd.DataFrame,
    integrity: dict[str, Any],
    health: DatabaseHealth,
    summary: dict[str, Any],
) -> Path:
    ensure_export_directory()

    timestamp = utc_now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_path = (
        EXPORT_DIR
        / (
            "hevemind_governance_snapshot_"
            + timestamp
            + ".xlsx"
        )
    )

    health_frame = pd.DataFrame(
        [
            {
                "database_exists": (
                    health.database_exists
                ),
                "database_size_bytes": (
                    health.database_size_bytes
                ),
                "database_size_display": (
                    format_filesize(
                        health.database_size_bytes
                    )
                ),
                "integrity_ok": (
                    health.integrity_ok
                ),
                "integrity_message": (
                    health.integrity_message
                ),
                "event_rows": (
                    health.event_rows
                ),
                "session_rows": (
                    health.session_rows
                ),
                "active_sessions": (
                    health.active_sessions
                ),
                "oldest_event_utc": (
                    health.oldest_event_utc
                ),
                "newest_event_utc": (
                    health.newest_event_utc
                ),
            }
        ]
    )

    summary_frame = pd.DataFrame(
        [
            summary
        ]
    )

    integrity_frame = pd.DataFrame(
        [
            integrity
        ]
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:
        summary_frame.to_excel(
            writer,
            sheet_name="Governance Summary",
            index=False,
        )

        health_frame.to_excel(
            writer,
            sheet_name="Database Health",
            index=False,
        )

        integrity_frame.to_excel(
            writer,
            sheet_name="Integrity",
            index=False,
        )

        events.to_excel(
            writer,
            sheet_name="Audit Events",
            index=False,
        )

        sessions.to_excel(
            writer,
            sheet_name="Sessions",
            index=False,
        )

        build_failed_authentication_table(
            events
        ).to_excel(
            writer,
            sheet_name="Failed Authentication",
            index=False,
        )

        build_export_history(
            events
        ).to_excel(
            writer,
            sheet_name="Export History",
            index=False,
        )

        build_wafer_access_summary(
            events
        ).to_excel(
            writer,
            sheet_name="Wafer Access",
            index=False,
        )

        build_user_activity_summary(
            events
        ).to_excel(
            writer,
            sheet_name="User Activity",
            index=False,
        )

    return output_path


# ============================================================
# STREAMLIT STYLING
# ============================================================
GOVERNANCE_CSS = """
<style>
    :root {
        --navy-950: #071426;
        --navy-900: #0b1f36;
        --navy-800: #123252;
        --blue-600: #1f6aa5;
        --slate-900: #172230;
        --slate-700: #405267;
        --slate-500: #708196;
        --slate-200: #dce3e9;
        --slate-100: #eef3f7;
        --white: #ffffff;
        --green: #1b7f5b;
        --green-bg: #e6f4ed;
        --amber: #a96b00;
        --amber-bg: #fff3d9;
        --red: #b43b3b;
        --red-bg: #fdeaea;
        --purple: #6f4f9c;
        --purple-bg: #f1ebf9;
    }

    .stApp {
        background:
            linear-gradient(
                180deg,
                #f5f8fb 0%,
                #edf3f7 100%
            );
    }

    .block-container {
        max-width: 1440px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .governance-header {
        background:
            linear-gradient(
                110deg,
                var(--navy-950) 0%,
                var(--navy-800) 72%,
                var(--blue-600) 100%
            );
        border-radius: 18px;
        padding: 1.25rem 1.45rem;
        color: #ffffff;
        box-shadow:
            0 14px 34px
            rgba(7, 20, 38, 0.17);
        margin-bottom: 1rem;
    }

    .governance-title {
        font-size: 1.8rem;
        font-weight: 780;
        letter-spacing: -0.025em;
    }

    .governance-subtitle {
        color: #d8e8f4;
        font-size: 0.9rem;
        margin-top: 0.36rem;
        line-height: 1.5;
    }

    .governance-status-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .governance-status {
        border-radius: 999px;
        padding: 0.32rem 0.68rem;
        font-size: 0.72rem;
        font-weight: 720;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #ffffff;
    }

    .governance-metric {
        min-height: 138px;
        background: #ffffff;
        border: 1px solid var(--slate-200);
        border-radius: 15px;
        padding: 0.95rem 1rem;
        box-shadow:
            0 7px 20px
            rgba(21, 48, 75, 0.06);
        height: 100%;
    }

    .governance-metric-label {
        color: var(--slate-500);
        font-size: 0.70rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .governance-metric-value {
        color: var(--navy-950);
        font-size: 1.58rem;
        font-weight: 790;
        margin-top: 0.42rem;
        overflow-wrap: anywhere;
    }

    .governance-metric-note {
        color: var(--slate-700);
        font-size: 0.77rem;
        line-height: 1.4;
        margin-top: 0.36rem;
    }

    .health-card {
        border-radius: 14px;
        padding: 0.95rem 1rem;
        border: 1px solid var(--slate-200);
        background: #ffffff;
        box-shadow:
            0 6px 18px
            rgba(21, 48, 75, 0.05);
        min-height: 122px;
    }

    .health-label {
        color: var(--slate-500);
        font-size: 0.68rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .health-value {
        color: var(--navy-950);
        font-size: 1.05rem;
        font-weight: 760;
        margin-top: 0.34rem;
    }

    .health-note {
        color: var(--slate-700);
        font-size: 0.76rem;
        margin-top: 0.28rem;
        line-height: 1.4;
    }

    .status-good {
        color: var(--green);
        background: var(--green-bg);
        border-color: #b6dbc9;
    }

    .status-warning {
        color: var(--amber);
        background: var(--amber-bg);
        border-color: #efd49b;
    }

    .status-bad {
        color: var(--red);
        background: var(--red-bg);
        border-color: #efbcbc;
    }

    .section-title {
        color: var(--navy-950);
        font-size: 1.18rem;
        font-weight: 760;
        margin-top: 0.35rem;
        margin-bottom: 0.12rem;
    }

    .section-subtitle {
        color: var(--slate-500);
        font-size: 0.82rem;
        margin-bottom: 0.75rem;
    }

    .governance-note {
        border-left: 4px solid var(--blue-600);
        background: #eef7fc;
        border-radius: 10px;
        padding: 0.78rem 0.9rem;
        color: var(--slate-700);
        font-size: 0.79rem;
        line-height: 1.45;
        margin-top: 0.7rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.28rem;
        border-bottom: 1px solid var(--slate-200);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 9px 9px 0 0;
        color: var(--slate-700);
        padding: 0.58rem 0.78rem;
    }

    .stTabs [aria-selected="true"] {
        background: #e8f2f9 !important;
        color: var(--navy-950) !important;
        border-bottom: 3px solid var(--blue-600) !important;
    }

    .stTabs [aria-selected="true"] p {
        color: var(--navy-950) !important;
        font-weight: 720 !important;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--slate-200);
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
        box-shadow:
            0 6px 18px
            rgba(21, 48, 75, 0.05);
    }

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                var(--navy-950) 0%,
                var(--navy-900) 100%
            );
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    footer {
        visibility: hidden;
    }
</style>
"""


# ============================================================
# STREAMLIT COMPONENTS
# ============================================================
def render_metric_card(
    *,
    label: str,
    value: str,
    note: str,
) -> None:
    import streamlit as st

    st.markdown(
        f"""
        <div class="governance-metric">
            <div class="governance-metric-label">
                {label}
            </div>
            <div class="governance-metric-value">
                {value}
            </div>
            <div class="governance-metric-note">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_health_card(
    *,
    label: str,
    value: str,
    note: str,
    status_class: str = "",
) -> None:
    import streamlit as st

    st.markdown(
        f"""
        <div class="health-card {status_class}">
            <div class="health-label">
                {label}
            </div>
            <div class="health-value">
                {value}
            </div>
            <div class="health-note">
                {note}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_administrator() -> dict[str, Any]:
    import streamlit as st

    user = st.session_state.get(
        "hevemind_user",
        {},
    )

    user_role = str(
        user.get(
            "role",
            "",
        )
    ).strip().lower()

    if (
        user_role not in ALLOWED_ADMIN_ROLES
        and not STANDALONE_ADMIN_MODE
    ):
        st.error(
            "Administrator access is required to open the "
            "HeveMind governance console."
        )

        st.stop()

    return (
        user
        if isinstance(
            user,
            dict,
        )
        else {}
    )


def record_governance_page_access(
    user: dict[str, Any],
) -> None:
    try:
        AUDIT_ENGINE.record_event(
            source_component=(
                "25_governance_dashboard"
            ),
            event_type=(
                "governance_console_opened"
            ),
            category="governance",
            outcome="success",
            user_email=user.get(
                "email"
            ),
            display_name=user.get(
                "display_name"
            ),
            role=user.get(
                "role"
            ),
            auth_source=user.get(
                "auth_source"
            ),
            session_id=(
                st_session_value(
                    "hevemind_audit_session_id"
                )
            ),
            resource_type=(
                "governance_console"
            ),
            resource_id="main",
            action="view",
            message=(
                "Administrator opened the governance console."
            ),
        )

    except Exception:
        # Governance rendering must not fail because logging is unavailable.
        pass


def st_session_value(
    key: str,
) -> Any:
    try:
        import streamlit as st

        return st.session_state.get(
            key
        )

    except Exception:
        return None


# ============================================================
# GOVERNANCE DASHBOARD
# ============================================================
def render_governance_dashboard() -> None:
    import altair as alt
    import streamlit as st

    AUDIT_ENGINE.initialise_database()

    st.markdown(
        GOVERNANCE_CSS,
        unsafe_allow_html=True,
    )

    user = require_administrator()

    if not st.session_state.get(
        "governance_access_recorded",
        False,
    ):
        record_governance_page_access(
            user
        )

        st.session_state[
            "governance_access_recorded"
        ] = True

    refresh_timestamp = (
        utc_now()
        .astimezone()
        .strftime(
            "%d %b %Y, %I:%M %p"
        )
    )

    st.markdown(
        f"""
        <div class="governance-header">
            <div class="governance-title">
                {APP_TITLE}
            </div>
            <div class="governance-subtitle">
                Centralised governance, audit integrity, authentication
                monitoring, user activity, session analytics and export
                oversight for the HeveMind platform
            </div>
            <div class="governance-status-row">
                <span class="governance-status">
                    Version {APP_VERSION}
                </span>
                <span class="governance-status">
                    Administrator access
                </span>
                <span class="governance-status">
                    Refreshed {refresh_timestamp}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    control_columns = st.columns(
        [
            0.24,
            0.26,
            0.24,
            0.26,
        ]
    )

    with control_columns[0]:
        summary_days = st.selectbox(
            "Monitoring period",
            options=[
                1,
                7,
                30,
                90,
                180,
                365,
            ],
            index=2,
            format_func=lambda value: (
                f"Last {value} day"
                if value == 1
                else f"Last {value} days"
            ),
        )

    with control_columns[1]:
        maximum_rows = st.selectbox(
            "Maximum event rows",
            options=[
                100,
                250,
                500,
                1000,
                2500,
                5000,
            ],
            index=3,
        )

    start_utc = (
        utc_now()
        - timedelta(
            days=int(
                summary_days
            )
        )
    ).isoformat()

    all_events = (
        AUDIT_ENGINE.query_events(
            start_utc=start_utc,
            limit=int(
                maximum_rows
            ),
        )
    )

    all_sessions = (
        AUDIT_ENGINE.query_sessions(
            start_utc=start_utc,
        )
    )

    summary = (
        AUDIT_ENGINE.audit_summary(
            days=int(
                summary_days
            )
        )
    )

    database_health = (
        read_database_health()
    )

    integrity_result = (
        AUDIT_ENGINE.verify_hash_chain()
    )

    available_users = (
        sorted(
            all_events[
                "user_email"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not all_events.empty
        else []
    )

    available_categories = (
        sorted(
            all_events[
                "category"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if not all_events.empty
        else []
    )

    with control_columns[2]:
        selected_user = st.selectbox(
            "User filter",
            options=[
                "All users",
                *available_users,
            ],
        )

    with control_columns[3]:
        selected_category = st.selectbox(
            "Category filter",
            options=[
                "All categories",
                *available_categories,
            ],
        )

    filtered_events = all_events.copy()

    if selected_user != "All users":
        filtered_events = filtered_events.loc[
            filtered_events[
                "user_email"
            ]
            == selected_user
        ].copy()

    if selected_category != "All categories":
        filtered_events = filtered_events.loc[
            filtered_events[
                "category"
            ]
            == selected_category
        ].copy()

    failed_or_denied = int(
        (
            filtered_events[
                "outcome"
            ].isin(
                [
                    "failure",
                    "denied",
                ]
            )
        ).sum()
    ) if not filtered_events.empty else 0

    active_today_events = (
        AUDIT_ENGINE.query_events(
            start_utc=(
                start_of_day_utc(
                    0
                ).isoformat()
            ),
            limit=5000,
        )
    )

    active_today_users = int(
        active_today_events[
            "user_email"
        ]
        .dropna()
        .nunique()
    ) if not active_today_events.empty else 0

    mean_session_minutes = (
        summary.get(
            "mean_session_minutes"
        )
    )

    metric_columns = st.columns(
        6
    )

    metric_payloads = [
        (
            "Audit events",
            str(
                len(
                    filtered_events
                )
            ),
            "Recorded events matching the active governance filters.",
        ),
        (
            "Active users today",
            str(
                active_today_users
            ),
            "Distinct users with recorded activity since 00:00 UTC.",
        ),
        (
            "Failed or denied",
            str(
                failed_or_denied
            ),
            "Events requiring authentication or access review.",
        ),
        (
            "Wafer accesses",
            str(
                summary.get(
                    "wafer_views",
                    0,
                )
            ),
            "Validated wafer reports opened in the selected period.",
        ),
        (
            "Exports",
            str(
                summary.get(
                    "exports",
                    0,
                )
            ),
            "Recorded engineering or governance export events.",
        ),
        (
            "Mean session",
            (
                f"{safe_numeric(mean_session_minutes):.1f} min"
                if mean_session_minutes is not None
                else "Unavailable"
            ),
            "Average duration for completed authenticated sessions.",
        ),
    ]

    for column, payload in zip(
        metric_columns,
        metric_payloads,
    ):
        with column:
            render_metric_card(
                label=payload[
                    0
                ],
                value=payload[
                    1
                ],
                note=payload[
                    2
                ],
            )

    st.write("")

    health_columns = st.columns(
        5
    )

    with health_columns[0]:
        render_health_card(
            label="Audit database",
            value=(
                "Available"
                if database_health.database_exists
                else "Unavailable"
            ),
            note=(
                str(
                    AUDIT_DATABASE_PATH
                )
            ),
            status_class=(
                "status-good"
                if database_health.database_exists
                else "status-bad"
            ),
        )

    with health_columns[1]:
        render_health_card(
            label="SQLite integrity",
            value=(
                "Passed"
                if database_health.integrity_ok
                else "Failed"
            ),
            note=(
                database_health.integrity_message
            ),
            status_class=(
                "status-good"
                if database_health.integrity_ok
                else "status-bad"
            ),
        )

    with health_columns[2]:
        render_health_card(
            label="Hash chain",
            value=(
                "Valid"
                if integrity_result.get(
                    "valid"
                )
                else "Invalid"
            ),
            note=(
                f"{integrity_result.get('checked_events', 0)} "
                "events verified"
            ),
            status_class=(
                "status-good"
                if integrity_result.get(
                    "valid"
                )
                else "status-bad"
            ),
        )

    with health_columns[3]:
        render_health_card(
            label="Database size",
            value=format_filesize(
                database_health.database_size_bytes
            ),
            note=(
                f"{database_health.event_rows} events and "
                f"{database_health.session_rows} sessions"
            ),
        )

    with health_columns[4]:
        render_health_card(
            label="Active sessions",
            value=str(
                database_health.active_sessions
            ),
            note=(
                "Sessions without a recorded end timestamp."
            ),
            status_class=(
                "status-warning"
                if database_health.active_sessions
                > 0
                else ""
            ),
        )

    st.write("")

    (
        overview_tab,
        authentication_tab,
        sessions_tab,
        access_tab,
        exports_tab,
        integrity_tab,
        raw_events_tab,
    ) = st.tabs(
        [
            "Governance Overview",
            "Authentication",
            "Sessions",
            "Wafer Access",
            "Export Oversight",
            "Integrity and Retention",
            "Audit Register",
        ]
    )

    with overview_tab:
        overview_columns = st.columns(
            2
        )

        login_timeline = build_login_timeline(
            filtered_events
        )

        user_activity = build_user_activity_summary(
            filtered_events
        )

        with overview_columns[0]:
            st.markdown(
                """
                <div class="section-title">
                    Authentication Activity
                </div>
                <div class="section-subtitle">
                    Successful and unsuccessful authentication events by day.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if login_timeline.empty:
                st.info(
                    "No authentication activity is available "
                    "for the selected period."
                )

            else:
                melted_timeline = (
                    login_timeline.melt(
                        id_vars=[
                            "date"
                        ],
                        value_vars=[
                            "successful_logins",
                            "failed_or_denied",
                        ],
                        var_name="authentication_result",
                        value_name="events",
                    )
                )

                authentication_chart = (
                    alt.Chart(
                        melted_timeline
                    )
                    .mark_line(
                        point=True,
                        strokeWidth=2.3,
                    )
                    .encode(
                        x=alt.X(
                            "date:T",
                            title="Date",
                        ),
                        y=alt.Y(
                            "events:Q",
                            title="Events",
                        ),
                        color=alt.Color(
                            "authentication_result:N",
                            title="Result",
                            scale=alt.Scale(
                                domain=[
                                    "successful_logins",
                                    "failed_or_denied",
                                ],
                                range=[
                                    "#1b7f5b",
                                    "#b43b3b",
                                ],
                            ),
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "date:T",
                                title="Date",
                            ),
                            alt.Tooltip(
                                "authentication_result:N",
                                title="Result",
                            ),
                            alt.Tooltip(
                                "events:Q",
                                title="Events",
                            ),
                        ],
                    )
                    .properties(
                        height=320,
                        background="#ffffff",
                    )
                )

                st.altair_chart(
                    authentication_chart,
                    use_container_width=True,
                )

        with overview_columns[1]:
            st.markdown(
                """
                <div class="section-title">
                    User Activity
                </div>
                <div class="section-subtitle">
                    Users ranked by recorded governance activity.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if user_activity.empty:
                st.info(
                    "No user activity is available "
                    "for the selected period."
                )

            else:
                top_user_activity = (
                    user_activity.head(
                        15
                    )
                )

                user_chart = (
                    alt.Chart(
                        top_user_activity
                    )
                    .mark_bar(
                        cornerRadiusEnd=5
                    )
                    .encode(
                        y=alt.Y(
                            "user_email:N",
                            sort="-x",
                            title="User",
                        ),
                        x=alt.X(
                            "events:Q",
                            title="Recorded events",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "user_email:N",
                                title="User",
                            ),
                            alt.Tooltip(
                                "events:Q",
                                title="Events",
                            ),
                            alt.Tooltip(
                                "wafer_views:Q",
                                title="Wafer views",
                            ),
                            alt.Tooltip(
                                "searches:Q",
                                title="Searches",
                            ),
                            alt.Tooltip(
                                "exports:Q",
                                title="Exports",
                            ),
                        ],
                    )
                    .properties(
                        height=320,
                        background="#ffffff",
                    )
                )

                st.altair_chart(
                    user_chart,
                    use_container_width=True,
                )

        st.markdown(
            """
            <div class="section-title">
                Governance Activity Register
            </div>
            <div class="section-subtitle">
                Consolidated activity by authenticated user.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if user_activity.empty:
            st.info(
                "No consolidated activity records are available."
            )

        else:
            st.dataframe(
                user_activity,
                use_container_width=True,
                hide_index=True,
                height=420,
            )

    with authentication_tab:
        failed_table = (
            build_failed_authentication_table(
                filtered_events
            )
        )

        authentication_events = (
            filtered_events.loc[
                filtered_events[
                    "category"
                ]
                == "authentication"
            ].copy()
            if not filtered_events.empty
            else pd.DataFrame()
        )

        authentication_metric_columns = (
            st.columns(
                4
            )
        )

        successful_authentication = int(
            (
                authentication_events[
                    "outcome"
                ]
                == "success"
            ).sum()
        ) if not authentication_events.empty else 0

        failed_authentication = int(
            (
                authentication_events[
                    "outcome"
                ]
                == "failure"
            ).sum()
        ) if not authentication_events.empty else 0

        denied_authentication = int(
            (
                authentication_events[
                    "outcome"
                ]
                == "denied"
            ).sum()
        ) if not authentication_events.empty else 0

        locked_accounts = int(
            authentication_events[
                "message"
            ]
            .fillna(
                ""
            )
            .astype(str)
            .str.contains(
                "lock",
                case=False,
                regex=False,
            )
            .sum()
        ) if not authentication_events.empty else 0

        authentication_payloads = [
            (
                "Successful",
                successful_authentication,
                "Authentication events completed successfully.",
            ),
            (
                "Failed",
                failed_authentication,
                "Invalid or unsuccessful authentication attempts.",
            ),
            (
                "Denied",
                denied_authentication,
                "Authenticated identities denied platform access.",
            ),
            (
                "Lock indicators",
                locked_accounts,
                "Events containing an account-lock condition.",
            ),
        ]

        for column, payload in zip(
            authentication_metric_columns,
            authentication_payloads,
        ):
            with column:
                render_metric_card(
                    label=payload[
                        0
                    ],
                    value=str(
                        payload[
                            1
                        ]
                    ),
                    note=payload[
                        2
                    ],
                )

        st.write("")

        st.markdown(
            """
            <div class="section-title">
                Failed and Denied Authentication
            </div>
            <div class="section-subtitle">
                Authentication events requiring administrative review.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if failed_table.empty:
            st.success(
                "No failed or denied authentication events "
                "match the current filters."
            )

        else:
            st.dataframe(
                failed_table,
                use_container_width=True,
                hide_index=True,
                height=520,
            )

    with sessions_tab:
        session_analytics = (
            build_session_analytics(
                all_sessions
            )
        )

        completed_sessions = (
            session_analytics.loc[
                session_analytics[
                    "session_status"
                ]
                == "Ended"
            ].copy()
            if not session_analytics.empty
            else pd.DataFrame()
        )

        session_metric_columns = st.columns(
            4
        )

        session_payloads = [
            (
                "Recorded sessions",
                len(
                    session_analytics
                ),
                "Sessions started in the selected monitoring period.",
            ),
            (
                "Active sessions",
                (
                    int(
                        (
                            session_analytics[
                                "session_status"
                            ]
                            == "Active"
                        ).sum()
                    )
                    if not session_analytics.empty
                    else 0
                ),
                "Sessions with no recorded termination.",
            ),
            (
                "Completed sessions",
                len(
                    completed_sessions
                ),
                "Sessions with a recorded end timestamp.",
            ),
            (
                "Median duration",
                (
                    f"{completed_sessions['duration_minutes'].median():.1f} min"
                    if (
                        not completed_sessions.empty
                        and completed_sessions[
                            "duration_minutes"
                        ].notna().any()
                    )
                    else "Unavailable"
                ),
                "Median duration for completed sessions.",
            ),
        ]

        for column, payload in zip(
            session_metric_columns,
            session_payloads,
        ):
            with column:
                render_metric_card(
                    label=payload[
                        0
                    ],
                    value=str(
                        payload[
                            1
                        ]
                    ),
                    note=payload[
                        2
                    ],
                )

        st.write("")

        if session_analytics.empty:
            st.info(
                "No session records are available for "
                "the selected period."
            )

        else:
            st.dataframe(
                session_analytics,
                use_container_width=True,
                hide_index=True,
                height=580,
            )

    with access_tab:
        wafer_summary = (
            build_wafer_access_summary(
                filtered_events
            )
        )

        access_columns = st.columns(
            2
        )

        with access_columns[0]:
            st.markdown(
                """
                <div class="section-title">
                    Most Accessed Wafer Reports
                </div>
                <div class="section-subtitle">
                    Validated wafer reports ranked by view frequency.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if wafer_summary.empty:
                st.info(
                    "No wafer-access events are available "
                    "for the selected period."
                )

            else:
                wafer_chart = (
                    alt.Chart(
                        wafer_summary.head(
                            20
                        )
                    )
                    .mark_bar(
                        cornerRadiusEnd=5
                    )
                    .encode(
                        y=alt.Y(
                            "wafer_id:N",
                            sort="-x",
                            title="Wafer",
                        ),
                        x=alt.X(
                            "views:Q",
                            title="Views",
                        ),
                        tooltip=[
                            alt.Tooltip(
                                "wafer_id:N",
                                title="Wafer",
                            ),
                            alt.Tooltip(
                                "views:Q",
                                title="Views",
                            ),
                            alt.Tooltip(
                                "unique_users:Q",
                                title="Unique users",
                            ),
                            alt.Tooltip(
                                "last_access_utc:N",
                                title="Last access",
                            ),
                        ],
                    )
                    .properties(
                        height=430,
                        background="#ffffff",
                    )
                )

                st.altair_chart(
                    wafer_chart,
                    use_container_width=True,
                )

        with access_columns[1]:
            st.markdown(
                """
                <div class="section-title">
                    Wafer Access Register
                </div>
                <div class="section-subtitle">
                    Frequency, unique users and most recent access.
                </div>
                """,
                unsafe_allow_html=True,
            )

            if wafer_summary.empty:
                st.info(
                    "No wafer-access register is available."
                )

            else:
                st.dataframe(
                    wafer_summary,
                    use_container_width=True,
                    hide_index=True,
                    height=430,
                )

    with exports_tab:
        export_history = (
            build_export_history(
                filtered_events
            )
        )

        export_metric_columns = st.columns(
            4
        )

        export_formats = (
            export_history[
                "export_format"
            ].value_counts()
            if not export_history.empty
            else pd.Series(
                dtype=int
            )
        )

        export_payloads = [
            (
                "Export events",
                len(
                    export_history
                ),
                "Exports matching the current governance filters.",
            ),
            (
                "CSV exports",
                safe_integer(
                    export_formats.get(
                        "csv",
                        export_formats.get(
                            "CSV",
                            0,
                        ),
                    )
                ),
                "Recorded comma-separated-value exports.",
            ),
            (
                "JSON exports",
                safe_integer(
                    export_formats.get(
                        "json",
                        export_formats.get(
                            "JSON",
                            0,
                        ),
                    )
                ),
                "Recorded structured JSON report exports.",
            ),
            (
                "Other formats",
                (
                    len(
                        export_history
                    )
                    - safe_integer(
                        export_formats.get(
                            "csv",
                            export_formats.get(
                                "CSV",
                                0,
                            ),
                        )
                    )
                    - safe_integer(
                        export_formats.get(
                            "json",
                            export_formats.get(
                                "JSON",
                                0,
                            ),
                        )
                    )
                ),
                "PDF, Excel or other registered export formats.",
            ),
        ]

        for column, payload in zip(
            export_metric_columns,
            export_payloads,
        ):
            with column:
                render_metric_card(
                    label=payload[
                        0
                    ],
                    value=str(
                        payload[
                            1
                        ]
                    ),
                    note=payload[
                        2
                    ],
                )

        st.write("")

        if export_history.empty:
            st.info(
                "No export events match the current filters."
            )

        else:
            st.dataframe(
                export_history,
                use_container_width=True,
                hide_index=True,
                height=560,
            )

    with integrity_tab:
        integrity_columns = st.columns(
            3
        )

        with integrity_columns[0]:
            render_health_card(
                label="Hash-chain verification",
                value=(
                    "Valid"
                    if integrity_result.get(
                        "valid"
                    )
                    else "Invalid"
                ),
                note=(
                    f"{integrity_result.get('checked_events', 0)} "
                    "events verified"
                ),
                status_class=(
                    "status-good"
                    if integrity_result.get(
                        "valid"
                    )
                    else "status-bad"
                ),
            )

        with integrity_columns[1]:
            render_health_card(
                label="Retention policy",
                value=(
                    f"{AUDIT_ENGINE.AUDIT_RETENTION_DAYS} days"
                ),
                note=(
                    "Configured audit-event retention period."
                ),
            )

        with integrity_columns[2]:
            render_health_card(
                label="Hash chaining",
                value=(
                    "Enabled"
                    if AUDIT_ENGINE.AUDIT_HASH_CHAIN_ENABLED
                    else "Disabled"
                ),
                note=(
                    "Tamper-evident sequential event hashing."
                ),
                status_class=(
                    "status-good"
                    if AUDIT_ENGINE.AUDIT_HASH_CHAIN_ENABLED
                    else "status-warning"
                ),
            )

        st.write("")

        st.markdown(
            """
            <div class="section-title">
                Integrity Verification Result
            </div>
            <div class="section-subtitle">
                Current sequential-hash validation result for the audit register.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.json(
            integrity_result,
            expanded=True,
        )

        st.markdown(
            """
            <div class="governance-note">
                Purging audit events changes the retained hash chain. Apply
                retention operations only under an approved governance
                procedure and preserve exported audit snapshots when required.
            </div>
            """,
            unsafe_allow_html=True,
        )

        maintenance_columns = st.columns(
            3
        )

        with maintenance_columns[0]:
            if st.button(
                "Verify integrity now",
                use_container_width=True,
            ):
                st.session_state[
                    "governance_integrity_result"
                ] = (
                    AUDIT_ENGINE.verify_hash_chain()
                )

                st.rerun()

        with maintenance_columns[1]:
            if st.button(
                "Export governance snapshot",
                use_container_width=True,
            ):
                output_path = (
                    create_governance_export(
                        events=filtered_events,
                        sessions=all_sessions,
                        integrity=integrity_result,
                        health=database_health,
                        summary=summary,
                    )
                )

                st.session_state[
                    "governance_export_path"
                ] = str(
                    output_path
                )

                try:
                    AUDIT_ENGINE.record_event(
                        source_component=(
                            "25_governance_dashboard"
                        ),
                        event_type=(
                            "governance_snapshot_exported"
                        ),
                        category="export",
                        outcome="success",
                        user_email=user.get(
                            "email"
                        ),
                        display_name=user.get(
                            "display_name"
                        ),
                        role=user.get(
                            "role"
                        ),
                        auth_source=user.get(
                            "auth_source"
                        ),
                        session_id=(
                            st.session_state.get(
                                "hevemind_audit_session_id"
                            )
                        ),
                        resource_type=(
                            "governance_snapshot"
                        ),
                        action="export",
                        details={
                            "export_format": "xlsx",
                            "record_count": len(
                                filtered_events
                            ),
                            "filename": (
                                output_path.name
                            ),
                        },
                    )

                except Exception:
                    pass

                st.success(
                    f"Governance snapshot created: {output_path}"
                )

        with maintenance_columns[2]:
            if st.button(
                "Apply retention purge",
                use_container_width=True,
            ):
                deleted_count = (
                    AUDIT_ENGINE.purge_expired_events()
                )

                st.warning(
                    f"Expired events deleted: {deleted_count}"
                )

                try:
                    AUDIT_ENGINE.record_event(
                        source_component=(
                            "25_governance_dashboard"
                        ),
                        event_type=(
                            "audit_retention_purge_executed"
                        ),
                        category="governance",
                        outcome="warning",
                        user_email=user.get(
                            "email"
                        ),
                        display_name=user.get(
                            "display_name"
                        ),
                        role=user.get(
                            "role"
                        ),
                        auth_source=user.get(
                            "auth_source"
                        ),
                        session_id=(
                            st.session_state.get(
                                "hevemind_audit_session_id"
                            )
                        ),
                        resource_type=(
                            "audit_database"
                        ),
                        resource_id=str(
                            AUDIT_DATABASE_PATH
                        ),
                        action="purge",
                        details={
                            "deleted_events": (
                                deleted_count
                            ),
                            "retention_days": (
                                AUDIT_ENGINE.AUDIT_RETENTION_DAYS
                            ),
                        },
                    )

                except Exception:
                    pass

    with raw_events_tab:
        st.markdown(
            """
            <div class="section-title">
                Audit Event Register
            </div>
            <div class="section-subtitle">
                Raw governance events matching the active user,
                category and monitoring-period filters.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if filtered_events.empty:
            st.info(
                "No audit events match the current filters."
            )

        else:
            visible_columns = [
                "timestamp_utc",
                "event_type",
                "category",
                "outcome",
                "user_email",
                "role",
                "source_component",
                "resource_type",
                "resource_id",
                "action",
                "message",
                "duration_ms",
                "session_id",
                "request_id",
            ]

            available_columns = [
                column
                for column in visible_columns
                if column
                in filtered_events.columns
            ]

            st.dataframe(
                filtered_events[
                    available_columns
                ],
                use_container_width=True,
                hide_index=True,
                height=620,
            )

            csv_payload = (
                filtered_events.to_csv(
                    index=False
                )
                .encode(
                    "utf-8"
                )
            )

            if st.download_button(
                "Download filtered audit register",
                data=csv_payload,
                file_name=(
                    "hevemind_filtered_audit_register.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            ):
                try:
                    AUDIT_ENGINE.record_event(
                        source_component=(
                            "25_governance_dashboard"
                        ),
                        event_type=(
                            "filtered_audit_register_downloaded"
                        ),
                        category="export",
                        outcome="success",
                        user_email=user.get(
                            "email"
                        ),
                        display_name=user.get(
                            "display_name"
                        ),
                        role=user.get(
                            "role"
                        ),
                        auth_source=user.get(
                            "auth_source"
                        ),
                        session_id=(
                            st.session_state.get(
                                "hevemind_audit_session_id"
                            )
                        ),
                        resource_type=(
                            "audit_register"
                        ),
                        action="download",
                        details={
                            "export_format": "csv",
                            "record_count": len(
                                filtered_events
                            ),
                            "filename": (
                                "hevemind_filtered_audit_register.csv"
                            ),
                        },
                    )

                except Exception:
                    pass


# ============================================================
# STANDALONE STREAMLIT ENTRY POINT
# ============================================================
def run_standalone_console() -> None:
    import streamlit as st

    st.set_page_config(
        page_title=(
            "HeveMind Governance Console"
        ),
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_governance_dashboard()


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HeveMind governance, security operations, "
            "audit analytics and administrative oversight."
        )
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print a governance summary to the terminal."
        ),
    )

    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_SUMMARY_DAYS,
        help=(
            "Governance summary period in days."
        ),
    )

    parser.add_argument(
        "--verify",
        action="store_true",
        help=(
            "Verify SQLite integrity and the audit hash chain."
        ),
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help=(
            "Create a complete Excel governance snapshot."
        ),
    )

    return parser.parse_args()


def terminal_summary(
    days: int,
) -> dict[str, Any]:
    AUDIT_ENGINE.initialise_database()

    start_utc = (
        utc_now()
        - timedelta(
            days=max(
                int(
                    days
                ),
                1,
            )
        )
    ).isoformat()

    events = AUDIT_ENGINE.query_events(
        start_utc=start_utc,
        limit=50000,
    )

    sessions = AUDIT_ENGINE.query_sessions(
        start_utc=start_utc,
    )

    summary = AUDIT_ENGINE.audit_summary(
        days=max(
            int(
                days
            ),
            1,
        )
    )

    health = read_database_health()

    integrity = (
        AUDIT_ENGINE.verify_hash_chain()
    )

    return {
        "generated_utc": utc_now_iso(),
        "period_days": max(
            int(
                days
            ),
            1,
        ),
        "summary": summary,
        "database_health": {
            "database_exists": (
                health.database_exists
            ),
            "database_size_bytes": (
                health.database_size_bytes
            ),
            "database_size_display": (
                format_filesize(
                    health.database_size_bytes
                )
            ),
            "integrity_ok": (
                health.integrity_ok
            ),
            "integrity_message": (
                health.integrity_message
            ),
            "event_rows": (
                health.event_rows
            ),
            "session_rows": (
                health.session_rows
            ),
            "active_sessions": (
                health.active_sessions
            ),
        },
        "hash_chain": integrity,
        "filtered_event_rows": len(
            events
        ),
        "filtered_session_rows": len(
            sessions
        ),
    }


def main() -> None:
    arguments = parse_arguments()

    action_requested = False

    if arguments.summary:
        print(
            json.dumps(
                terminal_summary(
                    days=arguments.days
                ),
                indent=4,
                ensure_ascii=False,
            )
        )

        action_requested = True

    if arguments.verify:
        health = read_database_health()

        result = {
            "database_integrity_ok": (
                health.integrity_ok
            ),
            "database_integrity_message": (
                health.integrity_message
            ),
            "hash_chain": (
                AUDIT_ENGINE.verify_hash_chain()
            ),
        }

        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
            )
        )

        action_requested = True

    if arguments.export:
        days = max(
            int(
                arguments.days
            ),
            1,
        )

        start_utc = (
            utc_now()
            - timedelta(
                days=days
            )
        ).isoformat()

        events = AUDIT_ENGINE.query_events(
            start_utc=start_utc,
            limit=50000,
        )

        sessions = AUDIT_ENGINE.query_sessions(
            start_utc=start_utc,
        )

        output_path = create_governance_export(
            events=events,
            sessions=sessions,
            integrity=(
                AUDIT_ENGINE.verify_hash_chain()
            ),
            health=read_database_health(),
            summary=(
                AUDIT_ENGINE.audit_summary(
                    days=days
                )
            ),
        )

        print(
            f"Governance snapshot exported: {output_path}"
        )

        action_requested = True

    if not action_requested:
        AUDIT_ENGINE.initialise_database()

        print(
            "\n"
            + "=" * 112
        )

        print(
            "HEVEMIND GOVERNANCE AND SECURITY OPERATIONS DASHBOARD"
        )

        print(
            "=" * 112
        )

        print(
            f"\nAudit engine:               {AUDIT_ENGINE_PATH}"
        )

        print(
            f"Audit database:             {AUDIT_DATABASE_PATH}"
        )

        print(
            f"Dashboard version:          {APP_VERSION}"
        )

        print(
            "\nRun with --help to view available commands."
        )


if __name__ == "__main__":
    try:
        import streamlit.runtime.scriptrunner

        run_standalone_console()

    except Exception:
        main()