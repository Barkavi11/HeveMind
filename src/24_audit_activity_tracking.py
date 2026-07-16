from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sqlite3
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Literal

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

LOG_DIR = ROOT_DIR / "logs"
EXPORT_DIR = LOG_DIR / "exports"

DATABASE_PATH = LOG_DIR / "hevemind_audit.db"
LEGACY_AUTH_LOG_PATH = LOG_DIR / "authentication_audit.jsonl"

DEFAULT_CSV_EXPORT_PATH = (
    EXPORT_DIR
    / "hevemind_audit_events.csv"
)

DEFAULT_SESSION_EXPORT_PATH = (
    EXPORT_DIR
    / "hevemind_audit_sessions.csv"
)


# ============================================================
# AUDIT CONFIGURATION
# ============================================================
AUDIT_SCHEMA_VERSION = "1.0"

AUDIT_HASH_CHAIN_ENABLED = (
    os.getenv(
        "HEVEMIND_AUDIT_HASH_CHAIN",
        "true",
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

AUDIT_RETENTION_DAYS = int(
    os.getenv(
        "HEVEMIND_AUDIT_RETENTION_DAYS",
        "365",
    )
)

AUDIT_STANDALONE_ADMIN_MODE = (
    os.getenv(
        "HEVEMIND_AUDIT_STANDALONE_ADMIN_MODE",
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

ALLOWED_EVENT_CATEGORIES = {
    "authentication",
    "session",
    "wafer_access",
    "search",
    "filter",
    "export",
    "navigation",
    "governance",
    "system",
}

ALLOWED_EVENT_OUTCOMES = {
    "success",
    "failure",
    "denied",
    "warning",
    "information",
}

SENSITIVE_DETAIL_KEYS = {
    "password",
    "password_hash",
    "token",
    "access_token",
    "refresh_token",
    "api_key",
    "secret",
    "client_secret",
    "authorization",
}


# ============================================================
# DATA MODELS
# ============================================================
@dataclass(frozen=True)
class AuditActor:
    email: str | None = None
    display_name: str | None = None
    role: str | None = None
    auth_source: str | None = None


@dataclass(frozen=True)
class AuditContext:
    session_id: str | None = None
    request_id: str | None = None
    source_component: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None


@dataclass(frozen=True)
class AuditEvent:
    event_type: str
    category: str
    outcome: str
    actor: AuditActor
    context: AuditContext
    resource_type: str | None = None
    resource_id: str | None = None
    action: str | None = None
    message: str | None = None
    details: dict[str, Any] | None = None
    duration_ms: float | None = None


# ============================================================
# GENERIC UTILITIES
# ============================================================
def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    return utc_now().isoformat()


def ensure_directories() -> None:
    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def normalise_optional_text(
    value: Any,
) -> str | None:
    if value is None:
        return None

    text = str(
        value
    ).strip()

    if not text:
        return None

    return text


def normalise_email(
    value: Any,
) -> str | None:
    text = normalise_optional_text(
        value
    )

    if text is None:
        return None

    return text.lower()


def json_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(
        value,
        datetime,
    ):
        return value.astimezone(
            timezone.utc
        ).isoformat()

    if isinstance(
        value,
        Path,
    ):
        return str(
            value
        )

    if isinstance(
        value,
        dict,
    ):
        return {
            str(
                key
            ): json_safe(
                item
            )
            for key, item in value.items()
        }

    if isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        return [
            json_safe(
                item
            )
            for item in value
        ]

    try:
        if pd.isna(
            value
        ):
            return None

    except (
        TypeError,
        ValueError,
    ):
        pass

    if hasattr(
        value,
        "item",
    ):
        try:
            return json_safe(
                value.item()
            )

        except Exception:
            pass

    return str(
        value
    )


def redact_sensitive_details(
    details: dict[str, Any] | None,
) -> dict[str, Any]:
    if not details:
        return {}

    def redact(
        value: Any,
        parent_key: str | None = None,
    ) -> Any:
        if (
            parent_key is not None
            and parent_key.strip().lower()
            in SENSITIVE_DETAIL_KEYS
        ):
            return "[REDACTED]"

        if isinstance(
            value,
            dict,
        ):
            return {
                str(
                    key
                ): redact(
                    item,
                    parent_key=str(
                        key
                    ),
                )
                for key, item in value.items()
            }

        if isinstance(
            value,
            (
                list,
                tuple,
                set,
            ),
        ):
            return [
                redact(
                    item
                )
                for item in value
            ]

        return json_safe(
            value
        )

    return redact(
        details
    )


def canonical_json(
    payload: dict[str, Any],
) -> str:
    return json.dumps(
        json_safe(
            payload
        ),
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    )


def calculate_event_hash(
    *,
    previous_hash: str | None,
    event_payload: dict[str, Any],
) -> str:
    canonical_payload = canonical_json(
        {
            "previous_hash": (
                previous_hash
                or ""
            ),
            "event": event_payload,
        }
    )

    return hashlib.sha256(
        canonical_payload.encode(
            "utf-8"
        )
    ).hexdigest()


# ============================================================
# DATABASE MANAGEMENT
# ============================================================
@contextmanager
def database_connection() -> Iterator[
    sqlite3.Connection
]:
    ensure_directories()

    connection = sqlite3.connect(
        DATABASE_PATH,
        timeout=30,
    )

    connection.row_factory = (
        sqlite3.Row
    )

    connection.execute(
        "PRAGMA foreign_keys = ON"
    )

    connection.execute(
        "PRAGMA journal_mode = WAL"
    )

    connection.execute(
        "PRAGMA synchronous = NORMAL"
    )

    try:
        yield connection
        connection.commit()

    except Exception:
        connection.rollback()
        raise

    finally:
        connection.close()


def initialise_database() -> None:
    with database_connection() as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS audit_metadata (
                metadata_key TEXT PRIMARY KEY,
                metadata_value TEXT NOT NULL,
                updated_utc TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS audit_sessions (
                session_id TEXT PRIMARY KEY,
                user_email TEXT,
                display_name TEXT,
                role TEXT,
                auth_source TEXT,
                started_utc TEXT NOT NULL,
                last_activity_utc TEXT NOT NULL,
                ended_utc TEXT,
                duration_seconds REAL,
                termination_reason TEXT,
                client_ip TEXT,
                user_agent TEXT,
                source_component TEXT
            );

            CREATE TABLE IF NOT EXISTS audit_events (
                event_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                event_type TEXT NOT NULL,
                category TEXT NOT NULL,
                outcome TEXT NOT NULL,
                user_email TEXT,
                display_name TEXT,
                role TEXT,
                auth_source TEXT,
                session_id TEXT,
                request_id TEXT,
                source_component TEXT,
                client_ip TEXT,
                user_agent TEXT,
                resource_type TEXT,
                resource_id TEXT,
                action TEXT,
                message TEXT,
                details_json TEXT NOT NULL,
                duration_ms REAL,
                previous_event_hash TEXT,
                event_hash TEXT,
                created_epoch REAL NOT NULL,
                FOREIGN KEY(session_id)
                    REFERENCES audit_sessions(session_id)
                    ON DELETE SET NULL
            );

            CREATE INDEX IF NOT EXISTS idx_audit_events_timestamp
            ON audit_events(timestamp_utc);

            CREATE INDEX IF NOT EXISTS idx_audit_events_user
            ON audit_events(user_email);

            CREATE INDEX IF NOT EXISTS idx_audit_events_category
            ON audit_events(category);

            CREATE INDEX IF NOT EXISTS idx_audit_events_type
            ON audit_events(event_type);

            CREATE INDEX IF NOT EXISTS idx_audit_events_resource
            ON audit_events(resource_type, resource_id);

            CREATE INDEX IF NOT EXISTS idx_audit_events_session
            ON audit_events(session_id);

            CREATE INDEX IF NOT EXISTS idx_audit_sessions_user
            ON audit_sessions(user_email);

            CREATE INDEX IF NOT EXISTS idx_audit_sessions_started
            ON audit_sessions(started_utc);
            """
        )

        connection.execute(
            """
            INSERT INTO audit_metadata (
                metadata_key,
                metadata_value,
                updated_utc
            )
            VALUES (?, ?, ?)
            ON CONFLICT(metadata_key)
            DO UPDATE SET
                metadata_value = excluded.metadata_value,
                updated_utc = excluded.updated_utc
            """,
            (
                "schema_version",
                AUDIT_SCHEMA_VERSION,
                utc_now_iso(),
            ),
        )


def get_last_event_hash(
    connection: sqlite3.Connection,
) -> str | None:
    row = connection.execute(
        """
        SELECT event_hash
        FROM audit_events
        WHERE event_hash IS NOT NULL
        ORDER BY created_epoch DESC, event_id DESC
        LIMIT 1
        """
    ).fetchone()

    if row is None:
        return None

    return normalise_optional_text(
        row[
            "event_hash"
        ]
    )


# ============================================================
# AUDIT LOGGER
# ============================================================
class AuditLogger:
    def __init__(
        self,
        source_component: str,
    ) -> None:
        self.source_component = (
            normalise_optional_text(
                source_component
            )
            or "unknown"
        )

        initialise_database()

    def start_session(
        self,
        actor: AuditActor,
        *,
        client_ip: str | None = None,
        user_agent: str | None = None,
        session_id: str | None = None,
    ) -> str:
        resolved_session_id = (
            normalise_optional_text(
                session_id
            )
            or str(
                uuid.uuid4()
            )
        )

        timestamp = utc_now_iso()

        with database_connection() as connection:
            connection.execute(
                """
                INSERT INTO audit_sessions (
                    session_id,
                    user_email,
                    display_name,
                    role,
                    auth_source,
                    started_utc,
                    last_activity_utc,
                    ended_utc,
                    duration_seconds,
                    termination_reason,
                    client_ip,
                    user_agent,
                    source_component
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?,
                    ?, NULL, NULL, NULL,
                    ?, ?, ?
                )
                ON CONFLICT(session_id)
                DO UPDATE SET
                    user_email = excluded.user_email,
                    display_name = excluded.display_name,
                    role = excluded.role,
                    auth_source = excluded.auth_source,
                    last_activity_utc = excluded.last_activity_utc,
                    client_ip = excluded.client_ip,
                    user_agent = excluded.user_agent,
                    source_component = excluded.source_component
                """,
                (
                    resolved_session_id,
                    normalise_email(
                        actor.email
                    ),
                    normalise_optional_text(
                        actor.display_name
                    ),
                    normalise_optional_text(
                        actor.role
                    ),
                    normalise_optional_text(
                        actor.auth_source
                    ),
                    timestamp,
                    timestamp,
                    normalise_optional_text(
                        client_ip
                    ),
                    normalise_optional_text(
                        user_agent
                    ),
                    self.source_component,
                ),
            )

        self.log(
            AuditEvent(
                event_type="session_started",
                category="session",
                outcome="success",
                actor=actor,
                context=AuditContext(
                    session_id=resolved_session_id,
                    source_component=(
                        self.source_component
                    ),
                    client_ip=client_ip,
                    user_agent=user_agent,
                ),
                action="start_session",
                message=(
                    "Authenticated application session started."
                ),
            )
        )

        return resolved_session_id

    def touch_session(
        self,
        session_id: str,
    ) -> None:
        with database_connection() as connection:
            connection.execute(
                """
                UPDATE audit_sessions
                SET last_activity_utc = ?
                WHERE session_id = ?
                """,
                (
                    utc_now_iso(),
                    session_id,
                ),
            )

    def end_session(
        self,
        session_id: str,
        actor: AuditActor,
        *,
        termination_reason: str,
    ) -> None:
        timestamp = utc_now()

        with database_connection() as connection:
            row = connection.execute(
                """
                SELECT started_utc
                FROM audit_sessions
                WHERE session_id = ?
                """,
                (
                    session_id,
                ),
            ).fetchone()

            duration_seconds: (
                float
                | None
            ) = None

            if row is not None:
                try:
                    started = datetime.fromisoformat(
                        str(
                            row[
                                "started_utc"
                            ]
                        )
                    )

                    duration_seconds = max(
                        (
                            timestamp
                            - started
                        ).total_seconds(),
                        0.0,
                    )

                except ValueError:
                    duration_seconds = None

            connection.execute(
                """
                UPDATE audit_sessions
                SET
                    last_activity_utc = ?,
                    ended_utc = ?,
                    duration_seconds = ?,
                    termination_reason = ?
                WHERE session_id = ?
                """,
                (
                    timestamp.isoformat(),
                    timestamp.isoformat(),
                    duration_seconds,
                    termination_reason,
                    session_id,
                ),
            )

        self.log(
            AuditEvent(
                event_type="session_ended",
                category="session",
                outcome="success",
                actor=actor,
                context=AuditContext(
                    session_id=session_id,
                    source_component=(
                        self.source_component
                    ),
                ),
                action="end_session",
                message=(
                    "Authenticated application session ended."
                ),
                details={
                    "termination_reason": (
                        termination_reason
                    ),
                    "duration_seconds": (
                        duration_seconds
                    ),
                },
            )
        )

    def log(
        self,
        event: AuditEvent,
    ) -> str:
        category = (
            event.category.strip().lower()
        )

        outcome = (
            event.outcome.strip().lower()
        )

        if category not in ALLOWED_EVENT_CATEGORIES:
            raise ValueError(
                f"Unsupported audit category: {category}"
            )

        if outcome not in ALLOWED_EVENT_OUTCOMES:
            raise ValueError(
                f"Unsupported audit outcome: {outcome}"
            )

        event_id = str(
            uuid.uuid4()
        )

        timestamp = utc_now()
        timestamp_iso = timestamp.isoformat()

        details = redact_sensitive_details(
            event.details
        )

        details_json = canonical_json(
            details
        )

        event_payload = {
            "event_id": event_id,
            "timestamp_utc": timestamp_iso,
            "event_type": event.event_type,
            "category": category,
            "outcome": outcome,
            "user_email": normalise_email(
                event.actor.email
            ),
            "display_name": normalise_optional_text(
                event.actor.display_name
            ),
            "role": normalise_optional_text(
                event.actor.role
            ),
            "auth_source": normalise_optional_text(
                event.actor.auth_source
            ),
            "session_id": normalise_optional_text(
                event.context.session_id
            ),
            "request_id": normalise_optional_text(
                event.context.request_id
            ),
            "source_component": (
                normalise_optional_text(
                    event.context.source_component
                )
                or self.source_component
            ),
            "client_ip": normalise_optional_text(
                event.context.client_ip
            ),
            "user_agent": normalise_optional_text(
                event.context.user_agent
            ),
            "resource_type": normalise_optional_text(
                event.resource_type
            ),
            "resource_id": normalise_optional_text(
                event.resource_id
            ),
            "action": normalise_optional_text(
                event.action
            ),
            "message": normalise_optional_text(
                event.message
            ),
            "details_json": details_json,
            "duration_ms": event.duration_ms,
            "created_epoch": timestamp.timestamp(),
        }

        with database_connection() as connection:
            previous_hash = (
                get_last_event_hash(
                    connection
                )
                if AUDIT_HASH_CHAIN_ENABLED
                else None
            )

            event_hash = (
                calculate_event_hash(
                    previous_hash=previous_hash,
                    event_payload=event_payload,
                )
                if AUDIT_HASH_CHAIN_ENABLED
                else None
            )

            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id,
                    timestamp_utc,
                    event_type,
                    category,
                    outcome,
                    user_email,
                    display_name,
                    role,
                    auth_source,
                    session_id,
                    request_id,
                    source_component,
                    client_ip,
                    user_agent,
                    resource_type,
                    resource_id,
                    action,
                    message,
                    details_json,
                    duration_ms,
                    previous_event_hash,
                    event_hash,
                    created_epoch
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?
                )
                """,
                (
                    event_id,
                    timestamp_iso,
                    event.event_type,
                    category,
                    outcome,
                    event_payload[
                        "user_email"
                    ],
                    event_payload[
                        "display_name"
                    ],
                    event_payload[
                        "role"
                    ],
                    event_payload[
                        "auth_source"
                    ],
                    event_payload[
                        "session_id"
                    ],
                    event_payload[
                        "request_id"
                    ],
                    event_payload[
                        "source_component"
                    ],
                    event_payload[
                        "client_ip"
                    ],
                    event_payload[
                        "user_agent"
                    ],
                    event_payload[
                        "resource_type"
                    ],
                    event_payload[
                        "resource_id"
                    ],
                    event_payload[
                        "action"
                    ],
                    event_payload[
                        "message"
                    ],
                    details_json,
                    event.duration_ms,
                    previous_hash,
                    event_hash,
                    timestamp.timestamp(),
                ),
            )

            if event.context.session_id:
                connection.execute(
                    """
                    UPDATE audit_sessions
                    SET last_activity_utc = ?
                    WHERE session_id = ?
                    """,
                    (
                        timestamp_iso,
                        event.context.session_id,
                    ),
                )

        return event_id

    def log_authentication(
        self,
        *,
        actor: AuditActor,
        success: bool,
        event_type: str,
        context: AuditContext | None = None,
        message: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type=event_type,
                category="authentication",
                outcome=(
                    "success"
                    if success
                    else "failure"
                ),
                actor=actor,
                context=(
                    context
                    or AuditContext(
                        source_component=(
                            self.source_component
                        )
                    )
                ),
                action=event_type,
                message=message,
                details=details,
            )
        )

    def log_wafer_view(
        self,
        *,
        actor: AuditActor,
        context: AuditContext,
        wafer_id: str,
        decision: str | None = None,
        failure_probability: float | None = None,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type="wafer_report_viewed",
                category="wafer_access",
                outcome="success",
                actor=actor,
                context=context,
                resource_type="wafer_report",
                resource_id=wafer_id,
                action="view",
                message=(
                    "Validated wafer report opened."
                ),
                details={
                    "decision": decision,
                    "failure_probability": (
                        failure_probability
                    ),
                },
            )
        )

    def log_search(
        self,
        *,
        actor: AuditActor,
        context: AuditContext,
        query: str,
        result_count: int,
        search_scope: str,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type="search_performed",
                category="search",
                outcome="success",
                actor=actor,
                context=context,
                resource_type="search",
                action="search",
                message=(
                    "Application search executed."
                ),
                details={
                    "query": query,
                    "result_count": result_count,
                    "search_scope": search_scope,
                },
            )
        )

    def log_filter(
        self,
        *,
        actor: AuditActor,
        context: AuditContext,
        filter_name: str,
        filter_value: Any,
        result_count: int | None = None,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type="filter_applied",
                category="filter",
                outcome="success",
                actor=actor,
                context=context,
                resource_type="dashboard_filter",
                resource_id=filter_name,
                action="apply_filter",
                message=(
                    "Dashboard filter applied."
                ),
                details={
                    "filter_name": filter_name,
                    "filter_value": filter_value,
                    "result_count": result_count,
                },
            )
        )

    def log_export(
        self,
        *,
        actor: AuditActor,
        context: AuditContext,
        export_format: str,
        resource_type: str,
        resource_id: str | None = None,
        record_count: int | None = None,
        filename: str | None = None,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type="report_exported",
                category="export",
                outcome="success",
                actor=actor,
                context=context,
                resource_type=resource_type,
                resource_id=resource_id,
                action="export",
                message=(
                    "Engineering data exported."
                ),
                details={
                    "export_format": export_format,
                    "record_count": record_count,
                    "filename": filename,
                },
            )
        )

    def log_navigation(
        self,
        *,
        actor: AuditActor,
        context: AuditContext,
        page_name: str,
    ) -> str:
        return self.log(
            AuditEvent(
                event_type="page_opened",
                category="navigation",
                outcome="information",
                actor=actor,
                context=context,
                resource_type="dashboard_page",
                resource_id=page_name,
                action="navigate",
                message=(
                    "Dashboard page opened."
                ),
            )
        )


# ============================================================
# PUBLIC CONVENIENCE FUNCTIONS
# ============================================================
_DEFAULT_LOGGERS: dict[
    str,
    AuditLogger,
] = {}


def get_logger(
    source_component: str,
) -> AuditLogger:
    key = (
        normalise_optional_text(
            source_component
        )
        or "unknown"
    )

    if key not in _DEFAULT_LOGGERS:
        _DEFAULT_LOGGERS[
            key
        ] = AuditLogger(
            source_component=key
        )

    return _DEFAULT_LOGGERS[
        key
    ]


def record_event(
    *,
    source_component: str,
    event_type: str,
    category: str,
    outcome: str,
    user_email: str | None = None,
    display_name: str | None = None,
    role: str | None = None,
    auth_source: str | None = None,
    session_id: str | None = None,
    request_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    action: str | None = None,
    message: str | None = None,
    details: dict[str, Any] | None = None,
    duration_ms: float | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> str:
    logger = get_logger(
        source_component
    )

    return logger.log(
        AuditEvent(
            event_type=event_type,
            category=category,
            outcome=outcome,
            actor=AuditActor(
                email=user_email,
                display_name=display_name,
                role=role,
                auth_source=auth_source,
            ),
            context=AuditContext(
                session_id=session_id,
                request_id=request_id,
                source_component=source_component,
                client_ip=client_ip,
                user_agent=user_agent,
            ),
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            message=message,
            details=details,
            duration_ms=duration_ms,
        )
    )


# ============================================================
# LEGACY AUTH LOG IMPORT
# ============================================================
def legacy_event_already_imported(
    connection: sqlite3.Connection,
    legacy_fingerprint: str,
) -> bool:
    row = connection.execute(
        """
        SELECT 1
        FROM audit_events
        WHERE json_extract(
            details_json,
            '$.legacy_fingerprint'
        ) = ?
        LIMIT 1
        """,
        (
            legacy_fingerprint,
        ),
    ).fetchone()

    return row is not None


def import_legacy_authentication_log() -> int:
    initialise_database()

    if not LEGACY_AUTH_LOG_PATH.exists():
        return 0

    imported_count = 0
    logger = get_logger(
        "21_secure_access_gateway"
    )

    with LEGACY_AUTH_LOG_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(
            file,
            start=1,
        ):
            stripped = line.strip()

            if not stripped:
                continue

            try:
                payload = json.loads(
                    stripped
                )

            except json.JSONDecodeError:
                continue

            fingerprint = hashlib.sha256(
                (
                    str(
                        line_number
                    )
                    + "|"
                    + stripped
                ).encode(
                    "utf-8"
                )
            ).hexdigest()

            with database_connection() as connection:
                if legacy_event_already_imported(
                    connection,
                    fingerprint,
                ):
                    continue

            event_type = str(
                payload.get(
                    "event_type",
                    "legacy_auth_event",
                )
            )

            success_value = payload.get(
                "success"
            )

            if success_value is True:
                outcome = "success"

            elif success_value is False:
                outcome = "failure"

            else:
                outcome = "information"

            logger.log(
                AuditEvent(
                    event_type=event_type,
                    category=(
                        "session"
                        if "session"
                        in event_type.lower()
                        else "authentication"
                    ),
                    outcome=outcome,
                    actor=AuditActor(
                        email=payload.get(
                            "email"
                        ),
                        role=payload.get(
                            "role"
                        ),
                        auth_source=payload.get(
                            "auth_source"
                        ),
                    ),
                    context=AuditContext(
                        source_component=(
                            "21_secure_access_gateway"
                        ),
                    ),
                    action=event_type,
                    message=payload.get(
                        "details"
                    ),
                    details={
                        "legacy_timestamp_utc": (
                            payload.get(
                                "timestamp_utc"
                            )
                        ),
                        "legacy_fingerprint": (
                            fingerprint
                        ),
                        "legacy_line_number": (
                            line_number
                        ),
                    },
                )
            )

            imported_count += 1

    return imported_count


# ============================================================
# QUERY AND REPORTING
# ============================================================
def query_events(
    *,
    start_utc: str | None = None,
    end_utc: str | None = None,
    user_email: str | None = None,
    category: str | None = None,
    event_type: str | None = None,
    outcome: str | None = None,
    resource_id: str | None = None,
    session_id: str | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    initialise_database()

    clauses: list[str] = []
    parameters: list[Any] = []

    if start_utc:
        clauses.append(
            "timestamp_utc >= ?"
        )
        parameters.append(
            start_utc
        )

    if end_utc:
        clauses.append(
            "timestamp_utc <= ?"
        )
        parameters.append(
            end_utc
        )

    if user_email:
        clauses.append(
            "LOWER(user_email) = LOWER(?)"
        )
        parameters.append(
            user_email
        )

    if category:
        clauses.append(
            "category = ?"
        )
        parameters.append(
            category
        )

    if event_type:
        clauses.append(
            "event_type = ?"
        )
        parameters.append(
            event_type
        )

    if outcome:
        clauses.append(
            "outcome = ?"
        )
        parameters.append(
            outcome
        )

    if resource_id:
        clauses.append(
            "resource_id = ?"
        )
        parameters.append(
            resource_id
        )

    if session_id:
        clauses.append(
            "session_id = ?"
        )
        parameters.append(
            session_id
        )

    where_sql = (
        "WHERE "
        + " AND ".join(
            clauses
        )
        if clauses
        else ""
    )

    limit_sql = ""

    if limit is not None:
        limit_sql = "LIMIT ?"
        parameters.append(
            int(
                limit
            )
        )

    query = f"""
        SELECT
            event_id,
            timestamp_utc,
            event_type,
            category,
            outcome,
            user_email,
            display_name,
            role,
            auth_source,
            session_id,
            request_id,
            source_component,
            client_ip,
            user_agent,
            resource_type,
            resource_id,
            action,
            message,
            details_json,
            duration_ms,
            previous_event_hash,
            event_hash
        FROM audit_events
        {where_sql}
        ORDER BY created_epoch DESC, event_id DESC
        {limit_sql}
    """

    with database_connection() as connection:
        dataframe = pd.read_sql_query(
            query,
            connection,
            params=parameters,
        )

    return dataframe


def query_sessions(
    *,
    start_utc: str | None = None,
    end_utc: str | None = None,
    user_email: str | None = None,
    active_only: bool = False,
) -> pd.DataFrame:
    initialise_database()

    clauses: list[str] = []
    parameters: list[Any] = []

    if start_utc:
        clauses.append(
            "started_utc >= ?"
        )
        parameters.append(
            start_utc
        )

    if end_utc:
        clauses.append(
            "started_utc <= ?"
        )
        parameters.append(
            end_utc
        )

    if user_email:
        clauses.append(
            "LOWER(user_email) = LOWER(?)"
        )
        parameters.append(
            user_email
        )

    if active_only:
        clauses.append(
            "ended_utc IS NULL"
        )

    where_sql = (
        "WHERE "
        + " AND ".join(
            clauses
        )
        if clauses
        else ""
    )

    query = f"""
        SELECT
            session_id,
            user_email,
            display_name,
            role,
            auth_source,
            started_utc,
            last_activity_utc,
            ended_utc,
            duration_seconds,
            termination_reason,
            client_ip,
            user_agent,
            source_component
        FROM audit_sessions
        {where_sql}
        ORDER BY started_utc DESC
    """

    with database_connection() as connection:
        return pd.read_sql_query(
            query,
            connection,
            params=parameters,
        )


def audit_summary(
    days: int = 30,
) -> dict[str, Any]:
    start_datetime = (
        utc_now()
        - timedelta(
            days=days
        )
    )

    events = query_events(
        start_utc=(
            start_datetime.isoformat()
        )
    )

    sessions = query_sessions(
        start_utc=(
            start_datetime.isoformat()
        )
    )

    if events.empty:
        return {
            "period_days": days,
            "total_events": 0,
            "unique_users": 0,
            "failed_events": 0,
            "wafer_views": 0,
            "exports": 0,
            "searches": 0,
            "sessions": int(
                len(
                    sessions
                )
            ),
            "mean_session_minutes": None,
        }

    mean_session_minutes: (
        float
        | None
    ) = None

    if (
        not sessions.empty
        and "duration_seconds"
        in sessions.columns
    ):
        numeric_duration = pd.to_numeric(
            sessions[
                "duration_seconds"
            ],
            errors="coerce",
        )

        if numeric_duration.notna().any():
            mean_session_minutes = float(
                numeric_duration.mean()
                / 60.0
            )

    return {
        "period_days": days,
        "total_events": int(
            len(
                events
            )
        ),
        "unique_users": int(
            events[
                "user_email"
            ]
            .dropna()
            .nunique()
        ),
        "failed_events": int(
            events[
                "outcome"
            ]
            .isin(
                [
                    "failure",
                    "denied",
                ]
            )
            .sum()
        ),
        "wafer_views": int(
            (
                events[
                    "event_type"
                ]
                == "wafer_report_viewed"
            ).sum()
        ),
        "exports": int(
            (
                events[
                    "category"
                ]
                == "export"
            ).sum()
        ),
        "searches": int(
            (
                events[
                    "category"
                ]
                == "search"
            ).sum()
        ),
        "sessions": int(
            len(
                sessions
            )
        ),
        "mean_session_minutes": (
            mean_session_minutes
        ),
    }


# ============================================================
# INTEGRITY VERIFICATION
# ============================================================
def verify_hash_chain() -> dict[str, Any]:
    initialise_database()

    with database_connection() as connection:
        rows = connection.execute(
            """
            SELECT
                event_id,
                timestamp_utc,
                event_type,
                category,
                outcome,
                user_email,
                display_name,
                role,
                auth_source,
                session_id,
                request_id,
                source_component,
                client_ip,
                user_agent,
                resource_type,
                resource_id,
                action,
                message,
                details_json,
                duration_ms,
                previous_event_hash,
                event_hash,
                created_epoch
            FROM audit_events
            ORDER BY created_epoch ASC, event_id ASC
            """
        ).fetchall()

    previous_hash: str | None = None
    checked_events = 0

    for row in rows:
        if row[
            "event_hash"
        ] is None:
            continue

        event_payload = {
            "event_id": row[
                "event_id"
            ],
            "timestamp_utc": row[
                "timestamp_utc"
            ],
            "event_type": row[
                "event_type"
            ],
            "category": row[
                "category"
            ],
            "outcome": row[
                "outcome"
            ],
            "user_email": row[
                "user_email"
            ],
            "display_name": row[
                "display_name"
            ],
            "role": row[
                "role"
            ],
            "auth_source": row[
                "auth_source"
            ],
            "session_id": row[
                "session_id"
            ],
            "request_id": row[
                "request_id"
            ],
            "source_component": row[
                "source_component"
            ],
            "client_ip": row[
                "client_ip"
            ],
            "user_agent": row[
                "user_agent"
            ],
            "resource_type": row[
                "resource_type"
            ],
            "resource_id": row[
                "resource_id"
            ],
            "action": row[
                "action"
            ],
            "message": row[
                "message"
            ],
            "details_json": row[
                "details_json"
            ],
            "duration_ms": row[
                "duration_ms"
            ],
            "created_epoch": row[
                "created_epoch"
            ],
        }

        expected_hash = calculate_event_hash(
            previous_hash=previous_hash,
            event_payload=event_payload,
        )

        if row[
            "previous_event_hash"
        ] != previous_hash:
            return {
                "valid": False,
                "checked_events": checked_events,
                "failed_event_id": row[
                    "event_id"
                ],
                "reason": (
                    "Previous-event hash mismatch."
                ),
            }

        if row[
            "event_hash"
        ] != expected_hash:
            return {
                "valid": False,
                "checked_events": checked_events,
                "failed_event_id": row[
                    "event_id"
                ],
                "reason": (
                    "Event hash mismatch."
                ),
            }

        previous_hash = row[
            "event_hash"
        ]

        checked_events += 1

    return {
        "valid": True,
        "checked_events": checked_events,
        "failed_event_id": None,
        "reason": None,
    }


# ============================================================
# DATA RETENTION
# ============================================================
def purge_expired_events(
    retention_days: int = (
        AUDIT_RETENTION_DAYS
    ),
) -> int:
    cutoff_epoch = (
        utc_now()
        - timedelta(
            days=retention_days
        )
    ).timestamp()

    with database_connection() as connection:
        cursor = connection.execute(
            """
            DELETE FROM audit_events
            WHERE created_epoch < ?
            """,
            (
                cutoff_epoch,
            ),
        )

        deleted_count = int(
            cursor.rowcount
            if cursor.rowcount is not None
            else 0
        )

        connection.execute(
            """
            DELETE FROM audit_sessions
            WHERE
                ended_utc IS NOT NULL
                AND started_utc < ?
            """,
            (
                datetime.fromtimestamp(
                    cutoff_epoch,
                    tz=timezone.utc,
                ).isoformat(),
            ),
        )

    return deleted_count


# ============================================================
# EXPORTS
# ============================================================
def export_events_csv(
    path: Path = (
        DEFAULT_CSV_EXPORT_PATH
    ),
) -> Path:
    dataframe = query_events()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    return path


def export_sessions_csv(
    path: Path = (
        DEFAULT_SESSION_EXPORT_PATH
    ),
) -> Path:
    dataframe = query_sessions()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe.to_csv(
        path,
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    return path


# ============================================================
# STREAMLIT AUDIT CONSOLE
# ============================================================
def run_streamlit_console() -> None:
    import streamlit as st

    st.set_page_config(
        page_title=(
            "HeveMind Audit Console"
        ),
        page_icon=None,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown(
        """
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
                --white: #ffffff;
                --green: #1b7f5b;
                --green-bg: #e6f4ed;
                --amber: #a96b00;
                --amber-bg: #fff3d9;
                --red: #b43b3b;
                --red-bg: #fdeaea;
            }

            .stApp {
                background:
                    linear-gradient(
                        180deg,
                        #f5f8fb 0%,
                        #edf3f7 100%
                    );
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

            .audit-header {
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
                    rgba(7, 20, 38, 0.16);
                margin-bottom: 1rem;
            }

            .audit-title {
                font-size: 1.75rem;
                font-weight: 780;
            }

            .audit-subtitle {
                color: #d8e8f4;
                font-size: 0.9rem;
                margin-top: 0.35rem;
            }

            .audit-metric {
                min-height: 120px;
                background: #ffffff;
                border: 1px solid var(--slate-200);
                border-radius: 14px;
                padding: 0.9rem 1rem;
                box-shadow:
                    0 7px 20px
                    rgba(21, 48, 75, 0.06);
            }

            .audit-metric-label {
                color: var(--slate-500);
                font-size: 0.72rem;
                font-weight: 740;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            .audit-metric-value {
                color: var(--navy-950);
                font-size: 1.55rem;
                font-weight: 780;
                margin-top: 0.38rem;
            }

            .audit-metric-note {
                color: var(--slate-700);
                font-size: 0.77rem;
                margin-top: 0.35rem;
            }

            [data-testid="stDataFrame"] {
                border: 1px solid var(--slate-200);
                border-radius: 12px;
                overflow: hidden;
                background: #ffffff;
            }

            footer {
                visibility: hidden;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    initialise_database()

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
        user_role != "admin"
        and not AUDIT_STANDALONE_ADMIN_MODE
    ):
        st.error(
            "Administrator access is required to open the audit console."
        )

        st.stop()

    st.markdown(
        """
        <div class="audit-header">
            <div class="audit-title">
                HeveMind Audit and Activity Console
            </div>
            <div class="audit-subtitle">
                Authentication, sessions, wafer access, searches,
                filters, exports and governance events
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    period_days = st.sidebar.selectbox(
        "Summary period",
        options=[
            1,
            7,
            30,
            90,
            365,
        ],
        index=2,
        format_func=lambda value: (
            f"Last {value} day"
            if value == 1
            else f"Last {value} days"
        ),
    )

    summary = audit_summary(
        days=int(
            period_days
        )
    )

    metric_columns = st.columns(
        6
    )

    metric_data = [
        (
            "Audit events",
            summary[
                "total_events"
            ],
            "Events recorded in the selected period.",
        ),
        (
            "Unique users",
            summary[
                "unique_users"
            ],
            "Distinct authenticated user accounts.",
        ),
        (
            "Failed or denied",
            summary[
                "failed_events"
            ],
            "Authentication or access events requiring attention.",
        ),
        (
            "Wafer views",
            summary[
                "wafer_views"
            ],
            "Validated wafer reports opened.",
        ),
        (
            "Exports",
            summary[
                "exports"
            ],
            "Data or engineering report exports.",
        ),
        (
            "Sessions",
            summary[
                "sessions"
            ],
            "Authenticated sessions recorded.",
        ),
    ]

    for column, (
        label,
        value,
        note,
    ) in zip(
        metric_columns,
        metric_data,
    ):
        with column:
            st.markdown(
                f"""
                <div class="audit-metric">
                    <div class="audit-metric-label">
                        {label}
                    </div>
                    <div class="audit-metric-value">
                        {value}
                    </div>
                    <div class="audit-metric-note">
                        {note}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    filter_columns = st.columns(
        5
    )

    with filter_columns[0]:
        selected_category = st.selectbox(
            "Category",
            options=[
                "All",
                *sorted(
                    ALLOWED_EVENT_CATEGORIES
                ),
            ],
        )

    with filter_columns[1]:
        selected_outcome = st.selectbox(
            "Outcome",
            options=[
                "All",
                *sorted(
                    ALLOWED_EVENT_OUTCOMES
                ),
            ],
        )

    all_events = query_events(
        limit=5000
    )

    available_users = sorted(
        all_events[
            "user_email"
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ) if not all_events.empty else []

    with filter_columns[2]:
        selected_user = st.selectbox(
            "User",
            options=[
                "All",
                *available_users,
            ],
        )

    with filter_columns[3]:
        selected_resource = st.text_input(
            "Resource ID",
            placeholder="Wafer or resource identifier",
        )

    with filter_columns[4]:
        maximum_rows = st.selectbox(
            "Maximum rows",
            options=[
                100,
                250,
                500,
                1000,
                5000,
            ],
            index=2,
        )

    start_utc = (
        utc_now()
        - timedelta(
            days=int(
                period_days
            )
        )
    ).isoformat()

    events = query_events(
        start_utc=start_utc,
        user_email=(
            None
            if selected_user == "All"
            else selected_user
        ),
        category=(
            None
            if selected_category == "All"
            else selected_category
        ),
        outcome=(
            None
            if selected_outcome == "All"
            else selected_outcome
        ),
        resource_id=(
            selected_resource.strip()
            or None
        ),
        limit=int(
            maximum_rows
        ),
    )

    tab_events, tab_sessions, tab_integrity = st.tabs(
        [
            "Audit Events",
            "Sessions",
            "Integrity and Export",
        ]
    )

    with tab_events:
        if events.empty:
            st.info(
                "No audit events match the selected filters."
            )

        else:
            display_columns = [
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
            ]

            st.dataframe(
                events[
                    display_columns
                ],
                use_container_width=True,
                hide_index=True,
                height=560,
            )

            category_counts = (
                events[
                    "category"
                ]
                .value_counts()
                .rename_axis(
                    "category"
                )
                .reset_index(
                    name="events"
                )
            )

            st.bar_chart(
                category_counts.set_index(
                    "category"
                ),
                height=300,
            )

    with tab_sessions:
        sessions = query_sessions(
            start_utc=start_utc,
            user_email=(
                None
                if selected_user == "All"
                else selected_user
            ),
        )

        if sessions.empty:
            st.info(
                "No sessions match the selected filters."
            )

        else:
            st.dataframe(
                sessions,
                use_container_width=True,
                hide_index=True,
                height=560,
            )

    with tab_integrity:
        integrity = verify_hash_chain()

        if integrity[
            "valid"
        ]:
            st.success(
                "Audit hash-chain verification passed."
            )

        else:
            st.error(
                "Audit hash-chain verification failed."
            )

        st.json(
            integrity,
            expanded=True,
        )

        export_columns = st.columns(
            2
        )

        with export_columns[0]:
            event_csv = events.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                "Download filtered audit events",
                data=event_csv,
                file_name=(
                    "hevemind_filtered_audit_events.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )

        with export_columns[1]:
            session_frame = query_sessions(
                start_utc=start_utc
            )

            session_csv = session_frame.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            st.download_button(
                "Download session register",
                data=session_csv,
                file_name=(
                    "hevemind_audit_sessions.csv"
                ),
                mime="text/csv",
                use_container_width=True,
            )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HeveMind audit logging, activity tracking, "
            "reporting and integrity verification."
        )
    )

    parser.add_argument(
        "--initialise",
        action="store_true",
        help=(
            "Initialise the audit SQLite database."
        ),
    )

    parser.add_argument(
        "--import-auth-log",
        action="store_true",
        help=(
            "Import the legacy authentication JSONL log."
        ),
    )

    parser.add_argument(
        "--summary",
        action="store_true",
        help=(
            "Print a JSON audit summary."
        ),
    )

    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help=(
            "Summary period in days."
        ),
    )

    parser.add_argument(
        "--verify-integrity",
        action="store_true",
        help=(
            "Verify the chained event hashes."
        ),
    )

    parser.add_argument(
        "--export-events",
        action="store_true",
        help=(
            "Export all audit events to CSV."
        ),
    )

    parser.add_argument(
        "--export-sessions",
        action="store_true",
        help=(
            "Export all sessions to CSV."
        ),
    )

    parser.add_argument(
        "--purge",
        action="store_true",
        help=(
            "Delete events older than the configured retention period."
        ),
    )

    parser.add_argument(
        "--standalone-console",
        action="store_true",
        help=(
            "Run the Streamlit audit console in the current process."
        ),
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.standalone_console:
        run_streamlit_console()
        return

    action_requested = False

    if arguments.initialise:
        initialise_database()
        print(
            f"Audit database initialised: {DATABASE_PATH}"
        )
        action_requested = True

    if arguments.import_auth_log:
        imported = import_legacy_authentication_log()
        print(
            f"Legacy authentication events imported: {imported}"
        )
        action_requested = True

    if arguments.summary:
        print(
            json.dumps(
                audit_summary(
                    days=max(
                        int(
                            arguments.days
                        ),
                        1,
                    )
                ),
                indent=4,
                ensure_ascii=False,
            )
        )
        action_requested = True

    if arguments.verify_integrity:
        result = verify_hash_chain()
        print(
            json.dumps(
                result,
                indent=4,
                ensure_ascii=False,
            )
        )
        action_requested = True

    if arguments.export_events:
        path = export_events_csv()
        print(
            f"Audit events exported: {path}"
        )
        action_requested = True

    if arguments.export_sessions:
        path = export_sessions_csv()
        print(
            f"Audit sessions exported: {path}"
        )
        action_requested = True

    if arguments.purge:
        deleted = purge_expired_events()
        print(
            f"Expired audit events deleted: {deleted}"
        )
        action_requested = True

    if not action_requested:
        initialise_database()

        print(
            "\n"
            + "=" * 104
        )

        print(
            "HEVEMIND AUDIT AND ACTIVITY TRACKING ENGINE"
        )

        print(
            "=" * 104
        )

        print(
            f"\nDatabase:                    {DATABASE_PATH}"
        )

        print(
            f"Hash chain enabled:          {AUDIT_HASH_CHAIN_ENABLED}"
        )

        print(
            f"Retention period:            {AUDIT_RETENTION_DAYS} days"
        )

        print(
            "\nRun with --help to view administration commands."
        )


if __name__ == "__main__":
    main()
