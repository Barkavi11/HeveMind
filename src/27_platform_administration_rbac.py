from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import importlib.util
import json
import os
import secrets
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "config"
LOG_DIR = ROOT_DIR / "logs"
BACKUP_DIR = ROOT_DIR / "backups" / "administration"

USERS_PATH = CONFIG_DIR / "users.json"
AUTH_SETTINGS_PATH = CONFIG_DIR / "auth_settings.json"
PLATFORM_SETTINGS_PATH = CONFIG_DIR / "platform_settings.json"
RBAC_POLICY_PATH = CONFIG_DIR / "rbac_policy.json"

AUDIT_ENGINE_PATH = SRC_DIR / "24_audit_activity_tracking.py"
FASTAPI_SERVICE_PATH = SRC_DIR / "19_fastapi_service.py"
DASHBOARD_PATH = SRC_DIR / "20_streamlit_dashboard.py"
SECURE_GATEWAY_PATH = SRC_DIR / "21_secure_access_gateway.py"
PDF_REPORT_PATH = SRC_DIR / "26_executive_pdf_report_generator.py"

AUDIT_DATABASE_PATH = LOG_DIR / "hevemind_audit.db"


# ============================================================
# PLATFORM CONFIGURATION
# ============================================================
APP_VERSION = os.getenv(
    "HEVEMIND_ADMIN_VERSION",
    "1.0.0",
).strip()

DEFAULT_API_BASE_URL = os.getenv(
    "HEVEMIND_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_PLATFORM_SETTINGS = {
    "schema_version": 1,
    "environment_name": "Validated Research Environment",
    "site_name": "Malaysia Engineering Demonstration",
    "maintenance_mode": False,
    "maintenance_message": (
        "HeveMind is temporarily unavailable while approved maintenance "
        "activities are completed."
    ),
    "allow_report_downloads": True,
    "allow_governance_exports": True,
    "allow_raw_json_downloads": True,
    "session_timeout_minutes": 30,
    "maximum_failed_attempts": 5,
    "lockout_minutes": 15,
    "audit_retention_days": 365,
    "api_base_url": DEFAULT_API_BASE_URL,
    "last_updated_utc": None,
    "last_updated_by": None,
}

DEFAULT_RBAC_POLICY = {
    "schema_version": 1,
    "roles": {
        "viewer": {
            "description": "Read-only executive access.",
            "permissions": [
                "dashboard.executive_view",
            ],
        },
        "engineer": {
            "description": "Engineering review and investigation access.",
            "permissions": [
                "dashboard.executive_view",
                "wafer.view",
                "queue.view",
                "sensor.view",
                "report.pdf_download",
                "report.json_download",
            ],
        },
        "admin": {
            "description": "Full platform administration and governance access.",
            "permissions": [
                "*",
            ],
        },
    },
}

ALLOWED_ROLES = {
    "viewer",
    "engineer",
    "admin",
}

SENSITIVE_FIELDS = {
    "password_hash",
    "password",
    "token",
    "secret",
    "api_key",
    "client_secret",
}


# ============================================================
# GENERIC UTILITIES
# ============================================================
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    return utc_now().isoformat()


def ensure_directories() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    ensure_directories()

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    temporary.write_text(
        json.dumps(
            payload,
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    temporary.replace(path)


def load_json(
    path: Path,
    default: dict[str, Any],
) -> dict[str, Any]:
    ensure_directories()

    if not path.exists():
        atomic_write_json(
            path,
            default,
        )
        return json.loads(
            json.dumps(default)
        )

    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON configuration file: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected JSON object in: {path}"
        )

    return payload


def safe_text(
    value: Any,
    fallback: str = "Unavailable",
) -> str:
    if value is None:
        return fallback

    try:
        if pd.isna(value):
            return fallback
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    return text or fallback


def normalise_email(
    value: Any,
) -> str:
    return safe_text(
        value,
        "",
    ).lower()


def json_safe(
    value: Any,
) -> Any:
    if value is None:
        return None

    if isinstance(
        value,
        (str, int, float, bool),
    ):
        return value

    if isinstance(value, datetime):
        return value.astimezone(
            timezone.utc
        ).isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            json_safe(item)
            for item in value
        ]

    return str(value)


def redact_payload(
    payload: dict[str, Any],
) -> dict[str, Any]:
    redacted: dict[str, Any] = {}

    for key, value in payload.items():
        if str(key).lower() in SENSITIVE_FIELDS:
            redacted[str(key)] = "[REDACTED]"

        elif isinstance(value, dict):
            redacted[str(key)] = redact_payload(value)

        elif isinstance(value, list):
            redacted[str(key)] = [
                (
                    redact_payload(item)
                    if isinstance(item, dict)
                    else json_safe(item)
                )
                for item in value
            ]

        else:
            redacted[str(key)] = json_safe(value)

    return redacted


def encode_password_hash(
    password: str,
    iterations: int = 390_000,
) -> str:
    salt = secrets.token_bytes(16)

    key = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        iterations,
    )

    return (
        f"pbkdf2_sha256${iterations}$"
        f"{base64.urlsafe_b64encode(salt).decode('ascii')}$"
        f"{base64.urlsafe_b64encode(key).decode('ascii')}"
    )


def verify_password(
    password: str,
    encoded_hash: str,
) -> bool:
    try:
        algorithm, iterations_text, salt_text, key_text = encoded_hash.split(
            "$",
            maxsplit=3,
        )

        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(
            salt_text.encode("ascii")
        )
        expected = base64.urlsafe_b64decode(
            key_text.encode("ascii")
        )

        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )

        return hmac.compare_digest(
            candidate,
            expected,
        )

    except Exception:
        return False


def validate_password_strength(
    password: str,
) -> None:
    checks = {
        "at least 12 characters": len(password) >= 12,
        "one uppercase letter": any(
            character.isupper()
            for character in password
        ),
        "one lowercase letter": any(
            character.islower()
            for character in password
        ),
        "one number": any(
            character.isdigit()
            for character in password
        ),
        "one special character": any(
            not character.isalnum()
            for character in password
        ),
    }

    failed = [
        name
        for name, passed in checks.items()
        if not passed
    ]

    if failed:
        raise ValueError(
            "Password must contain "
            + ", ".join(failed)
            + "."
        )


# ============================================================
# CONFIGURATION LOADERS
# ============================================================
def load_users_payload() -> dict[str, Any]:
    payload = load_json(
        USERS_PATH,
        {
            "schema_version": 1,
            "users": [],
        },
    )

    if not isinstance(
        payload.get("users"),
        list,
    ):
        raise RuntimeError(
            f"'users' must be a list in {USERS_PATH}"
        )

    return payload


def load_platform_settings() -> dict[str, Any]:
    stored = load_json(
        PLATFORM_SETTINGS_PATH,
        DEFAULT_PLATFORM_SETTINGS,
    )

    return {
        **DEFAULT_PLATFORM_SETTINGS,
        **stored,
    }


def load_rbac_policy() -> dict[str, Any]:
    stored = load_json(
        RBAC_POLICY_PATH,
        DEFAULT_RBAC_POLICY,
    )

    if not isinstance(
        stored.get("roles"),
        dict,
    ):
        raise RuntimeError(
            f"'roles' must be an object in {RBAC_POLICY_PATH}"
        )

    return stored


def load_auth_settings() -> dict[str, Any]:
    return load_json(
        AUTH_SETTINGS_PATH,
        {
            "auth_mode": "development",
            "development_email_domain": "hevemind.local",
            "production_email_domain": "infineon.com",
            "session_timeout_minutes": 30,
            "maximum_failed_attempts": 5,
            "lockout_minutes": 15,
        },
    )


# ============================================================
# RBAC
# ============================================================
def role_permissions(
    role: str,
    policy: dict[str, Any] | None = None,
) -> set[str]:
    resolved_policy = (
        policy
        or load_rbac_policy()
    )

    role_record = (
        resolved_policy
        .get("roles", {})
        .get(role, {})
    )

    permissions = role_record.get(
        "permissions",
        [],
    )

    if not isinstance(
        permissions,
        list,
    ):
        return set()

    return {
        str(permission).strip()
        for permission in permissions
        if str(permission).strip()
    }


def has_permission(
    role: str,
    permission: str,
    policy: dict[str, Any] | None = None,
) -> bool:
    permissions = role_permissions(
        role,
        policy,
    )

    return (
        "*"
        in permissions
        or permission
        in permissions
    )


def require_permission(
    role: str,
    permission: str,
) -> None:
    if not has_permission(
        role,
        permission,
    ):
        raise PermissionError(
            f"Role '{role}' does not have permission '{permission}'."
        )


# ============================================================
# USER MANAGEMENT
# ============================================================
def list_users_dataframe() -> pd.DataFrame:
    users = [
        user
        for user in load_users_payload()["users"]
        if isinstance(user, dict)
    ]

    rows = []

    for user in users:
        rows.append(
            {
                "email": normalise_email(
                    user.get("email")
                ),
                "display_name": safe_text(
                    user.get("display_name")
                ),
                "role": safe_text(
                    user.get("role"),
                    "viewer",
                ),
                "active": bool(
                    user.get("active", True)
                ),
                "failed_attempts": int(
                    user.get(
                        "failed_attempts",
                        0,
                    )
                    or 0
                ),
                "locked_until_epoch": float(
                    user.get(
                        "locked_until_epoch",
                        0,
                    )
                    or 0
                ),
                "created_utc": user.get(
                    "created_utc"
                ),
                "last_login_utc": user.get(
                    "last_login_utc"
                ),
            }
        )

    return pd.DataFrame(rows)


def find_user(
    email: str,
) -> dict[str, Any] | None:
    normalised = normalise_email(email)

    for user in load_users_payload()["users"]:
        if not isinstance(user, dict):
            continue

        if normalise_email(
            user.get("email")
        ) == normalised:
            return user

    return None


def upsert_user(
    *,
    email: str,
    display_name: str,
    role: str,
    active: bool,
    password: str | None,
) -> dict[str, Any]:
    normalised = normalise_email(email)

    if not normalised or "@" not in normalised:
        raise ValueError(
            "A valid email address is required."
        )

    if role not in ALLOWED_ROLES:
        raise ValueError(
            "Role must be viewer, engineer, or admin."
        )

    payload = load_users_payload()
    existing = find_user(
        normalised
    )

    if existing is None and not password:
        raise ValueError(
            "A password is required when creating a new local user."
        )

    if password:
        validate_password_strength(
            password
        )
        password_hash = encode_password_hash(
            password
        )
    else:
        password_hash = safe_text(
            existing.get("password_hash"),
            "",
        )

    record = {
        "email": normalised,
        "display_name": display_name.strip()
        or normalised,
        "role": role,
        "active": bool(active),
        "password_hash": password_hash,
        "failed_attempts": 0,
        "locked_until_epoch": 0,
        "created_utc": (
            existing.get("created_utc")
            if existing
            else utc_now_iso()
        ),
        "last_login_utc": (
            existing.get("last_login_utc")
            if existing
            else None
        ),
        "updated_utc": utc_now_iso(),
    }

    replaced = False

    for index, user in enumerate(
        payload["users"]
    ):
        if (
            isinstance(user, dict)
            and normalise_email(
                user.get("email")
            )
            == normalised
        ):
            payload["users"][index] = record
            replaced = True
            break

    if not replaced:
        payload["users"].append(
            record
        )

    atomic_write_json(
        USERS_PATH,
        payload,
    )

    return record


def set_user_active(
    email: str,
    active: bool,
) -> None:
    payload = load_users_payload()
    normalised = normalise_email(email)
    changed = False

    for user in payload["users"]:
        if not isinstance(user, dict):
            continue

        if normalise_email(
            user.get("email")
        ) == normalised:
            user["active"] = bool(active)
            user["updated_utc"] = utc_now_iso()
            changed = True
            break

    if not changed:
        raise KeyError(
            f"User not found: {email}"
        )

    atomic_write_json(
        USERS_PATH,
        payload,
    )


def unlock_user(
    email: str,
) -> None:
    payload = load_users_payload()
    normalised = normalise_email(email)
    changed = False

    for user in payload["users"]:
        if not isinstance(user, dict):
            continue

        if normalise_email(
            user.get("email")
        ) == normalised:
            user["failed_attempts"] = 0
            user["locked_until_epoch"] = 0
            user["updated_utc"] = utc_now_iso()
            changed = True
            break

    if not changed:
        raise KeyError(
            f"User not found: {email}"
        )

    atomic_write_json(
        USERS_PATH,
        payload,
    )


# ============================================================
# AUDIT INTEGRATION
# ============================================================
def load_audit_engine() -> Any | None:
    if not AUDIT_ENGINE_PATH.exists():
        return None

    specification = (
        importlib.util.spec_from_file_location(
            "hevemind_audit_activity_tracking",
            AUDIT_ENGINE_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        return None

    module = importlib.util.module_from_spec(
        specification
    )

    sys.modules[
        "hevemind_audit_activity_tracking"
    ] = module

    try:
        specification.loader.exec_module(
            module
        )
    except Exception:
        return None

    return module


AUDIT_ENGINE = load_audit_engine()


def record_admin_event(
    *,
    event_type: str,
    outcome: str,
    action: str,
    actor: dict[str, Any],
    resource_type: str,
    resource_id: str | None = None,
    details: dict[str, Any] | None = None,
    message: str | None = None,
) -> None:
    if AUDIT_ENGINE is None:
        return

    try:
        AUDIT_ENGINE.record_event(
            source_component=(
                "27_platform_administration"
            ),
            event_type=event_type,
            category="governance",
            outcome=outcome,
            user_email=actor.get("email"),
            display_name=actor.get(
                "display_name"
            ),
            role=actor.get("role"),
            auth_source=actor.get(
                "auth_source"
            ),
            session_id=actor.get(
                "session_id"
            ),
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            message=message,
            details=redact_payload(
                details or {}
            ),
        )

    except Exception:
        pass


# ============================================================
# HEALTH AND READINESS
# ============================================================
@dataclass(frozen=True)
class ComponentStatus:
    component: str
    status: str
    details: str


def file_status(
    label: str,
    path: Path,
) -> ComponentStatus:
    return ComponentStatus(
        component=label,
        status=(
            "Available"
            if path.exists()
            else "Missing"
        ),
        details=str(path),
    )


def collect_platform_status() -> pd.DataFrame:
    statuses = [
        file_status(
            "FastAPI service",
            FASTAPI_SERVICE_PATH,
        ),
        file_status(
            "Streamlit dashboard",
            DASHBOARD_PATH,
        ),
        file_status(
            "Secure access gateway",
            SECURE_GATEWAY_PATH,
        ),
        file_status(
            "PDF report generator",
            PDF_REPORT_PATH,
        ),
        file_status(
            "Audit engine",
            AUDIT_ENGINE_PATH,
        ),
        file_status(
            "Audit database",
            AUDIT_DATABASE_PATH,
        ),
        file_status(
            "User registry",
            USERS_PATH,
        ),
        file_status(
            "Platform settings",
            PLATFORM_SETTINGS_PATH,
        ),
        file_status(
            "RBAC policy",
            RBAC_POLICY_PATH,
        ),
    ]

    return pd.DataFrame(
        [
            {
                "component": item.component,
                "status": item.status,
                "details": item.details,
            }
            for item in statuses
        ]
    )


def api_health(
    api_base_url: str,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    import urllib.error
    import urllib.request

    url = api_base_url.rstrip(
        "/"
    ) + "/health"

    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": (
                "HeveMind-Administration/1.0"
            ),
        },
        method="GET",
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode(
                "utf-8"
            )

        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        payload = json.loads(raw)

        return {
            "reachable": True,
            "status_code": 200,
            "response_time_ms": elapsed_ms,
            "payload": payload,
            "error": None,
        }

    except Exception as error:
        elapsed_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return {
            "reachable": False,
            "status_code": None,
            "response_time_ms": elapsed_ms,
            "payload": None,
            "error": str(error),
        }


# ============================================================
# BACKUP AND RESTORE
# ============================================================
def create_configuration_backup(
    actor_email: str,
) -> Path:
    ensure_directories()

    timestamp = utc_now().strftime(
        "%Y%m%d_%H%M%S"
    )

    backup_path = (
        BACKUP_DIR
        / (
            "hevemind_admin_backup_"
            + timestamp
        )
    )

    backup_path.mkdir(
        parents=True,
        exist_ok=False,
    )

    copied_files = []

    for source in [
        USERS_PATH,
        AUTH_SETTINGS_PATH,
        PLATFORM_SETTINGS_PATH,
        RBAC_POLICY_PATH,
    ]:
        if source.exists():
            destination = (
                backup_path
                / source.name
            )

            shutil.copy2(
                source,
                destination,
            )

            copied_files.append(
                destination.name
            )

    manifest = {
        "backup_created_utc": utc_now_iso(),
        "backup_created_by": actor_email,
        "files": copied_files,
        "source_root": str(ROOT_DIR),
    }

    atomic_write_json(
        backup_path
        / "manifest.json",
        manifest,
    )

    return backup_path


def restore_configuration_backup(
    backup_path: Path,
) -> None:
    if not backup_path.exists():
        raise FileNotFoundError(
            f"Backup directory not found: {backup_path}"
        )

    manifest_path = (
        backup_path
        / "manifest.json"
    )

    if not manifest_path.exists():
        raise RuntimeError(
            "Backup manifest is missing."
        )

    manifest = load_json(
        manifest_path,
        {},
    )

    files = manifest.get(
        "files",
        [],
    )

    if not isinstance(files, list):
        raise RuntimeError(
            "Backup manifest is invalid."
        )

    allowed_destinations = {
        USERS_PATH.name: USERS_PATH,
        AUTH_SETTINGS_PATH.name: AUTH_SETTINGS_PATH,
        PLATFORM_SETTINGS_PATH.name: PLATFORM_SETTINGS_PATH,
        RBAC_POLICY_PATH.name: RBAC_POLICY_PATH,
    }

    for filename in files:
        source = (
            backup_path
            / str(filename)
        )

        destination = (
            allowed_destinations.get(
                str(filename)
            )
        )

        if (
            destination is None
            or not source.exists()
        ):
            continue

        shutil.copy2(
            source,
            destination,
        )


# ============================================================
# CONFIGURATION EXPORT
# ============================================================
def export_configuration_bundle() -> bytes:
    payload = {
        "generated_utc": utc_now_iso(),
        "platform_settings": redact_payload(
            load_platform_settings()
        ),
        "auth_settings": redact_payload(
            load_auth_settings()
        ),
        "rbac_policy": redact_payload(
            load_rbac_policy()
        ),
        "users": [
            redact_payload(user)
            for user in load_users_payload()[
                "users"
            ]
            if isinstance(user, dict)
        ],
        "component_status": (
            collect_platform_status()
            .to_dict(
                orient="records"
            )
        ),
    }

    return json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
    ).encode(
        "utf-8"
    )


# ============================================================
# STREAMLIT ADMINISTRATION CONSOLE
# ============================================================
ADMIN_CSS = """
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
        max-width: 1450px;
        padding-top: 1.7rem;
        padding-bottom: 3rem;
    }

    .admin-header {
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

    .admin-title {
        font-size: 1.8rem;
        font-weight: 780;
        letter-spacing: -0.025em;
    }

    .admin-subtitle {
        color: #d8e8f4;
        font-size: 0.9rem;
        margin-top: 0.36rem;
        line-height: 1.5;
    }

    .admin-status-row {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.9rem;
    }

    .admin-status-chip {
        border-radius: 999px;
        padding: 0.32rem 0.68rem;
        font-size: 0.72rem;
        font-weight: 720;
        background: rgba(255, 255, 255, 0.10);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #ffffff;
    }

    .admin-metric {
        min-height: 130px;
        background: #ffffff;
        border: 1px solid var(--slate-200);
        border-radius: 15px;
        padding: 0.95rem 1rem;
        box-shadow:
            0 7px 20px
            rgba(21, 48, 75, 0.06);
        height: 100%;
    }

    .admin-metric-label {
        color: var(--slate-500);
        font-size: 0.70rem;
        font-weight: 760;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .admin-metric-value {
        color: var(--navy-950);
        font-size: 1.55rem;
        font-weight: 790;
        margin-top: 0.42rem;
    }

    .admin-metric-note {
        color: var(--slate-700);
        font-size: 0.77rem;
        line-height: 1.4;
        margin-top: 0.36rem;
    }

    .admin-note {
        border-left: 4px solid var(--blue-600);
        background: #eef7fc;
        border-radius: 10px;
        padding: 0.78rem 0.9rem;
        color: var(--slate-700);
        font-size: 0.79rem;
        line-height: 1.45;
        margin-top: 0.7rem;
    }

    .warning-note {
        border-left: 4px solid var(--amber);
        background: var(--amber-bg);
        border-radius: 10px;
        padding: 0.78rem 0.9rem;
        color: #6b4c11;
        font-size: 0.79rem;
        line-height: 1.45;
        margin-top: 0.7rem;
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


def render_metric_card(
    label: str,
    value: str,
    note: str,
) -> None:
    import streamlit as st

    st.markdown(
        f"""
        <div class="admin-metric">
            <div class="admin-metric-label">{label}</div>
            <div class="admin-metric-value">{value}</div>
            <div class="admin-metric-note">{note}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def require_admin_user() -> dict[str, Any]:
    import streamlit as st

    user = st.session_state.get(
        "hevemind_user",
        {},
    )

    role = safe_text(
        user.get("role"),
        "",
    ).lower()

    standalone = (
        os.getenv(
            "HEVEMIND_ADMIN_STANDALONE_MODE",
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

    if (
        role != "admin"
        and not standalone
    ):
        st.error(
            "Administrator access is required."
        )
        st.stop()

    if not isinstance(user, dict):
        return {}

    return user


def render_administration_console() -> None:
    import streamlit as st

    st.markdown(
        ADMIN_CSS,
        unsafe_allow_html=True,
    )

    ensure_directories()
    load_platform_settings()
    load_rbac_policy()
    load_users_payload()

    actor = require_admin_user()

    actor_with_session = {
        **actor,
        "session_id": st.session_state.get(
            "hevemind_audit_session_id"
        ),
    }

    settings = load_platform_settings()
    users_df = list_users_dataframe()
    component_status = collect_platform_status()
    api_status = api_health(
        settings["api_base_url"]
    )

    active_users = (
        int(
            users_df["active"].sum()
        )
        if not users_df.empty
        else 0
    )

    administrators = (
        int(
            (
                users_df["role"]
                == "admin"
            ).sum()
        )
        if not users_df.empty
        else 0
    )

    available_components = int(
        (
            component_status["status"]
            == "Available"
        ).sum()
    )

    st.markdown(
        f"""
        <div class="admin-header">
            <div class="admin-title">
                HeveMind Platform Administration
            </div>
            <div class="admin-subtitle">
                User administration, role-based access control,
                operational settings, component readiness,
                configuration backup and controlled maintenance
            </div>
            <div class="admin-status-row">
                <span class="admin-status-chip">
                    Version {APP_VERSION}
                </span>
                <span class="admin-status-chip">
                    Administrator access
                </span>
                <span class="admin-status-chip">
                    Environment {settings["environment_name"]}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_columns = st.columns(5)

    metric_payloads = [
        (
            "Registered users",
            str(len(users_df)),
            "Local development identities in the user registry.",
        ),
        (
            "Active users",
            str(active_users),
            "Users currently permitted to authenticate.",
        ),
        (
            "Administrators",
            str(administrators),
            "Accounts with full administrative permission.",
        ),
        (
            "Available components",
            f"{available_components}/{len(component_status)}",
            "Required platform files currently available.",
        ),
        (
            "API status",
            (
                "Reachable"
                if api_status["reachable"]
                else "Unavailable"
            ),
            (
                f"{api_status['response_time_ms']:.1f} ms"
                if api_status["reachable"]
                else safe_text(
                    api_status["error"]
                )
            ),
        ),
    ]

    for column, payload in zip(
        metric_columns,
        metric_payloads,
    ):
        with column:
            render_metric_card(
                payload[0],
                payload[1],
                payload[2],
            )

    st.write("")

    (
        users_tab,
        roles_tab,
        settings_tab,
        readiness_tab,
        backup_tab,
        configuration_tab,
    ) = st.tabs(
        [
            "User Management",
            "Role Permissions",
            "Platform Settings",
            "Readiness",
            "Backup and Restore",
            "Configuration Export",
        ]
    )

    with users_tab:
        st.subheader(
            "Local Development User Management"
        )

        st.caption(
            "These accounts are for the local development authentication "
            "mode. Production access should use approved enterprise OIDC."
        )

        if users_df.empty:
            st.info(
                "No local users are configured."
            )
        else:
            display_users = users_df.copy()

            now_epoch = time.time()

            display_users[
                "lock_status"
            ] = display_users[
                "locked_until_epoch"
            ].map(
                lambda value: (
                    "Locked"
                    if float(value or 0)
                    > now_epoch
                    else "Not locked"
                )
            )

            st.dataframe(
                display_users[
                    [
                        "email",
                        "display_name",
                        "role",
                        "active",
                        "lock_status",
                        "failed_attempts",
                        "created_utc",
                        "last_login_utc",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=360,
            )

        st.divider()

        action_mode = st.radio(
            "User operation",
            options=[
                "Create or update",
                "Activate or deactivate",
                "Unlock account",
            ],
            horizontal=True,
        )

        if action_mode == "Create or update":
            with st.form(
                "admin_user_upsert_form"
            ):
                email = st.text_input(
                    "Email address"
                )

                display_name = st.text_input(
                    "Display name"
                )

                role = st.selectbox(
                    "Role",
                    options=[
                        "viewer",
                        "engineer",
                        "admin",
                    ],
                )

                active = st.checkbox(
                    "Account active",
                    value=True,
                )

                password = st.text_input(
                    "New password",
                    type="password",
                    help=(
                        "Leave blank only when updating an existing user "
                        "without changing the password."
                    ),
                )

                confirmation = st.text_input(
                    "Confirm password",
                    type="password",
                )

                submitted = st.form_submit_button(
                    "Save user",
                    use_container_width=True,
                )

            if submitted:
                try:
                    if password != confirmation:
                        raise ValueError(
                            "Passwords do not match."
                        )

                    record = upsert_user(
                        email=email,
                        display_name=display_name,
                        role=role,
                        active=active,
                        password=(
                            password
                            or None
                        ),
                    )

                    record_admin_event(
                        event_type=(
                            "platform_user_upserted"
                        ),
                        outcome="success",
                        action="upsert_user",
                        actor=actor_with_session,
                        resource_type="user_account",
                        resource_id=record["email"],
                        details={
                            "role": role,
                            "active": active,
                            "password_changed": bool(
                                password
                            ),
                        },
                        message=(
                            "Administrator created or updated a local user."
                        ),
                    )

                    st.success(
                        f"User saved: {record['email']}"
                    )

                    st.rerun()

                except Exception as error:
                    st.error(
                        str(error)
                    )

        elif action_mode == "Activate or deactivate":
            if users_df.empty:
                st.info(
                    "No users are available."
                )
            else:
                selected_email = st.selectbox(
                    "User",
                    options=users_df[
                        "email"
                    ].tolist(),
                    key="activation_user",
                )

                new_state = st.selectbox(
                    "New account state",
                    options=[
                        "Active",
                        "Inactive",
                    ],
                )

                if st.button(
                    "Apply account state",
                    use_container_width=True,
                ):
                    set_user_active(
                        selected_email,
                        new_state == "Active",
                    )

                    record_admin_event(
                        event_type=(
                            "platform_user_state_changed"
                        ),
                        outcome="success",
                        action="set_user_active",
                        actor=actor_with_session,
                        resource_type="user_account",
                        resource_id=selected_email,
                        details={
                            "active": (
                                new_state
                                == "Active"
                            )
                        },
                    )

                    st.success(
                        "Account state updated."
                    )

                    st.rerun()

        else:
            if users_df.empty:
                st.info(
                    "No users are available."
                )
            else:
                selected_email = st.selectbox(
                    "User",
                    options=users_df[
                        "email"
                    ].tolist(),
                    key="unlock_user",
                )

                if st.button(
                    "Unlock account",
                    use_container_width=True,
                ):
                    unlock_user(
                        selected_email
                    )

                    record_admin_event(
                        event_type=(
                            "platform_user_unlocked"
                        ),
                        outcome="success",
                        action="unlock_user",
                        actor=actor_with_session,
                        resource_type="user_account",
                        resource_id=selected_email,
                    )

                    st.success(
                        "Account unlocked."
                    )

                    st.rerun()

    with roles_tab:
        st.subheader(
            "Role-Based Access Control"
        )

        policy = load_rbac_policy()

        permission_rows = []

        for role_name, role_record in (
            policy.get(
                "roles",
                {}
            ).items()
        ):
            permissions = role_record.get(
                "permissions",
                [],
            )

            permission_rows.append(
                {
                    "role": role_name,
                    "description": role_record.get(
                        "description"
                    ),
                    "permissions": ", ".join(
                        permissions
                    ),
                }
            )

        st.dataframe(
            pd.DataFrame(
                permission_rows
            ),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown(
            """
            <div class="admin-note">
                The admin role uses the wildcard permission. Viewer and
                engineer permissions are explicit so that future features
                remain denied until deliberately assigned.
            </div>
            """,
            unsafe_allow_html=True,
        )

        test_columns = st.columns(2)

        with test_columns[0]:
            test_role = st.selectbox(
                "Test role",
                options=[
                    "viewer",
                    "engineer",
                    "admin",
                ],
            )

        with test_columns[1]:
            test_permission = st.text_input(
                "Permission to test",
                value="wafer.view",
            )

        if st.button(
            "Evaluate permission",
            use_container_width=True,
        ):
            allowed = has_permission(
                test_role,
                test_permission,
            )

            if allowed:
                st.success(
                    "Permission granted by the active RBAC policy."
                )
            else:
                st.warning(
                    "Permission denied by the active RBAC policy."
                )

    with settings_tab:
        st.subheader(
            "Operational Platform Settings"
        )

        with st.form(
            "platform_settings_form"
        ):
            environment_name = st.text_input(
                "Environment name",
                value=settings[
                    "environment_name"
                ],
            )

            site_name = st.text_input(
                "Site name",
                value=settings[
                    "site_name"
                ],
            )

            api_base_url = st.text_input(
                "API base URL",
                value=settings[
                    "api_base_url"
                ],
            )

            maintenance_mode = st.checkbox(
                "Maintenance mode",
                value=bool(
                    settings[
                        "maintenance_mode"
                    ]
                ),
            )

            maintenance_message = st.text_area(
                "Maintenance message",
                value=settings[
                    "maintenance_message"
                ],
            )

            setting_columns = st.columns(3)

            with setting_columns[0]:
                allow_report_downloads = st.checkbox(
                    "Allow PDF report downloads",
                    value=bool(
                        settings[
                            "allow_report_downloads"
                        ]
                    ),
                )

            with setting_columns[1]:
                allow_governance_exports = st.checkbox(
                    "Allow governance exports",
                    value=bool(
                        settings[
                            "allow_governance_exports"
                        ]
                    ),
                )

            with setting_columns[2]:
                allow_raw_json_downloads = st.checkbox(
                    "Allow raw JSON downloads",
                    value=bool(
                        settings[
                            "allow_raw_json_downloads"
                        ]
                    ),
                )

            session_timeout_minutes = st.number_input(
                "Session timeout in minutes",
                min_value=5,
                max_value=480,
                value=int(
                    settings[
                        "session_timeout_minutes"
                    ]
                ),
                step=5,
            )

            maximum_failed_attempts = st.number_input(
                "Maximum failed login attempts",
                min_value=1,
                max_value=20,
                value=int(
                    settings[
                        "maximum_failed_attempts"
                    ]
                ),
                step=1,
            )

            lockout_minutes = st.number_input(
                "Account lockout duration in minutes",
                min_value=1,
                max_value=1440,
                value=int(
                    settings[
                        "lockout_minutes"
                    ]
                ),
                step=1,
            )

            audit_retention_days = st.number_input(
                "Audit retention in days",
                min_value=30,
                max_value=3650,
                value=int(
                    settings[
                        "audit_retention_days"
                    ]
                ),
                step=30,
            )

            save_settings = st.form_submit_button(
                "Save platform settings",
                use_container_width=True,
            )

        if save_settings:
            updated = {
                **settings,
                "environment_name": (
                    environment_name.strip()
                ),
                "site_name": site_name.strip(),
                "api_base_url": api_base_url.strip().rstrip(
                    "/"
                ),
                "maintenance_mode": bool(
                    maintenance_mode
                ),
                "maintenance_message": (
                    maintenance_message.strip()
                ),
                "allow_report_downloads": bool(
                    allow_report_downloads
                ),
                "allow_governance_exports": bool(
                    allow_governance_exports
                ),
                "allow_raw_json_downloads": bool(
                    allow_raw_json_downloads
                ),
                "session_timeout_minutes": int(
                    session_timeout_minutes
                ),
                "maximum_failed_attempts": int(
                    maximum_failed_attempts
                ),
                "lockout_minutes": int(
                    lockout_minutes
                ),
                "audit_retention_days": int(
                    audit_retention_days
                ),
                "last_updated_utc": utc_now_iso(),
                "last_updated_by": actor.get(
                    "email"
                ),
            }

            atomic_write_json(
                PLATFORM_SETTINGS_PATH,
                updated,
            )

            record_admin_event(
                event_type=(
                    "platform_settings_updated"
                ),
                outcome="success",
                action="update_settings",
                actor=actor_with_session,
                resource_type=(
                    "platform_settings"
                ),
                resource_id="primary",
                details=redact_payload(
                    updated
                ),
            )

            st.success(
                "Platform settings saved."
            )

            st.rerun()

        if settings[
            "maintenance_mode"
        ]:
            st.markdown(
                f"""
                <div class="warning-note">
                    Maintenance mode is currently enabled.
                    Message: {settings["maintenance_message"]}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with readiness_tab:
        st.subheader(
            "Platform Readiness"
        )

        st.dataframe(
            component_status,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader(
            "FastAPI Health"
        )

        st.json(
            api_status,
            expanded=True,
        )

        all_available = bool(
            (
                component_status[
                    "status"
                ]
                == "Available"
            ).all()
        )

        if (
            all_available
            and api_status[
                "reachable"
            ]
        ):
            st.success(
                "All registered components are available and the API is reachable."
            )
        else:
            st.warning(
                "One or more components require attention."
            )

    with backup_tab:
        st.subheader(
            "Configuration Backup and Restore"
        )

        st.markdown(
            """
            <div class="warning-note">
                Restore operations replace active configuration files.
                Create a fresh backup before restoring an older snapshot.
            </div>
            """,
            unsafe_allow_html=True,
        )

        backup_columns = st.columns(2)

        with backup_columns[0]:
            if st.button(
                "Create configuration backup",
                use_container_width=True,
            ):
                backup_path = (
                    create_configuration_backup(
                        actor_email=safe_text(
                            actor.get("email"),
                            "unknown",
                        )
                    )
                )

                record_admin_event(
                    event_type=(
                        "platform_configuration_backup_created"
                    ),
                    outcome="success",
                    action="backup",
                    actor=actor_with_session,
                    resource_type=(
                        "configuration_backup"
                    ),
                    resource_id=backup_path.name,
                    details={
                        "backup_path": str(
                            backup_path
                        )
                    },
                )

                st.success(
                    f"Backup created: {backup_path}"
                )

        available_backups = sorted(
            [
                path
                for path in BACKUP_DIR.glob(
                    "hevemind_admin_backup_*"
                )
                if path.is_dir()
            ],
            reverse=True,
        )

        with backup_columns[1]:
            if not available_backups:
                st.info(
                    "No configuration backups are available."
                )
            else:
                selected_backup = st.selectbox(
                    "Backup snapshot",
                    options=[
                        str(path)
                        for path in available_backups
                    ],
                )

                confirmation_text = st.text_input(
                    "Type RESTORE to confirm"
                )

                if st.button(
                    "Restore selected backup",
                    use_container_width=True,
                    disabled=(
                        confirmation_text
                        != "RESTORE"
                    ),
                ):
                    restore_configuration_backup(
                        Path(
                            selected_backup
                        )
                    )

                    record_admin_event(
                        event_type=(
                            "platform_configuration_backup_restored"
                        ),
                        outcome="warning",
                        action="restore",
                        actor=actor_with_session,
                        resource_type=(
                            "configuration_backup"
                        ),
                        resource_id=Path(
                            selected_backup
                        ).name,
                    )

                    st.success(
                        "Configuration restored. Restart the application services."
                    )

    with configuration_tab:
        st.subheader(
            "Configuration Export"
        )

        st.caption(
            "Sensitive values, including password hashes and secrets, "
            "are redacted from this export."
        )

        export_bytes = (
            export_configuration_bundle()
        )

        if st.download_button(
            "Download redacted configuration bundle",
            data=export_bytes,
            file_name=(
                "hevemind_redacted_configuration.json"
            ),
            mime="application/json",
            use_container_width=True,
        ):
            record_admin_event(
                event_type=(
                    "platform_configuration_exported"
                ),
                outcome="success",
                action="export",
                actor=actor_with_session,
                resource_type=(
                    "platform_configuration"
                ),
                resource_id="redacted_bundle",
                details={
                    "export_format": "json"
                },
            )


# ============================================================
# COMMAND-LINE ADMINISTRATION
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HeveMind role-based access control, user administration, "
            "platform settings, readiness, backup and configuration export."
        )
    )

    parser.add_argument(
        "--initialise",
        action="store_true",
    )

    parser.add_argument(
        "--list-users",
        action="store_true",
    )

    parser.add_argument(
        "--create-user",
        action="store_true",
    )

    parser.add_argument(
        "--backup",
        action="store_true",
    )

    parser.add_argument(
        "--status",
        action="store_true",
    )

    parser.add_argument(
        "--export-config",
        type=Path,
    )

    return parser.parse_args()


def cli_create_user() -> None:
    email = input(
        "Email: "
    ).strip()

    display_name = input(
        "Display name: "
    ).strip()

    role = input(
        "Role [viewer/engineer/admin]: "
    ).strip().lower()

    active_text = input(
        "Active [yes/no]: "
    ).strip().lower()

    password = getpass.getpass(
        "Password: "
    )

    confirmation = getpass.getpass(
        "Confirm password: "
    )

    if password != confirmation:
        raise SystemExit(
            "Passwords do not match."
        )

    record = upsert_user(
        email=email,
        display_name=display_name,
        role=role,
        active=(
            active_text
            not in {
                "no",
                "n",
                "false",
                "0",
            }
        ),
        password=password,
    )

    print(
        f"User saved: {record['email']}"
    )


def main() -> None:
    arguments = parse_arguments()
    action_requested = False

    if arguments.initialise:
        ensure_directories()
        load_platform_settings()
        load_rbac_policy()
        load_users_payload()

        print(
            "Administration configuration initialised."
        )

        action_requested = True

    if arguments.list_users:
        dataframe = list_users_dataframe()

        if dataframe.empty:
            print(
                "No users configured."
            )
        else:
            print(
                dataframe.to_string(
                    index=False
                )
            )

        action_requested = True

    if arguments.create_user:
        cli_create_user()
        action_requested = True

    if arguments.backup:
        output_path = (
            create_configuration_backup(
                actor_email=(
                    "command-line-administrator"
                )
            )
        )

        print(
            f"Backup created: {output_path}"
        )

        action_requested = True

    if arguments.status:
        print(
            collect_platform_status().to_string(
                index=False
            )
        )

        settings = load_platform_settings()

        print(
            "\nAPI health:"
        )

        print(
            json.dumps(
                api_health(
                    settings[
                        "api_base_url"
                    ]
                ),
                indent=2,
                ensure_ascii=False,
            )
        )

        action_requested = True

    if arguments.export_config:
        arguments.export_config.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        arguments.export_config.write_bytes(
            export_configuration_bundle()
        )

        print(
            f"Configuration exported: {arguments.export_config}"
        )

        action_requested = True

    if not action_requested:
        ensure_directories()
        load_platform_settings()
        load_rbac_policy()
        load_users_payload()

        print(
            "\n"
            + "=" * 108
        )

        print(
            "HEVEMIND PLATFORM ADMINISTRATION AND RBAC"
        )

        print(
            "=" * 108
        )

        print(
            f"\nUser registry:           {USERS_PATH}"
        )

        print(
            f"Platform settings:       {PLATFORM_SETTINGS_PATH}"
        )

        print(
            f"RBAC policy:             {RBAC_POLICY_PATH}"
        )

        print(
            "\nRun with --help to view administration commands."
        )


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        import streamlit as st

        st.set_page_config(
            page_title="HeveMind Administration",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        render_administration_console()

    else:
        main()
