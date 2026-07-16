from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import runpy
import secrets
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "config"
LOG_DIR = ROOT_DIR / "logs"

DASHBOARD_PATH = SRC_DIR / "20_streamlit_dashboard.py"
USERS_PATH = CONFIG_DIR / "users.json"
AUTH_SETTINGS_PATH = CONFIG_DIR / "auth_settings.json"
AUDIT_LOG_PATH = LOG_DIR / "authentication_audit.jsonl"

ALLOWED_ROLES = {"viewer", "engineer", "admin"}

DEFAULT_SETTINGS = {
    "auth_mode": "development",
    "development_email_domain": "hevemind.local",
    "production_email_domain": "infineon.com",
    "session_timeout_minutes": 30,
    "maximum_failed_attempts": 5,
    "lockout_minutes": 15,
    "oidc_provider_name": "infineon",
    "allow_development_login": True,
}

DEFAULT_USERS = {
    "schema_version": 1,
    "users": [],
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_directories() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=4, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        atomic_write_json(path, default)
        return json.loads(json.dumps(default))

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON file: {path}") from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object in: {path}")

    return payload


def load_settings() -> dict[str, Any]:
    stored = load_json(AUTH_SETTINGS_PATH, DEFAULT_SETTINGS)
    settings = {**DEFAULT_SETTINGS, **stored}

    settings["auth_mode"] = os.getenv(
        "HEVEMIND_AUTH_MODE",
        str(settings["auth_mode"]),
    ).strip().lower()

    settings["production_email_domain"] = os.getenv(
        "HEVEMIND_ALLOWED_EMAIL_DOMAIN",
        str(settings["production_email_domain"]),
    ).strip().lower()

    settings["session_timeout_minutes"] = int(
        os.getenv(
            "HEVEMIND_SESSION_TIMEOUT_MINUTES",
            str(settings["session_timeout_minutes"]),
        )
    )

    settings["oidc_provider_name"] = os.getenv(
        "HEVEMIND_OIDC_PROVIDER",
        str(settings["oidc_provider_name"]),
    ).strip()

    return settings


def load_users_payload() -> dict[str, Any]:
    payload = load_json(USERS_PATH, DEFAULT_USERS)
    if not isinstance(payload.get("users"), list):
        raise RuntimeError(f"'users' must be a list in {USERS_PATH}")
    return payload


def normalise_email(email: str) -> str:
    return email.strip().lower()


def encode_password_hash(password: str, iterations: int = 390_000) -> str:
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


def verify_password(password: str, encoded_hash: str) -> bool:
    try:
        algorithm, iterations_text, salt_text, key_text = encoded_hash.split(
            "$",
            maxsplit=3,
        )
        if algorithm != "pbkdf2_sha256":
            return False

        iterations = int(iterations_text)
        salt = base64.urlsafe_b64decode(salt_text.encode("ascii"))
        expected = base64.urlsafe_b64decode(key_text.encode("ascii"))
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            iterations,
        )
        return hmac.compare_digest(candidate, expected)
    except (ValueError, TypeError, base64.binascii.Error):
        return False


def validate_password_strength(password: str) -> None:
    checks = {
        "at least 12 characters": len(password) >= 12,
        "one uppercase character": any(c.isupper() for c in password),
        "one lowercase character": any(c.islower() for c in password),
        "one number": any(c.isdigit() for c in password),
        "one special character": any(not c.isalnum() for c in password),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Password must contain " + ", ".join(failed) + ".")


def find_user(email: str) -> dict[str, Any] | None:
    normalised = normalise_email(email)
    for user in load_users_payload()["users"]:
        if not isinstance(user, dict):
            continue
        if normalise_email(str(user.get("email", ""))) == normalised:
            return user
    return None


def email_allowed(
    email: str,
    settings: dict[str, Any],
    auth_source: str,
) -> bool:
    normalised = normalise_email(email)
    if "@" not in normalised:
        return False

    domain_key = (
        "production_email_domain"
        if auth_source == "oidc"
        else "development_email_domain"
    )
    domain = str(settings[domain_key]).strip().lower()
    return bool(domain) and normalised.endswith(f"@{domain}")


def user_locked(user: dict[str, Any]) -> tuple[bool, int]:
    locked_until = user.get("locked_until_epoch", 0)
    if not isinstance(locked_until, (int, float)):
        return False, 0

    remaining = int(float(locked_until) - time.time())
    return remaining > 0, max(remaining, 0)


def update_user(
    email: str,
    **changes: Any,
) -> None:
    payload = load_users_payload()
    normalised = normalise_email(email)

    for user in payload["users"]:
        if not isinstance(user, dict):
            continue
        if normalise_email(str(user.get("email", ""))) == normalised:
            user.update(changes)
            atomic_write_json(USERS_PATH, payload)
            return


def audit(
    event_type: str,
    *,
    email: str | None = None,
    role: str | None = None,
    auth_source: str | None = None,
    success: bool | None = None,
    details: str | None = None,
) -> None:
    ensure_directories()
    record = {
        "timestamp_utc": utc_now_iso(),
        "event_type": event_type,
        "email": normalise_email(email) if email else None,
        "role": role,
        "auth_source": auth_source,
        "success": success,
        "details": details,
    }
    with AUDIT_LOG_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def create_or_update_user() -> None:
    ensure_directories()
    settings = load_settings()
    payload = load_users_payload()

    print("\nHeveMind local development user setup")
    print(
        "Development domain:",
        settings["development_email_domain"],
    )

    email = normalise_email(input("Email: "))
    if not email_allowed(email, settings, "development"):
        raise SystemExit(
            "Email must use the development domain "
            f"{settings['development_email_domain']}."
        )

    display_name = input("Display name: ").strip() or email
    role = input("Role [viewer/engineer/admin]: ").strip().lower()
    if role not in ALLOWED_ROLES:
        raise SystemExit("Role must be viewer, engineer, or admin.")

    password = getpass.getpass("Password: ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        raise SystemExit("Passwords do not match.")

    try:
        validate_password_strength(password)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    record = {
        "email": email,
        "display_name": display_name,
        "role": role,
        "active": True,
        "password_hash": encode_password_hash(password),
        "failed_attempts": 0,
        "locked_until_epoch": 0,
        "created_utc": utc_now_iso(),
        "last_login_utc": None,
    }

    replaced = False
    for index, existing in enumerate(payload["users"]):
        if (
            isinstance(existing, dict)
            and normalise_email(str(existing.get("email", ""))) == email
        ):
            payload["users"][index] = record
            replaced = True
            break

    if not replaced:
        payload["users"].append(record)

    atomic_write_json(USERS_PATH, payload)
    audit(
        "development_user_created_or_updated",
        email=email,
        role=role,
        auth_source="administration_cli",
        success=True,
    )

    print(f"\nUser saved successfully: {USERS_PATH}")


def list_users() -> None:
    users = [
        user
        for user in load_users_payload()["users"]
        if isinstance(user, dict)
    ]
    if not users:
        print("No local development users are configured.")
        return

    for user in users:
        print(
            f"{user.get('email')} | "
            f"role={user.get('role')} | "
            f"active={user.get('active', True)}"
        )


def handle_cli() -> bool:
    parser = argparse.ArgumentParser()
    parser.add_argument("--create-user", action="store_true")
    parser.add_argument("--list-users", action="store_true")
    parser.add_argument("--generate-password-hash", action="store_true")
    args = parser.parse_args()

    if args.create_user:
        create_or_update_user()
        return True

    if args.list_users:
        list_users()
        return True

    if args.generate_password_hash:
        password = getpass.getpass("Password: ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            raise SystemExit("Passwords do not match.")
        validate_password_strength(password)
        print(encode_password_hash(password))
        return True

    return False


if __name__ == "__main__" and handle_cli():
    raise SystemExit(0)


import streamlit as st


st.set_page_config(
    page_title="HeveMind Secure Access",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


LOGIN_CSS = """
<style>
:root {
    --navy-950: #071426;
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
}
.stApp {
    background: linear-gradient(180deg, #f4f8fb 0%, #eaf1f6 100%);
}

.block-container {
    max-width: 1120px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}
.brand-card {
    background: linear-gradient(
        145deg,
        var(--navy-950) 0%,
        var(--navy-800) 70%,
        var(--blue-600) 100%
    );
    border-radius: 22px;
    padding: 3rem 2.7rem;
    min-height: 560px;
    color: var(--white);
    box-shadow: 0 22px 55px rgba(7, 20, 38, 0.16);
}
.brand-title {
    font-size: 2.7rem;
    font-weight: 780;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin-bottom: 1rem;
}
.brand-subtitle {
    color: #d7e8f4;
    font-size: 1rem;
    line-height: 1.6;
}
.brand-feature {
    margin-top: 1rem;
    padding: 0.9rem 1rem;
    border-radius: 12px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.13);
    color: #eaf4fa;
    font-size: 0.9rem;
    line-height: 1.5;
}
.login-heading {
    color: var(--slate-900);
    font-size: 1.8rem;
    font-weight: 760;
    margin-bottom: 0.45rem;
}
.login-subtitle {
    color: var(--slate-500);
    font-size: 0.9rem;
    line-height: 1.5;
    margin-bottom: 1.3rem;
}
.security-note {
    border-left: 4px solid var(--blue-600);
    background: #eef7fc;
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    color: var(--slate-700);
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 1rem;
}
.warning-note {
    border-left: 4px solid var(--amber);
    background: var(--amber-bg);
    border-radius: 10px;
    padding: 0.8rem 0.95rem;
    color: #6b4c11;
    font-size: 0.82rem;
    line-height: 1.45;
    margin-top: 1rem;
}
.session-banner {
    background:
        linear-gradient(
            145deg,
            #123252 0%,
            #17456f 100%
        );
    border: 1px solid rgba(255, 255, 255, 0.14);
    color: #ffffff;
    border-radius: 14px;
    padding: 1rem 1.05rem;
    font-weight: 650;
    line-height: 1.6;
    margin-bottom: 0.8rem;
    box-shadow: 0 8px 20px rgba(0, 0, 0, 0.16);
}

[data-testid="stSidebar"] {
    min-width: 310px;
    max-width: 310px;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.35rem;
    padding-left: 1.2rem;
    padding-right: 1.2rem;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    background: rgba(255, 255, 255, 0.10);
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.24);
    border-radius: 10px;
    font-weight: 700;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255, 255, 255, 0.18);
    border-color: rgba(255, 255, 255, 0.42);
    color: #ffffff;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255, 255, 255, 0.18);
}

.stTextInput input {
    border-radius: 10px;
}
.stButton > button {
    border-radius: 10px;
    font-weight: 700;
    min-height: 2.8rem;
}
footer {
    visibility: hidden;
}
</style>
"""

is_authenticated = st.session_state.get(
    "hevemind_authenticated",
    False,
)

st.markdown(
    LOGIN_CSS,
    unsafe_allow_html=True,
)

if not is_authenticated:
    st.markdown(
        """
        <style>
            [data-testid="stSidebar"] {
                display: none;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

@dataclass(frozen=True)
class AuthenticatedUser:
    email: str
    display_name: str
    role: str
    auth_source: str


def escape_html(value: Any) -> str:
    import html

    return html.escape(
        str(value),
        quote=True,
    )


def clear_session() -> None:
    for key in [
        "hevemind_authenticated",
        "hevemind_user",
        "hevemind_login_time",
        "hevemind_last_activity",
    ]:
        st.session_state.pop(key, None)


def store_session(user: AuthenticatedUser) -> None:
    now = time.time()
    st.session_state["hevemind_authenticated"] = True
    st.session_state["hevemind_user"] = {
        "email": user.email,
        "display_name": user.display_name,
        "role": user.role,
        "auth_source": user.auth_source,
    }
    st.session_state["hevemind_login_time"] = now
    st.session_state["hevemind_last_activity"] = now


def session_valid(settings: dict[str, Any]) -> bool:
    if not st.session_state.get("hevemind_authenticated", False):
        return False

    last_activity = st.session_state.get("hevemind_last_activity")
    if not isinstance(last_activity, (int, float)):
        clear_session()
        return False

    timeout_seconds = int(settings["session_timeout_minutes"]) * 60
    if time.time() - float(last_activity) > timeout_seconds:
        user = st.session_state.get("hevemind_user", {})
        audit(
            "session_timeout",
            email=str(user.get("email", "")),
            role=str(user.get("role", "")),
            auth_source=str(user.get("auth_source", "")),
            success=True,
        )
        clear_session()
        return False

    st.session_state["hevemind_last_activity"] = time.time()
    return True


def current_user(settings: dict[str, Any]) -> dict[str, Any] | None:
    if not session_valid(settings):
        return None
    user = st.session_state.get("hevemind_user")
    return user if isinstance(user, dict) else None


def authenticate_local(
    email: str,
    password: str,
    settings: dict[str, Any],
) -> tuple[AuthenticatedUser | None, str | None]:
    normalised = normalise_email(email)
    user = find_user(normalised)
    generic_error = (
        "The email or password is invalid, "
        "or this account is not authorised."
    )

    if (
        user is None
        or not user.get("active", True)
        or not email_allowed(normalised, settings, "development")
    ):
        audit(
            "login_attempt",
            email=normalised,
            auth_source="development",
            success=False,
            details="Unknown, inactive, or unauthorised account.",
        )
        return None, generic_error

    locked, seconds_remaining = user_locked(user)
    if locked:
        minutes = max(1, seconds_remaining // 60 + 1)
        audit(
            "login_attempt",
            email=normalised,
            role=str(user.get("role", "viewer")),
            auth_source="development",
            success=False,
            details="Account locked.",
        )
        return (
            None,
            f"This account is temporarily locked. Try again in {minutes} minutes.",
        )

    if not verify_password(
        password,
        str(user.get("password_hash", "")),
    ):
        failed_attempts = int(user.get("failed_attempts", 0) or 0) + 1
        maximum = int(settings["maximum_failed_attempts"])
        locked_until = 0.0

        if failed_attempts >= maximum:
            locked_until = (
                time.time()
                + int(settings["lockout_minutes"]) * 60
            )

        update_user(
            normalised,
            failed_attempts=0 if locked_until else failed_attempts,
            locked_until_epoch=locked_until,
        )

        audit(
            "login_attempt",
            email=normalised,
            role=str(user.get("role", "viewer")),
            auth_source="development",
            success=False,
            details=(
                "Invalid password. Account locked."
                if locked_until
                else f"Invalid password. Attempt {failed_attempts} of {maximum}."
            ),
        )
        return None, generic_error

    role = str(user.get("role", "viewer")).strip().lower()
    if role not in ALLOWED_ROLES:
        role = "viewer"

    display_name = str(
        user.get("display_name", normalised)
    ).strip()

    update_user(
        normalised,
        failed_attempts=0,
        locked_until_epoch=0,
        last_login_utc=utc_now_iso(),
    )

    audit(
        "login_attempt",
        email=normalised,
        role=role,
        auth_source="development",
        success=True,
    )

    return (
        AuthenticatedUser(
            email=normalised,
            display_name=display_name,
            role=role,
            auth_source="development",
        ),
        None,
    )


def extract_oidc_value(
    oidc_user: Any,
    keys: list[str],
) -> Any:
    for key in keys:
        value = getattr(oidc_user, key, None)
        if value not in (None, ""):
            return value
        if isinstance(oidc_user, dict):
            value = oidc_user.get(key)
            if value not in (None, ""):
                return value
    return None


def extract_oidc_role(oidc_user: Any) -> str:
    candidates: list[str] = []

    for key in ["role", "roles", "groups"]:
        value = extract_oidc_value(oidc_user, [key])

        if isinstance(value, str):
            candidates.append(value.lower())
        elif isinstance(value, (list, tuple, set)):
            candidates.extend(str(item).lower() for item in value)

    if any("admin" in value for value in candidates):
        return "admin"
    if any("engineer" in value for value in candidates):
        return "engineer"
    return "viewer"


def authenticate_oidc(
    settings: dict[str, Any],
) -> AuthenticatedUser | None:
    if not hasattr(st, "login"):
        st.error(
            "This Streamlit version does not support OIDC login."
        )
        return None

    oidc_user = getattr(st, "user", None)
    is_logged_in = bool(
        extract_oidc_value(oidc_user, ["is_logged_in"])
    )

    if not is_logged_in:
        if st.button(
            "Sign in with Infineon corporate account",
            use_container_width=True,
        ):
            st.login(str(settings["oidc_provider_name"]))
        return None

    email = normalise_email(
        str(
            extract_oidc_value(
                oidc_user,
                ["email", "preferred_username", "upn"],
            )
            or ""
        )
    )

    if not email_allowed(email, settings, "oidc"):
        audit(
            "oidc_access_denied",
            email=email,
            auth_source="oidc",
            success=False,
            details="Identity outside approved domain.",
        )
        st.error(
            "Your authenticated account is not authorised "
            "to access HeveMind."
        )
        if st.button("Sign out", use_container_width=True):
            st.logout()
        return None

    display_name = str(
        extract_oidc_value(
            oidc_user,
            ["name", "given_name"],
        )
        or email
    ).strip()

    role = extract_oidc_role(oidc_user)

    audit(
        "oidc_login",
        email=email,
        role=role,
        auth_source="oidc",
        success=True,
    )

    return AuthenticatedUser(
        email=email,
        display_name=display_name,
        role=role,
        auth_source="oidc",
    )


def render_login(settings: dict[str, Any]) -> None:
    brand_column, login_column = st.columns(
        [1.05, 0.95],
        gap="large",
    )

    with brand_column:
        st.markdown(
            """
            <div class="brand-card">
                <div class="brand-title">HeveMind</div>
                <div class="brand-subtitle">
                    Secure semiconductor decision-support access for
                    authorised engineering personnel.
                </div>
                <div class="brand-feature">
                    Calibrated failure-risk assessment
                </div>
                <div class="brand-feature">
                    Confidence, uncertainty and abstention evidence
                </div>
                <div class="brand-feature">
                    Explainability, historical cases and sensor priorities
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with login_column:
        st.markdown(
            """
            <div class="login-heading">Authorised access</div>
            <div class="login-subtitle">
                Sign in using an approved identity. Access attempts
                are recorded in the local authentication audit log.
            </div>
            """,
            unsafe_allow_html=True,
        )

        auth_mode = str(settings["auth_mode"]).strip().lower()
        authenticated_user: AuthenticatedUser | None = None

        if auth_mode == "oidc":
            authenticated_user = authenticate_oidc(settings)

        elif auth_mode == "development":
            users = [
                user
                for user in load_users_payload()["users"]
                if isinstance(user, dict) and user.get("active", True)
            ]

            if not users:
                st.error("No local development user is configured.")
                st.code(
                    "cd /Users/barkavi/Desktop/HeveMind\n"
                    "/usr/local/bin/python3 "
                    "src/21_secure_access_gateway.py --create-user",
                    language="bash",
                )
            else:
                with st.form(
                    "development_login_form",
                    clear_on_submit=False,
                ):
                    email = st.text_input(
                        "Email",
                        placeholder=(
                            "name@"
                            + str(
                                settings[
                                    "development_email_domain"
                                ]
                            )
                        ),
                    )
                    password = st.text_input(
                        "Password",
                        type="password",
                    )
                    submitted = st.form_submit_button(
                        "Sign in",
                        use_container_width=True,
                    )

                if submitted:
                    authenticated_user, error = authenticate_local(
                        email,
                        password,
                        settings,
                    )
                    if error:
                        st.error(error)

            st.markdown(
                """
                <div class="warning-note">
                    Local authentication is for testing only.
                    Production access must use Infineon-approved OIDC/SSO.
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            st.error(
                "Unsupported auth_mode. Use development or oidc."
            )

        if authenticated_user is not None:
            store_session(authenticated_user)
            st.rerun()


def render_dashboard(user: dict[str, Any]) -> None:
    display_name = escape_html(
        user.get(
            "display_name",
            user.get(
                "email",
                "User",
            ),
        )
    )

    role_label = escape_html(
        str(
            user.get(
                "role",
                "viewer",
            )
        ).title()
    )

    st.sidebar.markdown(
        f"""
        <div class="session-banner">
            <div style="font-size:0.76rem;opacity:0.78;letter-spacing:0.04em;text-transform:uppercase;">
                Signed in as
            </div>
            <div style="font-size:1.05rem;font-weight:750;margin-top:0.18rem;">
                {display_name}
            </div>
            <div style="font-size:0.82rem;opacity:0.88;margin-top:0.2rem;">
                Role: {role_label}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.sidebar.button("Sign out", use_container_width=True):
        audit(
            "logout",
            email=str(user.get("email", "")),
            role=str(user.get("role", "")),
            auth_source=str(user.get("auth_source", "")),
            success=True,
        )
        auth_source = str(user.get("auth_source", ""))
        clear_session()

        if auth_source == "oidc" and hasattr(st, "logout"):
            st.logout()

        st.rerun()

    if not DASHBOARD_PATH.exists():
        st.error(f"Dashboard not found: {DASHBOARD_PATH}")
        return

    st.session_state["hevemind_user"] = user
    runpy.run_path(str(DASHBOARD_PATH), run_name="__main__")


def main() -> None:
    ensure_directories()
    settings = load_settings()
    user = current_user(settings)

    if user is None:
        render_login(settings)
        return

    render_dashboard(user)


if __name__ == "__main__":
    main()
