from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
CONFIG_DIR = ROOT_DIR / "config"
LOG_DIR = ROOT_DIR / "logs"
REPORTS_DIR = ROOT_DIR / "reports"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
EXPORT_DIR = REPORTS_DIR / "deployment_manager"
RUNTIME_DIR = ROOT_DIR / ".runtime"

FASTAPI_SCRIPT = SRC_DIR / "19_fastapi_service.py"
SECURE_GATEWAY_SCRIPT = SRC_DIR / "21_secure_access_gateway.py"
AUDIT_SCRIPT = SRC_DIR / "24_audit_activity_tracking.py"
GOVERNANCE_SCRIPT = SRC_DIR / "25_governance_security_dashboard.py"
PDF_REPORT_SCRIPT = SRC_DIR / "26_executive_pdf_report_generator.py"
ADMIN_SCRIPT = SRC_DIR / "27_platform_administration_rbac.py"

PLATFORM_SETTINGS = CONFIG_DIR / "platform_settings.json"
RBAC_POLICY = CONFIG_DIR / "rbac_policy.json"
USERS_FILE = CONFIG_DIR / "users.json"
AUTH_SETTINGS = CONFIG_DIR / "auth_settings.json"
AUDIT_DATABASE = LOG_DIR / "hevemind_audit.db"

FASTAPI_PID = RUNTIME_DIR / "fastapi.pid"
STREAMLIT_PID = RUNTIME_DIR / "streamlit.pid"
FASTAPI_LOG = LOG_DIR / "fastapi_service.log"
STREAMLIT_LOG = LOG_DIR / "streamlit_service.log"

API_BASE_URL = os.getenv(
    "HEVEMIND_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

STREAMLIT_URL = os.getenv(
    "HEVEMIND_STREAMLIT_URL",
    "http://127.0.0.1:8501",
).rstrip("/")

STREAMLIT_PORT = int(
    os.getenv(
        "HEVEMIND_STREAMLIT_PORT",
        "8501",
    )
)

PYTHON_EXECUTABLE = os.getenv(
    "HEVEMIND_PYTHON_EXECUTABLE",
    sys.executable,
)

STARTUP_TIMEOUT_SECONDS = int(
    os.getenv(
        "HEVEMIND_STARTUP_TIMEOUT_SECONDS",
        "30",
    )
)

MINIMUM_DISK_FREE_GB = float(
    os.getenv(
        "HEVEMIND_MINIMUM_DISK_FREE_GB",
        "2",
    )
)

REQUIRED_PACKAGES = [
    "pandas",
    "numpy",
    "fastapi",
    "uvicorn",
    "streamlit",
    "joblib",
    "scikit-learn",
    "reportlab",
    "openpyxl",
]

CRITICAL_FILES = {
    "FastAPI service": FASTAPI_SCRIPT,
    "Secure gateway": SECURE_GATEWAY_SCRIPT,
    "Audit engine": AUDIT_SCRIPT,
    "Governance dashboard": GOVERNANCE_SCRIPT,
    "PDF report generator": PDF_REPORT_SCRIPT,
    "Administration console": ADMIN_SCRIPT,
    "Platform settings": PLATFORM_SETTINGS,
    "RBAC policy": RBAC_POLICY,
    "User registry": USERS_FILE,
    "Authentication settings": AUTH_SETTINGS,
}


@dataclass(frozen=True)
class CheckResult:
    name: str
    category: str
    status: str
    critical: bool
    message: str
    details: dict[str, Any]


@dataclass(frozen=True)
class ServiceStatus:
    name: str
    pid: int | None
    running: bool
    healthy: bool
    url: str
    response_time_ms: float | None
    message: str


def utc_now_iso() -> str:
    return datetime.now(
        timezone.utc
    ).isoformat()


def ensure_directories() -> None:
    for path in [
        CONFIG_DIR,
        LOG_DIR,
        REPORTS_DIR,
        ARTIFACTS_DIR,
        EXPORT_DIR,
        RUNTIME_DIR,
    ]:
        path.mkdir(
            parents=True,
            exist_ok=True,
        )


def json_safe(value: Any) -> Any:
    if value is None or isinstance(
        value,
        (
            str,
            int,
            float,
            bool,
        ),
    ):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, datetime):
        return value.astimezone(
            timezone.utc
        ).isoformat()

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

    if hasattr(value, "__dict__"):
        return json_safe(value.__dict__)

    return str(value)


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
            json_safe(payload),
            indent=4,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"Expected JSON object in {path}"
        )
    return payload


def format_bytes(value: int) -> str:
    amount = float(value)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if amount < 1024:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return f"{amount:.1f} PB"


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(
            path.read_text(
                encoding="utf-8"
            ).strip()
        )
    except Exception:
        return None


def process_exists(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def clean_stale_pid(path: Path) -> None:
    pid = read_pid(path)
    if pid is None or not process_exists(pid):
        path.unlink(missing_ok=True)


def terminate_process(
    pid: int,
    timeout: int = 10,
) -> bool:
    if not process_exists(pid):
        return True

    try:
        os.kill(pid, signal.SIGTERM)
    except Exception:
        return True

    deadline = time.time() + timeout

    while time.time() < deadline:
        if not process_exists(pid):
            return True
        time.sleep(0.25)

    try:
        os.kill(pid, signal.SIGKILL)
    except Exception:
        return True

    time.sleep(0.5)
    return not process_exists(pid)


def http_probe(
    url: str,
    expect_json: bool,
    timeout: float = 5,
) -> tuple[bool, float | None, Any, str | None]:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": (
                "application/json"
                if expect_json
                else "text/html"
            ),
            "User-Agent": (
                "HeveMind-Deployment-Manager/1.0"
            ),
        },
    )

    started = time.perf_counter()

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout,
        ) as response:
            raw = response.read().decode(
                "utf-8",
                errors="replace",
            )

        elapsed = (
            time.perf_counter()
            - started
        ) * 1000

        payload: Any = raw

        if expect_json:
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                payload = {
                    "raw": raw
                }

        return True, elapsed, payload, None

    except Exception as error:
        elapsed = (
            time.perf_counter()
            - started
        ) * 1000
        return False, elapsed, None, str(error)


def check_python() -> CheckResult:
    passed = sys.version_info >= (3, 11)
    return CheckResult(
        name="Python version",
        category="Environment",
        status="PASS" if passed else "FAIL",
        critical=True,
        message=platform.python_version(),
        details={
            "executable": sys.executable,
            "minimum": "3.11",
        },
    )


def check_disk() -> CheckResult:
    usage = shutil.disk_usage(ROOT_DIR)
    free_gb = usage.free / (1024 ** 3)
    passed = free_gb >= MINIMUM_DISK_FREE_GB
    return CheckResult(
        name="Disk space",
        category="Environment",
        status="PASS" if passed else "FAIL",
        critical=True,
        message=f"{free_gb:.2f} GB free",
        details={
            "total": format_bytes(usage.total),
            "used": format_bytes(usage.used),
            "free": format_bytes(usage.free),
        },
    )


def check_package(package: str) -> CheckResult:
    try:
        version = importlib.metadata.version(
            package
        )
        return CheckResult(
            name=package,
            category="Packages",
            status="PASS",
            critical=True,
            message=version,
            details={},
        )
    except importlib.metadata.PackageNotFoundError:
        return CheckResult(
            name=package,
            category="Packages",
            status="FAIL",
            critical=True,
            message="Not installed",
            details={},
        )


def check_file(
    label: str,
    path: Path,
) -> CheckResult:
    exists = path.exists()
    return CheckResult(
        name=label,
        category="Required Files",
        status="PASS" if exists else "FAIL",
        critical=True,
        message=str(path),
        details={
            "exists": exists,
            "size_bytes": (
                path.stat().st_size
                if exists and path.is_file()
                else None
            ),
        },
    )


def check_json(
    label: str,
    path: Path,
) -> CheckResult:
    if not path.exists():
        return CheckResult(
            name=label,
            category="Configuration",
            status="FAIL",
            critical=True,
            message="File missing",
            details={
                "path": str(path)
            },
        )

    try:
        payload = load_json(path)
        return CheckResult(
            name=label,
            category="Configuration",
            status="PASS",
            critical=True,
            message="Valid JSON",
            details={
                "keys": sorted(
                    payload.keys()
                )
            },
        )
    except Exception as error:
        return CheckResult(
            name=label,
            category="Configuration",
            status="FAIL",
            critical=True,
            message=str(error),
            details={
                "path": str(path)
            },
        )


def check_audit_database() -> CheckResult:
    if not AUDIT_DATABASE.exists():
        return CheckResult(
            name="Audit database",
            category="Data Services",
            status="FAIL",
            critical=True,
            message="Missing",
            details={
                "path": str(AUDIT_DATABASE)
            },
        )

    import sqlite3

    try:
        with sqlite3.connect(
            AUDIT_DATABASE
        ) as connection:
            row = connection.execute(
                "PRAGMA integrity_check"
            ).fetchone()

        message = str(
            row[0]
            if row
            else "Unknown"
        )
        passed = message.lower() == "ok"

        return CheckResult(
            name="Audit database",
            category="Data Services",
            status="PASS" if passed else "FAIL",
            critical=True,
            message=message,
            details={
                "size": format_bytes(
                    AUDIT_DATABASE.stat().st_size
                )
            },
        )
    except Exception as error:
        return CheckResult(
            name="Audit database",
            category="Data Services",
            status="FAIL",
            critical=True,
            message=str(error),
            details={},
        )


def check_artifacts() -> CheckResult:
    artifacts = []
    if ARTIFACTS_DIR.exists():
        artifacts = [
            path
            for path in ARTIFACTS_DIR.rglob("*")
            if path.is_file()
        ]

    return CheckResult(
        name="Model and report artifacts",
        category="Model Assets",
        status="PASS" if artifacts else "WARN",
        critical=False,
        message=f"{len(artifacts)} files found",
        details={
            "examples": [
                str(
                    path.relative_to(ROOT_DIR)
                )
                for path in artifacts[:15]
            ]
        },
    )


def check_api_health() -> CheckResult:
    ok, elapsed, payload, error = http_probe(
        API_BASE_URL + "/health",
        True,
    )
    return CheckResult(
        name="FastAPI health",
        category="Runtime Services",
        status="PASS" if ok else "FAIL",
        critical=True,
        message=(
            f"{elapsed:.1f} ms"
            if ok and elapsed is not None
            else str(error)
        ),
        details={
            "url": API_BASE_URL + "/health",
            "payload": payload,
        },
    )


def check_api_ready() -> CheckResult:
    ok, elapsed, payload, error = http_probe(
        API_BASE_URL + "/ready",
        True,
    )
    return CheckResult(
        name="FastAPI readiness",
        category="Runtime Services",
        status="PASS" if ok else "WARN",
        critical=False,
        message=(
            f"{elapsed:.1f} ms"
            if ok and elapsed is not None
            else str(error)
        ),
        details={
            "url": API_BASE_URL + "/ready",
            "payload": payload,
        },
    )


def check_streamlit() -> CheckResult:
    ok, elapsed, _, error = http_probe(
        STREAMLIT_URL,
        False,
    )
    return CheckResult(
        name="Streamlit reachability",
        category="Runtime Services",
        status="PASS" if ok else "WARN",
        critical=False,
        message=(
            f"{elapsed:.1f} ms"
            if ok and elapsed is not None
            else str(error)
        ),
        details={
            "url": STREAMLIT_URL
        },
    )


def collect_checks(
    include_runtime: bool = True,
) -> list[CheckResult]:
    ensure_directories()

    checks = [
        check_python(),
        check_disk(),
    ]

    checks.extend(
        check_package(package)
        for package in REQUIRED_PACKAGES
    )

    checks.extend(
        check_file(label, path)
        for label, path in CRITICAL_FILES.items()
    )

    checks.extend(
        [
            check_json(
                "Platform settings",
                PLATFORM_SETTINGS,
            ),
            check_json(
                "RBAC policy",
                RBAC_POLICY,
            ),
            check_json(
                "User registry",
                USERS_FILE,
            ),
            check_json(
                "Authentication settings",
                AUTH_SETTINGS,
            ),
            check_audit_database(),
            check_artifacts(),
        ]
    )

    if include_runtime:
        checks.extend(
            [
                check_api_health(),
                check_api_ready(),
                check_streamlit(),
            ]
        )

    return checks


def readiness_summary(
    checks: list[CheckResult],
) -> dict[str, Any]:
    passed = sum(
        check.status == "PASS"
        for check in checks
    )
    warnings = sum(
        check.status == "WARN"
        for check in checks
    )
    failed = sum(
        check.status == "FAIL"
        for check in checks
    )
    critical_failures = sum(
        check.status == "FAIL"
        and check.critical
        for check in checks
    )

    total_weight = sum(
        2 if check.critical else 1
        for check in checks
    )

    achieved = sum(
        (
            2 if check.critical else 1
        )
        * (
            1
            if check.status == "PASS"
            else 0.5
            if check.status == "WARN"
            else 0
        )
        for check in checks
    )

    score = (
        100 * achieved / total_weight
        if total_weight
        else 0
    )

    return {
        "score": round(score, 2),
        "passed": passed,
        "warnings": warnings,
        "failed": failed,
        "critical_failures": critical_failures,
        "total_checks": len(checks),
        "production_ready": (
            critical_failures == 0
            and score >= 90
        ),
    }


def fastapi_status() -> ServiceStatus:
    clean_stale_pid(FASTAPI_PID)
    pid = read_pid(FASTAPI_PID)
    running = process_exists(pid)
    ok, elapsed, _, error = http_probe(
        API_BASE_URL + "/health",
        True,
    )
    return ServiceStatus(
        name="FastAPI",
        pid=pid,
        running=running,
        healthy=ok,
        url=API_BASE_URL,
        response_time_ms=elapsed,
        message="Healthy" if ok else str(error),
    )


def streamlit_status() -> ServiceStatus:
    clean_stale_pid(STREAMLIT_PID)
    pid = read_pid(STREAMLIT_PID)
    running = process_exists(pid)
    ok, elapsed, _, error = http_probe(
        STREAMLIT_URL,
        False,
    )
    return ServiceStatus(
        name="Streamlit",
        pid=pid,
        running=running,
        healthy=ok,
        url=STREAMLIT_URL,
        response_time_ms=elapsed,
        message="Reachable" if ok else str(error),
    )


def service_statuses() -> list[ServiceStatus]:
    return [
        fastapi_status(),
        streamlit_status(),
    ]


def start_fastapi() -> ServiceStatus:
    current = fastapi_status()
    if current.running:
        return current

    ensure_directories()

    with FASTAPI_LOG.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                str(FASTAPI_SCRIPT),
            ],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    FASTAPI_PID.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    deadline = (
        time.time()
        + STARTUP_TIMEOUT_SECONDS
    )

    while time.time() < deadline:
        status = fastapi_status()
        if status.healthy:
            return status
        if not process_exists(process.pid):
            break
        time.sleep(0.5)

    return fastapi_status()


def start_streamlit() -> ServiceStatus:
    current = streamlit_status()
    if current.running:
        return current

    ensure_directories()

    with STREAMLIT_LOG.open(
        "a",
        encoding="utf-8",
    ) as log_file:
        process = subprocess.Popen(
            [
                PYTHON_EXECUTABLE,
                "-m",
                "streamlit",
                "run",
                str(SECURE_GATEWAY_SCRIPT),
                "--server.port",
                str(STREAMLIT_PORT),
                "--server.address",
                "127.0.0.1",
                "--browser.gatherUsageStats",
                "false",
            ],
            cwd=str(ROOT_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            text=True,
        )

    STREAMLIT_PID.write_text(
        str(process.pid),
        encoding="utf-8",
    )

    deadline = (
        time.time()
        + STARTUP_TIMEOUT_SECONDS
    )

    while time.time() < deadline:
        status = streamlit_status()
        if status.healthy:
            return status
        if not process_exists(process.pid):
            break
        time.sleep(0.5)

    return streamlit_status()


def stop_fastapi() -> ServiceStatus:
    pid = read_pid(FASTAPI_PID)
    if pid is not None:
        terminate_process(pid)
    FASTAPI_PID.unlink(missing_ok=True)
    return fastapi_status()


def stop_streamlit() -> ServiceStatus:
    pid = read_pid(STREAMLIT_PID)
    if pid is not None:
        terminate_process(pid)
    STREAMLIT_PID.unlink(missing_ok=True)
    return streamlit_status()


def start_all() -> list[ServiceStatus]:
    return [
        start_fastapi(),
        start_streamlit(),
    ]


def stop_all() -> list[ServiceStatus]:
    return [
        stop_streamlit(),
        stop_fastapi(),
    ]


def restart_all() -> list[ServiceStatus]:
    stop_all()
    time.sleep(1)
    return start_all()


def tail_file(
    path: Path,
    lines: int = 120,
) -> str:
    if not path.exists():
        return f"Log file does not exist: {path}"

    content = path.read_text(
        encoding="utf-8",
        errors="replace",
    ).splitlines()

    return "\n".join(
        content[-lines:]
    )


def deployment_report() -> dict[str, Any]:
    checks = collect_checks(
        include_runtime=True
    )

    return {
        "generated_utc": utc_now_iso(),
        "root_dir": str(ROOT_DIR),
        "readiness": readiness_summary(checks),
        "checks": [
            asdict(check)
            for check in checks
        ],
        "services": [
            asdict(status)
            for status in service_statuses()
        ],
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "hostname": socket.gethostname(),
        },
    }


def export_reports() -> tuple[Path, Path]:
    ensure_directories()
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )
    payload = deployment_report()

    json_path = (
        EXPORT_DIR
        / f"deployment_readiness_{timestamp}.json"
    )
    xlsx_path = (
        EXPORT_DIR
        / f"deployment_readiness_{timestamp}.xlsx"
    )

    atomic_write_json(
        json_path,
        payload,
    )

    with pd.ExcelWriter(
        xlsx_path,
        engine="openpyxl",
    ) as writer:
        pd.DataFrame(
            [payload["readiness"]]
        ).to_excel(
            writer,
            sheet_name="Readiness",
            index=False,
        )
        pd.DataFrame(
            payload["checks"]
        ).to_excel(
            writer,
            sheet_name="Checks",
            index=False,
        )
        pd.DataFrame(
            payload["services"]
        ).to_excel(
            writer,
            sheet_name="Services",
            index=False,
        )
        pd.DataFrame(
            [payload["system"]]
        ).to_excel(
            writer,
            sheet_name="System",
            index=False,
        )

    return json_path, xlsx_path


DEPLOYMENT_CSS = """
<style>
:root {
    --navy-950: #071426;
    --navy-900: #0b1f36;
    --navy-800: #123252;
    --blue-600: #1f6aa5;
    --slate-700: #405267;
    --slate-500: #708196;
    --slate-200: #dce3e9;
    --white: #ffffff;
}
.stApp {
    background: linear-gradient(180deg, #f5f8fb 0%, #edf3f7 100%);
}
.block-container {
    max-width: 1450px;
    padding-top: 1.7rem;
    padding-bottom: 3rem;
}
.deployment-header {
    background: linear-gradient(
        110deg,
        var(--navy-950) 0%,
        var(--navy-800) 72%,
        var(--blue-600) 100%
    );
    border-radius: 18px;
    padding: 1.25rem 1.45rem;
    color: #ffffff;
    box-shadow: 0 14px 34px rgba(7, 20, 38, 0.17);
    margin-bottom: 1rem;
}
.deployment-title {
    font-size: 1.8rem;
    font-weight: 780;
}
.deployment-subtitle {
    color: #d8e8f4;
    font-size: 0.9rem;
    margin-top: 0.36rem;
}
.metric-card {
    min-height: 125px;
    background: #ffffff;
    border: 1px solid var(--slate-200);
    border-radius: 15px;
    padding: 0.95rem 1rem;
    box-shadow: 0 7px 20px rgba(21, 48, 75, 0.06);
}
.metric-label {
    color: var(--slate-500);
    font-size: 0.70rem;
    font-weight: 760;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-value {
    color: var(--navy-950);
    font-size: 1.55rem;
    font-weight: 790;
    margin-top: 0.42rem;
}
.metric-note {
    color: var(--slate-700);
    font-size: 0.77rem;
    margin-top: 0.36rem;
}
.stTabs [aria-selected="true"] {
    background: #e8f2f9 !important;
    color: var(--navy-950) !important;
    border-bottom: 3px solid var(--blue-600) !important;
}
footer {
    visibility: hidden;
}
</style>
"""


def render_metric(
    label: str,
    value: str,
    note: str,
) -> None:
    import streamlit as st

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


def require_admin() -> None:
    import streamlit as st

    user = st.session_state.get(
        "hevemind_user",
        {},
    )

    role = str(
        user.get("role", "")
    ).lower()

    standalone = (
        os.getenv(
            "HEVEMIND_DEPLOYMENT_STANDALONE_MODE",
            "false",
        ).lower()
        in {
            "1",
            "true",
            "yes",
            "on",
        }
    )

    if role != "admin" and not standalone:
        st.error(
            "Administrator access is required."
        )
        st.stop()


def render_deployment_console() -> None:
    import streamlit as st

    st.markdown(
        DEPLOYMENT_CSS,
        unsafe_allow_html=True,
    )

    require_admin()

    checks = collect_checks(
        include_runtime=True
    )
    summary = readiness_summary(
        checks
    )
    statuses = service_statuses()

    st.markdown(
        f"""
        <div class="deployment-header">
            <div class="deployment-title">
                HeveMind Enterprise Deployment Manager
            </div>
            <div class="deployment-subtitle">
                Environment validation, readiness scoring, service lifecycle,
                diagnostics, logs and deployment exports
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    columns = st.columns(5)

    metrics = [
        (
            "Readiness",
            f"{summary['score']:.1f}%",
            "Weighted deployment score.",
        ),
        (
            "Passed",
            str(summary["passed"]),
            "Checks completed successfully.",
        ),
        (
            "Warnings",
            str(summary["warnings"]),
            "Non-critical items requiring review.",
        ),
        (
            "Failures",
            str(summary["failed"]),
            "Checks that did not meet requirements.",
        ),
        (
            "Production ready",
            "Yes" if summary["production_ready"] else "No",
            "Requires no critical failures and score of at least 90%.",
        ),
    ]

    for column, metric in zip(
        columns,
        metrics,
    ):
        with column:
            render_metric(
                metric[0],
                metric[1],
                metric[2],
            )

    lifecycle_tab, checks_tab, services_tab, logs_tab, export_tab = st.tabs(
        [
            "Lifecycle",
            "Checks",
            "Services",
            "Logs",
            "Export",
        ]
    )

    with lifecycle_tab:
        action_columns = st.columns(3)

        with action_columns[0]:
            if st.button(
                "Start all services",
                use_container_width=True,
            ):
                st.session_state["deployment_action"] = [
                    asdict(item)
                    for item in start_all()
                ]
                st.rerun()

        with action_columns[1]:
            if st.button(
                "Restart all services",
                use_container_width=True,
            ):
                st.session_state["deployment_action"] = [
                    asdict(item)
                    for item in restart_all()
                ]
                st.rerun()

        with action_columns[2]:
            if st.button(
                "Stop all services",
                use_container_width=True,
            ):
                st.session_state["deployment_action"] = [
                    asdict(item)
                    for item in stop_all()
                ]
                st.rerun()

        if st.session_state.get(
            "deployment_action"
        ):
            st.json(
                st.session_state[
                    "deployment_action"
                ]
            )

        st.info(
            "Lifecycle controls manage only services started through this deployment manager."
        )

    with checks_tab:
        checks_df = pd.DataFrame(
            [
                {
                    **asdict(check),
                    "details": json.dumps(
                        check.details,
                        ensure_ascii=False,
                    ),
                }
                for check in checks
            ]
        )

        category = st.selectbox(
            "Category",
            options=[
                "All",
                *sorted(
                    checks_df[
                        "category"
                    ].unique()
                ),
            ],
        )

        if category != "All":
            checks_df = checks_df.loc[
                checks_df[
                    "category"
                ]
                == category
            ]

        st.dataframe(
            checks_df,
            use_container_width=True,
            hide_index=True,
            height=600,
        )

    with services_tab:
        st.dataframe(
            pd.DataFrame(
                [
                    asdict(status)
                    for status in statuses
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )

    with logs_tab:
        log_name = st.selectbox(
            "Log",
            options=[
                "FastAPI",
                "Streamlit",
            ],
        )

        lines = st.slider(
            "Lines",
            min_value=20,
            max_value=500,
            value=120,
            step=20,
        )

        selected = (
            FASTAPI_LOG
            if log_name == "FastAPI"
            else STREAMLIT_LOG
        )

        st.code(
            tail_file(
                selected,
                lines,
            ),
            language="text",
        )

    with export_tab:
        payload = deployment_report()

        st.download_button(
            "Download readiness JSON",
            data=json.dumps(
                payload,
                indent=2,
                ensure_ascii=False,
            ).encode("utf-8"),
            file_name=(
                "hevemind_deployment_readiness.json"
            ),
            mime="application/json",
            use_container_width=True,
        )

        if st.button(
            "Create JSON and Excel reports",
            use_container_width=True,
        ):
            json_path, xlsx_path = export_reports()
            st.success(
                f"Created: {json_path} and {xlsx_path}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "HeveMind enterprise deployment manager."
        )
    )

    parser.add_argument(
        "--status",
        action="store_true",
    )
    parser.add_argument(
        "--start",
        action="store_true",
    )
    parser.add_argument(
        "--stop",
        action="store_true",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
    )
    parser.add_argument(
        "--export",
        action="store_true",
    )
    parser.add_argument(
        "--no-runtime-checks",
        action="store_true",
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    action = False

    if arguments.start:
        print(
            pd.DataFrame(
                [
                    asdict(item)
                    for item in start_all()
                ]
            ).to_string(
                index=False
            )
        )
        action = True

    if arguments.stop:
        print(
            pd.DataFrame(
                [
                    asdict(item)
                    for item in stop_all()
                ]
            ).to_string(
                index=False
            )
        )
        action = True

    if arguments.restart:
        print(
            pd.DataFrame(
                [
                    asdict(item)
                    for item in restart_all()
                ]
            ).to_string(
                index=False
            )
        )
        action = True

    if arguments.status:
        print(
            pd.DataFrame(
                [
                    asdict(item)
                    for item in service_statuses()
                ]
            ).to_string(
                index=False
            )
        )
        action = True

    if arguments.diagnostics:
        checks = collect_checks(
            include_runtime=(
                not arguments.no_runtime_checks
            )
        )

        print(
            json.dumps(
                {
                    "readiness": readiness_summary(
                        checks
                    ),
                    "checks": [
                        asdict(check)
                        for check in checks
                    ],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        action = True

    if arguments.export:
        json_path, xlsx_path = export_reports()
        print(
            f"JSON report:  {json_path}"
        )
        print(
            f"Excel report: {xlsx_path}"
        )
        action = True

    if not action:
        payload = deployment_report()
        summary = payload["readiness"]

        print(
            "\n"
            + "=" * 108
        )
        print(
            "HEVEMIND ENTERPRISE DEPLOYMENT MANAGER"
        )
        print(
            "=" * 108
        )
        print(
            f"\nReadiness score:    {summary['score']:.1f}%"
        )
        print(
            f"Production ready:   {summary['production_ready']}"
        )
        print(
            f"Critical failures:  {summary['critical_failures']}"
        )
        print(
            "\nRun with --help to view commands."
        )


if __name__ == "__main__":
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is not None:
        import streamlit as st

        st.set_page_config(
            page_title="HeveMind Deployment Manager",
            page_icon=None,
            layout="wide",
            initial_sidebar_state="expanded",
        )

        render_deployment_console()

    else:
        main()