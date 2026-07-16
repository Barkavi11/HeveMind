from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import math
import os
import sys
import textwrap
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"
REPORTS_DIR = ROOT_DIR / "reports"
EXPORT_DIR = REPORTS_DIR / "executive_pdf_reports"

AUDIT_ENGINE_PATH = SRC_DIR / "24_audit_activity_tracking.py"

DEFAULT_API_BASE_URL = os.getenv(
    "HEVEMIND_API_BASE_URL",
    "http://127.0.0.1:8000",
).rstrip("/")

DEFAULT_TIMEOUT_SECONDS = float(
    os.getenv(
        "HEVEMIND_REPORT_API_TIMEOUT_SECONDS",
        "20",
    )
)

DEFAULT_ORGANISATION_LABEL = os.getenv(
    "HEVEMIND_REPORT_ORGANISATION",
    "HeveMind Engineering Decision Support",
).strip()

DEFAULT_SITE_LABEL = os.getenv(
    "HEVEMIND_REPORT_SITE",
    "Validated Research Environment",
).strip()

DEFAULT_REPORT_VERSION = os.getenv(
    "HEVEMIND_REPORT_VERSION",
    "1.0.0",
).strip()

DEFAULT_GENERATED_BY = os.getenv(
    "HEVEMIND_REPORT_GENERATED_BY",
    "HeveMind Report Service",
).strip()

DEFAULT_DISCLAIMER = (
    "This report supports engineering prioritisation and does not independently "
    "authorise release, rework, scrap, maintenance, or process changes. Final "
    "decisions remain subject to qualified human review and local operating procedures."
)

DECISION_ORDER = [
    "Insufficient Evidence",
    "High Risk",
    "Engineering Review",
    "Low Risk",
]

DECISION_COLOURS = {
    "Low Risk": "#1B7F5B",
    "Engineering Review": "#D89A1D",
    "High Risk": "#B43B3B",
    "Insufficient Evidence": "#6F4F9C",
}

RISK_TEXT = {
    "Low Risk": "Continue under standard monitoring unless local procedures indicate otherwise.",
    "Engineering Review": "Perform structured engineering review before release or operational action.",
    "High Risk": "Prioritise immediate engineering assessment and targeted sensor investigation.",
    "Insufficient Evidence": "Do not rely on automated output alone; verify data quality and gather additional evidence.",
}


# ============================================================
# OPTIONAL DEPENDENCIES
# ============================================================
try:
    import qrcode
except ImportError:
    qrcode = None

try:
    from reportlab.graphics.charts.barcharts import HorizontalBarChart
    from reportlab.graphics.charts.legends import Legend
    from reportlab.graphics.shapes import Drawing, String
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        BaseDocTemplate,
        Frame,
        Image,
        KeepTogether,
        PageBreak,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
except ImportError as error:
    raise RuntimeError(
        "ReportLab is required. Install it with: "
        "/usr/local/bin/python3 -m pip install reportlab"
    ) from error


# ============================================================
# DATA MODELS
# ============================================================
@dataclass(frozen=True)
class ReportIdentity:
    wafer_id: str
    report_id: str
    generated_utc: str
    report_version: str
    generated_by: str


@dataclass(frozen=True)
class ReportContext:
    organisation_label: str
    site_label: str
    api_base_url: str
    source_scope: str
    disclaimer: str


@dataclass(frozen=True)
class ReportArtifacts:
    output_pdf: Path
    output_json: Path | None
    sha256: str
    pages_estimated: int


# ============================================================
# GENERAL UTILITIES
# ============================================================
def ensure_output_directory() -> None:
    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def utc_now() -> datetime:
    return datetime.now(
        timezone.utc
    )


def utc_now_iso() -> str:
    return utc_now().isoformat()


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


def safe_float(
    value: Any,
    fallback: float | None = None,
) -> float | None:
    if value is None:
        return fallback

    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return fallback

    if math.isnan(numeric) or math.isinf(numeric):
        return fallback

    return numeric


def safe_int(
    value: Any,
    fallback: int = 0,
) -> int:
    numeric = safe_float(value)
    if numeric is None:
        return fallback
    return int(round(numeric))


def format_percentage(
    value: Any,
    digits: int = 1,
    fallback: str = "Unavailable",
) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return fallback

    if abs(numeric) <= 1.0:
        numeric *= 100.0

    return f"{numeric:.{digits}f}%"


def format_decimal(
    value: Any,
    digits: int = 3,
    fallback: str = "Unavailable",
) -> str:
    numeric = safe_float(value)
    if numeric is None:
        return fallback
    return f"{numeric:.{digits}f}"


def format_sensor_name(
    value: Any,
    fallback: str = "Unavailable",
) -> str:
    text = safe_text(value, fallback)
    if text.lower().startswith("sensor_"):
        return "Sensor " + text.split("_", 1)[1]
    return text


def slugify(value: str) -> str:
    output = []
    for character in value:
        if character.isalnum() or character in {"-", "_"}:
            output.append(character)
        else:
            output.append("_")

    slug = "".join(output).strip("_")
    return slug or "report"


def json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and (
            math.isnan(value) or math.isinf(value)
        ):
            return None
        return value

    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]

    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass

    return str(value)


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def nested_get(
    payload: dict[str, Any],
    candidates: Sequence[str],
    fallback: Any = None,
) -> Any:
    for candidate in candidates:
        current: Any = payload
        found = True

        for part in candidate.split("."):
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                found = False
                break

        if found and current is not None:
            return current

    return fallback


def normalise_rows(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")

    if isinstance(value, dict):
        if "items" in value and isinstance(value["items"], list):
            return [
                item
                for item in value["items"]
                if isinstance(item, dict)
            ]

        return [value]

    if isinstance(value, list):
        return [
            item
            for item in value
            if isinstance(item, dict)
        ]

    return []


# ============================================================
# API CLIENT
# ============================================================
def fetch_json(
    url: str,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url=url,
        headers={
            "Accept": "application/json",
            "User-Agent": "HeveMind-Report-Generator/1.0",
        },
        method="GET",
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            raw = response.read().decode("utf-8")

    except urllib.error.HTTPError as error:
        body = error.read().decode(
            "utf-8",
            errors="replace",
        )
        raise RuntimeError(
            f"API request failed with HTTP {error.code}: {body}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Unable to connect to the HeveMind API: {error}"
        ) from error

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "The API response was not valid JSON."
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "The API response must be a JSON object."
        )

    return payload


def fetch_wafer_report(
    wafer_id: str,
    api_base_url: str = DEFAULT_API_BASE_URL,
) -> dict[str, Any]:
    encoded_id = urllib.parse.quote(
        wafer_id,
        safe="",
    )

    candidate_paths = [
        f"/wafers/{encoded_id}",
        f"/wafer/{encoded_id}",
        f"/reports/{encoded_id}",
        f"/report/{encoded_id}",
    ]

    last_error: Exception | None = None

    for path in candidate_paths:
        try:
            payload = fetch_json(
                api_base_url.rstrip("/") + path
            )

            if "report" in payload and isinstance(payload["report"], dict):
                return payload["report"]

            if "item" in payload and isinstance(payload["item"], dict):
                return payload["item"]

            return payload

        except Exception as error:
            last_error = error

    raise RuntimeError(
        f"Unable to retrieve wafer report {wafer_id}. "
        f"Last error: {last_error}"
    )


def load_report_from_json(
    path: Path,
) -> dict[str, Any]:
    try:
        payload = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )
    except json.JSONDecodeError as error:
        raise RuntimeError(
            f"Invalid JSON report file: {path}"
        ) from error

    if not isinstance(payload, dict):
        raise RuntimeError(
            "The report JSON must contain a JSON object."
        )

    return payload


# ============================================================
# OPTIONAL AUDIT INTEGRATION
# ============================================================
def load_audit_engine() -> Any | None:
    if not AUDIT_ENGINE_PATH.exists():
        return None

    specification = importlib.util.spec_from_file_location(
        "hevemind_audit_activity_tracking",
        AUDIT_ENGINE_PATH,
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


def audit_report_generation(
    *,
    wafer_id: str,
    output_pdf: Path,
    report_sha256: str,
    user_email: str | None,
    display_name: str | None,
    role: str | None,
    auth_source: str | None,
    session_id: str | None,
) -> None:
    audit_engine = load_audit_engine()

    if audit_engine is None:
        return

    try:
        audit_engine.record_event(
            source_component="26_executive_pdf_report_generator",
            event_type="engineering_pdf_report_generated",
            category="export",
            outcome="success",
            user_email=user_email,
            display_name=display_name,
            role=role,
            auth_source=auth_source,
            session_id=session_id,
            resource_type="wafer_report",
            resource_id=wafer_id,
            action="generate_pdf",
            message="Executive engineering PDF report generated.",
            details={
                "export_format": "pdf",
                "filename": output_pdf.name,
                "sha256": report_sha256,
            },
        )
    except Exception:
        pass


# ============================================================
# REPORT DATA EXTRACTION
# ============================================================
def extract_identity(
    report: dict[str, Any],
) -> ReportIdentity:
    wafer_id = safe_text(
        nested_get(
            report,
            [
                "wafer_id",
                "query_wafer_id",
                "id",
                "summary.wafer_id",
                "record.wafer_id",
            ],
            "Unknown Wafer",
        )
    )

    return ReportIdentity(
        wafer_id=wafer_id,
        report_id=str(uuid.uuid4()),
        generated_utc=utc_now_iso(),
        report_version=DEFAULT_REPORT_VERSION,
        generated_by=DEFAULT_GENERATED_BY,
    )


def extract_primary_metrics(
    report: dict[str, Any],
) -> dict[str, Any]:
    return {
        "decision": safe_text(
            nested_get(
                report,
                [
                    "uncertainty_adjusted_decision",
                    "operational_decision",
                    "decision",
                    "summary.decision",
                ],
                "Unavailable",
            )
        ),
        "failure_probability": safe_float(
            nested_get(
                report,
                [
                    "calibrated_failure_probability",
                    "failure_probability",
                    "probability",
                    "summary.calibrated_failure_probability",
                ]
            )
        ),
        "prediction_confidence": safe_float(
            nested_get(
                report,
                [
                    "prediction_confidence",
                    "confidence",
                    "summary.prediction_confidence",
                ]
            )
        ),
        "combined_uncertainty": safe_float(
            nested_get(
                report,
                [
                    "combined_uncertainty",
                    "uncertainty",
                    "summary.combined_uncertainty",
                ]
            )
        ),
        "data_confidence": safe_float(
            nested_get(
                report,
                [
                    "data_confidence",
                    "reliability.data_confidence",
                    "summary.data_confidence",
                ]
            )
        ),
        "data_familiarity": safe_text(
            nested_get(
                report,
                [
                    "data_familiarity_band",
                    "data_familiarity",
                    "familiarity_band",
                    "summary.data_familiarity_band",
                ],
                "Unavailable",
            )
        ),
        "evidence_status": safe_text(
            nested_get(
                report,
                [
                    "evidence_agreement",
                    "evidence_status",
                    "summary.evidence_agreement",
                ],
                "Unavailable",
            )
        ),
        "priority_sensor": format_sensor_name(
            nested_get(
                report,
                [
                    "priority_1_sensor",
                    "priority_sensor",
                    "summary.priority_1_sensor",
                ],
                "Unavailable",
            )
        ),
        "recommended_action": safe_text(
            nested_get(
                report,
                [
                    "recommended_action",
                    "engineering_recommendation",
                    "recommendation",
                    "summary.recommended_action",
                ],
                "Unavailable",
            )
        ),
        "abstention_reason": safe_text(
            nested_get(
                report,
                [
                    "abstention_reason",
                    "reliability.abstention_reason",
                    "summary.abstention_reason",
                ],
                "No abstention",
            ),
            "No abstention",
        ),
        "missing_sensor_rate": safe_float(
            nested_get(
                report,
                [
                    "missing_sensor_rate",
                    "reliability.missing_sensor_rate",
                    "summary.missing_sensor_rate",
                ]
            )
        ),
    }


def extract_sensor_priorities(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        nested_get(
            report,
            [
                "investigation_priorities",
                "sensor_investigation_priorities",
                "sensor_priorities",
                "priority_sensors",
            ]
        ),
        nested_get(
            report,
            [
                "details.investigation_priorities",
                "technical_record.investigation_priorities",
            ]
        ),
    ]

    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        rows = normalise_rows(candidate)
        if rows:
            break

    normalised: list[dict[str, Any]] = []

    for index, row in enumerate(rows, start=1):
        normalised.append(
            {
                "rank": safe_int(
                    row.get(
                        "priority_rank",
                        row.get(
                            "rank",
                            index,
                        ),
                    ),
                    index,
                ),
                "sensor": format_sensor_name(
                    row.get(
                        "feature",
                        row.get(
                            "sensor",
                            row.get(
                                "priority_sensor",
                                "Unavailable",
                            ),
                        ),
                    )
                ),
                "priority_level": safe_text(
                    row.get(
                        "priority_level",
                        row.get(
                            "level",
                            "Unavailable",
                        ),
                    )
                ),
                "priority_score": safe_float(
                    row.get(
                        "priority_score",
                        row.get(
                            "score",
                            row.get(
                                "rank_aggregation_score",
                            ),
                        ),
                    )
                ),
                "measurement_status": safe_text(
                    row.get(
                        "measurement_status",
                        row.get(
                            "status",
                            "Unavailable",
                        ),
                    )
                ),
                "explanation": safe_text(
                    row.get(
                        "engineering_explanation",
                        row.get(
                            "explanation",
                            row.get(
                                "reason",
                                "Unavailable",
                            ),
                        ),
                    )
                ),
            }
        )

    return sorted(
        normalised,
        key=lambda item: item["rank"],
    )


def extract_sensor_attributions(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        nested_get(
            report,
            [
                "local_explanations",
                "sensor_attribution",
                "shap_explanations",
                "shap_values",
            ]
        ),
        nested_get(
            report,
            [
                "details.local_explanations",
                "technical_record.local_explanations",
            ]
        ),
    ]

    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        rows = normalise_rows(candidate)
        if rows:
            break

    normalised = []

    for row in rows:
        contribution = safe_float(
            row.get(
                "shap_value",
                row.get(
                    "contribution",
                    row.get(
                        "absolute_contribution",
                    ),
                ),
            )
        )

        normalised.append(
            {
                "sensor": format_sensor_name(
                    row.get(
                        "feature",
                        row.get(
                            "sensor",
                            "Unavailable",
                        ),
                    )
                ),
                "contribution": contribution,
                "direction": safe_text(
                    row.get(
                        "contribution_direction",
                        row.get(
                            "direction",
                            (
                                "Increases estimated risk"
                                if contribution is not None and contribution > 0
                                else "Decreases estimated risk"
                            ),
                        ),
                    )
                ),
                "measurement_status": safe_text(
                    row.get(
                        "measurement_status",
                        row.get(
                            "status",
                            "Unavailable",
                        ),
                    )
                ),
            }
        )

    return sorted(
        normalised,
        key=lambda item: abs(
            item["contribution"]
            if item["contribution"] is not None
            else 0.0
        ),
        reverse=True,
    )


def extract_historical_cases(
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    candidates = [
        nested_get(
            report,
            [
                "historical_neighbours",
                "historical_cases",
                "neighbours",
                "similar_cases",
            ]
        ),
        nested_get(
            report,
            [
                "details.historical_neighbours",
                "technical_record.historical_neighbours",
            ]
        ),
    ]

    rows: list[dict[str, Any]] = []

    for candidate in candidates:
        rows = normalise_rows(candidate)
        if rows:
            break

    normalised = []

    for index, row in enumerate(rows, start=1):
        normalised.append(
            {
                "rank": safe_int(
                    row.get(
                        "neighbour_rank",
                        row.get(
                            "rank",
                            index,
                        ),
                    ),
                    index,
                ),
                "wafer_id": safe_text(
                    row.get(
                        "historical_wafer_id",
                        row.get(
                            "wafer_id",
                            row.get(
                                "neighbour_id",
                                "Unavailable",
                            ),
                        ),
                    )
                ),
                "similarity_score": safe_float(
                    row.get(
                        "similarity_score",
                        row.get(
                            "similarity",
                        ),
                    )
                ),
                "historical_status": safe_text(
                    row.get(
                        "historical_status",
                        row.get(
                            "status",
                            row.get(
                                "outcome",
                                "Unavailable",
                            ),
                        ),
                    )
                ),
                "historical_failure": row.get(
                    "historical_failure",
                    row.get(
                        "actual_failure",
                    ),
                ),
            }
        )

    return sorted(
        normalised,
        key=lambda item: item["rank"],
    )


# ============================================================
# REPORTLAB STYLES
# ============================================================
class ReportTheme:
    NAVY_950 = colors.HexColor("#071426")
    NAVY_900 = colors.HexColor("#0B1F36")
    NAVY_800 = colors.HexColor("#123252")
    BLUE_600 = colors.HexColor("#1F6AA5")
    BLUE_100 = colors.HexColor("#E8F2F9")
    SLATE_900 = colors.HexColor("#172230")
    SLATE_700 = colors.HexColor("#405267")
    SLATE_500 = colors.HexColor("#708196")
    SLATE_300 = colors.HexColor("#C7D2DC")
    SLATE_200 = colors.HexColor("#DCE3E9")
    SLATE_100 = colors.HexColor("#EEF3F7")
    WHITE = colors.white
    GREEN = colors.HexColor("#1B7F5B")
    GREEN_BG = colors.HexColor("#E6F4ED")
    AMBER = colors.HexColor("#A96B00")
    AMBER_BG = colors.HexColor("#FFF3D9")
    RED = colors.HexColor("#B43B3B")
    RED_BG = colors.HexColor("#FDEAEA")
    PURPLE = colors.HexColor("#6F4F9C")
    PURPLE_BG = colors.HexColor("#F1EBF9")


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()

    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=24,
            leading=29,
            textColor=ReportTheme.WHITE,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=15,
            textColor=colors.HexColor("#D8E8F4"),
            alignment=TA_LEFT,
        ),
        "section_heading": ParagraphStyle(
            "SectionHeading",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=ReportTheme.NAVY_950,
            spaceBefore=7,
            spaceAfter=7,
        ),
        "subheading": ParagraphStyle(
            "Subheading",
            parent=sample["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=10.5,
            leading=13,
            textColor=ReportTheme.NAVY_900,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.8,
            leading=12.2,
            textColor=ReportTheme.SLATE_700,
            spaceAfter=5,
        ),
        "body_small": ParagraphStyle(
            "BodySmall",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=ReportTheme.SLATE_700,
        ),
        "metric_label": ParagraphStyle(
            "MetricLabel",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=6.5,
            leading=8,
            textColor=ReportTheme.SLATE_500,
            alignment=TA_LEFT,
        ),
        "metric_value": ParagraphStyle(
            "MetricValue",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=17,
            textColor=ReportTheme.NAVY_950,
            alignment=TA_LEFT,
        ),
        "metric_note": ParagraphStyle(
            "MetricNote",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=9,
            textColor=ReportTheme.SLATE_700,
            alignment=TA_LEFT,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7,
            leading=8.5,
            textColor=ReportTheme.WHITE,
            alignment=TA_LEFT,
        ),
        "table_body": ParagraphStyle(
            "TableBody",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.8,
            leading=8.5,
            textColor=ReportTheme.SLATE_700,
            alignment=TA_LEFT,
        ),
        "footer": ParagraphStyle(
            "Footer",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=6.5,
            leading=8,
            textColor=ReportTheme.SLATE_500,
            alignment=TA_CENTER,
        ),
        "callout": ParagraphStyle(
            "Callout",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.3,
            leading=11.5,
            textColor=ReportTheme.SLATE_700,
        ),
    }


# ============================================================
# PAGE TEMPLATE
# ============================================================
class NumberedReportTemplate(BaseDocTemplate):
    def __init__(
        self,
        filename: str,
        identity: ReportIdentity,
        context: ReportContext,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            filename,
            **kwargs,
        )

        self.identity = identity
        self.context = context

        page_frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )

        self.addPageTemplates(
            [
                PageTemplate(
                    id="report",
                    frames=[
                        page_frame
                    ],
                    onPage=self._draw_page,
                )
            ]
        )

    def _draw_page(
        self,
        canvas: Any,
        document: Any,
    ) -> None:
        canvas.saveState()

        page_width, page_height = A4

        if document.page == 1:
            canvas.setFillColor(
                ReportTheme.NAVY_950
            )

            canvas.rect(
                0,
                page_height - 76 * mm,
                page_width,
                76 * mm,
                fill=1,
                stroke=0,
            )

            canvas.setFillColor(
                ReportTheme.BLUE_600
            )

            canvas.rect(
                page_width - 58 * mm,
                page_height - 76 * mm,
                58 * mm,
                76 * mm,
                fill=1,
                stroke=0,
            )

        canvas.setStrokeColor(
            ReportTheme.SLATE_200
        )

        canvas.line(
            18 * mm,
            14 * mm,
            page_width - 18 * mm,
            14 * mm,
        )

        canvas.setFont(
            "Helvetica",
            6.5,
        )

        canvas.setFillColor(
            ReportTheme.SLATE_500
        )

        footer_left = (
            f"{self.identity.wafer_id} | "
            f"Report {self.identity.report_id[:8]} | "
            f"Version {self.identity.report_version}"
        )

        canvas.drawString(
            18 * mm,
            9 * mm,
            footer_left,
        )

        canvas.drawRightString(
            page_width - 18 * mm,
            9 * mm,
            f"Page {document.page}",
        )

        canvas.restoreState()


# ============================================================
# REPORT COMPONENTS
# ============================================================
def paragraph(
    text: Any,
    style: ParagraphStyle,
) -> Paragraph:
    return Paragraph(
        safe_text(text),
        style,
    )


def section_heading(
    title: str,
    styles: dict[str, ParagraphStyle],
) -> Paragraph:
    return paragraph(
        title,
        styles["section_heading"],
    )


def decision_colours(
    decision: str,
) -> tuple[Any, Any]:
    if decision == "Low Risk":
        return ReportTheme.GREEN, ReportTheme.GREEN_BG

    if decision == "Engineering Review":
        return ReportTheme.AMBER, ReportTheme.AMBER_BG

    if decision == "High Risk":
        return ReportTheme.RED, ReportTheme.RED_BG

    if decision == "Insufficient Evidence":
        return ReportTheme.PURPLE, ReportTheme.PURPLE_BG

    return ReportTheme.SLATE_700, ReportTheme.SLATE_100


def build_cover_block(
    identity: ReportIdentity,
    context: ReportContext,
    metrics: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> list[Any]:
    decision = metrics["decision"]
    decision_colour, decision_background = decision_colours(
        decision
    )

    cover_table = Table(
        [
            [
                paragraph(
                    "HeveMind Executive Engineering Report",
                    styles["cover_title"],
                ),
                "",
            ],
            [
                paragraph(
                    (
                        f"{context.organisation_label}<br/>"
                        f"{context.site_label}<br/>"
                        "Validated wafer-level decision-support output"
                    ),
                    styles["cover_subtitle"],
                ),
                "",
            ],
        ],
        colWidths=[
            130 * mm,
            35 * mm,
        ],
    )

    cover_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    ReportTheme.NAVY_950,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, 0),
                    14,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 1),
                    (-1, 1),
                    15,
                ),
                (
                    "SPAN",
                    (0, 0),
                    (1, 0),
                ),
                (
                    "SPAN",
                    (0, 1),
                    (1, 1),
                ),
            ]
        )
    )

    identity_table = Table(
        [
            [
                paragraph(
                    "WAFER IDENTIFIER",
                    styles["metric_label"],
                ),
                paragraph(
                    "OPERATIONAL DECISION",
                    styles["metric_label"],
                ),
                paragraph(
                    "GENERATED",
                    styles["metric_label"],
                ),
            ],
            [
                paragraph(
                    identity.wafer_id,
                    styles["metric_value"],
                ),
                paragraph(
                    decision,
                    styles["metric_value"],
                ),
                paragraph(
                    datetime.fromisoformat(
                        identity.generated_utc
                    )
                    .astimezone()
                    .strftime(
                        "%d %b %Y %I:%M %p"
                    ),
                    styles["metric_value"],
                ),
            ],
        ],
        colWidths=[
            56 * mm,
            58 * mm,
            51 * mm,
        ],
    )

    identity_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    ReportTheme.WHITE,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    ReportTheme.SLATE_200,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    ReportTheme.SLATE_200,
                ),
                (
                    "BACKGROUND",
                    (1, 0),
                    (1, 1),
                    decision_background,
                ),
                (
                    "TEXTCOLOR",
                    (1, 0),
                    (1, 1),
                    decision_colour,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "MIDDLE",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
            ]
        )
    )

    return [
        cover_table,
        Spacer(
            1,
            8 * mm,
        ),
        identity_table,
        Spacer(
            1,
            7 * mm,
        ),
    ]


def build_metric_cards(
    metrics: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> Table:
    cards = [
        (
            "Failure probability",
            format_percentage(
                metrics[
                    "failure_probability"
                ],
                1,
            ),
            "Beta-calibrated primary estimate.",
        ),
        (
            "Prediction confidence",
            format_percentage(
                metrics[
                    "prediction_confidence"
                ],
                1,
            ),
            "Uncertainty-derived record confidence.",
        ),
        (
            "Combined uncertainty",
            format_percentage(
                metrics[
                    "combined_uncertainty"
                ],
                1,
            ),
            "Higher values indicate lower certainty.",
        ),
        (
            "Data familiarity",
            metrics[
                "data_familiarity"
            ],
            "OOD-derived familiarity interpretation.",
        ),
        (
            "Priority sensor",
            metrics[
                "priority_sensor"
            ],
            "First measurement recommended for review.",
        ),
    ]

    cells = []

    for label, value, note in cards:
        cell = Table(
            [
                [
                    paragraph(
                        label.upper(),
                        styles[
                            "metric_label"
                        ],
                    )
                ],
                [
                    paragraph(
                        value,
                        styles[
                            "metric_value"
                        ],
                    )
                ],
                [
                    paragraph(
                        note,
                        styles[
                            "metric_note"
                        ],
                    )
                ],
            ],
            colWidths=[
                31 * mm
            ],
        )

        cell.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        ReportTheme.WHITE,
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.6,
                        ReportTheme.SLATE_200,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        6,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                ]
            )
        )

        cells.append(
            cell
        )

    table = Table(
        [
            cells
        ],
        colWidths=[
            33 * mm
        ]
        * len(
            cells
        ),
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    return table


def build_decision_callout(
    metrics: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> Table:
    decision = metrics[
        "decision"
    ]

    decision_colour, decision_background = (
        decision_colours(
            decision
        )
    )

    action = metrics[
        "recommended_action"
    ]

    if action == "Unavailable":
        action = RISK_TEXT.get(
            decision,
            "Engineering review is required.",
        )

    content = [
        [
            paragraph(
                decision,
                styles[
                    "subheading"
                ],
            )
        ],
        [
            paragraph(
                action,
                styles[
                    "callout"
                ],
            )
        ],
        [
            paragraph(
                (
                    f"Evidence status: {metrics['evidence_status']} | "
                    f"Abstention: {metrics['abstention_reason']}"
                ),
                styles[
                    "body_small"
                ],
            )
        ],
    ]

    table = Table(
        content,
        colWidths=[
            165 * mm
        ],
    )

    table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    decision_background,
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    decision_colour,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.9,
                    decision_colour,
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    10,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    return table


def make_simple_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[Any]],
    styles: dict[str, ParagraphStyle],
    col_widths: Sequence[float],
) -> Table:
    data: list[list[Any]] = [
        [
            paragraph(
                header,
                styles[
                    "table_header"
                ],
            )
            for header in headers
        ]
    ]

    for row in rows:
        data.append(
            [
                paragraph(
                    value,
                    styles[
                        "table_body"
                    ],
                )
                for value in row
            ]
        )

    table = Table(
        data,
        colWidths=list(
            col_widths
        ),
        repeatRows=1,
    )

    style_commands = [
        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            ReportTheme.NAVY_800,
        ),
        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            ReportTheme.WHITE,
        ),
        (
            "GRID",
            (0, 0),
            (-1, -1),
            0.45,
            ReportTheme.SLATE_200,
        ),
        (
            "VALIGN",
            (0, 0),
            (-1, -1),
            "TOP",
        ),
        (
            "LEFTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "RIGHTPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            5,
        ),
    ]

    for row_index in range(
        1,
        len(data),
    ):
        if row_index % 2 == 0:
            style_commands.append(
                (
                    "BACKGROUND",
                    (0, row_index),
                    (-1, row_index),
                    ReportTheme.SLATE_100,
                )
            )

    table.setStyle(
        TableStyle(
            style_commands
        )
    )

    return table


def build_sensor_priority_table(
    priorities: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
) -> Any:
    if not priorities:
        return paragraph(
            "No sensor-priority table was available in the validated report.",
            styles["body"],
        )

    rows = []

    for item in priorities[
        :10
    ]:
        rows.append(
            [
                str(
                    item[
                        "rank"
                    ]
                ),
                item[
                    "sensor"
                ],
                item[
                    "priority_level"
                ],
                (
                    format_percentage(
                        item[
                            "priority_score"
                        ],
                        1,
                    )
                    if item[
                        "priority_score"
                    ]
                    is not None
                    else "Unavailable"
                ),
                item[
                    "measurement_status"
                ],
                item[
                    "explanation"
                ],
            ]
        )

    return make_simple_table(
        headers=[
            "Rank",
            "Sensor",
            "Priority",
            "Score",
            "Measurement status",
            "Engineering explanation",
        ],
        rows=rows,
        styles=styles,
        col_widths=[
            10 * mm,
            24 * mm,
            22 * mm,
            17 * mm,
            30 * mm,
            62 * mm,
        ],
    )


def build_historical_table(
    cases: list[dict[str, Any]],
    styles: dict[str, ParagraphStyle],
) -> Any:
    if not cases:
        return paragraph(
            "No historical-neighbour records were available in the validated report.",
            styles["body"],
        )

    rows = []

    for item in cases[
        :10
    ]:
        historical_failure = item[
            "historical_failure"
        ]

        if isinstance(
            historical_failure,
            bool,
        ):
            failure_label = (
                "Failure"
                if historical_failure
                else "Pass"
            )
        else:
            failure_label = safe_text(
                historical_failure
            )

        rows.append(
            [
                str(
                    item[
                        "rank"
                    ]
                ),
                item[
                    "wafer_id"
                ],
                (
                    format_percentage(
                        item[
                            "similarity_score"
                        ],
                        1,
                    )
                    if item[
                        "similarity_score"
                    ]
                    is not None
                    else "Unavailable"
                ),
                item[
                    "historical_status"
                ],
                failure_label,
            ]
        )

    return make_simple_table(
        headers=[
            "Rank",
            "Historical wafer",
            "Similarity",
            "Status",
            "Observed outcome",
        ],
        rows=rows,
        styles=styles,
        col_widths=[
            12 * mm,
            45 * mm,
            30 * mm,
            42 * mm,
            36 * mm,
        ],
    )


def build_attribution_chart(
    attributions: list[dict[str, Any]],
) -> Drawing | None:
    usable = [
        item
        for item in attributions
        if item[
            "contribution"
        ]
        is not None
    ][
        :10
    ]

    if not usable:
        return None

    labels = [
        item[
            "sensor"
        ]
        for item in usable
    ]

    values = [
        abs(
            float(
                item[
                    "contribution"
                ]
            )
        )
        for item in usable
    ]

    drawing = Drawing(
        480,
        235,
    )

    chart = HorizontalBarChart()

    chart.x = 115
    chart.y = 28
    chart.height = 175
    chart.width = 325
    chart.data = [
        values
    ]
    chart.categoryAxis.categoryNames = labels
    chart.categoryAxis.labels.fontName = "Helvetica"
    chart.categoryAxis.labels.fontSize = 7
    chart.categoryAxis.labels.fillColor = ReportTheme.SLATE_700
    chart.categoryAxis.labels.dx = -4
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max(values) * 1.15
    chart.valueAxis.labels.fontName = "Helvetica"
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.labels.fillColor = ReportTheme.SLATE_500
    chart.bars[0].fillColor = ReportTheme.BLUE_600
    chart.bars[0].strokeColor = ReportTheme.BLUE_600
    chart.barSpacing = 3

    drawing.add(
        chart
    )

    drawing.add(
        String(
            115,
            215,
            "Absolute local contribution",
            fontName="Helvetica-Bold",
            fontSize=9,
            fillColor=ReportTheme.NAVY_950,
        )
    )

    return drawing


def build_qr_image(
    identity: ReportIdentity,
    context: ReportContext,
    report_sha256_placeholder: str,
) -> Image | None:
    if qrcode is None:
        return None

    payload = {
        "report_id": identity.report_id,
        "wafer_id": identity.wafer_id,
        "generated_utc": identity.generated_utc,
        "report_version": identity.report_version,
        "source_scope": context.source_scope,
        "sha256": report_sha256_placeholder,
    }

    qr = qrcode.QRCode(
        version=3,
        box_size=5,
        border=2,
    )

    qr.add_data(
        json.dumps(
            payload,
            separators=(
                ",",
                ":",
            ),
        )
    )

    qr.make(
        fit=True
    )

    image = qr.make_image(
        fill_color="black",
        back_color="white",
    )

    buffer = io.BytesIO()
    image.save(
        buffer,
        format="PNG",
    )
    buffer.seek(0)

    return Image(
        buffer,
        width=28 * mm,
        height=28 * mm,
    )


def build_methodology_note(
    styles: dict[str, ParagraphStyle],
) -> Table:
    stages = [
        "Primary prediction",
        "Probability calibration",
        "Uncertainty assessment",
        "Historical evidence",
        "Sensor prioritisation",
        "Engineering recommendation",
    ]

    cells = []

    for index, stage in enumerate(
        stages,
        start=1,
    ):
        cell = Table(
            [
                [
                    paragraph(
                        f"Stage {index:02d}",
                        styles[
                            "metric_label"
                        ],
                    )
                ],
                [
                    paragraph(
                        stage,
                        styles[
                            "table_body"
                        ],
                    )
                ],
            ],
            colWidths=[
                26 * mm
            ],
        )

        cell.setStyle(
            TableStyle(
                [
                    (
                        "BACKGROUND",
                        (0, 0),
                        (-1, -1),
                        ReportTheme.WHITE,
                    ),
                    (
                        "BOX",
                        (0, 0),
                        (-1, -1),
                        0.5,
                        ReportTheme.SLATE_200,
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        5,
                    ),
                    (
                        "TOPPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                    (
                        "BOTTOMPADDING",
                        (0, 0),
                        (-1, -1),
                        4,
                    ),
                ]
            )
        )

        cells.append(
            cell
        )

    workflow = Table(
        [
            cells[:3],
            cells[3:],
        ],
        colWidths=[
            55 * mm,
            55 * mm,
            55 * mm,
        ],
    )

    workflow.setStyle(
        TableStyle(
            [
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    2,
                ),
            ]
        )
    )

    return workflow


# ============================================================
# PDF GENERATION
# ============================================================
def generate_pdf_report(
    report: dict[str, Any],
    *,
    output_path: Path | None = None,
    organisation_label: str = DEFAULT_ORGANISATION_LABEL,
    site_label: str = DEFAULT_SITE_LABEL,
    api_base_url: str = DEFAULT_API_BASE_URL,
    source_scope: str = "Validated held-out report",
    disclaimer: str = DEFAULT_DISCLAIMER,
    save_source_json: bool = True,
    user_email: str | None = None,
    display_name: str | None = None,
    role: str | None = None,
    auth_source: str | None = None,
    session_id: str | None = None,
) -> ReportArtifacts:
    ensure_output_directory()

    identity = extract_identity(
        report
    )

    context = ReportContext(
        organisation_label=organisation_label,
        site_label=site_label,
        api_base_url=api_base_url,
        source_scope=source_scope,
        disclaimer=disclaimer,
    )

    metrics = extract_primary_metrics(
        report
    )

    priorities = extract_sensor_priorities(
        report
    )

    attributions = extract_sensor_attributions(
        report
    )

    historical_cases = extract_historical_cases(
        report
    )

    if output_path is None:
        timestamp = utc_now().strftime(
            "%Y%m%d_%H%M%S"
        )

        filename = (
            slugify(
                identity.wafer_id
            )
            + "_hevemind_engineering_report_"
            + timestamp
            + ".pdf"
        )

        output_path = (
            EXPORT_DIR
            / filename
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    styles = build_styles()

    document = NumberedReportTemplate(
        str(
            output_path
        ),
        identity=identity,
        context=context,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title=(
            f"HeveMind Executive Engineering Report - "
            f"{identity.wafer_id}"
        ),
        author=identity.generated_by,
        subject=(
            "Validated semiconductor wafer decision-support report"
        ),
        creator=(
            "HeveMind Executive PDF Report Generator"
        ),
    )

    story: list[Any] = []

    story.extend(
        build_cover_block(
            identity=identity,
            context=context,
            metrics=metrics,
            styles=styles,
        )
    )

    story.append(
        section_heading(
            "Executive Summary",
            styles,
        )
    )

    story.append(
        build_decision_callout(
            metrics,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        build_metric_cards(
            metrics,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    summary_text = (
        f"The retained operational decision for wafer "
        f"<b>{identity.wafer_id}</b> is <b>{metrics['decision']}</b>. "
        f"The calibrated failure probability is "
        f"<b>{format_percentage(metrics['failure_probability'], 2)}</b>, "
        f"with prediction confidence of "
        f"<b>{format_percentage(metrics['prediction_confidence'], 1)}</b> "
        f"and combined uncertainty of "
        f"<b>{format_percentage(metrics['combined_uncertainty'], 1)}</b>. "
        f"The record is classified as "
        f"<b>{metrics['data_familiarity']}</b> relative to the development "
        f"data, and the first recommended sensor for statistical review is "
        f"<b>{metrics['priority_sensor']}</b>."
    )

    story.append(
        paragraph(
            summary_text,
            styles["body"],
        )
    )

    story.append(
        Spacer(
            1,
            2 * mm,
        )
    )

    story.append(
        section_heading(
            "Decision Pathway",
            styles,
        )
    )

    story.append(
        build_methodology_note(
            styles
        )
    )

    story.append(
        PageBreak()
    )

    story.append(
        section_heading(
            "Reliability and Evidence Context",
            styles,
        )
    )

    reliability_rows = [
        [
            "Operational decision",
            metrics[
                "decision"
            ],
        ],
        [
            "Evidence status",
            metrics[
                "evidence_status"
            ],
        ],
        [
            "Data familiarity",
            metrics[
                "data_familiarity"
            ],
        ],
        [
            "Data confidence",
            format_percentage(
                metrics[
                    "data_confidence"
                ],
                1,
            ),
        ],
        [
            "Missing sensor rate",
            format_percentage(
                metrics[
                    "missing_sensor_rate"
                ],
                1,
            ),
        ],
        [
            "Abstention reason",
            metrics[
                "abstention_reason"
            ],
        ],
    ]

    story.append(
        make_simple_table(
            headers=[
                "Reliability measure",
                "Validated report value",
            ],
            rows=reliability_rows,
            styles=styles,
            col_widths=[
                55 * mm,
                110 * mm,
            ],
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        section_heading(
            "Sensor Investigation Priorities",
            styles,
        )
    )

    story.append(
        build_sensor_priority_table(
            priorities,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    story.append(
        section_heading(
            "Local Sensor Attribution",
            styles,
        )
    )

    attribution_chart = build_attribution_chart(
        attributions
    )

    if attribution_chart is None:
        story.append(
            paragraph(
                (
                    "No local attribution chart was available in the "
                    "validated report. This does not invalidate the retained "
                    "decision; it indicates that the report payload did not "
                    "contain usable local contribution values."
                ),
                styles["body"],
            )
        )
    else:
        story.append(
            attribution_chart
        )

        attribution_rows = []

        for item in attributions[
            :10
        ]:
            attribution_rows.append(
                [
                    item[
                        "sensor"
                    ],
                    format_decimal(
                        item[
                            "contribution"
                        ],
                        5,
                    ),
                    item[
                        "direction"
                    ],
                    item[
                        "measurement_status"
                    ],
                ]
            )

        story.append(
            make_simple_table(
                headers=[
                    "Sensor",
                    "Contribution",
                    "Direction",
                    "Measurement status",
                ],
                rows=attribution_rows,
                styles=styles,
                col_widths=[
                    35 * mm,
                    30 * mm,
                    60 * mm,
                    40 * mm,
                ],
            )
        )

    story.append(
        PageBreak()
    )

    story.append(
        section_heading(
            "Historical Case-Based Evidence",
            styles,
        )
    )

    story.append(
        paragraph(
            (
                "Historical similarity is supporting statistical evidence. "
                "It does not establish a physical root cause and does not "
                "override the retained calibrated decision."
            ),
            styles["body"],
        )
    )

    story.append(
        build_historical_table(
            historical_cases,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    story.append(
        section_heading(
            "Engineering Recommendation",
            styles,
        )
    )

    story.append(
        build_decision_callout(
            metrics,
            styles,
        )
    )

    story.append(
        Spacer(
            1,
            5 * mm,
        )
    )

    technical_metadata = [
        [
            "Report identifier",
            identity[
                "report_id"
            ]
            if isinstance(identity, dict)
            else identity.report_id,
        ],
        [
            "Wafer identifier",
            identity.wafer_id,
        ],
        [
            "Generated UTC",
            identity.generated_utc,
        ],
        [
            "Generated by",
            identity.generated_by,
        ],
        [
            "Report version",
            identity.report_version,
        ],
        [
            "Source scope",
            context.source_scope,
        ],
        [
            "API base URL",
            context.api_base_url,
        ],
    ]

    qr_image = build_qr_image(
        identity=identity,
        context=context,
        report_sha256_placeholder=(
            "Calculated after generation"
        ),
    )

    metadata_table = make_simple_table(
        headers=[
            "Technical metadata",
            "Value",
        ],
        rows=technical_metadata,
        styles=styles,
        col_widths=[
            52 * mm,
            88 * mm,
        ],
    )

    if qr_image is not None:
        metadata_layout = Table(
            [
                [
                    metadata_table,
                    qr_image,
                ]
            ],
            colWidths=[
                140 * mm,
                28 * mm,
            ],
        )

        metadata_layout.setStyle(
            TableStyle(
                [
                    (
                        "VALIGN",
                        (0, 0),
                        (-1, -1),
                        "TOP",
                    ),
                    (
                        "LEFTPADDING",
                        (0, 0),
                        (-1, -1),
                        0,
                    ),
                    (
                        "RIGHTPADDING",
                        (0, 0),
                        (-1, -1),
                        0,
                    ),
                ]
            )
        )

        story.append(
            metadata_layout
        )
    else:
        story.append(
            metadata_table
        )

    story.append(
        Spacer(
            1,
            6 * mm,
        )
    )

    story.append(
        section_heading(
            "Interpretation Limits",
            styles,
        )
    )

    story.append(
        paragraph(
            context.disclaimer,
            styles["body"],
        )
    )

    story.append(
        paragraph(
            (
                "SHAP and local sensor attribution describe statistical "
                "influence on the fitted model. Historical similarity describes "
                "proximity to development records. Neither component independently "
                "proves a physical failure mechanism."
            ),
            styles["body"],
        )
    )

    story.append(
        Spacer(
            1,
            8 * mm,
        )
    )

    signature_table = Table(
        [
            [
                paragraph(
                    "ENGINEERING REVIEWER",
                    styles[
                        "metric_label"
                    ],
                ),
                paragraph(
                    "REVIEW DATE",
                    styles[
                        "metric_label"
                    ],
                ),
                paragraph(
                    "DISPOSITION REFERENCE",
                    styles[
                        "metric_label"
                    ],
                ),
            ],
            [
                " ",
                " ",
                " ",
            ],
        ],
        colWidths=[
            70 * mm,
            45 * mm,
            50 * mm,
        ],
        rowHeights=[
            8 * mm,
            18 * mm,
        ],
    )

    signature_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    ReportTheme.SLATE_100,
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.6,
                    ReportTheme.SLATE_300,
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    ReportTheme.SLATE_300,
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5,
                ),
            ]
        )
    )

    story.append(
        signature_table
    )

    document.build(
        story
    )

    report_sha256 = calculate_sha256(
        output_path
    )

    output_json: Path | None = None

    if save_source_json:
        output_json = output_path.with_suffix(
            ".json"
        )

        output_json.write_text(
            json.dumps(
                {
                    "report_identity": json_safe(
                        identity.__dict__
                    ),
                    "report_context": json_safe(
                        context.__dict__
                    ),
                    "report_sha256": report_sha256,
                    "source_report": json_safe(
                        report
                    ),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    audit_report_generation(
        wafer_id=identity.wafer_id,
        output_pdf=output_path,
        report_sha256=report_sha256,
        user_email=user_email,
        display_name=display_name,
        role=role,
        auth_source=auth_source,
        session_id=session_id,
    )

    return ReportArtifacts(
        output_pdf=output_path,
        output_json=output_json,
        sha256=report_sha256,
        pages_estimated=max(
            3,
            2
            + int(
                bool(
                    priorities
                )
            )
            + int(
                bool(
                    historical_cases
                )
            ),
        ),
    )


# ============================================================
# BATCH GENERATION
# ============================================================
def generate_batch(
    wafer_ids: Iterable[str],
    *,
    api_base_url: str,
    output_directory: Path,
) -> list[ReportArtifacts]:
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for wafer_id in wafer_ids:
        report = fetch_wafer_report(
            wafer_id=wafer_id,
            api_base_url=api_base_url,
        )

        output_path = (
            output_directory
            / (
                slugify(
                    wafer_id
                )
                + "_hevemind_engineering_report.pdf"
            )
        )

        artifacts = generate_pdf_report(
            report,
            output_path=output_path,
            api_base_url=api_base_url,
        )

        results.append(
            artifacts
        )

    return results


# ============================================================
# STREAMLIT COMPONENT
# ============================================================
def render_streamlit_report_generator() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="HeveMind PDF Report Generator",
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
            }

            .stApp {
                background:
                    linear-gradient(
                        180deg,
                        #f5f8fb 0%,
                        #edf3f7 100%
                    );
            }

            .report-header {
                background:
                    linear-gradient(
                        110deg,
                        var(--navy-950) 0%,
                        var(--navy-800) 72%,
                        var(--blue-600) 100%
                    );
                color: #ffffff;
                border-radius: 18px;
                padding: 1.2rem 1.4rem;
                box-shadow:
                    0 14px 34px
                    rgba(7, 20, 38, 0.17);
                margin-bottom: 1rem;
            }

            .report-title {
                font-size: 1.75rem;
                font-weight: 780;
            }

            .report-subtitle {
                color: #d8e8f4;
                font-size: 0.9rem;
                margin-top: 0.35rem;
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
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="report-header">
            <div class="report-title">
                HeveMind Executive PDF Report Generator
            </div>
            <div class="report-subtitle">
                Generate a structured engineering report from a validated
                wafer record served by the deployment API.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    api_base_url = st.sidebar.text_input(
        "API base URL",
        value=DEFAULT_API_BASE_URL,
    )

    user = st.session_state.get(
        "hevemind_user",
        {},
    )

    wafer_id = st.text_input(
        "Validated wafer identifier",
        placeholder="WF-000731",
    )

    save_source_json = st.checkbox(
        "Save source JSON beside the PDF",
        value=True,
    )

    if st.button(
        "Generate engineering PDF",
        use_container_width=True,
    ):
        if not wafer_id.strip():
            st.error(
                "Enter a wafer identifier."
            )
            return

        with st.spinner(
            "Retrieving the validated report and generating the PDF..."
        ):
            report = fetch_wafer_report(
                wafer_id.strip(),
                api_base_url=api_base_url,
            )

            artifacts = generate_pdf_report(
                report,
                api_base_url=api_base_url,
                save_source_json=save_source_json,
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
                session_id=st.session_state.get(
                    "hevemind_audit_session_id"
                ),
            )

        st.success(
            f"Report generated: {artifacts.output_pdf}"
        )

        pdf_bytes = artifacts.output_pdf.read_bytes()

        st.download_button(
            "Download engineering PDF",
            data=pdf_bytes,
            file_name=artifacts.output_pdf.name,
            mime="application/pdf",
            use_container_width=True,
        )

        st.code(
            f"SHA-256: {artifacts.sha256}",
            language="text",
        )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate HeveMind executive engineering PDF reports "
            "from validated wafer records."
        )
    )

    source_group = parser.add_mutually_exclusive_group(
        required=False
    )

    source_group.add_argument(
        "--wafer-id",
        type=str,
        help=(
            "Validated wafer identifier to retrieve from the API."
        ),
    )

    source_group.add_argument(
        "--input-json",
        type=Path,
        help=(
            "Local validated report JSON file."
        ),
    )

    parser.add_argument(
        "--api-base-url",
        type=str,
        default=DEFAULT_API_BASE_URL,
        help=(
            "HeveMind FastAPI base URL."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        help=(
            "Output PDF path."
        ),
    )

    parser.add_argument(
        "--organisation",
        type=str,
        default=DEFAULT_ORGANISATION_LABEL,
    )

    parser.add_argument(
        "--site",
        type=str,
        default=DEFAULT_SITE_LABEL,
    )

    parser.add_argument(
        "--no-source-json",
        action="store_true",
        help=(
            "Do not save the source JSON beside the PDF."
        ),
    )

    parser.add_argument(
        "--batch-file",
        type=Path,
        help=(
            "Text file containing one wafer identifier per line."
        ),
    )

    parser.add_argument(
        "--batch-output-dir",
        type=Path,
        default=EXPORT_DIR,
    )

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    if arguments.batch_file:
        wafer_ids = [
            line.strip()
            for line in arguments.batch_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        results = generate_batch(
            wafer_ids,
            api_base_url=arguments.api_base_url,
            output_directory=arguments.batch_output_dir,
        )

        print(
            json.dumps(
                [
                    {
                        "output_pdf": str(
                            item.output_pdf
                        ),
                        "output_json": (
                            str(
                                item.output_json
                            )
                            if item.output_json
                            else None
                        ),
                        "sha256": item.sha256,
                    }
                    for item in results
                ],
                indent=2,
            )
        )

        return

    if arguments.input_json:
        report = load_report_from_json(
            arguments.input_json
        )

        source_scope = (
            "Validated report loaded from local JSON"
        )

    elif arguments.wafer_id:
        report = fetch_wafer_report(
            arguments.wafer_id,
            api_base_url=arguments.api_base_url,
        )

        source_scope = (
            "Validated report retrieved from deployment API"
        )

    else:
        print(
            "\n"
            + "=" * 110
        )

        print(
            "HEVEMIND EXECUTIVE ENGINEERING PDF REPORT GENERATOR"
        )

        print(
            "=" * 110
        )

        print(
            f"\nDefault API:             {DEFAULT_API_BASE_URL}"
        )

        print(
            f"Output directory:        {EXPORT_DIR}"
        )

        print(
            "\nUse --wafer-id or --input-json to generate a report."
        )

        return

    artifacts = generate_pdf_report(
        report,
        output_path=arguments.output,
        organisation_label=arguments.organisation,
        site_label=arguments.site,
        api_base_url=arguments.api_base_url,
        source_scope=source_scope,
        save_source_json=not arguments.no_source_json,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "HEVEMIND EXECUTIVE ENGINEERING PDF REPORT GENERATED"
    )

    print(
        "=" * 110
    )

    print(
        f"\nPDF:                     {artifacts.output_pdf}"
    )

    print(
        f"Source JSON:             {artifacts.output_json}"
    )

    print(
        f"SHA-256:                 {artifacts.sha256}"
    )


if __name__ == "__main__":
    main()
