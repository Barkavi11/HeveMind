from __future__ import annotations

import importlib.util
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
import numpy as np
import importlib.util
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import uvicorn
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Query,
    Request,
    status,
)
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=os.getenv(
        "HEVEMIND_LOG_LEVEL",
        "INFO",
    ).upper(),
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(
    "hevemind_api"
)


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(
    __file__
).resolve().parents[1]

SRC_DIR = ROOT_DIR / "src"

BACKEND_MODULE_PATH = (
    SRC_DIR
    / "18_validated_report_backend.py"
)

REPORTS_DIR = ROOT_DIR / "reports"

BACKEND_INDEX_PATH = (
    REPORTS_DIR
    / "deployment_backend"
    / "wafer_report_index.csv"
)


# ============================================================
# API CONFIGURATION
# ============================================================
API_TITLE = (
    "HeveMind Semiconductor Decision-Support API"
)

API_DESCRIPTION = """
HeveMind exposes validated held-out wafer reports produced by the
semiconductor decision-support pipeline.

The API serves:

- Beta-calibrated failure probabilities
- Four-level operational decisions
- Confidence and uncertainty
- Data familiarity and OOD evidence
- SHAP-based statistical attribution
- Historical nearest-neighbour evidence
- Sensor investigation priorities
- Engineer-facing recommendations

Important limitation: this service does not yet accept a new raw sensor
record for production inference. It serves validated reports generated
by the existing HeveMind analytical pipeline.
""".strip()

API_VERSION = "1.0.0"

DEFAULT_HOST = os.getenv(
    "HEVEMIND_API_HOST",
    "127.0.0.1",
)

DEFAULT_PORT = int(
    os.getenv(
        "HEVEMIND_API_PORT",
        "8000",
    )
)

API_KEY = os.getenv(
    "HEVEMIND_API_KEY",
    "",
).strip()

ALLOWED_ORIGINS_RAW = os.getenv(
    "HEVEMIND_ALLOWED_ORIGINS",
    "http://localhost:8501,"
    "http://127.0.0.1:8501,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000",
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in ALLOWED_ORIGINS_RAW.split(",")
    if origin.strip()
]

MAX_PAGE_SIZE = 100
DEFAULT_PAGE_SIZE = 25


# ============================================================
# ENUM-LIKE CONSTANTS
# ============================================================
VALID_DECISIONS = {
    "Low Risk",
    "Engineering Review",
    "High Risk",
    "Insufficient Evidence",
}

RISK_PRIORITY = {
    "Insufficient Evidence": 0,
    "High Risk": 1,
    "Engineering Review": 2,
    "Low Risk": 3,
}


# ============================================================
# RESPONSE MODELS
# ============================================================
class APIModel(BaseModel):
    model_config = ConfigDict(
        extra="allow",
    )


class ErrorDetail(APIModel):
    code: str
    message: str
    request_id: str | None = None
    details: Any | None = None


class ErrorResponse(APIModel):
    error: ErrorDetail


class HealthResponse(APIModel):
    status: Literal[
        "healthy",
        "degraded",
    ]
    service: str
    version: str
    timestamp_utc: str
    report_store_loaded: bool
    available_wafer_reports: int
    raw_record_inference_supported: bool


class ReadinessCheck(APIModel):
    name: str
    status: Literal[
        "pass",
        "fail",
    ]
    details: str


class ReadinessResponse(APIModel):
    ready: bool
    checks: list[ReadinessCheck]
    timestamp_utc: str


class WaferIndexItem(APIModel):
    wafer_id: str
    calibrated_failure_probability: float | None = None
    uncertainty_adjusted_decision: str | None = None
    prediction_confidence: float | None = None
    combined_uncertainty: float | None = None
    data_familiarity_band: str | None = None
    evidence_agreement: str | None = None
    priority_1_sensor: str | None = None
    recommended_action: str | None = None


class PaginationMeta(APIModel):
    page: int
    page_size: int
    total_records: int
    total_pages: int
    returned_records: int


class WaferListResponse(APIModel):
    items: list[WaferIndexItem]
    pagination: PaginationMeta
    filters: dict[str, Any]


class WaferReportResponse(APIModel):
    project: str
    report_type: str
    wafer_id: str
    primary_prediction: dict[str, Any]
    data_reliability: dict[str, Any]
    evidence_assessment: dict[str, Any]
    historical_evidence: dict[str, Any]
    explainability: dict[str, Any]
    sensor_investigation: dict[str, Any]
    methodological_limits: dict[str, Any]


class ExplanationResponse(APIModel):
    wafer_id: str
    decision: str | None = None
    failure_probability: float | None = None
    top_risk_features: Any | None = None
    top_protective_features: Any | None = None
    explanation_text: str | None = None
    local_shap_contributions: list[dict[str, Any]]
    interpretation_limit: str


class HistoricalEvidenceResponse(APIModel):
    wafer_id: str
    failed_neighbour_count: int | None = None
    passed_neighbour_count: int | None = None
    historical_weighted_failure_rate: float | None = None
    mean_similarity: float | None = None
    maximum_similarity: float | None = None
    historical_evidence_level: str | None = None
    nearest_historical_records: list[dict[str, Any]]
    interpretation_limit: str


class InvestigationResponse(APIModel):
    wafer_id: str
    operational_decision: str | None = None
    priority_1_sensor: str | None = None
    top_5_sensors: Any | None = None
    inspection_summary: str | None = None
    ranked_sensor_priorities: list[dict[str, Any]]
    interpretation_limit: str


class DecisionSummaryItem(APIModel):
    decision: str
    records: int
    record_rate: float
    mean_failure_probability: float | None = None
    mean_prediction_confidence: float | None = None
    mean_combined_uncertainty: float | None = None


class DecisionSummaryResponse(APIModel):
    total_records: int
    decisions: list[DecisionSummaryItem]


class MetadataResponse(APIModel):
    project: str
    api_version: str
    available_reports: int
    operational_decisions: list[str]
    architecture: dict[str, Any]
    methodological_limits: list[str]


# ============================================================
# APPLICATION STATE
# ============================================================
class ApplicationState:
    report_store: Any | None = None
    wafer_index: pd.DataFrame | None = None
    backend_module: Any | None = None
    startup_error: str | None = None


STATE = ApplicationState()


# ============================================================
# BACKEND MODULE LOADING
# ============================================================
def load_backend_module() -> Any:
    if not BACKEND_MODULE_PATH.exists():
        raise FileNotFoundError(
            "Validated report backend was not found: "
            f"{BACKEND_MODULE_PATH}"
        )

    module_name = (
        "hevemind_validated_report_backend"
    )

    specification = (
        importlib.util.spec_from_file_location(
            module_name,
            BACKEND_MODULE_PATH,
        )
    )

    if (
        specification is None
        or specification.loader is None
    ):
        raise ImportError(
            "Unable to build the import specification for "
            f"{BACKEND_MODULE_PATH}"
        )

    module = (
        importlib.util.module_from_spec(
            specification
        )
    )

    specification.loader.exec_module(
        module
    )

    if not hasattr(
        module,
        "HeveMindReportStore",
    ):
        raise AttributeError(
            "Script 18 does not expose "
            "HeveMindReportStore."
        )

    return module


def initialise_report_store() -> None:
    LOGGER.info(
        "Initialising HeveMind validated report store"
    )

    backend_module = load_backend_module()

    report_store = (
        backend_module.HeveMindReportStore()
    )

    wafer_index = report_store.build_index()

    if wafer_index.empty:
        raise ValueError(
            "The validated report index is empty."
        )

    if "wafer_id" not in wafer_index.columns:
        raise ValueError(
            "The report index does not contain wafer_id."
        )

    wafer_index[
        "wafer_id"
    ] = wafer_index[
        "wafer_id"
    ].astype(str)

    STATE.backend_module = backend_module
    STATE.report_store = report_store
    STATE.wafer_index = wafer_index
    STATE.startup_error = None

    LOGGER.info(
        "HeveMind API loaded %s validated wafer reports",
        len(
            wafer_index
        ),
    )


# ============================================================
# APPLICATION LIFESPAN
# ============================================================
@asynccontextmanager
async def lifespan(
    application: FastAPI,
):
    try:
        initialise_report_store()

    except Exception as error:
        STATE.startup_error = str(
            error
        )

        LOGGER.exception(
            "HeveMind API startup failed: %s",
            error,
        )

    yield

    LOGGER.info(
        "HeveMind API shutdown complete"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    GZipMiddleware,
    minimum_size=1000,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=(
        ALLOWED_ORIGINS
        if ALLOWED_ORIGINS
        else []
    ),
    allow_credentials=True,
    allow_methods=[
        "GET",
        "OPTIONS",
    ],
    allow_headers=[
        "*",
    ],
)


# ============================================================
# MIDDLEWARE
# ============================================================
@app.middleware(
    "http"
)
async def request_context_middleware(
    request: Request,
    call_next,
):
    request_id = (
        request.headers.get(
            "X-Request-ID"
        )
        or str(
            uuid.uuid4()
        )
    )

    request.state.request_id = (
        request_id
    )

    start_time = time.perf_counter()

    response = await call_next(
        request
    )

    elapsed_ms = (
        time.perf_counter()
        - start_time
    ) * 1000.0

    response.headers[
        "X-Request-ID"
    ] = request_id

    response.headers[
        "X-Process-Time-Ms"
    ] = f"{elapsed_ms:.2f}"

    LOGGER.info(
        "%s %s -> %s in %.2f ms | request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        elapsed_ms,
        request_id,
    )

    return response


# ============================================================
# AUTHENTICATION
# ============================================================
def require_api_key(
    x_api_key: str | None = Header(
        default=None,
        alias="X-API-Key",
    ),
) -> None:
    """
    Authentication is disabled when HEVEMIND_API_KEY is empty.

    For deployment, define HEVEMIND_API_KEY and send it through
    the X-API-Key request header.
    """
    if not API_KEY:
        return

    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=(
                status.HTTP_401_UNAUTHORIZED
            ),
            detail={
                "code": "invalid_api_key",
                "message": (
                    "A valid X-API-Key header is required."
                ),
            },
        )


# ============================================================
# DEPENDENCIES
# ============================================================
def get_store() -> Any:
    if STATE.report_store is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "code": "report_store_unavailable",
                "message": (
                    STATE.startup_error
                    or (
                        "The validated report store "
                        "is unavailable."
                    )
                ),
            },
        )

    return STATE.report_store


def get_index() -> pd.DataFrame:
    if STATE.wafer_index is None:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
            ),
            detail={
                "code": "report_index_unavailable",
                "message": (
                    STATE.startup_error
                    or (
                        "The validated wafer index "
                        "is unavailable."
                    )
                ),
            },
        )

    return STATE.wafer_index


def get_existing_report(
    wafer_id: str,
    store: Any,
) -> dict[str, Any]:
    try:
        return store.get_wafer_report(
            wafer_id
        )

    except KeyError:
        raise HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
            ),
            detail={
                "code": "wafer_not_found",
                "message": (
                    f"Wafer ID was not found: "
                    f"{wafer_id}"
                ),
            },
        ) from None


# ============================================================
# ERROR HANDLERS
# ============================================================
@app.exception_handler(
    HTTPException
)
async def http_exception_handler(
    request: Request,
    exception: HTTPException,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    if isinstance(
        exception.detail,
        dict,
    ):
        code = str(
            exception.detail.get(
                "code",
                "http_error",
            )
        )

        message = str(
            exception.detail.get(
                "message",
                exception.detail,
            )
        )

        details = exception.detail.get(
            "details"
        )

    else:
        code = "http_error"
        message = str(
            exception.detail
        )
        details = None

    payload = ErrorResponse(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    )

    return JSONResponse(
        status_code=(
            exception.status_code
        ),
        content=payload.model_dump(
            mode="json"
        ),
        headers=exception.headers,
    )


@app.exception_handler(
    RequestValidationError
)
async def validation_exception_handler(
    request: Request,
    exception: RequestValidationError,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    payload = ErrorResponse(
        error=ErrorDetail(
            code="request_validation_error",
            message=(
                "The request parameters are invalid."
            ),
            request_id=request_id,
            details=exception.errors(),
        )
    )

    return JSONResponse(
        status_code=(
            status.HTTP_422_UNPROCESSABLE_ENTITY
        ),
        content=payload.model_dump(
            mode="json"
        ),
    )


@app.exception_handler(
    Exception
)
async def unhandled_exception_handler(
    request: Request,
    exception: Exception,
) -> JSONResponse:
    request_id = getattr(
        request.state,
        "request_id",
        None,
    )

    LOGGER.exception(
        "Unhandled API error | request_id=%s",
        request_id,
    )

    payload = ErrorResponse(
        error=ErrorDetail(
            code="internal_server_error",
            message=(
                "An unexpected server error occurred."
            ),
            request_id=request_id,
        )
    )

    return JSONResponse(
        status_code=(
            status.HTTP_500_INTERNAL_SERVER_ERROR
        ),
        content=payload.model_dump(
            mode="json"
        ),
    )


# ============================================================
# DATAFRAME UTILITIES
# ============================================================
def frame_to_records(
    dataframe: pd.DataFrame,
) -> list[dict[str, Any]]:
    """
    Convert a DataFrame into JSON-safe Python records.

    This explicitly handles pandas missing values, NumPy scalars,
    timestamps, infinities and categorical values before Pydantic
    response-model validation.
    """
    records: list[dict[str, Any]] = []

    for raw_record in dataframe.to_dict(
        orient="records"
    ):
        clean_record: dict[str, Any] = {}

        for key, value in raw_record.items():
            if value is None:
                clean_record[key] = None
                continue

            if isinstance(
                value,
                pd.Timestamp,
            ):
                clean_record[key] = (
                    value.isoformat()
                )
                continue

            if isinstance(
                value,
                (
                    np.integer,
                    np.floating,
                    np.bool_,
                ),
            ):
                native_value = value.item()

                if (
                    isinstance(
                        native_value,
                        float,
                    )
                    and not np.isfinite(
                        native_value
                    )
                ):
                    clean_record[key] = None

                else:
                    clean_record[key] = (
                        native_value
                    )

                continue

            if (
                isinstance(
                    value,
                    float,
                )
                and not np.isfinite(
                    value
                )
            ):
                clean_record[key] = None
                continue

            try:
                if pd.isna(
                    value
                ):
                    clean_record[key] = None
                    continue

            except (
                TypeError,
                ValueError,
            ):
                pass

            clean_record[key] = value

        records.append(
            clean_record
        )

    return records


def normalise_decision_filter(
    decision: str | None,
) -> str | None:
    if decision is None:
        return None

    stripped = decision.strip()

    lower_mapping = {
        valid.lower(): valid
        for valid in VALID_DECISIONS
    }

    normalised = lower_mapping.get(
        stripped.lower()
    )

    if normalised is None:
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "code": "invalid_decision_filter",
                "message": (
                    "decision must be one of: "
                    + ", ".join(
                        sorted(
                            VALID_DECISIONS
                        )
                    )
                ),
            },
        )

    return normalised


def calculate_total_pages(
    total_records: int,
    page_size: int,
) -> int:
    if total_records == 0:
        return 0

    return int(
        np.ceil(
            total_records
            / page_size
        )
    )


# ============================================================
# SYSTEM ENDPOINTS
# ============================================================
@app.get(
    "/",
    tags=[
        "System",
    ],
)
def root() -> dict[str, Any]:
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "documentation": "/docs",
        "health": "/health",
        "readiness": "/ready",
        "scope": (
            "Validated held-out wafer reports"
        ),
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=[
        "System",
    ],
)
def health() -> HealthResponse:
    loaded = (
        STATE.report_store is not None
        and STATE.wafer_index is not None
    )

    count = (
        len(
            STATE.wafer_index
        )
        if STATE.wafer_index is not None
        else 0
    )

    return HealthResponse(
        status=(
            "healthy"
            if loaded
            else "degraded"
        ),
        service=API_TITLE,
        version=API_VERSION,
        timestamp_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        report_store_loaded=loaded,
        available_wafer_reports=count,
        raw_record_inference_supported=False,
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    tags=[
        "System",
    ],
)
def readiness() -> ReadinessResponse:
    checks: list[ReadinessCheck] = []

    backend_exists = (
        BACKEND_MODULE_PATH.exists()
    )

    checks.append(
        ReadinessCheck(
            name="validated_report_backend",
            status=(
                "pass"
                if backend_exists
                else "fail"
            ),
            details=str(
                BACKEND_MODULE_PATH
            ),
        )
    )

    store_loaded = (
        STATE.report_store is not None
    )

    checks.append(
        ReadinessCheck(
            name="report_store_loaded",
            status=(
                "pass"
                if store_loaded
                else "fail"
            ),
            details=(
                (
                    f"{len(STATE.report_store.reference_ids)} "
                    "reports loaded"
                )
                if store_loaded
                else (
                    STATE.startup_error
                    or "Store not loaded"
                )
            ),
        )
    )

    index_loaded = (
        STATE.wafer_index is not None
        and not STATE.wafer_index.empty
    )

    checks.append(
        ReadinessCheck(
            name="wafer_index_loaded",
            status=(
                "pass"
                if index_loaded
                else "fail"
            ),
            details=(
                (
                    f"{len(STATE.wafer_index)} "
                    "index rows loaded"
                )
                if index_loaded
                else "Index unavailable"
            ),
        )
    )

    ready = all(
        check.status == "pass"
        for check in checks
    )

    return ReadinessResponse(
        ready=ready,
        checks=checks,
        timestamp_utc=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


@app.get(
    "/metadata",
    response_model=MetadataResponse,
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "System",
    ],
)
def metadata(
    index: pd.DataFrame = Depends(
        get_index
    ),
) -> MetadataResponse:
    return MetadataResponse(
        project="HeveMind",
        api_version=API_VERSION,
        available_reports=int(
            len(
                index
            )
        ),
        operational_decisions=sorted(
            VALID_DECISIONS,
            key=lambda value: (
                RISK_PRIORITY[
                    value
                ]
            ),
        ),
        architecture={
            "primary_predictive_layer": (
                "Beta-calibrated Balanced Random Forest"
            ),
            "decision_policy": (
                "Four-level uncertainty-adjusted policy"
            ),
            "supporting_layers": [
                "Uncertainty",
                "OOD-based data familiarity",
                "SHAP attribution",
                "Historical similarity",
                "Sensor investigation priority",
                "Evidence aggregation",
            ],
            "backend": (
                "Script 18 validated report store"
            ),
        },
        methodological_limits=[
            (
                "The API currently serves validated "
                "held-out reports only."
            ),
            (
                "It does not accept a new raw sensor "
                "record for production inference."
            ),
            (
                "SHAP attribution is not proof of "
                "physical causality."
            ),
            (
                "Historical similarity is not a "
                "failure probability."
            ),
            (
                "Sensor priority is an investigation "
                "ranking, not confirmed root cause."
            ),
        ],
    )


# ============================================================
# WAFER LIST AND SEARCH ENDPOINTS
# ============================================================
@app.get(
    "/wafers",
    response_model=WaferListResponse,
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafers",
    ],
)
def list_wafers(
    page: int = Query(
        default=1,
        ge=1,
    ),
    page_size: int = Query(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
    ),
    decision: str | None = Query(
        default=None,
        description=(
            "Low Risk, Engineering Review, "
            "High Risk, or Insufficient Evidence."
        ),
    ),
    min_probability: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),
    max_probability: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),
    min_confidence: float | None = Query(
        default=None,
        ge=0.0,
        le=1.0,
    ),
    evidence_conflict: bool | None = Query(
        default=None,
    ),
    search: str | None = Query(
        default=None,
        min_length=1,
    ),
    sort_by: Literal[
        "risk",
        "probability",
        "uncertainty",
        "confidence",
        "wafer_id",
    ] = Query(
        default="risk",
    ),
    sort_order: Literal[
        "asc",
        "desc",
    ] = Query(
        default="desc",
    ),
    index: pd.DataFrame = Depends(
        get_index
    ),
) -> WaferListResponse:
    if (
        min_probability is not None
        and max_probability is not None
        and min_probability > max_probability
    ):
        raise HTTPException(
            status_code=(
                status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "code": "invalid_probability_range",
                "message": (
                    "min_probability cannot exceed "
                    "max_probability."
                ),
            },
        )

    filtered = index.copy()

    decision_filter = (
        normalise_decision_filter(
            decision
        )
    )

    if decision_filter is not None:
        filtered = filtered.loc[
            filtered[
                "uncertainty_adjusted_decision"
            ]
            == decision_filter
        ]

    if min_probability is not None:
        filtered = filtered.loc[
            filtered[
                "calibrated_failure_probability"
            ]
            >= min_probability
        ]

    if max_probability is not None:
        filtered = filtered.loc[
            filtered[
                "calibrated_failure_probability"
            ]
            <= max_probability
        ]

    if min_confidence is not None:
        filtered = filtered.loc[
            filtered[
                "prediction_confidence"
            ]
            >= min_confidence
        ]

    if evidence_conflict is not None:
        if "evidence_conflict_flag" in filtered.columns:
            conflict_values = (
                filtered[
                    "evidence_conflict_flag"
                ]
                .astype(str)
                .str.lower()
                .map(
                    {
                        "true": True,
                        "false": False,
                        "1": True,
                        "0": False,
                    }
                )
            )

            filtered = filtered.loc[
                conflict_values
                == evidence_conflict
            ]

        elif evidence_conflict:
            filtered = filtered.iloc[
                0:0
            ]

    if search is not None:
        search_text = (
            search.strip().lower()
        )

        filtered = filtered.loc[
            filtered[
                "wafer_id"
            ]
            .astype(str)
            .str.lower()
            .str.contains(
                search_text,
                regex=False,
            )
        ]

    ascending = (
        sort_order == "asc"
    )

    if sort_by == "risk":
        filtered[
            "_risk_priority"
        ] = (
            filtered[
                "uncertainty_adjusted_decision"
            ]
            .map(
                RISK_PRIORITY
            )
            .fillna(
                99
            )
        )

        filtered = filtered.sort_values(
            by=[
                "_risk_priority",
                "calibrated_failure_probability",
                "combined_uncertainty",
            ],
            ascending=[
                ascending,
                not ascending,
                not ascending,
            ],
        ).drop(
            columns=[
                "_risk_priority"
            ]
        )

    else:
        sort_column_mapping = {
            "probability": (
                "calibrated_failure_probability"
            ),
            "uncertainty": (
                "combined_uncertainty"
            ),
            "confidence": (
                "prediction_confidence"
            ),
            "wafer_id": (
                "wafer_id"
            ),
        }

        filtered = filtered.sort_values(
            by=sort_column_mapping[
                sort_by
            ],
            ascending=ascending,
        )

    total_records = int(
        len(
            filtered
        )
    )

    total_pages = calculate_total_pages(
        total_records=total_records,
        page_size=page_size,
    )

    start = (
        page - 1
    ) * page_size

    end = (
        start
        + page_size
    )

    page_df = filtered.iloc[
        start:end
    ].copy()

    response_columns = [
        "wafer_id",
        "calibrated_failure_probability",
        "uncertainty_adjusted_decision",
        "prediction_confidence",
        "combined_uncertainty",
        "data_familiarity_band",
        "evidence_agreement",
        "priority_1_sensor",
        "recommended_action",
    ]

    for column in response_columns:
        if column not in page_df.columns:
            page_df[
                column
            ] = None

    safe_records = frame_to_records(
        page_df[
            response_columns
        ]
    )

    items: list[WaferIndexItem] = []

    for record in safe_records:
        try:
            items.append(
                WaferIndexItem(
                    **record
                )
            )

        except Exception as error:
            LOGGER.exception(
                "Unable to serialise wafer index record: %s",
                record,
            )

            raise HTTPException(
                status_code=(
                    status.HTTP_500_INTERNAL_SERVER_ERROR
                ),
                detail={
                    "code": (
                        "wafer_index_serialisation_error"
                    ),
                    "message": (
                        "A wafer index record could not "
                        "be serialised safely."
                    ),
                    "details": {
                        "wafer_id": record.get(
                            "wafer_id"
                        ),
                        "error": str(
                            error
                        ),
                    },
                },
            ) from error

    return WaferListResponse(
        items=items,
        pagination=PaginationMeta(
            page=page,
            page_size=page_size,
            total_records=total_records,
            total_pages=total_pages,
            returned_records=len(
                items
            ),
        ),
        filters={
            "decision": decision_filter,
            "min_probability": min_probability,
            "max_probability": max_probability,
            "min_confidence": min_confidence,
            "evidence_conflict": evidence_conflict,
            "search": search,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    )


@app.get(
    "/wafers/summary/decisions",
    response_model=DecisionSummaryResponse,
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafers",
    ],
)
def decision_summary(
    index: pd.DataFrame = Depends(
        get_index
    ),
) -> DecisionSummaryResponse:
    rows: list[DecisionSummaryItem] = []

    for decision, group in index.groupby(
        "uncertainty_adjusted_decision",
        observed=False,
        dropna=False,
    ):
        rows.append(
            DecisionSummaryItem(
                decision=str(
                    decision
                ),
                records=int(
                    len(
                        group
                    )
                ),
                record_rate=float(
                    len(
                        group
                    )
                    / len(
                        index
                    )
                ),
                mean_failure_probability=float(
                    group[
                        "calibrated_failure_probability"
                    ].mean()
                ),
                mean_prediction_confidence=float(
                    group[
                        "prediction_confidence"
                    ].mean()
                ),
                mean_combined_uncertainty=float(
                    group[
                        "combined_uncertainty"
                    ].mean()
                ),
            )
        )

    rows.sort(
        key=lambda item: (
            RISK_PRIORITY.get(
                item.decision,
                99,
            )
        )
    )

    return DecisionSummaryResponse(
        total_records=int(
            len(
                index
            )
        ),
        decisions=rows,
    )


# ============================================================
# INDIVIDUAL WAFER ENDPOINTS
# ============================================================
@app.get(
    "/wafers/{wafer_id}",
    response_model=WaferReportResponse,
    responses={
        404: {
            "model": ErrorResponse
        },
        503: {
            "model": ErrorResponse
        },
    },
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafer Report",
    ],
)
def wafer_report(
    wafer_id: str,
    store: Any = Depends(
        get_store
    ),
) -> WaferReportResponse:
    report = get_existing_report(
        wafer_id=wafer_id,
        store=store,
    )

    return WaferReportResponse(
        **report
    )


@app.get(
    "/wafers/{wafer_id}/explanation",
    response_model=ExplanationResponse,
    responses={
        404: {
            "model": ErrorResponse
        },
    },
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafer Report",
    ],
)
def wafer_explanation(
    wafer_id: str,
    store: Any = Depends(
        get_store
    ),
) -> ExplanationResponse:
    report = get_existing_report(
        wafer_id=wafer_id,
        store=store,
    )

    primary = report[
        "primary_prediction"
    ]

    explanation = report[
        "explainability"
    ]

    return ExplanationResponse(
        wafer_id=report[
            "wafer_id"
        ],
        decision=primary.get(
            "operational_decision"
        ),
        failure_probability=primary.get(
            "calibrated_failure_probability"
        ),
        top_risk_features=explanation.get(
            "top_risk_features"
        ),
        top_protective_features=(
            explanation.get(
                "top_protective_features"
            )
        ),
        explanation_text=explanation.get(
            "explanation_text"
        ),
        local_shap_contributions=(
            explanation.get(
                "local_shap_contributions",
                [],
            )
        ),
        interpretation_limit=(
            "SHAP values describe model attribution. "
            "They do not confirm physical causality."
        ),
    )


@app.get(
    "/wafers/{wafer_id}/history",
    response_model=HistoricalEvidenceResponse,
    responses={
        404: {
            "model": ErrorResponse
        },
    },
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafer Report",
    ],
)
def wafer_history(
    wafer_id: str,
    store: Any = Depends(
        get_store
    ),
) -> HistoricalEvidenceResponse:
    report = get_existing_report(
        wafer_id=wafer_id,
        store=store,
    )

    historical = report[
        "historical_evidence"
    ]

    return HistoricalEvidenceResponse(
        wafer_id=report[
            "wafer_id"
        ],
        failed_neighbour_count=(
            historical.get(
                "failed_neighbour_count"
            )
        ),
        passed_neighbour_count=(
            historical.get(
                "passed_neighbour_count"
            )
        ),
        historical_weighted_failure_rate=(
            historical.get(
                "historical_weighted_failure_rate"
            )
        ),
        mean_similarity=historical.get(
            "mean_similarity"
        ),
        maximum_similarity=historical.get(
            "maximum_similarity"
        ),
        historical_evidence_level=(
            historical.get(
                "historical_evidence_level"
            )
        ),
        nearest_historical_records=(
            historical.get(
                "nearest_historical_records",
                [],
            )
        ),
        interpretation_limit=(
            "Historical proximity supplies case-based "
            "statistical evidence only. Similarity is "
            "not a failure probability or causal proof."
        ),
    )


@app.get(
    "/wafers/{wafer_id}/investigation",
    response_model=InvestigationResponse,
    responses={
        404: {
            "model": ErrorResponse
        },
    },
    dependencies=[
        Depends(
            require_api_key
        ),
    ],
    tags=[
        "Wafer Report",
    ],
)
def wafer_investigation(
    wafer_id: str,
    store: Any = Depends(
        get_store
    ),
) -> InvestigationResponse:
    report = get_existing_report(
        wafer_id=wafer_id,
        store=store,
    )

    primary = report[
        "primary_prediction"
    ]

    investigation = report[
        "sensor_investigation"
    ]

    return InvestigationResponse(
        wafer_id=report[
            "wafer_id"
        ],
        operational_decision=primary.get(
            "operational_decision"
        ),
        priority_1_sensor=investigation.get(
            "priority_1_sensor"
        ),
        top_5_sensors=investigation.get(
            "top_5_sensors"
        ),
        inspection_summary=investigation.get(
            "inspection_summary"
        ),
        ranked_sensor_priorities=(
            investigation.get(
                "ranked_sensor_priorities",
                [],
            )
        ),
        interpretation_limit=(
            "The ranking identifies statistical "
            "investigation priorities. It does not "
            "identify a confirmed physical root cause."
        ),
    )


# ============================================================
# DEVELOPMENT ENTRY POINT
# ============================================================
def main() -> None:
    uvicorn.run(
        "19_fastapi_service:app",
        host=DEFAULT_HOST,
        port=DEFAULT_PORT,
        reload=False,
        log_level=os.getenv(
            "HEVEMIND_UVICORN_LOG_LEVEL",
            "info",
        ),
    )


if __name__ == "__main__":
    main()
