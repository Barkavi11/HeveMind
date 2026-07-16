from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

REPORTS_DIR = ROOT_DIR / "reports"

UNCERTAINTY_PATH = (
    REPORTS_DIR
    / "uncertainty_engine"
    / "tables"
    / "test_uncertainty_scores.csv"
)

EXPLAINABILITY_SUMMARY_PATH = (
    REPORTS_DIR
    / "explainability_engine"
    / "tables"
    / "wafer_explanation_summary.csv"
)

LOCAL_SHAP_PATH = (
    REPORTS_DIR
    / "explainability_engine"
    / "tables"
    / "local_shap_explanations.csv"
)

SIMILARITY_SUMMARY_PATH = (
    REPORTS_DIR
    / "historical_similarity_engine"
    / "tables"
    / "historical_similarity_summary.csv"
)

NEIGHBOUR_RESULTS_PATH = (
    REPORTS_DIR
    / "historical_similarity_engine"
    / "tables"
    / "historical_neighbour_results.csv"
)

EVIDENCE_REPORT_PATH = (
    REPORTS_DIR
    / "evidence_aggregation_engine"
    / "tables"
    / "engineering_evidence_report.csv"
)

SENSOR_PRIORITY_PATH = (
    REPORTS_DIR
    / "sensor_investigation_priority"
    / "tables"
    / "sensor_investigation_priority.csv"
)

WAFER_PRIORITY_SUMMARY_PATH = (
    REPORTS_DIR
    / "sensor_investigation_priority"
    / "tables"
    / "wafer_investigation_summary.csv"
)

OUTPUT_DIR = (
    REPORTS_DIR
    / "deployment_backend"
)

JSON_OUTPUT_DIR = OUTPUT_DIR / "wafer_reports"

BACKEND_INDEX_PATH = (
    OUTPUT_DIR
    / "wafer_report_index.csv"
)

BACKEND_SUMMARY_PATH = (
    OUTPUT_DIR
    / "deployment_backend_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"

TOP_LOCAL_SHAP = 10
TOP_HISTORICAL_NEIGHBOURS = 10
TOP_SENSOR_PRIORITIES = 10


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        JSON_OUTPUT_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# GENERIC UTILITIES
# ============================================================
def json_safe(
    value: Any,
) -> Any:
    if isinstance(
        value,
        dict,
    ):
        return {
            str(key): json_safe(item)
            for key, item in value.items()
        }

    if isinstance(
        value,
        (list, tuple),
    ):
        return [
            json_safe(item)
            for item in value
        ]

    if isinstance(
        value,
        np.ndarray,
    ):
        return [
            json_safe(item)
            for item in value.tolist()
        ]

    if isinstance(
        value,
        (
            np.integer,
            np.floating,
        ),
    ):
        return value.item()

    if isinstance(
        value,
        pd.Timestamp,
    ):
        return value.isoformat()

    if pd.isna(value):
        return None

    return value


def clean_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    return {
        key: json_safe(value)
        for key, value in record.items()
    }


def normalise_id(
    value: Any,
) -> str:
    if pd.isna(value):
        raise ValueError(
            "Wafer ID cannot be missing."
        )

    return str(
        value
    ).strip()


def safe_filename(
    wafer_id: Any,
) -> str:
    text = normalise_id(
        wafer_id
    )

    invalid_characters = [
        "/",
        "\\",
        ":",
        "*",
        "?",
        "\"",
        "<",
        ">",
        "|",
        " ",
    ]

    for character in invalid_characters:
        text = text.replace(
            character,
            "_",
        )

    return text


def load_csv_required(
    path: Path,
    required_columns: set[str],
    description: str,
    id_column: str | None = ID_COLUMN,
) -> pd.DataFrame:
    """
    Load and validate a required CSV file.

    Parameters
    ----------
    path:
        CSV file path.

    required_columns:
        Columns that must exist.

    description:
        Human-readable table description.

    id_column:
        Identifier column to normalize. Use None when the table has
        no single wafer identifier, or provide an alternative such
        as 'query_wafer_id'.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    dataframe = pd.read_csv(
        path
    )

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{description} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if id_column is not None:
        if id_column not in dataframe.columns:
            raise ValueError(
                f"{description} does not contain the identifier "
                f"column '{id_column}'. Available columns: "
                f"{list(dataframe.columns)}"
            )

        dataframe[
            id_column
        ] = dataframe[
            id_column
        ].apply(
            normalise_id
        )

    return dataframe


def validate_one_row_per_wafer(
    dataframe: pd.DataFrame,
    description: str,
) -> None:
    duplicate_count = int(
        dataframe[
            ID_COLUMN
        ].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{description} contains {duplicate_count} "
            f"duplicate wafer IDs."
        )


def validate_same_wafer_set(
    reference_ids: set[str],
    dataframe: pd.DataFrame,
    description: str,
) -> None:
    observed_ids = set(
        dataframe[
            ID_COLUMN
        ].astype(str)
    )

    missing_ids = (
        reference_ids
        - observed_ids
    )

    extra_ids = (
        observed_ids
        - reference_ids
    )

    if missing_ids or extra_ids:
        raise ValueError(
            f"{description} wafer-ID mismatch. "
            f"Missing={len(missing_ids)}, "
            f"extra={len(extra_ids)}."
        )


# ============================================================
# DATA CONTAINER
# ============================================================
class HeveMindReportStore:
    """
    Read-only backend over validated HeveMind report outputs.

    This class does not retrain a model, create a new probability,
    alter the retained four-level decision, or claim physical causality.
    """

    def __init__(self) -> None:
        LOGGER.info(
            "Loading validated HeveMind report tables"
        )

        self.uncertainty = load_csv_required(
            UNCERTAINTY_PATH,
            {
                ID_COLUMN,
                "calibrated_failure_probability",
                "prediction_confidence",
                "combined_uncertainty",
                "ood_percentile",
                "uncertainty_adjusted_decision",
            },
            "Uncertainty results",
        )

        self.explainability_summary = (
            load_csv_required(
                EXPLAINABILITY_SUMMARY_PATH,
                {
                    ID_COLUMN,
                    "top_risk_features",
                    "top_protective_features",
                },
                "Explainability summary",
            )
        )

        self.local_shap = load_csv_required(
            LOCAL_SHAP_PATH,
            {
                ID_COLUMN,
                "rank",
                "feature",
                "feature_value",
                "shap_contribution",
                "absolute_shap_contribution",
                "contribution_direction",
            },
            "Local SHAP explanations",
        )

        self.similarity_summary = (
            load_csv_required(
                SIMILARITY_SUMMARY_PATH,
                {
                    ID_COLUMN,
                    "failed_neighbour_count",
                    "historical_weighted_failure_rate",
                    "mean_similarity",
                },
                "Historical similarity summary",
            )
        )

        self.neighbour_results = (
            load_csv_required(
                NEIGHBOUR_RESULTS_PATH,
                {
                    "query_wafer_id",
                    "neighbour_rank",
                    "historical_wafer_id",
                    "historical_status",
                    "similarity_score",
                },
                "Historical neighbour results",
                id_column="query_wafer_id",
            )
        )



        self.evidence_report = (
            load_csv_required(
                EVIDENCE_REPORT_PATH,
                {
                    ID_COLUMN,
                    "uncertainty_adjusted_decision",
                    "evidence_agreement",
                    "recommended_action",
                    "engineering_evidence_narrative",
                },
                "Engineering evidence report",
            )
        )

        self.sensor_priority = (
            load_csv_required(
                SENSOR_PRIORITY_PATH,
                {
                    ID_COLUMN,
                    "investigation_rank",
                    "feature",
                    "priority_level",
                    "investigation_priority_score",
                    "deviation_level",
                    "investigation_reason",
                },
                "Sensor investigation priorities",
            )
        )

        self.wafer_priority_summary = (
            load_csv_required(
                WAFER_PRIORITY_SUMMARY_PATH,
                {
                    ID_COLUMN,
                    "priority_1_sensor",
                    "top_5_sensors",
                    "inspection_summary",
                },
                "Wafer investigation summary",
            )
        )

        for dataframe, description in [
            (
                self.uncertainty,
                "Uncertainty results",
            ),
            (
                self.explainability_summary,
                "Explainability summary",
            ),
            (
                self.similarity_summary,
                "Historical similarity summary",
            ),
            (
                self.evidence_report,
                "Engineering evidence report",
            ),
            (
                self.wafer_priority_summary,
                "Wafer investigation summary",
            ),
        ]:
            validate_one_row_per_wafer(
                dataframe,
                description,
            )

        self.reference_ids = set(
            self.uncertainty[
                ID_COLUMN
            ].astype(str)
        )

        for dataframe, description in [
            (
                self.explainability_summary,
                "Explainability summary",
            ),
            (
                self.similarity_summary,
                "Historical similarity summary",
            ),
            (
                self.evidence_report,
                "Engineering evidence report",
            ),
            (
                self.wafer_priority_summary,
                "Wafer investigation summary",
            ),
        ]:
            validate_same_wafer_set(
                self.reference_ids,
                dataframe,
                description,
            )

        self._build_lookups()

        LOGGER.info(
            "Loaded %s validated wafer reports",
            len(
                self.reference_ids
            ),
        )

    def _build_lookups(
        self,
    ) -> None:
        self.uncertainty_lookup = (
            self.uncertainty
            .set_index(
                ID_COLUMN
            )
        )

        self.explainability_lookup = (
            self.explainability_summary
            .set_index(
                ID_COLUMN
            )
        )

        self.similarity_lookup = (
            self.similarity_summary
            .set_index(
                ID_COLUMN
            )
        )

        self.evidence_lookup = (
            self.evidence_report
            .set_index(
                ID_COLUMN
            )
        )

        self.priority_summary_lookup = (
            self.wafer_priority_summary
            .set_index(
                ID_COLUMN
            )
        )

    def list_wafer_ids(
        self,
    ) -> list[str]:
        return sorted(
            self.reference_ids
        )

    def wafer_exists(
        self,
        wafer_id: Any,
    ) -> bool:
        return (
            normalise_id(
                wafer_id
            )
            in self.reference_ids
        )

    @staticmethod
    def _series_to_record(
        series: pd.Series,
    ) -> dict[str, Any]:
        return clean_record(
            series.to_dict()
        )

    def get_local_shap(
        self,
        wafer_id: str,
    ) -> list[dict[str, Any]]:
        output = (
            self.local_shap.loc[
                self.local_shap[
                    ID_COLUMN
                ]
                == wafer_id
            ]
            .sort_values(
                by="rank"
            )
            .head(
                TOP_LOCAL_SHAP
            )
        )

        return [
            clean_record(
                row
            )
            for row in output.to_dict(
                orient="records"
            )
        ]

    def get_historical_neighbours(
        self,
        wafer_id: str,
    ) -> list[dict[str, Any]]:
        output = (
            self.neighbour_results.loc[
                self.neighbour_results[
                    "query_wafer_id"
                ]
                == wafer_id
            ]
            .sort_values(
                by="neighbour_rank"
            )
            .head(
                TOP_HISTORICAL_NEIGHBOURS
            )
        )

        return [
            clean_record(
                row
            )
            for row in output.to_dict(
                orient="records"
            )
        ]

    def get_sensor_priorities(
        self,
        wafer_id: str,
    ) -> list[dict[str, Any]]:
        output = (
            self.sensor_priority.loc[
                self.sensor_priority[
                    ID_COLUMN
                ]
                == wafer_id
            ]
            .sort_values(
                by="investigation_rank"
            )
            .head(
                TOP_SENSOR_PRIORITIES
            )
        )

        return [
            clean_record(
                row
            )
            for row in output.to_dict(
                orient="records"
            )
        ]

    def get_wafer_report(
        self,
        wafer_id: Any,
    ) -> dict[str, Any]:
        normalised_id = normalise_id(
            wafer_id
        )

        if normalised_id not in self.reference_ids:
            raise KeyError(
                f"Wafer ID was not found: {normalised_id}"
            )

        uncertainty_record = (
            self._series_to_record(
                self.uncertainty_lookup.loc[
                    normalised_id
                ]
            )
        )

        explainability_record = (
            self._series_to_record(
                self.explainability_lookup.loc[
                    normalised_id
                ]
            )
        )

        similarity_record = (
            self._series_to_record(
                self.similarity_lookup.loc[
                    normalised_id
                ]
            )
        )

        evidence_record = (
            self._series_to_record(
                self.evidence_lookup.loc[
                    normalised_id
                ]
            )
        )

        priority_summary_record = (
            self._series_to_record(
                self.priority_summary_lookup.loc[
                    normalised_id
                ]
            )
        )

        return {
            "project": "HeveMind",
            "report_type": (
                "Validated held-out wafer engineering report"
            ),
            "wafer_id": normalised_id,
            "primary_prediction": {
                "calibrated_failure_probability": (
                    uncertainty_record.get(
                        "calibrated_failure_probability"
                    )
                ),
                "operational_decision": (
                    uncertainty_record.get(
                        "uncertainty_adjusted_decision"
                    )
                ),
                "prediction_confidence": (
                    uncertainty_record.get(
                        "prediction_confidence"
                    )
                ),
                "combined_uncertainty": (
                    uncertainty_record.get(
                        "combined_uncertainty"
                    )
                ),
                "probability_interval_lower": (
                    uncertainty_record.get(
                        "probability_interval_lower"
                    )
                ),
                "probability_interval_upper": (
                    uncertainty_record.get(
                        "probability_interval_upper"
                    )
                ),
            },
            "data_reliability": {
                "data_confidence": (
                    uncertainty_record.get(
                        "data_confidence"
                    )
                ),
                "missing_sensor_rate": (
                    uncertainty_record.get(
                        "missing_sensor_rate"
                    )
                ),
                "ood_percentile": (
                    uncertainty_record.get(
                        "ood_percentile"
                    )
                ),
                "data_familiarity_score": (
                    evidence_record.get(
                        "data_familiarity_score"
                    )
                ),
                "data_familiarity_band": (
                    evidence_record.get(
                        "data_familiarity_band"
                    )
                ),
                "abstention_reason": (
                    uncertainty_record.get(
                        "abstention_reason"
                    )
                ),
            },
            "evidence_assessment": {
                "model_evidence_strength": (
                    evidence_record.get(
                        "model_evidence_strength"
                    )
                ),
                "prediction_reliability": (
                    evidence_record.get(
                        "prediction_reliability"
                    )
                ),
                "evidence_agreement": (
                    evidence_record.get(
                        "evidence_agreement"
                    )
                ),
                "evidence_conflict_flag": (
                    evidence_record.get(
                        "evidence_conflict_flag"
                    )
                ),
                "recommended_action": (
                    evidence_record.get(
                        "recommended_action"
                    )
                ),
                "engineering_evidence_narrative": (
                    evidence_record.get(
                        "engineering_evidence_narrative"
                    )
                ),
            },
            "historical_evidence": {
                "failed_neighbour_count": (
                    similarity_record.get(
                        "failed_neighbour_count"
                    )
                ),
                "passed_neighbour_count": (
                    similarity_record.get(
                        "passed_neighbour_count"
                    )
                ),
                "historical_weighted_failure_rate": (
                    similarity_record.get(
                        "historical_weighted_failure_rate"
                    )
                ),
                "mean_similarity": (
                    similarity_record.get(
                        "mean_similarity"
                    )
                ),
                "maximum_similarity": (
                    similarity_record.get(
                        "maximum_similarity"
                    )
                ),
                "historical_evidence_level": (
                    similarity_record.get(
                        "historical_evidence_level"
                    )
                ),
                "nearest_historical_records": (
                    self.get_historical_neighbours(
                        normalised_id
                    )
                ),
            },
            "explainability": {
                "top_risk_features": (
                    explainability_record.get(
                        "top_risk_features"
                    )
                ),
                "top_protective_features": (
                    explainability_record.get(
                        "top_protective_features"
                    )
                ),
                "explanation_text": (
                    explainability_record.get(
                        "explanation_text"
                    )
                ),
                "local_shap_contributions": (
                    self.get_local_shap(
                        normalised_id
                    )
                ),
            },
            "sensor_investigation": {
                "priority_1_sensor": (
                    priority_summary_record.get(
                        "priority_1_sensor"
                    )
                ),
                "top_5_sensors": (
                    priority_summary_record.get(
                        "top_5_sensors"
                    )
                ),
                "inspection_summary": (
                    priority_summary_record.get(
                        "inspection_summary"
                    )
                ),
                "ranked_sensor_priorities": (
                    self.get_sensor_priorities(
                        normalised_id
                    )
                ),
            },
            "methodological_limits": {
                "new_failure_probability_created": False,
                "primary_decision_changed": False,
                "physical_root_cause_claimed": False,
                "historical_similarity_is_probability": False,
                "shap_is_causal_proof": False,
                "scope": (
                    "This backend serves validated held-out reports. "
                    "It does not yet perform inference on a newly "
                    "submitted raw sensor record."
                ),
            },
        }

    def save_wafer_report(
        self,
        wafer_id: Any,
        output_path: Path | None = None,
    ) -> Path:
        report = self.get_wafer_report(
            wafer_id
        )

        normalised_id = normalise_id(
            wafer_id
        )

        if output_path is None:
            output_path = (
                JSON_OUTPUT_DIR
                / (
                    safe_filename(
                        normalised_id
                    )
                    + "_engineering_report.json"
                )
            )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                json_safe(
                    report
                ),
                file,
                indent=4,
                ensure_ascii=False,
            )

        return output_path

    def build_index(
        self,
    ) -> pd.DataFrame:
        columns = [
            ID_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
            "calibrated_failure_probability",
            "uncertainty_adjusted_decision",
            "prediction_confidence",
            "combined_uncertainty",
        ]

        available_columns = [
            column
            for column in columns
            if column in self.uncertainty.columns
        ]

        output = self.uncertainty[
            available_columns
        ].copy()

        evidence_columns = [
            ID_COLUMN,
            "data_familiarity_band",
            "evidence_agreement",
            "recommended_action",
        ]

        available_evidence_columns = [
            column
            for column in evidence_columns
            if column in self.evidence_report.columns
        ]

        output = output.merge(
            self.evidence_report[
                available_evidence_columns
            ],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

        priority_columns = [
            ID_COLUMN,
            "priority_1_sensor",
            "top_5_sensors",
        ]

        available_priority_columns = [
            column
            for column in priority_columns
            if column
            in self.wafer_priority_summary.columns
        ]

        output = output.merge(
            self.wafer_priority_summary[
                available_priority_columns
            ],
            on=ID_COLUMN,
            how="left",
            validate="one_to_one",
        )

        return output.sort_values(
            by=[
                "calibrated_failure_probability",
                "combined_uncertainty",
            ],
            ascending=[
                False,
                False,
            ],
        ).reset_index(
            drop=True
        )


# ============================================================
# CONSOLE FORMATTING
# ============================================================
def print_report_summary(
    report: dict[str, Any],
) -> None:
    primary = report[
        "primary_prediction"
    ]

    reliability = report[
        "data_reliability"
    ]

    evidence = report[
        "evidence_assessment"
    ]

    historical = report[
        "historical_evidence"
    ]

    sensor = report[
        "sensor_investigation"
    ]

    print("\n" + "=" * 112)
    print(
        "HEVEMIND VALIDATED WAFER ENGINEERING REPORT"
    )
    print("=" * 112)

    print(
        f"\nWafer ID:                      "
        f"{report['wafer_id']}"
    )

    print(
        f"Operational decision:          "
        f"{primary['operational_decision']}"
    )

    probability = primary[
        "calibrated_failure_probability"
    ]

    if probability is not None:
        print(
            f"Failure probability:           "
            f"{probability:.2%}"
        )

    confidence = primary[
        "prediction_confidence"
    ]

    if confidence is not None:
        print(
            f"Prediction confidence:         "
            f"{confidence:.2%}"
        )

    uncertainty = primary[
        "combined_uncertainty"
    ]

    if uncertainty is not None:
        print(
            f"Combined uncertainty:          "
            f"{uncertainty:.2%}"
        )

    print(
        f"Data familiarity:              "
        f"{reliability['data_familiarity_band']}"
    )

    print(
        f"Evidence agreement:            "
        f"{evidence['evidence_agreement']}"
    )

    print(
        f"Failed historical neighbours:  "
        f"{historical['failed_neighbour_count']}"
    )

    historical_rate = historical[
        "historical_weighted_failure_rate"
    ]

    if historical_rate is not None:
        print(
            f"Historical failure evidence:   "
            f"{historical_rate:.2%}"
        )

    print(
        f"Priority sensor:               "
        f"{sensor['priority_1_sensor']}"
    )

    print(
        f"\nRecommended action:\n"
        f"{evidence['recommended_action']}"
    )

    print(
        f"\nEvidence narrative:\n"
        f"{evidence['engineering_evidence_narrative']}"
    )

    print(
        "\nImportant limitation:\n"
        "The sensor ranking and SHAP values are statistical "
        "investigation evidence, not confirmed physical root causes."
    )


# ============================================================
# COMMAND-LINE INTERFACE
# ============================================================
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read validated HeveMind wafer reports for "
            "dashboard and API integration."
        )
    )

    parser.add_argument(
        "--wafer-id",
        type=str,
        help=(
            "Wafer ID to retrieve."
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help=(
            "List available wafer IDs."
        ),
    )

    parser.add_argument(
        "--build-index",
        action="store_true",
        help=(
            "Build the dashboard/API wafer index."
        ),
    )

    parser.add_argument(
        "--save-json",
        action="store_true",
        help=(
            "Save the selected wafer report as JSON."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        help=(
            "Optional JSON output path."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    arguments = parse_arguments()

    store = HeveMindReportStore()

    if arguments.list:
        wafer_ids = store.list_wafer_ids()

        print(
            f"Available wafer reports: "
            f"{len(wafer_ids)}"
        )

        for wafer_id in wafer_ids:
            print(
                wafer_id
            )

    if arguments.build_index:
        index_df = store.build_index()

        index_df.to_csv(
            BACKEND_INDEX_PATH,
            index=False,
        )

        LOGGER.info(
            "Saved backend index: %s",
            BACKEND_INDEX_PATH,
        )

    if arguments.wafer_id:
        report = store.get_wafer_report(
            arguments.wafer_id
        )

        print_report_summary(
            report
        )

        if arguments.save_json:
            output_path = (
                Path(
                    arguments.output
                ).expanduser().resolve()
                if arguments.output
                else None
            )

            saved_path = store.save_wafer_report(
                wafer_id=(
                    arguments.wafer_id
                ),
                output_path=output_path,
            )

            LOGGER.info(
                "Saved wafer report: %s",
                saved_path,
            )

    if (
        not arguments.list
        and not arguments.build_index
        and not arguments.wafer_id
    ):
        index_df = store.build_index()

        index_df.to_csv(
            BACKEND_INDEX_PATH,
            index=False,
        )

        summary = {
            "project": "HeveMind",
            "stage": (
                "Validated held-out report backend"
            ),
            "available_wafer_reports": int(
                len(
                    store.reference_ids
                )
            ),
            "backend_index": str(
                BACKEND_INDEX_PATH
            ),
            "serves_new_raw_sensor_records": False,
            "creates_new_failure_probability": False,
            "changes_primary_decision": False,
            "intended_consumers": [
                "Streamlit dashboard",
                "REST API",
                "CLI report inspection",
            ],
            "important_limit": (
                "This backend currently serves fully validated "
                "held-out wafer reports. Raw-record production "
                "inference requires a separately validated artifact "
                "bundle and inference adapter."
            ),
        }

        with BACKEND_SUMMARY_PATH.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                summary,
                file,
                indent=4,
            )

        print("\n" + "=" * 112)
        print(
            "HEVEMIND VALIDATED REPORT BACKEND"
        )
        print("=" * 112)

        print(
            f"\nAvailable wafer reports:       "
            f"{len(store.reference_ids)}"
        )

        print(
            f"Backend index:                 "
            f"{BACKEND_INDEX_PATH}"
        )

        print(
            "\nThe backend is ready for dashboard and API "
            "integration over validated held-out reports."
        )


if __name__ == "__main__":
    try:
        main()

    except Exception as error:
        LOGGER.exception(
            "Deployment backend failed: %s",
            error,
        )

        sys.exit(1)
