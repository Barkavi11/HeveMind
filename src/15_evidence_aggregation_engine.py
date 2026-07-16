from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# WARNINGS AND LOGGING
# ============================================================
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
SPLITS_DIR = DATA_DIR / "processed" / "splits"
TEST_PATH = SPLITS_DIR / "test.parquet"

REPORTS_DIR = ROOT_DIR / "reports"

UNCERTAINTY_PATH = (
    REPORTS_DIR
    / "uncertainty_engine"
    / "tables"
    / "test_uncertainty_scores.csv"
)

SIMILARITY_PATH = (
    REPORTS_DIR
    / "historical_similarity_engine"
    / "tables"
    / "historical_similarity_summary.csv"
)

EXPLAINABILITY_PATH = (
    REPORTS_DIR
    / "explainability_engine"
    / "tables"
    / "wafer_explanation_summary.csv"
)

CALIBRATION_SUMMARY_PATH = (
    REPORTS_DIR
    / "cross_fitted_calibration_policy"
    / "cross_fitted_calibration_summary.json"
)

OUTPUT_DIR = (
    REPORTS_DIR
    / "evidence_aggregation_engine"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

ENGINEERING_REPORT_PATH = (
    TABLES_DIR
    / "engineering_evidence_report.csv"
)

AGREEMENT_SUMMARY_PATH = (
    TABLES_DIR
    / "evidence_agreement_summary.csv"
)

DECISION_SUMMARY_PATH = (
    TABLES_DIR
    / "evidence_by_operational_decision.csv"
)

CONFLICT_CASES_PATH = (
    TABLES_DIR
    / "evidence_conflict_cases.csv"
)

OFFLINE_EVALUATION_PATH = (
    TABLES_DIR
    / "offline_evidence_evaluation.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "evidence_aggregation_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
TIMESTAMP_COLUMN = "timestamp"

DEFAULT_LOWER_THRESHOLD = 0.035
DEFAULT_UPPER_THRESHOLD = 0.080

HIGH_FAMILIARITY_MINIMUM = 0.75
MODERATE_FAMILIARITY_MINIMUM = 0.40

STRONG_HISTORICAL_FAILURE_RATE = 0.30
MODERATE_HISTORICAL_FAILURE_RATE = 0.15

STRONG_FAILED_NEIGHBOUR_COUNT = 3
MODERATE_FAILED_NEIGHBOUR_COUNT = 1

HIGH_UNCERTAINTY_MINIMUM = 0.80
MODERATE_UNCERTAINTY_MINIMUM = 0.60

LOW_CONFIDENCE_MAXIMUM = 0.25
MODERATE_CONFIDENCE_MAXIMUM = 0.50


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_test_data() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Held-out test dataset was not found: {TEST_PATH}"
        )

    dataframe = pd.read_parquet(
        TEST_PATH
    )

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_csv_required(
    path: Path,
    required_columns: set[str],
    description: str,
) -> pd.DataFrame:
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
            f"{description} is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_json_if_available(
    path: Path,
) -> dict[str, Any]:
    if not path.exists():
        LOGGER.warning(
            "Optional calibration summary was not found: %s",
            path,
        )
        return {}

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(
            file
        )


def validate_unique_ids(
    dataframe: pd.DataFrame,
    description: str,
) -> None:
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"{description} does not contain {ID_COLUMN}."
        )

    duplicate_count = int(
        dataframe[ID_COLUMN].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{description} contains {duplicate_count} duplicate wafer IDs."
        )


# ============================================================
# THRESHOLD EXTRACTION
# ============================================================
def extract_operational_thresholds(
    calibration_summary: dict[str, Any],
) -> tuple[float, float]:
    selected_policy = calibration_summary.get(
        "selected_policy",
        {},
    )

    lower_threshold = float(
        selected_policy.get(
            "lower_threshold",
            DEFAULT_LOWER_THRESHOLD,
        )
    )

    upper_threshold = float(
        selected_policy.get(
            "upper_threshold",
            DEFAULT_UPPER_THRESHOLD,
        )
    )

    if upper_threshold <= lower_threshold:
        raise ValueError(
            "Operational thresholds are invalid: "
            f"lower={lower_threshold}, upper={upper_threshold}"
        )

    return (
        lower_threshold,
        upper_threshold,
    )


# ============================================================
# EVIDENCE DESCRIPTORS
# ============================================================
def assign_model_evidence_strength(
    probability: float,
    lower_threshold: float,
    upper_threshold: float,
) -> str:
    if probability >= max(
        upper_threshold * 2.0,
        upper_threshold + 0.05,
    ):
        return "Very Strong Risk Signal"

    if probability >= upper_threshold:
        return "Strong Risk Signal"

    if probability >= lower_threshold:
        return "Intermediate Risk Signal"

    if probability < lower_threshold / 2.0:
        return "Strong Low-Risk Signal"

    return "Moderate Low-Risk Signal"


def assign_prediction_reliability(
    confidence: float,
    uncertainty: float,
) -> str:
    if (
        confidence <= LOW_CONFIDENCE_MAXIMUM
        or uncertainty >= HIGH_UNCERTAINTY_MINIMUM
    ):
        return "Low Reliability"

    if (
        confidence <= MODERATE_CONFIDENCE_MAXIMUM
        or uncertainty >= MODERATE_UNCERTAINTY_MINIMUM
    ):
        return "Moderate Reliability"

    return "High Reliability"


def assign_data_familiarity(
    ood_percentile: float,
) -> tuple[float, str]:
    familiarity_score = float(
        np.clip(
            1.0 - ood_percentile,
            0.0,
            1.0,
        )
    )

    if familiarity_score >= HIGH_FAMILIARITY_MINIMUM:
        familiarity_band = "Very Familiar"

    elif familiarity_score >= MODERATE_FAMILIARITY_MINIMUM:
        familiarity_band = "Familiar"

    elif familiarity_score >= 0.15:
        familiarity_band = "Unusual"

    else:
        familiarity_band = "Outside Typical Historical Experience"

    return (
        familiarity_score,
        familiarity_band,
    )


def assign_historical_evidence(
    weighted_failure_rate: float,
    failed_neighbour_count: int,
) -> str:
    if (
        failed_neighbour_count >= STRONG_FAILED_NEIGHBOUR_COUNT
        and weighted_failure_rate >= STRONG_HISTORICAL_FAILURE_RATE
    ):
        return "Strong Historical Failure Evidence"

    if (
        failed_neighbour_count >= MODERATE_FAILED_NEIGHBOUR_COUNT
        and weighted_failure_rate >= MODERATE_HISTORICAL_FAILURE_RATE
    ):
        return "Moderate Historical Failure Evidence"

    if failed_neighbour_count >= 1:
        return "Limited Historical Failure Evidence"

    return "No Retrieved Historical Failure Evidence"


def assign_evidence_agreement(
    decision: str,
    historical_evidence: str,
    reliability: str,
    familiarity_band: str,
) -> str:
    historical_risk_support = historical_evidence in {
        "Strong Historical Failure Evidence",
        "Moderate Historical Failure Evidence",
    }

    historical_no_support = (
        historical_evidence
        == "No Retrieved Historical Failure Evidence"
    )

    low_familiarity = familiarity_band in {
        "Unusual",
        "Outside Typical Historical Experience",
    }

    if decision == "Insufficient Evidence":
        return "Insufficient Evidence"

    if decision == "High Risk":
        if historical_risk_support and reliability != "Low Reliability":
            return "Convergent Risk Evidence"

        if low_familiarity or reliability == "Low Reliability":
            return "High Risk with Reliability Caution"

        return "Model-Led Risk Evidence"

    if decision == "Engineering Review":
        if historical_risk_support:
            return "Review with Supporting Historical Risk"

        if low_familiarity or reliability == "Low Reliability":
            return "Review Driven by Reliability Concern"

        return "Mixed or Intermediate Evidence"

    if decision == "Low Risk":
        if (
            historical_no_support
            and reliability != "Low Reliability"
            and not low_familiarity
        ):
            return "Convergent Low-Risk Evidence"

        if historical_risk_support:
            return "Low-Risk Decision with Historical Conflict"

        if low_familiarity or reliability == "Low Reliability":
            return "Low-Risk Decision with Reliability Caution"

        return "Low-Risk Decision with Limited Support"

    return "Unclassified Evidence Pattern"


def assign_conflict_flag(
    decision: str,
    agreement: str,
) -> bool:
    conflict_patterns = {
        "Low-Risk Decision with Historical Conflict",
        "Low-Risk Decision with Reliability Caution",
        "High Risk with Reliability Caution",
    }

    return bool(
        agreement in conflict_patterns
        or decision == "Insufficient Evidence"
    )


# ============================================================
# ENGINEER-FRIENDLY TEXT
# ============================================================
def create_recommended_action(
    decision: str,
    agreement: str,
) -> str:
    if decision == "Low Risk":
        if agreement in {
            "Low-Risk Decision with Historical Conflict",
            "Low-Risk Decision with Reliability Caution",
        }:
            return (
                "Do not rely on automatic release alone. "
                "Verify the highlighted evidence before continuing."
            )

        return (
            "Continue processing under standard monitoring."
        )

    if decision == "Engineering Review":
        return (
            "Prioritise engineering assessment using the SHAP-ranked "
            "sensor evidence and retrieved historical cases."
        )

    if decision == "High Risk":
        return (
            "Hold for inspection or prioritised engineering assessment "
            "before continuing."
        )

    if decision == "Insufficient Evidence":
        return (
            "Repeat or verify available measurements and obtain "
            "engineering review before making a release decision."
        )

    return (
        "Obtain engineering review."
    )


def clean_feature_text(
    value: Any,
) -> str:
    if pd.isna(value):
        return ""

    text = str(
        value
    ).strip()

    if text.lower() in {
        "nan",
        "none",
    }:
        return ""

    return text


def create_evidence_narrative(
    decision: str,
    probability: float,
    model_evidence_strength: str,
    reliability: str,
    familiarity_band: str,
    historical_evidence: str,
    failed_neighbour_count: int,
    weighted_failure_rate: float,
    top_risk_features: str,
    agreement: str,
) -> str:
    feature_phrase = (
        top_risk_features
        if top_risk_features
        else "no dominant positive sensor attribution was available"
    )

    return (
        f"The primary operational decision is {decision}. "
        f"The retained Beta-calibrated model estimated a failure "
        f"probability of {probability:.2%}, corresponding to "
        f"{model_evidence_strength}. Prediction reliability is "
        f"{reliability}, and the record is {familiarity_band.lower()} "
        f"relative to the development data. The ten retrieved "
        f"development neighbours contained {failed_neighbour_count} "
        f"observed failures, with a similarity-weighted historical "
        f"failure rate of {weighted_failure_rate:.2%}; this is classified "
        f"as {historical_evidence}. The principal model-attribution "
        f"features were {feature_phrase}. Overall evidence status: "
        f"{agreement}. Historical similarity and SHAP attribution are "
        f"supporting statistical evidence only and do not establish a "
        f"physical root cause."
    )


# ============================================================
# REPORT CONSTRUCTION
# ============================================================
def build_engineering_report(
    test_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    explainability_df: pd.DataFrame,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    base_columns = [
        ID_COLUMN,
        TARGET_COLUMN,
        STATUS_COLUMN,
        TIMESTAMP_COLUMN,
    ]

    available_base_columns = [
        column
        for column in base_columns
        if column in test_df.columns
    ]

    report = test_df[
        available_base_columns
    ].copy()

    uncertainty_columns = [
        ID_COLUMN,
        "calibrated_failure_probability",
        "prediction_confidence",
        "combined_uncertainty",
        "data_confidence",
        "ood_percentile",
        "missing_sensor_rate",
        "probability_interval_lower",
        "probability_interval_upper",
        "probability_interval_width",
        "uncertainty_adjusted_decision",
        "abstention_reason",
    ]

    similarity_columns = [
        ID_COLUMN,
        "failed_neighbour_count",
        "passed_neighbour_count",
        "historical_weighted_failure_rate",
        "historical_raw_failure_rate",
        "mean_similarity",
        "maximum_similarity",
        "historical_evidence_level",
    ]

    explanation_columns = [
        ID_COLUMN,
        "positive_shap_total",
        "negative_shap_total",
        "top_risk_features",
        "top_protective_features",
        "explanation_text",
    ]

    available_uncertainty_columns = [
        column
        for column in uncertainty_columns
        if column in uncertainty_df.columns
    ]

    available_similarity_columns = [
        column
        for column in similarity_columns
        if column in similarity_df.columns
    ]

    available_explanation_columns = [
        column
        for column in explanation_columns
        if column in explainability_df.columns
    ]

    report = report.merge(
        uncertainty_df[
            available_uncertainty_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    report = report.merge(
        similarity_df[
            available_similarity_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    report = report.merge(
        explainability_df[
            available_explanation_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    required_after_merge = {
        "calibrated_failure_probability",
        "prediction_confidence",
        "combined_uncertainty",
        "ood_percentile",
        "uncertainty_adjusted_decision",
        "failed_neighbour_count",
        "historical_weighted_failure_rate",
    }

    missing_after_merge = required_after_merge.difference(
        report.columns
    )

    if missing_after_merge:
        raise ValueError(
            "Aggregated evidence report is missing columns: "
            f"{sorted(missing_after_merge)}"
        )

    report[
        "model_evidence_strength"
    ] = report[
        "calibrated_failure_probability"
    ].apply(
        lambda probability: assign_model_evidence_strength(
            probability=float(
                probability
            ),
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
    )

    report[
        "prediction_reliability"
    ] = report.apply(
        lambda row: assign_prediction_reliability(
            confidence=float(
                row[
                    "prediction_confidence"
                ]
            ),
            uncertainty=float(
                row[
                    "combined_uncertainty"
                ]
            ),
        ),
        axis=1,
    )

    familiarity_results = report[
        "ood_percentile"
    ].apply(
        lambda value: assign_data_familiarity(
            float(
                value
            )
        )
    )

    report[
        "data_familiarity_score"
    ] = [
        result[0]
        for result in familiarity_results
    ]

    report[
        "data_familiarity_band"
    ] = [
        result[1]
        for result in familiarity_results
    ]

    report[
        "derived_historical_evidence"
    ] = report.apply(
        lambda row: assign_historical_evidence(
            weighted_failure_rate=float(
                row[
                    "historical_weighted_failure_rate"
                ]
            ),
            failed_neighbour_count=int(
                row[
                    "failed_neighbour_count"
                ]
            ),
        ),
        axis=1,
    )

    report[
        "evidence_agreement"
    ] = report.apply(
        lambda row: assign_evidence_agreement(
            decision=str(
                row[
                    "uncertainty_adjusted_decision"
                ]
            ),
            historical_evidence=str(
                row[
                    "derived_historical_evidence"
                ]
            ),
            reliability=str(
                row[
                    "prediction_reliability"
                ]
            ),
            familiarity_band=str(
                row[
                    "data_familiarity_band"
                ]
            ),
        ),
        axis=1,
    )

    report[
        "evidence_conflict_flag"
    ] = report.apply(
        lambda row: assign_conflict_flag(
            decision=str(
                row[
                    "uncertainty_adjusted_decision"
                ]
            ),
            agreement=str(
                row[
                    "evidence_agreement"
                ]
            ),
        ),
        axis=1,
    )

    if "top_risk_features" not in report.columns:
        report[
            "top_risk_features"
        ] = ""

    if "top_protective_features" not in report.columns:
        report[
            "top_protective_features"
        ] = ""

    report[
        "top_risk_features"
    ] = report[
        "top_risk_features"
    ].apply(
        clean_feature_text
    )

    report[
        "top_protective_features"
    ] = report[
        "top_protective_features"
    ].apply(
        clean_feature_text
    )

    report[
        "recommended_action"
    ] = report.apply(
        lambda row: create_recommended_action(
            decision=str(
                row[
                    "uncertainty_adjusted_decision"
                ]
            ),
            agreement=str(
                row[
                    "evidence_agreement"
                ]
            ),
        ),
        axis=1,
    )

    report[
        "engineering_evidence_narrative"
    ] = report.apply(
        lambda row: create_evidence_narrative(
            decision=str(
                row[
                    "uncertainty_adjusted_decision"
                ]
            ),
            probability=float(
                row[
                    "calibrated_failure_probability"
                ]
            ),
            model_evidence_strength=str(
                row[
                    "model_evidence_strength"
                ]
            ),
            reliability=str(
                row[
                    "prediction_reliability"
                ]
            ),
            familiarity_band=str(
                row[
                    "data_familiarity_band"
                ]
            ),
            historical_evidence=str(
                row[
                    "derived_historical_evidence"
                ]
            ),
            failed_neighbour_count=int(
                row[
                    "failed_neighbour_count"
                ]
            ),
            weighted_failure_rate=float(
                row[
                    "historical_weighted_failure_rate"
                ]
            ),
            top_risk_features=str(
                row[
                    "top_risk_features"
                ]
            ),
            agreement=str(
                row[
                    "evidence_agreement"
                ]
            ),
        ),
        axis=1,
    )

    report[
        "primary_decision_retained"
    ] = True

    report[
        "new_failure_probability_created"
    ] = False

    report[
        "physical_root_cause_claimed"
    ] = False

    return report


# ============================================================
# SUMMARIES
# ============================================================
def build_agreement_summary(
    report: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for agreement, group in report.groupby(
        "evidence_agreement",
        observed=False,
        dropna=False,
    ):
        records = int(
            len(group)
        )

        actual_failures = (
            int(
                group[
                    TARGET_COLUMN
                ].sum()
            )
            if TARGET_COLUMN in group.columns
            else np.nan
        )

        rows.append(
            {
                "evidence_agreement": str(
                    agreement
                ),
                "records": records,
                "record_rate": float(
                    records
                    / len(report)
                ),
                "actual_failures": (
                    actual_failures
                ),
                "observed_failure_rate": (
                    float(
                        actual_failures
                        / records
                    )
                    if (
                        records > 0
                        and np.isfinite(
                            actual_failures
                        )
                    )
                    else np.nan
                ),
                "mean_failure_probability": float(
                    group[
                        "calibrated_failure_probability"
                    ].mean()
                ),
                "mean_prediction_confidence": float(
                    group[
                        "prediction_confidence"
                    ].mean()
                ),
                "mean_combined_uncertainty": float(
                    group[
                        "combined_uncertainty"
                    ].mean()
                ),
                "mean_historical_failure_rate": float(
                    group[
                        "historical_weighted_failure_rate"
                    ].mean()
                ),
                "mean_data_familiarity": float(
                    group[
                        "data_familiarity_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        by="records",
        ascending=False,
    ).reset_index(
        drop=True
    )


def build_decision_summary(
    report: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for decision, group in report.groupby(
        "uncertainty_adjusted_decision",
        observed=False,
        dropna=False,
    ):
        records = int(
            len(group)
        )

        actual_failures = (
            int(
                group[
                    TARGET_COLUMN
                ].sum()
            )
            if TARGET_COLUMN in group.columns
            else np.nan
        )

        rows.append(
            {
                "operational_decision": str(
                    decision
                ),
                "records": records,
                "actual_failures": (
                    actual_failures
                ),
                "observed_failure_rate": (
                    float(
                        actual_failures
                        / records
                    )
                    if (
                        records > 0
                        and np.isfinite(
                            actual_failures
                        )
                    )
                    else np.nan
                ),
                "mean_failure_probability": float(
                    group[
                        "calibrated_failure_probability"
                    ].mean()
                ),
                "mean_prediction_confidence": float(
                    group[
                        "prediction_confidence"
                    ].mean()
                ),
                "mean_historical_failure_rate": float(
                    group[
                        "historical_weighted_failure_rate"
                    ].mean()
                ),
                "evidence_conflict_records": int(
                    group[
                        "evidence_conflict_flag"
                    ].sum()
                ),
                "very_familiar_or_familiar_rate": float(
                    group[
                        "data_familiarity_band"
                    ]
                    .isin(
                        [
                            "Very Familiar",
                            "Familiar",
                        ]
                    )
                    .mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_offline_evaluation(
    report: pd.DataFrame,
) -> pd.DataFrame:
    if TARGET_COLUMN not in report.columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []

    for column in [
        "model_evidence_strength",
        "prediction_reliability",
        "data_familiarity_band",
        "derived_historical_evidence",
        "evidence_agreement",
    ]:
        for value, group in report.groupby(
            column,
            observed=False,
            dropna=False,
        ):
            rows.append(
                {
                    "evidence_variable": column,
                    "evidence_value": str(
                        value
                    ),
                    "records": int(
                        len(group)
                    ),
                    "actual_failures": int(
                        group[
                            TARGET_COLUMN
                        ].sum()
                    ),
                    "observed_failure_rate": float(
                        group[
                            TARGET_COLUMN
                        ].mean()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FIGURES
# ============================================================
def save_agreement_distribution_plot(
    agreement_summary: pd.DataFrame,
) -> None:
    plot_df = agreement_summary.sort_values(
        by="records",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(11, 7)
    )

    axis.barh(
        plot_df[
            "evidence_agreement"
        ],
        plot_df[
            "records"
        ],
    )

    axis.set_title(
        "Engineering Evidence Agreement Categories"
    )

    axis.set_xlabel(
        "Number of Test Records"
    )

    axis.set_ylabel(
        "Evidence Agreement"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "evidence_agreement_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_probability_vs_historical_plot(
    report: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    color_values = (
        report[
            TARGET_COLUMN
        ]
        if TARGET_COLUMN in report.columns
        else report[
            "evidence_conflict_flag"
        ].astype(int)
    )

    scatter = axis.scatter(
        report[
            "calibrated_failure_probability"
        ],
        report[
            "historical_weighted_failure_rate"
        ],
        c=color_values,
        alpha=0.7,
    )

    axis.set_title(
        "Primary Model Risk versus Historical Failure Evidence"
    )

    axis.set_xlabel(
        "Beta-Calibrated Failure Probability"
    )

    axis.set_ylabel(
        "Similarity-Weighted Historical Failure Rate"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label=(
            "Actual Outcome"
            if TARGET_COLUMN in report.columns
            else "Evidence Conflict"
        ),
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "model_risk_vs_historical_evidence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_familiarity_uncertainty_plot(
    report: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    scatter = axis.scatter(
        report[
            "data_familiarity_score"
        ],
        report[
            "combined_uncertainty"
        ],
        c=report[
            "calibrated_failure_probability"
        ],
        alpha=0.7,
    )

    axis.set_title(
        "Data Familiarity versus Prediction Uncertainty"
    )

    axis.set_xlabel(
        "Data Familiarity Score"
    )

    axis.set_ylabel(
        "Combined Uncertainty"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="Calibrated Failure Probability",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "data_familiarity_vs_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# SAVE UTILITIES
# ============================================================
def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:
    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
            default=str,
        )


# ============================================================
# CONSOLE OUTPUT
# ============================================================
def print_console_summary(
    report: pd.DataFrame,
    agreement_summary: pd.DataFrame,
    decision_summary: pd.DataFrame,
    lower_threshold: float,
    upper_threshold: float,
) -> None:
    print("\n" + "=" * 126)
    print(
        "HEVEMIND EVIDENCE AGGREGATION AND ENGINEERING REPORT ENGINE"
    )
    print("=" * 126)

    print("\nPrimary decision policy retained:")

    print(
        f"Low/Review threshold:            "
        f"{lower_threshold:.4f}"
    )

    print(
        f"Review/High threshold:           "
        f"{upper_threshold:.4f}"
    )

    print(
        f"Test records aggregated:         "
        f"{len(report)}"
    )

    print(
        f"New probability model created:   "
        f"No"
    )

    print(
        f"Primary decisions changed:       "
        f"No"
    )

    print(
        f"Evidence conflict records:       "
        f"{int(report['evidence_conflict_flag'].sum())}"
    )

    print("\nEvidence agreement categories:")

    print(
        agreement_summary[
            [
                "evidence_agreement",
                "records",
                "actual_failures",
                "observed_failure_rate",
                "mean_failure_probability",
                "mean_historical_failure_rate",
            ]
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nOperational-decision evidence summary:")

    print(
        decision_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nSaved outputs:")

    print(
        f"Engineering evidence report:     "
        f"{ENGINEERING_REPORT_PATH}"
    )

    print(
        f"Evidence agreement summary:      "
        f"{AGREEMENT_SUMMARY_PATH}"
    )

    print(
        f"Evidence conflict cases:         "
        f"{CONFLICT_CASES_PATH}"
    )

    print(
        f"Report directory:                "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading retained prediction and supporting-evidence outputs"
    )

    test_df = load_test_data()

    uncertainty_df = load_csv_required(
        UNCERTAINTY_PATH,
        {
            ID_COLUMN,
            "calibrated_failure_probability",
            "prediction_confidence",
            "combined_uncertainty",
            "ood_percentile",
            "uncertainty_adjusted_decision",
        },
        "Test uncertainty results",
    )

    similarity_df = load_csv_required(
        SIMILARITY_PATH,
        {
            ID_COLUMN,
            "failed_neighbour_count",
            "historical_weighted_failure_rate",
            "mean_similarity",
        },
        "Historical similarity results",
    )

    explainability_df = load_csv_required(
        EXPLAINABILITY_PATH,
        {
            ID_COLUMN,
            "top_risk_features",
            "top_protective_features",
        },
        "Wafer explainability summary",
    )

    calibration_summary = load_json_if_available(
        CALIBRATION_SUMMARY_PATH
    )

    (
        lower_threshold,
        upper_threshold,
    ) = extract_operational_thresholds(
        calibration_summary
    )

    validate_unique_ids(
        test_df,
        "Test dataset",
    )

    validate_unique_ids(
        uncertainty_df,
        "Test uncertainty results",
    )

    validate_unique_ids(
        similarity_df,
        "Historical similarity results",
    )

    validate_unique_ids(
        explainability_df,
        "Wafer explainability summary",
    )

    LOGGER.info(
        "Aggregating evidence without creating a new prediction model"
    )

    report = build_engineering_report(
        test_df=test_df,
        uncertainty_df=uncertainty_df,
        similarity_df=similarity_df,
        explainability_df=explainability_df,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    report.to_csv(
        ENGINEERING_REPORT_PATH,
        index=False,
    )

    agreement_summary = build_agreement_summary(
        report
    )

    agreement_summary.to_csv(
        AGREEMENT_SUMMARY_PATH,
        index=False,
    )

    decision_summary = build_decision_summary(
        report
    )

    decision_summary.to_csv(
        DECISION_SUMMARY_PATH,
        index=False,
    )

    conflict_cases = report.loc[
        report[
            "evidence_conflict_flag"
        ]
    ].copy()

    conflict_cases.to_csv(
        CONFLICT_CASES_PATH,
        index=False,
    )

    offline_evaluation = build_offline_evaluation(
        report
    )

    offline_evaluation.to_csv(
        OFFLINE_EVALUATION_PATH,
        index=False,
    )

    LOGGER.info(
        "Generating evidence-aggregation figures"
    )

    save_agreement_distribution_plot(
        agreement_summary
    )

    save_probability_vs_historical_plot(
        report
    )

    save_familiarity_uncertainty_plot(
        report
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Transparent evidence aggregation and "
            "engineer-facing reporting"
        ),
        "test_records": int(
            len(report)
        ),
        "operational_thresholds": {
            "lower_threshold": (
                lower_threshold
            ),
            "upper_threshold": (
                upper_threshold
            ),
        },
        "primary_predictive_layer": (
            "Beta-calibrated Balanced Random Forest"
        ),
        "primary_decision_source": (
            "Uncertainty-adjusted four-level operational policy"
        ),
        "new_failure_probability_created": False,
        "primary_decisions_changed": False,
        "evidence_sources": [
            "Calibrated failure probability",
            "Prediction confidence",
            "Combined uncertainty",
            "Data familiarity derived from OOD percentile",
            "Historical nearest-neighbour outcomes",
            "SHAP sensor attribution",
        ],
        "evidence_conflict_records": int(
            report[
                "evidence_conflict_flag"
            ].sum()
        ),
        "evidence_agreement_summary": (
            agreement_summary.to_dict(
                orient="records"
            )
        ),
        "decision_summary": (
            decision_summary.to_dict(
                orient="records"
            )
        ),
        "methodological_controls": {
            "test_target_used_to_generate_decision": False,
            "test_target_used_to_generate_evidence_category": False,
            "test_target_used_only_for_offline_evaluation": True,
            "manual_predictive_weights_used": False,
            "meta_model_used": False,
            "historical_similarity_presented_as_probability": False,
            "shap_presented_as_causality": False,
            "physical_sensor_meanings_invented": False,
        },
        "important_warning": (
            "This module aggregates traceable statistical evidence "
            "without generating a new failure probability or changing "
            "the retained operational decision. Historical similarity "
            "and SHAP attribution do not establish physical causality."
        ),
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print_console_summary(
        report=report,
        agreement_summary=(
            agreement_summary
        ),
        decision_summary=(
            decision_summary
        ),
        lower_threshold=(
            lower_threshold
        ),
        upper_threshold=(
            upper_threshold
        ),
    )


if __name__ == "__main__":
    main()
