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

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VALIDATION_PATH = SPLITS_DIR / "validation.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

REPORTS_DIR = ROOT_DIR / "reports"

EXPLAINABILITY_DIR = (
    REPORTS_DIR
    / "explainability_engine"
)

LOCAL_SHAP_PATH = (
    EXPLAINABILITY_DIR
    / "tables"
    / "local_shap_explanations.csv"
)

GLOBAL_SHAP_PATH = (
    EXPLAINABILITY_DIR
    / "tables"
    / "global_shap_importance.csv"
)

FAILURE_FEATURE_SUMMARY_PATH = (
    EXPLAINABILITY_DIR
    / "tables"
    / "actual_failure_feature_summary.csv"
)

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

OUTPUT_DIR = (
    REPORTS_DIR
    / "sensor_investigation_priority"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
LOCAL_FIGURES_DIR = FIGURES_DIR / "wafer_sensor_priority"

PRIORITY_RESULTS_PATH = (
    TABLES_DIR
    / "sensor_investigation_priority.csv"
)

WAFER_SUMMARY_PATH = (
    TABLES_DIR
    / "wafer_investigation_summary.csv"
)

GLOBAL_PRIORITY_PATH = (
    TABLES_DIR
    / "global_sensor_investigation_summary.csv"
)

DECISION_PRIORITY_PATH = (
    TABLES_DIR
    / "sensor_priority_by_decision.csv"
)

FAILURE_PRIORITY_PATH = (
    TABLES_DIR
    / "sensor_priority_in_actual_failures.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "sensor_investigation_priority_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
TIMESTAMP_COLUMN = "timestamp"

SENSOR_PREFIX = "sensor_"

TOP_SENSORS_PER_WAFER = 10
TOP_GLOBAL_DISPLAY = 30
MAX_LOCAL_PLOTS = 30

ROBUST_Z_EPSILON = 1e-12

# Rank-aggregation components are equally weighted.
# This is not a predictive probability.
LOCAL_SHAP_RANK_WEIGHT = 0.25
ROBUST_DEVIATION_RANK_WEIGHT = 0.25
GLOBAL_IMPORTANCE_RANK_WEIGHT = 0.25
FAILURE_RECURRENCE_RANK_WEIGHT = 0.25


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOCAL_FIGURES_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required parquet file was not found: {path}"
        )

    dataframe = pd.read_parquet(path)

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_csv(
    path: Path,
    required_columns: set[str],
    description: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{description} was not found: {path}"
        )

    dataframe = pd.read_csv(path)

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            f"{description} is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_development_and_test() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    train_df = load_parquet(TRAIN_PATH)
    validation_df = load_parquet(VALIDATION_PATH)
    test_df = load_parquet(TEST_PATH)

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        axis=0,
        ignore_index=True,
    )

    test_df = test_df.reset_index(drop=True)

    return development_df, test_df


def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    sensor_columns = [
        column
        for column in dataframe.columns
        if column.startswith(SENSOR_PREFIX)
    ]

    if not sensor_columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return sensor_columns


# ============================================================
# DEVELOPMENT SENSOR REFERENCE
# ============================================================
def build_development_sensor_reference(
    development_df: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for sensor in sensor_columns:
        values = pd.to_numeric(
            development_df[sensor],
            errors="coerce",
        )

        observed_values = values.dropna()

        median = float(
            observed_values.median()
        ) if not observed_values.empty else np.nan

        first_quartile = float(
            observed_values.quantile(0.25)
        ) if not observed_values.empty else np.nan

        third_quartile = float(
            observed_values.quantile(0.75)
        ) if not observed_values.empty else np.nan

        iqr = (
            third_quartile
            - first_quartile
        )

        rows.append(
            {
                "feature": sensor,
                "development_median": median,
                "development_q1": first_quartile,
                "development_q3": third_quartile,
                "development_iqr": float(iqr),
                "development_missing_rate": float(
                    values.isna().mean()
                ),
                "development_observed_count": int(
                    values.notna().sum()
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# EXTERNAL EVIDENCE TABLES
# ============================================================
def prepare_global_importance(
    global_shap_df: pd.DataFrame,
) -> pd.DataFrame:
    output = global_shap_df.copy()

    output = output.loc[
        output["feature"]
        .astype(str)
        .str.startswith(SENSOR_PREFIX)
    ].copy()

    if "relative_importance" not in output.columns:
        total = float(
            output["mean_absolute_shap"].sum()
        )

        output["relative_importance"] = (
            output["mean_absolute_shap"] / total
            if total > 0
            else 0.0
        )

    maximum_importance = float(
        output["mean_absolute_shap"].max()
    )

    output[
        "global_importance_normalized"
    ] = (
        output["mean_absolute_shap"]
        / maximum_importance
        if maximum_importance > 0
        else 0.0
    )

    return output[
        [
            "feature",
            "mean_absolute_shap",
            "relative_importance",
            "global_importance_normalized",
        ]
    ].copy()


def prepare_failure_recurrence(
    failure_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    if failure_feature_df.empty:
        return pd.DataFrame(
            columns=[
                "feature",
                "failure_records",
                "failure_mean_absolute_shap",
                "failure_positive_contribution_rate",
                "failure_recurrence_normalized",
            ]
        )

    output = failure_feature_df.copy()

    output = output.rename(
        columns={
            "mean_absolute_shap": (
                "failure_mean_absolute_shap"
            ),
            "positive_contribution_rate": (
                "failure_positive_contribution_rate"
            ),
        }
    )

    maximum_failure_records = float(
        output["failure_records"].max()
    )

    output[
        "failure_recurrence_normalized"
    ] = (
        output["failure_records"]
        / maximum_failure_records
        if maximum_failure_records > 0
        else 0.0
    )

    required_columns = [
        "feature",
        "failure_records",
        "failure_mean_absolute_shap",
        "failure_positive_contribution_rate",
        "failure_recurrence_normalized",
    ]

    for column in required_columns:
        if column not in output.columns:
            output[column] = 0.0

    return output[
        required_columns
    ].copy()


# ============================================================
# RANKING UTILITIES
# ============================================================
def percentile_rank(
    values: pd.Series,
    ascending: bool = True,
) -> pd.Series:
    """
    Produce values from 0 to 1.

    A value closer to 1 indicates stronger investigation priority.
    """
    numeric_values = pd.to_numeric(
        values,
        errors="coerce",
    ).fillna(0.0)

    if numeric_values.nunique() <= 1:
        return pd.Series(
            np.full(
                len(numeric_values),
                0.5,
            ),
            index=numeric_values.index,
            dtype=float,
        )

    return numeric_values.rank(
        method="average",
        pct=True,
        ascending=ascending,
    )


def assign_priority_level(
    priority_score: float,
    positive_shap: float,
    robust_z: float,
    is_missing: bool,
) -> str:
    if is_missing:
        return "Data Verification"

    if (
        priority_score >= 0.80
        and positive_shap > 0
    ):
        return "Very High"

    if (
        priority_score >= 0.65
        and positive_shap > 0
    ):
        return "High"

    if priority_score >= 0.45:
        return "Moderate"

    return "Low"


def assign_deviation_level(
    absolute_robust_z: float,
    is_missing: bool,
) -> str:
    if is_missing:
        return "Missing Measurement"

    if not np.isfinite(
        absolute_robust_z
    ):
        return "Unavailable"

    if absolute_robust_z >= 3.0:
        return "Extreme Deviation"

    if absolute_robust_z >= 1.5:
        return "Moderate Deviation"

    if absolute_robust_z >= 1.0:
        return "Mild Deviation"

    return "Within Typical Range"


def build_reason_text(
    feature: str,
    positive_shap: float,
    absolute_shap: float,
    absolute_robust_z: float,
    is_missing: bool,
    global_importance_normalized: float,
    failure_recurrence_normalized: float,
) -> str:
    reasons: list[str] = []

    if is_missing:
        reasons.append(
            "measurement is missing and should be verified"
        )

    if positive_shap > 0:
        reasons.append(
            "the feature increased this wafer's predicted failure risk"
        )

    elif absolute_shap > 0:
        reasons.append(
            "the feature materially influenced the model prediction"
        )

    if np.isfinite(
        absolute_robust_z
    ):
        if absolute_robust_z >= 3.0:
            reasons.append(
                "the observed value shows an extreme deviation "
                "from the development reference"
            )

        elif absolute_robust_z >= 1.5:
            reasons.append(
                "the observed value shows a moderate deviation "
                "from the development reference"
            )

    if global_importance_normalized >= 0.60:
        reasons.append(
            "the feature is globally influential in the fitted model"
        )

    if failure_recurrence_normalized >= 0.60:
        reasons.append(
            "the feature recurs frequently in explanations "
            "for observed failure records"
        )

    if not reasons:
        reasons.append(
            "the feature was among the strongest available "
            "local model contributors"
        )

    return (
        f"{feature}: "
        + "; ".join(reasons)
        + "."
    )


# ============================================================
# SENSOR-LEVEL PRIORITY RESULTS
# ============================================================
def build_sensor_priority_results(
    test_df: pd.DataFrame,
    local_shap_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    sensor_reference_df: pd.DataFrame,
    global_importance_df: pd.DataFrame,
    failure_recurrence_df: pd.DataFrame,
) -> pd.DataFrame:
    test_lookup = test_df.set_index(
        ID_COLUMN
    )

    uncertainty_lookup = uncertainty_df.set_index(
        ID_COLUMN
    )

    similarity_lookup = similarity_df.set_index(
        ID_COLUMN
    )

    reference_lookup = sensor_reference_df.set_index(
        "feature"
    )

    global_lookup = global_importance_df.set_index(
        "feature"
    )

    failure_lookup = failure_recurrence_df.set_index(
        "feature"
    )

    rows: list[dict[str, Any]] = []

    for wafer_id, wafer_shap_df in local_shap_df.groupby(
        ID_COLUMN,
        observed=False,
    ):
        if wafer_id not in test_lookup.index:
            raise KeyError(
                f"Wafer {wafer_id} is missing from the test dataset."
            )

        if wafer_id not in uncertainty_lookup.index:
            raise KeyError(
                f"Wafer {wafer_id} is missing from uncertainty results."
            )

        test_record = test_lookup.loc[
            wafer_id
        ]

        uncertainty_record = uncertainty_lookup.loc[
            wafer_id
        ]

        similarity_record = (
            similarity_lookup.loc[wafer_id]
            if wafer_id in similarity_lookup.index
            else None
        )

        wafer_candidates: list[
            dict[str, Any]
        ] = []

        for _, shap_record in wafer_shap_df.iterrows():
            feature = str(
                shap_record["feature"]
            )

            if not feature.startswith(
                SENSOR_PREFIX
            ):
                continue

            if feature not in test_record.index:
                continue

            feature_value = pd.to_numeric(
                pd.Series(
                    [
                        test_record[
                            feature
                        ]
                    ]
                ),
                errors="coerce",
            ).iloc[0]

            is_missing = bool(
                pd.isna(
                    feature_value
                )
            )

            if feature in reference_lookup.index:
                reference_record = (
                    reference_lookup.loc[
                        feature
                    ]
                )

                development_median = float(
                    reference_record[
                        "development_median"
                    ]
                )

                development_iqr = float(
                    reference_record[
                        "development_iqr"
                    ]
                )

                development_missing_rate = float(
                    reference_record[
                        "development_missing_rate"
                    ]
                )

            else:
                development_median = np.nan
                development_iqr = np.nan
                development_missing_rate = np.nan

            if (
                is_missing
                or not np.isfinite(
                    development_median
                )
            ):
                robust_z = np.nan

            else:
                denominator = (
                    development_iqr
                    if np.isfinite(
                        development_iqr
                    )
                    and abs(
                        development_iqr
                    )
                    > ROBUST_Z_EPSILON
                    else 1.0
                )

                robust_z = float(
                    (
                        feature_value
                        - development_median
                    )
                    / denominator
                )

            absolute_robust_z = (
                abs(robust_z)
                if np.isfinite(
                    robust_z
                )
                else np.nan
            )

            if feature in global_lookup.index:
                global_record = (
                    global_lookup.loc[
                        feature
                    ]
                )

                global_mean_absolute_shap = float(
                    global_record[
                        "mean_absolute_shap"
                    ]
                )

                global_relative_importance = float(
                    global_record[
                        "relative_importance"
                    ]
                )

                global_importance_normalized = float(
                    global_record[
                        "global_importance_normalized"
                    ]
                )

            else:
                global_mean_absolute_shap = 0.0
                global_relative_importance = 0.0
                global_importance_normalized = 0.0

            if feature in failure_lookup.index:
                failure_record = (
                    failure_lookup.loc[
                        feature
                    ]
                )

                failure_records = int(
                    failure_record[
                        "failure_records"
                    ]
                )

                failure_recurrence_normalized = float(
                    failure_record[
                        "failure_recurrence_normalized"
                    ]
                )

                failure_positive_rate = float(
                    failure_record[
                        "failure_positive_contribution_rate"
                    ]
                )

            else:
                failure_records = 0
                failure_recurrence_normalized = 0.0
                failure_positive_rate = 0.0

            shap_contribution = float(
                shap_record[
                    "shap_contribution"
                ]
            )

            absolute_shap = float(
                shap_record[
                    "absolute_shap_contribution"
                ]
            )

            positive_shap = max(
                shap_contribution,
                0.0,
            )

            wafer_candidates.append(
                {
                    ID_COLUMN: wafer_id,
                    TARGET_COLUMN: int(
                        test_record[
                            TARGET_COLUMN
                        ]
                    ),
                    STATUS_COLUMN: str(
                        test_record[
                            STATUS_COLUMN
                        ]
                    ),
                    "decision": str(
                        uncertainty_record[
                            "uncertainty_adjusted_decision"
                        ]
                    ),
                    "calibrated_failure_probability": float(
                        uncertainty_record[
                            "calibrated_failure_probability"
                        ]
                    ),
                    "prediction_confidence": float(
                        uncertainty_record[
                            "prediction_confidence"
                        ]
                    ),
                    "combined_uncertainty": float(
                        uncertainty_record[
                            "combined_uncertainty"
                        ]
                    ),
                    "data_familiarity": float(
                        1.0
                        - uncertainty_record[
                            "ood_percentile"
                        ]
                    ),
                    "historical_weighted_failure_rate": (
                        float(
                            similarity_record[
                                "historical_weighted_failure_rate"
                            ]
                        )
                        if similarity_record is not None
                        else np.nan
                    ),
                    "feature": feature,
                    "feature_value": (
                        float(feature_value)
                        if not is_missing
                        else np.nan
                    ),
                    "measurement_missing": is_missing,
                    "development_median": (
                        development_median
                    ),
                    "development_iqr": (
                        development_iqr
                    ),
                    "development_missing_rate": (
                        development_missing_rate
                    ),
                    "robust_z": robust_z,
                    "absolute_robust_z": (
                        absolute_robust_z
                    ),
                    "shap_contribution": (
                        shap_contribution
                    ),
                    "positive_shap_contribution": (
                        positive_shap
                    ),
                    "absolute_shap_contribution": (
                        absolute_shap
                    ),
                    "global_mean_absolute_shap": (
                        global_mean_absolute_shap
                    ),
                    "global_relative_importance": (
                        global_relative_importance
                    ),
                    "global_importance_normalized": (
                        global_importance_normalized
                    ),
                    "failure_explanation_records": (
                        failure_records
                    ),
                    "failure_recurrence_normalized": (
                        failure_recurrence_normalized
                    ),
                    "failure_positive_contribution_rate": (
                        failure_positive_rate
                    ),
                }
            )

        if not wafer_candidates:
            continue

        wafer_priority_df = pd.DataFrame(
            wafer_candidates
        )

        wafer_priority_df[
            "local_shap_rank_score"
        ] = percentile_rank(
            wafer_priority_df[
                "positive_shap_contribution"
            ],
            ascending=True,
        )

        wafer_priority_df[
            "robust_deviation_rank_score"
        ] = percentile_rank(
            wafer_priority_df[
                "absolute_robust_z"
            ],
            ascending=True,
        )

        wafer_priority_df[
            "global_importance_rank_score"
        ] = percentile_rank(
            wafer_priority_df[
                "global_importance_normalized"
            ],
            ascending=True,
        )

        wafer_priority_df[
            "failure_recurrence_rank_score"
        ] = percentile_rank(
            wafer_priority_df[
                "failure_recurrence_normalized"
            ],
            ascending=True,
        )

        wafer_priority_df[
            "investigation_priority_score"
        ] = (
            LOCAL_SHAP_RANK_WEIGHT
            * wafer_priority_df[
                "local_shap_rank_score"
            ]
            + ROBUST_DEVIATION_RANK_WEIGHT
            * wafer_priority_df[
                "robust_deviation_rank_score"
            ]
            + GLOBAL_IMPORTANCE_RANK_WEIGHT
            * wafer_priority_df[
                "global_importance_rank_score"
            ]
            + FAILURE_RECURRENCE_RANK_WEIGHT
            * wafer_priority_df[
                "failure_recurrence_rank_score"
            ]
        )

        wafer_priority_df[
            "priority_level"
        ] = wafer_priority_df.apply(
            lambda row: assign_priority_level(
                priority_score=float(
                    row[
                        "investigation_priority_score"
                    ]
                ),
                positive_shap=float(
                    row[
                        "positive_shap_contribution"
                    ]
                ),
                robust_z=float(
                    row[
                        "absolute_robust_z"
                    ]
                )
                if pd.notna(
                    row[
                        "absolute_robust_z"
                    ]
                )
                else np.nan,
                is_missing=bool(
                    row[
                        "measurement_missing"
                    ]
                ),
            ),
            axis=1,
        )

        wafer_priority_df[
            "deviation_level"
        ] = wafer_priority_df.apply(
            lambda row: assign_deviation_level(
                absolute_robust_z=float(
                    row[
                        "absolute_robust_z"
                    ]
                )
                if pd.notna(
                    row[
                        "absolute_robust_z"
                    ]
                )
                else np.nan,
                is_missing=bool(
                    row[
                        "measurement_missing"
                    ]
                ),
            ),
            axis=1,
        )

        wafer_priority_df[
            "investigation_reason"
        ] = wafer_priority_df.apply(
            lambda row: build_reason_text(
                feature=str(
                    row["feature"]
                ),
                positive_shap=float(
                    row[
                        "positive_shap_contribution"
                    ]
                ),
                absolute_shap=float(
                    row[
                        "absolute_shap_contribution"
                    ]
                ),
                absolute_robust_z=float(
                    row[
                        "absolute_robust_z"
                    ]
                )
                if pd.notna(
                    row[
                        "absolute_robust_z"
                    ]
                )
                else np.nan,
                is_missing=bool(
                    row[
                        "measurement_missing"
                    ]
                ),
                global_importance_normalized=float(
                    row[
                        "global_importance_normalized"
                    ]
                ),
                failure_recurrence_normalized=float(
                    row[
                        "failure_recurrence_normalized"
                    ]
                ),
            ),
            axis=1,
        )

        wafer_priority_df = (
            wafer_priority_df
            .sort_values(
                by=[
                    "investigation_priority_score",
                    "positive_shap_contribution",
                    "absolute_robust_z",
                ],
                ascending=[
                    False,
                    False,
                    False,
                ],
                na_position="last",
            )
            .head(
                TOP_SENSORS_PER_WAFER
            )
            .reset_index(
                drop=True
            )
        )

        wafer_priority_df[
            "investigation_rank"
        ] = np.arange(
            1,
            len(
                wafer_priority_df
            )
            + 1,
        )

        rows.extend(
            wafer_priority_df.to_dict(
                orient="records"
            )
        )

    output = pd.DataFrame(rows)

    if output.empty:
        raise ValueError(
            "No sensor-priority records were generated."
        )

    return output


# ============================================================
# SUMMARY TABLES
# ============================================================
def build_wafer_summary(
    priority_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for wafer_id, group in priority_df.groupby(
        ID_COLUMN,
        observed=False,
    ):
        group = group.sort_values(
            "investigation_rank"
        )

        top_features = group.head(5)

        rows.append(
            {
                ID_COLUMN: wafer_id,
                TARGET_COLUMN: int(
                    group[
                        TARGET_COLUMN
                    ].iloc[0]
                ),
                STATUS_COLUMN: str(
                    group[
                        STATUS_COLUMN
                    ].iloc[0]
                ),
                "decision": str(
                    group[
                        "decision"
                    ].iloc[0]
                ),
                "calibrated_failure_probability": float(
                    group[
                        "calibrated_failure_probability"
                    ].iloc[0]
                ),
                "prediction_confidence": float(
                    group[
                        "prediction_confidence"
                    ].iloc[0]
                ),
                "combined_uncertainty": float(
                    group[
                        "combined_uncertainty"
                    ].iloc[0]
                ),
                "data_familiarity": float(
                    group[
                        "data_familiarity"
                    ].iloc[0]
                ),
                "historical_weighted_failure_rate": float(
                    group[
                        "historical_weighted_failure_rate"
                    ].iloc[0]
                )
                if pd.notna(
                    group[
                        "historical_weighted_failure_rate"
                    ].iloc[0]
                )
                else np.nan,
                "priority_1_sensor": (
                    top_features[
                        "feature"
                    ].iloc[0]
                ),
                "priority_1_score": float(
                    top_features[
                        "investigation_priority_score"
                    ].iloc[0]
                ),
                "priority_1_level": (
                    top_features[
                        "priority_level"
                    ].iloc[0]
                ),
                "top_5_sensors": ", ".join(
                    top_features[
                        "feature"
                    ].astype(str)
                ),
                "very_high_or_high_sensor_count": int(
                    group[
                        "priority_level"
                    ]
                    .isin(
                        [
                            "Very High",
                            "High",
                        ]
                    )
                    .sum()
                ),
                "missing_priority_sensor_count": int(
                    group[
                        "measurement_missing"
                    ].sum()
                ),
                "inspection_summary": (
                    create_wafer_summary_text(
                        group
                    )
                ),
            }
        )

    return pd.DataFrame(rows)


def create_wafer_summary_text(
    wafer_group: pd.DataFrame,
) -> str:
    sorted_group = wafer_group.sort_values(
        "investigation_rank"
    )

    top_group = sorted_group.head(3)

    sensor_text = ", ".join(
        top_group["feature"].astype(str)
    )

    decision = str(
        sorted_group[
            "decision"
        ].iloc[0]
    )

    probability = float(
        sorted_group[
            "calibrated_failure_probability"
        ].iloc[0]
    )

    return (
        f"The record is assigned to {decision} with a calibrated "
        f"failure probability of {probability:.2%}. "
        f"The first statistical sensor measurements to investigate "
        f"are {sensor_text}. This ranking is based on model "
        f"attribution, robust deviation, global importance and "
        f"recurrence in observed failure explanations. It does not "
        f"establish a physical root cause."
    )


def build_global_sensor_summary(
    priority_df: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        priority_df
        .groupby(
            "feature",
            observed=False,
        )
        .agg(
            wafers_ranked=(
                ID_COLUMN,
                "nunique",
            ),
            average_investigation_rank=(
                "investigation_rank",
                "mean",
            ),
            average_priority_score=(
                "investigation_priority_score",
                "mean",
            ),
            priority_1_count=(
                "investigation_rank",
                lambda values: int(
                    np.sum(
                        values == 1
                    )
                ),
            ),
            actual_failure_wafers=(
                TARGET_COLUMN,
                "sum",
            ),
            positive_shap_rate=(
                "positive_shap_contribution",
                lambda values: float(
                    np.mean(
                        values > 0
                    )
                ),
            ),
            mean_absolute_robust_z=(
                "absolute_robust_z",
                "mean",
            ),
            missing_measurement_rate=(
                "measurement_missing",
                "mean",
            ),
        )
        .reset_index()
    )

    return output.sort_values(
        by=[
            "priority_1_count",
            "average_priority_score",
            "actual_failure_wafers",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_decision_sensor_summary(
    priority_df: pd.DataFrame,
) -> pd.DataFrame:
    output = (
        priority_df
        .groupby(
            [
                "decision",
                "feature",
            ],
            observed=False,
        )
        .agg(
            records=(
                ID_COLUMN,
                "nunique",
            ),
            mean_priority_score=(
                "investigation_priority_score",
                "mean",
            ),
            mean_rank=(
                "investigation_rank",
                "mean",
            ),
            priority_1_count=(
                "investigation_rank",
                lambda values: int(
                    np.sum(
                        values == 1
                    )
                ),
            ),
            mean_positive_shap=(
                "positive_shap_contribution",
                "mean",
            ),
            mean_absolute_robust_z=(
                "absolute_robust_z",
                "mean",
            ),
        )
        .reset_index()
    )

    output[
        "rank_within_decision"
    ] = (
        output
        .groupby(
            "decision",
            observed=False,
        )[
            "mean_priority_score"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    return (
        output.loc[
            output[
                "rank_within_decision"
            ]
            <= TOP_GLOBAL_DISPLAY
        ]
        .sort_values(
            by=[
                "decision",
                "rank_within_decision",
            ]
        )
        .reset_index(
            drop=True
        )
    )


def build_failure_sensor_summary(
    priority_df: pd.DataFrame,
) -> pd.DataFrame:
    failure_df = priority_df.loc[
        priority_df[
            TARGET_COLUMN
        ]
        == 1
    ].copy()

    if failure_df.empty:
        return pd.DataFrame()

    output = (
        failure_df
        .groupby(
            "feature",
            observed=False,
        )
        .agg(
            failure_wafers=(
                ID_COLUMN,
                "nunique",
            ),
            mean_priority_score=(
                "investigation_priority_score",
                "mean",
            ),
            mean_investigation_rank=(
                "investigation_rank",
                "mean",
            ),
            priority_1_failure_count=(
                "investigation_rank",
                lambda values: int(
                    np.sum(
                        values == 1
                    )
                ),
            ),
            mean_positive_shap=(
                "positive_shap_contribution",
                "mean",
            ),
            mean_absolute_robust_z=(
                "absolute_robust_z",
                "mean",
            ),
        )
        .reset_index()
    )

    return output.sort_values(
        by=[
            "priority_1_failure_count",
            "failure_wafers",
            "mean_priority_score",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_global_priority_plot(
    global_summary_df: pd.DataFrame,
) -> None:
    plot_df = (
        global_summary_df
        .head(
            TOP_GLOBAL_DISPLAY
        )
        .sort_values(
            by="average_priority_score",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 10)
    )

    axis.barh(
        plot_df["feature"],
        plot_df[
            "average_priority_score"
        ],
    )

    axis.set_title(
        "Global Sensor Investigation Priority"
    )

    axis.set_xlabel(
        "Average Investigation Priority Score"
    )

    axis.set_ylabel(
        "Anonymous Sensor Feature"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "global_sensor_investigation_priority.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_local_priority_plot(
    wafer_id: Any,
    wafer_df: pd.DataFrame,
) -> None:
    plot_df = (
        wafer_df
        .sort_values(
            by="investigation_rank",
            ascending=False,
        )
        .head(
            TOP_SENSORS_PER_WAFER
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    labels = [
        (
            f"{feature} "
            f"({priority_level})"
        )
        for feature, priority_level in zip(
            plot_df["feature"],
            plot_df["priority_level"],
            strict=True,
        )
    ]

    axis.barh(
        labels,
        plot_df[
            "investigation_priority_score"
        ],
    )

    axis.set_title(
        f"Sensor Investigation Priority: {wafer_id}"
    )

    axis.set_xlabel(
        "Rank-Aggregated Priority Score"
    )

    axis.set_ylabel(
        "Anonymous Sensor"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    figure.tight_layout()

    safe_id = str(
        wafer_id
    ).replace(
        "/",
        "_",
    )

    figure.savefig(
        LOCAL_FIGURES_DIR
        / f"{safe_id}_sensor_priority.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def select_local_plot_wafers(
    wafer_summary_df: pd.DataFrame,
) -> list[Any]:
    priority_mapping = {
        "Insufficient Evidence": 0,
        "High Risk": 1,
        "Engineering Review": 2,
        "Low Risk": 3,
    }

    selection_df = wafer_summary_df.copy()

    selection_df[
        "actual_failure_priority"
    ] = (
        1
        - selection_df[
            TARGET_COLUMN
        ].astype(int)
    )

    selection_df[
        "decision_priority"
    ] = (
        selection_df["decision"]
        .map(
            priority_mapping
        )
        .fillna(99)
    )

    selection_df = selection_df.sort_values(
        by=[
            "actual_failure_priority",
            "decision_priority",
            "priority_1_score",
            "calibrated_failure_probability",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )

    return (
        selection_df[
            ID_COLUMN
        ]
        .head(
            MAX_LOCAL_PLOTS
        )
        .tolist()
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
    priority_df: pd.DataFrame,
    wafer_summary_df: pd.DataFrame,
    global_summary_df: pd.DataFrame,
    plotted_count: int,
) -> None:
    print("\n" + "=" * 124)

    print(
        "HEVEMIND SENSOR INVESTIGATION PRIORITY ENGINE"
    )

    print("=" * 124)

    print("\nCoverage:")

    print(
        f"Test wafers ranked:              "
        f"{wafer_summary_df[ID_COLUMN].nunique()}"
    )

    print(
        f"Sensor-priority rows:            "
        f"{len(priority_df)}"
    )

    print(
        f"Sensors ranked per wafer:        "
        f"{TOP_SENSORS_PER_WAFER}"
    )

    print(
        f"Local priority plots:            "
        f"{plotted_count}"
    )

    print("\nPriority levels:")

    print(
        priority_df[
            "priority_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nTop global investigation sensors:")

    print(
        global_summary_df[
            [
                "feature",
                "wafers_ranked",
                "priority_1_count",
                "average_priority_score",
                "actual_failure_wafers",
                "mean_absolute_robust_z",
            ]
        ]
        .head(20)
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nDecision coverage:")

    print(
        wafer_summary_df[
            "decision"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nSaved outputs:")

    print(
        f"Sensor priority results:         "
        f"{PRIORITY_RESULTS_PATH}"
    )

    print(
        f"Wafer summaries:                 "
        f"{WAFER_SUMMARY_PATH}"
    )

    print(
        f"Global sensor summary:           "
        f"{GLOBAL_PRIORITY_PATH}"
    )

    print(
        f"Decision-specific summary:       "
        f"{DECISION_PRIORITY_PATH}"
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
        "Loading development, test and decision-support outputs"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    local_shap_df = load_csv(
        LOCAL_SHAP_PATH,
        {
            ID_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
            "decision",
            "feature",
            "shap_contribution",
            "absolute_shap_contribution",
        },
        "Local SHAP explanations",
    )

    global_shap_df = load_csv(
        GLOBAL_SHAP_PATH,
        {
            "feature",
            "mean_absolute_shap",
        },
        "Global SHAP importance",
    )

    failure_feature_df = load_csv(
        FAILURE_FEATURE_SUMMARY_PATH,
        {
            "feature",
            "failure_records",
            "mean_absolute_shap",
            "positive_contribution_rate",
        },
        "Actual-failure feature summary",
    )

    uncertainty_df = load_csv(
        UNCERTAINTY_PATH,
        {
            ID_COLUMN,
            TARGET_COLUMN,
            "calibrated_failure_probability",
            "prediction_confidence",
            "combined_uncertainty",
            "ood_percentile",
            "uncertainty_adjusted_decision",
        },
        "Test uncertainty results",
    )

    similarity_df = load_csv(
        SIMILARITY_PATH,
        {
            ID_COLUMN,
            "historical_weighted_failure_rate",
        },
        "Historical-similarity summary",
    )

    sensor_columns = get_sensor_columns(
        development_df
    )

    LOGGER.info(
        "Building development sensor reference"
    )

    sensor_reference_df = (
        build_development_sensor_reference(
            development_df=development_df,
            sensor_columns=sensor_columns,
        )
    )

    global_importance_df = (
        prepare_global_importance(
            global_shap_df
        )
    )

    failure_recurrence_df = (
        prepare_failure_recurrence(
            failure_feature_df
        )
    )

    LOGGER.info(
        "Calculating wafer-level sensor investigation priorities"
    )

    priority_df = (
        build_sensor_priority_results(
            test_df=test_df,
            local_shap_df=local_shap_df,
            uncertainty_df=uncertainty_df,
            similarity_df=similarity_df,
            sensor_reference_df=(
                sensor_reference_df
            ),
            global_importance_df=(
                global_importance_df
            ),
            failure_recurrence_df=(
                failure_recurrence_df
            ),
        )
    )

    priority_df.to_csv(
        PRIORITY_RESULTS_PATH,
        index=False,
    )

    wafer_summary_df = build_wafer_summary(
        priority_df
    )

    wafer_summary_df.to_csv(
        WAFER_SUMMARY_PATH,
        index=False,
    )

    global_summary_df = (
        build_global_sensor_summary(
            priority_df
        )
    )

    global_summary_df.to_csv(
        GLOBAL_PRIORITY_PATH,
        index=False,
    )

    decision_summary_df = (
        build_decision_sensor_summary(
            priority_df
        )
    )

    decision_summary_df.to_csv(
        DECISION_PRIORITY_PATH,
        index=False,
    )

    failure_summary_df = (
        build_failure_sensor_summary(
            priority_df
        )
    )

    failure_summary_df.to_csv(
        FAILURE_PRIORITY_PATH,
        index=False,
    )

    LOGGER.info(
        "Generating investigation-priority figures"
    )

    save_global_priority_plot(
        global_summary_df
    )

    selected_wafer_ids = (
        select_local_plot_wafers(
            wafer_summary_df
        )
    )

    plotted_count = 0

    for wafer_id in selected_wafer_ids:
        wafer_priority_df = (
            priority_df.loc[
                priority_df[
                    ID_COLUMN
                ]
                == wafer_id
            ]
            .copy()
        )

        if wafer_priority_df.empty:
            continue

        save_local_priority_plot(
            wafer_id=wafer_id,
            wafer_df=wafer_priority_df,
        )

        plotted_count += 1

    summary = {
        "project": "HeveMind",
        "stage": (
            "Sensor investigation priority engine"
        ),
        "test_wafers_ranked": int(
            wafer_summary_df[
                ID_COLUMN
            ].nunique()
        ),
        "sensor_priority_rows": int(
            len(priority_df)
        ),
        "sensors_ranked_per_wafer": int(
            TOP_SENSORS_PER_WAFER
        ),
        "rank_aggregation_components": {
            "local_positive_shap_rank": (
                LOCAL_SHAP_RANK_WEIGHT
            ),
            "robust_deviation_rank": (
                ROBUST_DEVIATION_RANK_WEIGHT
            ),
            "global_importance_rank": (
                GLOBAL_IMPORTANCE_RANK_WEIGHT
            ),
            "failure_recurrence_rank": (
                FAILURE_RECURRENCE_RANK_WEIGHT
            ),
        },
        "priority_level_counts": (
            priority_df[
                "priority_level"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        ),
        "top_global_investigation_sensors": (
            global_summary_df
            .head(30)
            .to_dict(
                orient="records"
            )
        ),
        "top_failure_investigation_sensors": (
            failure_summary_df
            .head(30)
            .to_dict(
                orient="records"
            )
        ),
        "methodological_controls": {
            "development_statistics_only": True,
            "test_target_used_for_ranking": False,
            "physical_sensor_meanings_assigned": False,
            "priority_score_is_probability": False,
            "priority_score_is_causal_estimate": False,
        },
        "important_warning": (
            "The investigation priority score is a transparent "
            "rank aggregation of statistical evidence. It is not "
            "a failure probability, causal estimate, physical root "
            "cause, or verified maintenance recommendation."
        ),
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print_console_summary(
        priority_df=priority_df,
        wafer_summary_df=(
            wafer_summary_df
        ),
        global_summary_df=(
            global_summary_df
        ),
        plotted_count=plotted_count,
    )


if __name__ == "__main__":
    main()