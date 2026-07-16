from __future__ import annotations

import json
import logging
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.ensemble import BalancedRandomForestClassifier
from scipy.stats import mannwhitneyu
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


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
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VALIDATION_PATH = SPLITS_DIR / "validation.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

REPORTS_DIR = ROOT_DIR / "reports"

QUALITY_SCORES_PATH = (
    REPORTS_DIR
    / "quality_engine"
    / "tables"
    / "record_quality_scores.csv"
)

ANOMALY_SCORES_PATH = (
    REPORTS_DIR
    / "quality_engine"
    / "tables"
    / "anomaly_scores.csv"
)

OUTPUT_DIR = REPORTS_DIR / "feature_error_diagnostics"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODEL_DIR = (
    ARTIFACTS_DIR
    / "models"
    / "feature_augmented"
)

FEATURE_STATS_PATH = (
    ARTIFACTS_DIR
    / "metadata"
    / "feature_augmentation_statistics.json"
)

DIAGNOSTIC_SENSOR_PATH = (
    TABLES_DIR
    / "development_sensor_error_diagnostics.csv"
)

TOP_MISSED_FAILURE_FEATURES_PATH = (
    TABLES_DIR
    / "top_missed_failure_sensor_patterns.csv"
)

TOP_FALSE_ALARM_FEATURES_PATH = (
    TABLES_DIR
    / "top_false_alarm_sensor_patterns.csv"
)

MODEL_COMPARISON_PATH = (
    TABLES_DIR
    / "augmented_model_comparison.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    TABLES_DIR
    / "augmented_model_threshold_analysis.csv"
)

DEVELOPMENT_PREDICTIONS_PATH = (
    TABLES_DIR
    / "augmented_development_oof_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR
    / "augmented_test_predictions.csv"
)

TEST_METRICS_PATH = (
    TABLES_DIR
    / "augmented_test_metrics.csv"
)

BEST_MODEL_PATH = (
    MODEL_DIR
    / "best_feature_augmented_pipeline.joblib"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "feature_error_diagnostics_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
TIMESTAMP_COLUMN = "timestamp"
SENSOR_PREFIX = "sensor_"

RANDOM_STATE = 42
CV_SPLITS = 5

MINIMUM_FAILURE_RECALL = 0.75

MISSED_FAILURE_COST = 10.0
FALSE_ALARM_COST = 1.0

THRESHOLD_MINIMUM = 0.005
THRESHOLD_MAXIMUM = 0.995
THRESHOLD_STEP = 0.005

NUMBER_OF_SENSOR_GROUPS = 6
MINIMUM_GROUP_SIZE_FOR_TEST = 3

TOP_DIAGNOSTIC_FEATURES = 50


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        OUTPUT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        MODEL_DIR,
        FEATURE_STATS_PATH.parent,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# DATA LOADING
# ============================================================
def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset split not found: {path}"
        )

    dataframe = pd.read_parquet(path)

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_development_and_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)
    test_df = load_split(TEST_PATH)

    development_df = pd.concat(
        [
            train_df,
            validation_df,
        ],
        axis=0,
        ignore_index=True,
    )

    development_df = development_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return development_df, test_df


def load_quality_scores() -> pd.DataFrame:
    if not QUALITY_SCORES_PATH.exists():
        raise FileNotFoundError(
            "Quality-score file not found. "
            "Run src/07_quality_imputation_anomaly.py first."
        )

    dataframe = pd.read_csv(
        QUALITY_SCORES_PATH
    )

    required_columns = {
        ID_COLUMN,
        "dataset_split",
        "missing_sensor_count",
        "missing_sensor_rate",
        "observed_sensor_count",
        "mean_observed_sensor_reliability",
        "completeness_score",
        "data_quality_score",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Quality-score file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_anomaly_scores() -> pd.DataFrame:
    if not ANOMALY_SCORES_PATH.exists():
        raise FileNotFoundError(
            "Anomaly-score file not found. "
            "Run src/07_quality_imputation_anomaly.py first."
        )

    dataframe = pd.read_csv(
        ANOMALY_SCORES_PATH
    )

    required_columns = {
        ID_COLUMN,
        "dataset_split",
        "anomaly_score",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Anomaly-score file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


# ============================================================
# SENSOR UTILITIES
# ============================================================
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


def calculate_scale_pos_weight(
    target: pd.Series,
) -> float:
    negative_count = int(
        (target == 0).sum()
    )

    positive_count = int(
        (target == 1).sum()
    )

    if positive_count == 0:
        raise ValueError(
            "No failure samples were found."
        )

    return float(
        negative_count / positive_count
    )


# ============================================================
# QUALITY AND ANOMALY MERGE
# ============================================================
def merge_context_features(
    dataframe: pd.DataFrame,
    quality_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    """
    Merge quality and anomaly features without creating duplicate
    missingness columns.

    Some missingness fields may already exist in the processed SECOM
    dataset. Existing versions are removed before the quality-engine
    values are merged.
    """
    output = dataframe.copy()

    quality_split = quality_df.loc[
        quality_df["dataset_split"] == split_name
    ].copy()

    anomaly_split = anomaly_df.loc[
        anomaly_df["dataset_split"] == split_name
    ].copy()

    quality_columns = [
        ID_COLUMN,
        "missing_sensor_count",
        "missing_sensor_rate",
        "observed_sensor_count",
        "mean_observed_sensor_reliability",
        "completeness_score",
        "data_quality_score",
    ]

    anomaly_columns = [
        ID_COLUMN,
        "anomaly_score",
    ]

    required_quality_columns = set(
        quality_columns
    )

    missing_quality_columns = (
        required_quality_columns
        .difference(quality_split.columns)
    )

    if missing_quality_columns:
        raise ValueError(
            "Quality-score table is missing required columns: "
            f"{sorted(missing_quality_columns)}"
        )

    missing_anomaly_columns = (
        set(anomaly_columns)
        .difference(anomaly_split.columns)
    )

    if missing_anomaly_columns:
        raise ValueError(
            "Anomaly-score table is missing required columns: "
            f"{sorted(missing_anomaly_columns)}"
        )

    # Remove existing context columns before merging to prevent
    # pandas from creating _x and _y suffixes.
    existing_context_columns = [
        column
        for column in (
            quality_columns[1:]
            + anomaly_columns[1:]
        )
        if column in output.columns
    ]

    if existing_context_columns:
        LOGGER.info(
            "Removing existing context columns before merge: %s",
            existing_context_columns,
        )

        output = output.drop(
            columns=existing_context_columns
        )

    output = output.merge(
        quality_split[quality_columns],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    output = output.merge(
        anomaly_split[anomaly_columns],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    required_context_columns = [
        "missing_sensor_count",
        "missing_sensor_rate",
        "observed_sensor_count",
        "mean_observed_sensor_reliability",
        "completeness_score",
        "data_quality_score",
        "anomaly_score",
    ]

    missing_context_columns = [
        column
        for column in required_context_columns
        if column not in output.columns
    ]

    if missing_context_columns:
        raise ValueError(
            "Merged dataset is missing context columns: "
            f"{missing_context_columns}"
        )

    unmatched_rows = output[
        required_context_columns
    ].isna().all(axis=1)

    if unmatched_rows.any():
        unmatched_ids = output.loc[
            unmatched_rows,
            ID_COLUMN,
        ].head(10).tolist()

        raise ValueError(
            "Some wafer records could not be matched to quality "
            "and anomaly results. Example wafer IDs: "
            f"{unmatched_ids}"
        )

    return output


# ============================================================
# DEVELOPMENT STATISTICS
# ============================================================
def calculate_development_statistics(
    development_df: pd.DataFrame,
    sensor_columns: list[str],
) -> dict[str, dict[str, float]]:
    statistics: dict[str, dict[str, float]] = {}

    for column in sensor_columns:
        series = development_df[column]

        median = float(
            series.median()
        )

        q1 = float(
            series.quantile(0.25)
        )

        q3 = float(
            series.quantile(0.75)
        )

        iqr = float(
            q3 - q1
        )

        mean = float(
            series.mean()
        )

        standard_deviation = float(
            series.std(ddof=1)
        )

        if not np.isfinite(iqr):
            iqr = 0.0

        if not np.isfinite(standard_deviation):
            standard_deviation = 0.0

        statistics[column] = {
            "median": median,
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "mean": mean,
            "standard_deviation": standard_deviation,
        }

    return statistics


def save_development_statistics(
    statistics: dict[str, dict[str, float]],
) -> None:
    with FEATURE_STATS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            statistics,
            file,
            indent=4,
        )


# ============================================================
# SENSOR GROUPS
# ============================================================
def create_sensor_groups(
    sensor_columns: list[str],
) -> dict[str, list[str]]:
    grouped_arrays = np.array_split(
        np.array(sensor_columns),
        NUMBER_OF_SENSOR_GROUPS,
    )

    return {
        f"sensor_zone_{index + 1:02d}": (
            group.tolist()
        )
        for index, group in enumerate(
            grouped_arrays
        )
    }


# ============================================================
# FEATURE AUGMENTATION
# ============================================================
def create_augmented_features(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    development_statistics: dict[str, dict[str, float]],
    sensor_groups: dict[str, list[str]],
) -> pd.DataFrame:
    """
    Create augmented features using statistics fitted only on the
    development dataset.

    Important:
    - Sensor zones are anonymous analytical groups.
    - They must not be interpreted as verified semiconductor process stages.
    - No target information is used during feature construction.
    """
    missing_sensor_columns = [
        column
        for column in sensor_columns
        if column not in dataframe.columns
    ]

    if missing_sensor_columns:
        raise ValueError(
            "The input dataframe is missing sensor columns: "
            f"{missing_sensor_columns[:10]}"
        )

    required_context_columns = [
        "missing_sensor_count",
        "missing_sensor_rate",
        "observed_sensor_count",
        "mean_observed_sensor_reliability",
        "completeness_score",
        "data_quality_score",
        "anomaly_score",
    ]

    missing_context_columns = [
        column
        for column in required_context_columns
        if column not in dataframe.columns
    ]

    if missing_context_columns:
        raise ValueError(
            "The input dataframe is missing required context columns: "
            f"{missing_context_columns}"
        )

    missing_statistics = [
        column
        for column in sensor_columns
        if column not in development_statistics
    ]

    if missing_statistics:
        raise ValueError(
            "Development statistics are missing for sensor columns: "
            f"{missing_statistics[:10]}"
        )

    # ========================================================
    # BASE SENSOR MATRIX
    # ========================================================
    sensor_data = dataframe[
        sensor_columns
    ].copy()

    output_parts: list[pd.DataFrame] = [
        sensor_data.copy()
    ]

    # ========================================================
    # GLOBAL ROW-LEVEL SENSOR AGGREGATES
    # ========================================================
    aggregate_features = pd.DataFrame(
        index=dataframe.index
    )

    aggregate_features[
        "aggregate_sensor_mean"
    ] = sensor_data.mean(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_median"
    ] = sensor_data.median(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_std"
    ] = sensor_data.std(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_min"
    ] = sensor_data.min(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_max"
    ] = sensor_data.max(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_range"
    ] = (
        aggregate_features[
            "aggregate_sensor_max"
        ]
        - aggregate_features[
            "aggregate_sensor_min"
        ]
    )

    aggregate_features[
        "aggregate_sensor_abs_mean"
    ] = sensor_data.abs().mean(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_abs_median"
    ] = sensor_data.abs().median(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_missing_count"
    ] = sensor_data.isna().sum(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_missing_rate"
    ] = (
        aggregate_features[
            "aggregate_sensor_missing_count"
        ]
        / len(sensor_columns)
    )

    aggregate_features[
        "aggregate_sensor_observed_count"
    ] = (
        len(sensor_columns)
        - aggregate_features[
            "aggregate_sensor_missing_count"
        ]
    )

    aggregate_features[
        "aggregate_sensor_positive_count"
    ] = (
        sensor_data > 0
    ).sum(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_negative_count"
    ] = (
        sensor_data < 0
    ).sum(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_zero_count"
    ] = (
        sensor_data == 0
    ).sum(
        axis=1
    )

    aggregate_features[
        "aggregate_sensor_positive_rate"
    ] = (
        aggregate_features[
            "aggregate_sensor_positive_count"
        ]
        / np.maximum(
            aggregate_features[
                "aggregate_sensor_observed_count"
            ],
            1,
        )
    )

    aggregate_features[
        "aggregate_sensor_negative_rate"
    ] = (
        aggregate_features[
            "aggregate_sensor_negative_count"
        ]
        / np.maximum(
            aggregate_features[
                "aggregate_sensor_observed_count"
            ],
            1,
        )
    )

    output_parts.append(
        aggregate_features
    )

    # ========================================================
    # ROBUST SENSOR DEVIATION FEATURES
    # ========================================================
    robust_z_columns: dict[
        str,
        pd.Series,
    ] = {}

    extreme_deviation_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    moderate_deviation_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    mild_deviation_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    positive_extreme_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    negative_extreme_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    absolute_robust_z_sum = np.zeros(
        len(dataframe),
        dtype=float,
    )

    squared_robust_z_sum = np.zeros(
        len(dataframe),
        dtype=float,
    )

    valid_robust_z_count = np.zeros(
        len(dataframe),
        dtype=float,
    )

    for column in sensor_columns:
        statistics = development_statistics[
            column
        ]

        median = float(
            statistics["median"]
        )

        iqr = float(
            statistics["iqr"]
        )

        denominator = (
            iqr
            if np.isfinite(iqr)
            and abs(iqr) > 1e-12
            else 1.0
        )

        robust_z = (
            dataframe[column]
            - median
        ) / denominator

        robust_z_columns[
            column
        ] = robust_z

        valid_mask = robust_z.notna().to_numpy(
            dtype=float
        )

        robust_values = (
            robust_z
            .fillna(0.0)
            .to_numpy(
                dtype=float
            )
        )

        absolute_values = np.abs(
            robust_values
        )

        mild_deviation_count += (
            absolute_values >= 1.0
        ).astype(
            float
        )

        moderate_deviation_count += (
            absolute_values >= 1.5
        ).astype(
            float
        )

        extreme_deviation_count += (
            absolute_values >= 3.0
        ).astype(
            float
        )

        positive_extreme_count += (
            robust_values >= 3.0
        ).astype(
            float
        )

        negative_extreme_count += (
            robust_values <= -3.0
        ).astype(
            float
        )

        absolute_robust_z_sum += (
            absolute_values
        )

        squared_robust_z_sum += (
            robust_values ** 2
        )

        valid_robust_z_count += (
            valid_mask
        )

    robust_z_data = pd.DataFrame(
        robust_z_columns,
        index=dataframe.index,
    )

    robust_features = pd.DataFrame(
        index=dataframe.index
    )

    robust_features[
        "robust_mild_sensor_count"
    ] = mild_deviation_count

    robust_features[
        "robust_moderate_sensor_count"
    ] = moderate_deviation_count

    robust_features[
        "robust_extreme_sensor_count"
    ] = extreme_deviation_count

    robust_features[
        "robust_positive_extreme_count"
    ] = positive_extreme_count

    robust_features[
        "robust_negative_extreme_count"
    ] = negative_extreme_count

    robust_features[
        "robust_extreme_direction_balance"
    ] = (
        positive_extreme_count
        - negative_extreme_count
    )

    robust_features[
        "robust_mild_sensor_rate"
    ] = (
        mild_deviation_count
        / np.maximum(
            valid_robust_z_count,
            1.0,
        )
    )

    robust_features[
        "robust_moderate_sensor_rate"
    ] = (
        moderate_deviation_count
        / np.maximum(
            valid_robust_z_count,
            1.0,
        )
    )

    robust_features[
        "robust_extreme_sensor_rate"
    ] = (
        extreme_deviation_count
        / np.maximum(
            valid_robust_z_count,
            1.0,
        )
    )

    robust_features[
        "mean_absolute_robust_z"
    ] = (
        absolute_robust_z_sum
        / np.maximum(
            valid_robust_z_count,
            1.0,
        )
    )

    robust_features[
        "root_mean_square_robust_z"
    ] = np.sqrt(
        squared_robust_z_sum
        / np.maximum(
            valid_robust_z_count,
            1.0,
        )
    )

    robust_features[
        "maximum_absolute_robust_z"
    ] = (
        robust_z_data
        .abs()
        .max(
            axis=1
        )
    )

    robust_features[
        "median_absolute_robust_z"
    ] = (
        robust_z_data
        .abs()
        .median(
            axis=1
        )
    )

    robust_features[
        "robust_z_standard_deviation"
    ] = robust_z_data.std(
        axis=1
    )

    output_parts.append(
        robust_features
    )

    # ========================================================
    # SENSOR-ZONE AGGREGATE FEATURES
    # ========================================================
    zone_feature_frames: list[
        pd.DataFrame
    ] = []

    for (
        group_name,
        group_columns,
    ) in sensor_groups.items():
        valid_group_columns = [
            column
            for column in group_columns
            if column in sensor_data.columns
        ]

        if not valid_group_columns:
            continue

        group_data = sensor_data[
            valid_group_columns
        ]

        group_robust_z = robust_z_data[
            valid_group_columns
        ]

        zone_features = pd.DataFrame(
            index=dataframe.index
        )

        zone_features[
            f"{group_name}_mean"
        ] = group_data.mean(
            axis=1
        )

        zone_features[
            f"{group_name}_median"
        ] = group_data.median(
            axis=1
        )

        zone_features[
            f"{group_name}_std"
        ] = group_data.std(
            axis=1
        )

        zone_features[
            f"{group_name}_min"
        ] = group_data.min(
            axis=1
        )

        zone_features[
            f"{group_name}_max"
        ] = group_data.max(
            axis=1
        )

        zone_features[
            f"{group_name}_range"
        ] = (
            zone_features[
                f"{group_name}_max"
            ]
            - zone_features[
                f"{group_name}_min"
            ]
        )

        zone_features[
            f"{group_name}_abs_mean"
        ] = group_data.abs().mean(
            axis=1
        )

        zone_features[
            f"{group_name}_missing_count"
        ] = group_data.isna().sum(
            axis=1
        )

        zone_features[
            f"{group_name}_missing_rate"
        ] = group_data.isna().mean(
            axis=1
        )

        zone_features[
            f"{group_name}_observed_count"
        ] = group_data.notna().sum(
            axis=1
        )

        zone_features[
            f"{group_name}_mean_abs_robust_z"
        ] = (
            group_robust_z
            .abs()
            .mean(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_median_abs_robust_z"
        ] = (
            group_robust_z
            .abs()
            .median(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_max_abs_robust_z"
        ] = (
            group_robust_z
            .abs()
            .max(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_mild_count"
        ] = (
            group_robust_z
            .abs()
            .ge(1.0)
            .sum(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_moderate_count"
        ] = (
            group_robust_z
            .abs()
            .ge(1.5)
            .sum(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_extreme_count"
        ] = (
            group_robust_z
            .abs()
            .ge(3.0)
            .sum(
                axis=1
            )
        )

        zone_features[
            f"{group_name}_extreme_rate"
        ] = (
            zone_features[
                f"{group_name}_extreme_count"
            ]
            / np.maximum(
                zone_features[
                    f"{group_name}_observed_count"
                ],
                1,
            )
        )

        zone_feature_frames.append(
            zone_features
        )

    output_parts.extend(
        zone_feature_frames
    )

    # ========================================================
    # CONTEXT FEATURES
    # ========================================================
    context_features = dataframe[
        required_context_columns
    ].copy()

    output_parts.append(
        context_features
    )

    # ========================================================
    # INTERACTION FEATURES
    # ========================================================
    interaction_features = pd.DataFrame(
        index=dataframe.index
    )

    interaction_features[
        "anomaly_quality_interaction"
    ] = (
        dataframe["anomaly_score"]
        * (
            1.0
            - dataframe[
                "data_quality_score"
            ]
        )
    )

    interaction_features[
        "anomaly_completeness_interaction"
    ] = (
        dataframe["anomaly_score"]
        * dataframe[
            "completeness_score"
        ]
    )

    interaction_features[
        "anomaly_missingness_interaction"
    ] = (
        dataframe["anomaly_score"]
        * dataframe[
            "missing_sensor_rate"
        ]
    )

    interaction_features[
        "quality_missingness_interaction"
    ] = (
        dataframe["data_quality_score"]
        * dataframe[
            "missing_sensor_rate"
        ]
    )

    interaction_features[
        "reliability_anomaly_interaction"
    ] = (
        dataframe[
            "mean_observed_sensor_reliability"
        ]
        * dataframe[
            "anomaly_score"
        ]
    )

    interaction_features[
        "extreme_quality_interaction"
    ] = (
        robust_features[
            "robust_extreme_sensor_count"
        ]
        * dataframe[
            "data_quality_score"
        ]
    )

    interaction_features[
        "extreme_anomaly_interaction"
    ] = (
        robust_features[
            "robust_extreme_sensor_count"
        ]
        * dataframe[
            "anomaly_score"
        ]
    )

    interaction_features[
        "deviation_anomaly_interaction"
    ] = (
        robust_features[
            "mean_absolute_robust_z"
        ]
        * dataframe[
            "anomaly_score"
        ]
    )

    interaction_features[
        "deviation_quality_interaction"
    ] = (
        robust_features[
            "mean_absolute_robust_z"
        ]
        * dataframe[
            "data_quality_score"
        ]
    )

    interaction_features[
        "range_anomaly_interaction"
    ] = (
        aggregate_features[
            "aggregate_sensor_range"
        ]
        * dataframe[
            "anomaly_score"
        ]
    )

    interaction_features[
        "variability_anomaly_interaction"
    ] = (
        aggregate_features[
            "aggregate_sensor_std"
        ]
        * dataframe[
            "anomaly_score"
        ]
    )

    interaction_features[
        "missingness_extreme_interaction"
    ] = (
        dataframe[
            "missing_sensor_rate"
        ]
        * robust_features[
            "robust_extreme_sensor_count"
        ]
    )

    output_parts.append(
        interaction_features
    )

    # ========================================================
    # FINAL CONCATENATION
    # ========================================================
    output = pd.concat(
        output_parts,
        axis=1,
    )

    duplicated_columns = output.columns[
        output.columns.duplicated()
    ].tolist()

    if duplicated_columns:
        raise ValueError(
            "Duplicate augmented feature names were created: "
            f"{duplicated_columns[:20]}"
        )

    output = output.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    all_missing_columns = [
        column
        for column in output.columns
        if output[column].isna().all()
    ]

    if all_missing_columns:
        LOGGER.warning(
            "Dropping %s augmented features containing only missing values",
            len(all_missing_columns),
        )

        output = output.drop(
            columns=all_missing_columns
        )

    LOGGER.info(
        "Created augmented feature matrix with %s rows and %s columns",
        output.shape[0],
        output.shape[1],
    )

    return output

# ============================================================
# MODEL REGISTRY
# ============================================================
def build_model_registry(
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    baseline_random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                BalancedRandomForestClassifier(
                    n_estimators=700,
                    max_depth=None,
                    min_samples_split=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    sampling_strategy="all",
                    replacement=True,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    augmented_random_forest = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                BalancedRandomForestClassifier(
                    n_estimators=800,
                    max_depth=None,
                    min_samples_split=6,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    sampling_strategy="all",
                    replacement=True,
                    bootstrap=False,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    augmented_xgboost = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
            (
                "model",
                XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    n_estimators=700,
                    learning_rate=0.025,
                    max_depth=4,
                    min_child_weight=3,
                    subsample=0.85,
                    colsample_bytree=0.70,
                    gamma=0.10,
                    reg_alpha=0.10,
                    reg_lambda=2.0,
                    scale_pos_weight=scale_pos_weight,
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return {
        "Baseline Balanced Random Forest": (
            baseline_random_forest
        ),
        "Augmented Balanced Random Forest": (
            augmented_random_forest
        ),
        "Augmented XGBoost": (
            augmented_xgboost
        ),
    }


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================
def generate_oof_probabilities(
    model: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
) -> np.ndarray:
    cross_validator = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        estimator=clone(model),
        X=x_data,
        y=y_data,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return probabilities


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================
def calculate_threshold_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    balanced_accuracy = balanced_accuracy_score(
        y_true,
        predictions,
    )

    operational_cost = (
        MISSED_FAILURE_COST * fn
        + FALSE_ALARM_COST * fp
    )

    false_alarms_per_detection = (
        float(fp / tp)
        if tp > 0
        else np.inf
    )

    return {
        "threshold": float(threshold),
        "recall_fail": float(recall),
        "precision_fail": float(precision),
        "f1_fail": float(f1),
        "balanced_accuracy": float(
            balanced_accuracy
        ),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_alarms_per_detected_failure": (
            false_alarms_per_detection
        ),
        "operational_cost": float(
            operational_cost
        ),
    }


def build_threshold_table(
    model_name: str,
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for threshold in np.arange(
        THRESHOLD_MINIMUM,
        THRESHOLD_MAXIMUM + THRESHOLD_STEP,
        THRESHOLD_STEP,
    ):
        metrics = calculate_threshold_metrics(
            y_true=y_true.to_numpy(),
            probabilities=probabilities,
            threshold=float(threshold),
        )

        metrics["model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows)


def select_operational_threshold(
    threshold_table: pd.DataFrame,
) -> dict[str, Any]:
    eligible = threshold_table.loc[
        threshold_table["recall_fail"]
        >= MINIMUM_FAILURE_RECALL
    ].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
            by=[
                "operational_cost",
                "false_positive",
                "precision_fail",
                "balanced_accuracy",
                "threshold",
            ],
            ascending=[
                True,
                True,
                False,
                False,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "Minimum operational cost while maintaining "
            f"failure recall >= {MINIMUM_FAILURE_RECALL:.2f}"
        )

    else:
        selected = threshold_table.sort_values(
            by=[
                "recall_fail",
                "operational_cost",
                "false_positive",
            ],
            ascending=[
                False,
                True,
                True,
            ],
        ).iloc[0]

        selection_reason = (
            "No threshold achieved the minimum recall requirement"
        )

    result = selected.to_dict()
    result["selection_reason"] = selection_reason

    return result


# ============================================================
# OOF ERROR GROUPS
# ============================================================
def assign_error_groups(
    target: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.Series:
    predictions = (
        probabilities >= threshold
    ).astype(int)

    conditions = [
        (
            (target.to_numpy() == 1)
            & (predictions == 1)
        ),
        (
            (target.to_numpy() == 1)
            & (predictions == 0)
        ),
        (
            (target.to_numpy() == 0)
            & (predictions == 1)
        ),
        (
            (target.to_numpy() == 0)
            & (predictions == 0)
        ),
    ]

    labels = [
        "Detected Failure",
        "Missed Failure",
        "False Alarm",
        "Correct Pass",
    ]

    return pd.Series(
        np.select(
            conditions,
            labels,
            default="Unknown",
        ),
        index=target.index,
        name="error_group",
    )


# ============================================================
# EFFECT SIZE
# ============================================================
def calculate_rank_biserial_effect(
    group_a: pd.Series,
    group_b: pd.Series,
) -> float:
    group_a = group_a.dropna()
    group_b = group_b.dropna()

    if (
        len(group_a) < MINIMUM_GROUP_SIZE_FOR_TEST
        or len(group_b) < MINIMUM_GROUP_SIZE_FOR_TEST
    ):
        return np.nan

    result = mannwhitneyu(
        group_a,
        group_b,
        alternative="two-sided",
    )

    u_value = float(
        result.statistic
    )

    sample_product = (
        len(group_a)
        * len(group_b)
    )

    rank_biserial = (
        2.0
        * u_value
        / sample_product
        - 1.0
    )

    return float(
        rank_biserial
    )


def calculate_group_comparison(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
    error_groups: pd.Series,
    group_a_name: str,
    group_b_name: str,
    comparison_name: str,
) -> pd.DataFrame:
    group_a_mask = (
        error_groups == group_a_name
    )

    group_b_mask = (
        error_groups == group_b_name
    )

    rows: list[dict[str, Any]] = []

    for column in feature_columns:
        group_a = dataframe.loc[
            group_a_mask,
            column,
        ].dropna()

        group_b = dataframe.loc[
            group_b_mask,
            column,
        ].dropna()

        if (
            len(group_a) < MINIMUM_GROUP_SIZE_FOR_TEST
            or len(group_b) < MINIMUM_GROUP_SIZE_FOR_TEST
        ):
            p_value = np.nan
            effect_size = np.nan

        else:
            test_result = mannwhitneyu(
                group_a,
                group_b,
                alternative="two-sided",
            )

            p_value = float(
                test_result.pvalue
            )

            effect_size = (
                calculate_rank_biserial_effect(
                    group_a,
                    group_b,
                )
            )

        median_a = (
            float(group_a.median())
            if not group_a.empty
            else np.nan
        )

        median_b = (
            float(group_b.median())
            if not group_b.empty
            else np.nan
        )

        rows.append(
            {
                "comparison": comparison_name,
                "feature": column,
                "group_a": group_a_name,
                "group_b": group_b_name,
                "group_a_count": int(
                    len(group_a)
                ),
                "group_b_count": int(
                    len(group_b)
                ),
                "group_a_median": median_a,
                "group_b_median": median_b,
                "median_difference": (
                    median_a - median_b
                    if (
                        np.isfinite(median_a)
                        and np.isfinite(median_b)
                    )
                    else np.nan
                ),
                "rank_biserial_effect": effect_size,
                "absolute_effect": (
                    abs(effect_size)
                    if np.isfinite(effect_size)
                    else np.nan
                ),
                "mann_whitney_p_value": p_value,
            }
        )

    return pd.DataFrame(rows)


def perform_sensor_error_diagnostics(
    development_df: pd.DataFrame,
    sensor_columns: list[str],
    error_groups: pd.Series,
) -> pd.DataFrame:
    missed_vs_detected = calculate_group_comparison(
        dataframe=development_df,
        feature_columns=sensor_columns,
        error_groups=error_groups,
        group_a_name="Missed Failure",
        group_b_name="Detected Failure",
        comparison_name=(
            "Missed Failure vs Detected Failure"
        ),
    )

    false_alarm_vs_correct_pass = (
        calculate_group_comparison(
            dataframe=development_df,
            feature_columns=sensor_columns,
            error_groups=error_groups,
            group_a_name="False Alarm",
            group_b_name="Correct Pass",
            comparison_name=(
                "False Alarm vs Correct Pass"
            ),
        )
    )

    diagnostics = pd.concat(
        [
            missed_vs_detected,
            false_alarm_vs_correct_pass,
        ],
        axis=0,
        ignore_index=True,
    )

    diagnostics["adjusted_significance_flag"] = (
        diagnostics[
            "mann_whitney_p_value"
        ]
        < (
            0.05
            / max(
                len(sensor_columns),
                1,
            )
        )
    )

    return diagnostics.sort_values(
        by=[
            "comparison",
            "absolute_effect",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)


# ============================================================
# MODEL BENCHMARK
# ============================================================
def benchmark_models(
    models: dict[str, Pipeline],
    x_baseline: pd.DataFrame,
    x_augmented: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, float],
]:
    rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []

    probability_mapping: dict[
        str,
        np.ndarray,
    ] = {}

    threshold_mapping: dict[
        str,
        float,
    ] = {}

    for model_name, model in models.items():
        LOGGER.info(
            "Evaluating model: %s",
            model_name,
        )

        if model_name == (
            "Baseline Balanced Random Forest"
        ):
            x_data = x_baseline
        else:
            x_data = x_augmented

        probabilities = generate_oof_probabilities(
            model=model,
            x_data=x_data,
            y_data=target,
        )

        threshold_table = build_threshold_table(
            model_name=model_name,
            y_true=target,
            probabilities=probabilities,
        )

        threshold_result = (
            select_operational_threshold(
                threshold_table
            )
        )

        threshold = float(
            threshold_result["threshold"]
        )

        metrics = calculate_threshold_metrics(
            y_true=target.to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )

        rows.append(
            {
                "model": model_name,
                "feature_set": (
                    "Raw Sensors"
                    if model_name
                    == "Baseline Balanced Random Forest"
                    else "Augmented"
                ),
                "roc_auc": float(
                    roc_auc_score(
                        target,
                        probabilities,
                    )
                ),
                "pr_auc": float(
                    average_precision_score(
                        target,
                        probabilities,
                    )
                ),
                "brier_score": float(
                    brier_score_loss(
                        target,
                        probabilities,
                    )
                ),
                **metrics,
                "selection_reason": (
                    threshold_result[
                        "selection_reason"
                    ]
                ),
            }
        )

        threshold_tables.append(
            threshold_table
        )

        probability_mapping[
            model_name
        ] = probabilities

        threshold_mapping[
            model_name
        ] = threshold

    comparison_df = pd.DataFrame(rows)

    comparison_df = comparison_df.sort_values(
        by=[
            "operational_cost",
            "false_positive",
            "precision_fail",
            "pr_auc",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    all_thresholds_df = pd.concat(
        threshold_tables,
        axis=0,
        ignore_index=True,
    )

    return (
        comparison_df,
        all_thresholds_df,
        probability_mapping,
        threshold_mapping,
    )


def select_best_model(
    comparison_df: pd.DataFrame,
) -> str:
    eligible = comparison_df.loc[
        comparison_df["recall_fail"]
        >= MINIMUM_FAILURE_RECALL
    ].copy()

    if eligible.empty:
        eligible = comparison_df.copy()

    selected = eligible.sort_values(
        by=[
            "operational_cost",
            "false_positive",
            "precision_fail",
            "balanced_accuracy",
            "pr_auc",
        ],
        ascending=[
            True,
            True,
            False,
            False,
            False,
        ],
    ).iloc[0]

    return str(
        selected["model"]
    )


# ============================================================
# TEST EVALUATION
# ============================================================
def evaluate_final_model(
    model: Pipeline,
    x_development: pd.DataFrame,
    y_development: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[
    Pipeline,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    final_model = clone(
        model
    )

    final_model.fit(
        x_development,
        y_development,
    )

    probabilities = final_model.predict_proba(
        x_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    metrics = calculate_threshold_metrics(
        y_true=y_test.to_numpy(),
        probabilities=probabilities,
        threshold=threshold,
    )

    metrics.update(
        {
            "roc_auc": float(
                roc_auc_score(
                    y_test,
                    probabilities,
                )
            ),
            "pr_auc": float(
                average_precision_score(
                    y_test,
                    probabilities,
                )
            ),
            "brier_score": float(
                brier_score_loss(
                    y_test,
                    probabilities,
                )
            ),
        }
    )

    return (
        final_model,
        metrics,
        probabilities,
        predictions,
    )


# ============================================================
# OUTPUT TABLES
# ============================================================
def build_development_prediction_table(
    development_df: pd.DataFrame,
    probability_mapping: dict[str, np.ndarray],
    threshold_mapping: dict[str, float],
) -> pd.DataFrame:
    output = development_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    for model_name, probabilities in probability_mapping.items():
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        threshold = threshold_mapping[
            model_name
        ]

        output[
            f"{safe_name}_fail_probability"
        ] = probabilities

        output[
            f"{safe_name}_predicted_target"
        ] = (
            probabilities >= threshold
        ).astype(int)

        output[
            f"{safe_name}_threshold"
        ] = threshold

    return output


def build_test_prediction_table(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    predictions: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    output = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
            "data_quality_score",
            "anomaly_score",
            "missing_sensor_rate",
        ]
    ].copy()

    output["fail_probability"] = probabilities
    output["predicted_target"] = predictions

    output["predicted_status"] = np.where(
        predictions == 1,
        "Fail",
        "Pass",
    )

    output["decision_threshold"] = threshold
    output["model_name"] = model_name

    output["error_type"] = np.select(
        [
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output["predicted_target"] == 0
                )
            ),
            (
                (output[TARGET_COLUMN] == 0)
                & (
                    output["predicted_target"] == 1
                )
            ),
        ],
        [
            "Missed Failure",
            "False Alarm",
        ],
        default="Correct",
    )

    return output


# ============================================================
# FIGURES
# ============================================================
def save_model_comparison_plot(
    comparison_df: pd.DataFrame,
) -> None:
    plot_df = comparison_df.sort_values(
        by="operational_cost",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        plot_df["model"],
        plot_df["operational_cost"],
    )

    axis.set_title(
        "Operational Cost After Feature Augmentation"
    )

    axis.set_xlabel(
        "Weighted Operational Cost"
    )

    axis.set_ylabel(
        "Model"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "augmented_model_operational_cost.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_false_alarm_plot(
    comparison_df: pd.DataFrame,
) -> None:
    plot_df = comparison_df.sort_values(
        by="false_positive",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        plot_df["model"],
        plot_df["false_positive"],
    )

    axis.set_title(
        "False Alarms After Feature Augmentation"
    )

    axis.set_xlabel(
        "False Alarms"
    )

    axis.set_ylabel(
        "Model"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "augmented_model_false_alarms.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_sensor_effect_plot(
    diagnostics_df: pd.DataFrame,
    comparison_name: str,
    output_filename: str,
) -> None:
    plot_df = diagnostics_df.loc[
        diagnostics_df["comparison"]
        == comparison_name
    ].copy()

    plot_df = (
        plot_df
        .dropna(
            subset=["absolute_effect"]
        )
        .head(25)
        .sort_values(
            by="absolute_effect",
            ascending=True,
        )
    )

    if plot_df.empty:
        return

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.barh(
        plot_df["feature"],
        plot_df["absolute_effect"],
    )

    axis.set_title(
        comparison_name
    )

    axis.set_xlabel(
        "Absolute Rank-Biserial Effect Size"
    )

    axis.set_ylabel(
        "Sensor"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / output_filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# SUMMARY
# ============================================================
def save_summary(
    payload: dict[str, Any],
) -> None:
    with SUMMARY_PATH.open(
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
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading development, test, quality and anomaly data"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    quality_df = load_quality_scores()
    anomaly_df = load_anomaly_scores()

    development_df = merge_context_features(
        dataframe=development_df,
        quality_df=quality_df,
        anomaly_df=anomaly_df,
        split_name="development",
    )

    test_df = merge_context_features(
        dataframe=test_df,
        quality_df=quality_df,
        anomaly_df=anomaly_df,
        split_name="test",
    )

    sensor_columns = get_sensor_columns(
        development_df
    )

    y_development = development_df[
        TARGET_COLUMN
    ].astype(int)

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    LOGGER.info(
        "Calculating development-only sensor statistics"
    )

    development_statistics = (
        calculate_development_statistics(
            development_df,
            sensor_columns,
        )
    )

    save_development_statistics(
        development_statistics
    )

    sensor_groups = create_sensor_groups(
        sensor_columns
    )

    LOGGER.info(
        "Creating baseline and augmented feature matrices"
    )

    x_development_baseline = development_df[
        sensor_columns
    ].copy()

    x_test_baseline = test_df[
        sensor_columns
    ].copy()

    x_development_augmented = (
        create_augmented_features(
            dataframe=development_df,
            sensor_columns=sensor_columns,
            development_statistics=(
                development_statistics
            ),
            sensor_groups=sensor_groups,
        )
    )

    x_test_augmented = create_augmented_features(
        dataframe=test_df,
        sensor_columns=sensor_columns,
        development_statistics=(
            development_statistics
        ),
        sensor_groups=sensor_groups,
    )

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_development
        )
    )

    models = build_model_registry(
        scale_pos_weight=scale_pos_weight
    )

    (
        comparison_df,
        threshold_analysis_df,
        probability_mapping,
        threshold_mapping,
    ) = benchmark_models(
        models=models,
        x_baseline=x_development_baseline,
        x_augmented=x_development_augmented,
        target=y_development,
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    threshold_analysis_df.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    selected_model_name = select_best_model(
        comparison_df
    )

    selected_threshold = float(
        threshold_mapping[
            selected_model_name
        ]
    )

    LOGGER.info(
        "Selected model: %s",
        selected_model_name,
    )

    development_prediction_table = (
        build_development_prediction_table(
            development_df=development_df,
            probability_mapping=probability_mapping,
            threshold_mapping=threshold_mapping,
        )
    )

    development_prediction_table.to_csv(
        DEVELOPMENT_PREDICTIONS_PATH,
        index=False,
    )

    diagnostic_model_name = (
        "Baseline Balanced Random Forest"
    )

    diagnostic_probabilities = probability_mapping[
        diagnostic_model_name
    ]

    diagnostic_threshold = threshold_mapping[
        diagnostic_model_name
    ]

    error_groups = assign_error_groups(
        target=y_development,
        probabilities=diagnostic_probabilities,
        threshold=diagnostic_threshold,
    )

    LOGGER.info(
        "Performing development-only sensor error diagnostics"
    )

    diagnostic_df = perform_sensor_error_diagnostics(
        development_df=development_df,
        sensor_columns=sensor_columns,
        error_groups=error_groups,
    )

    diagnostic_df.to_csv(
        DIAGNOSTIC_SENSOR_PATH,
        index=False,
    )

    missed_failure_features = (
        diagnostic_df.loc[
            diagnostic_df["comparison"]
            == "Missed Failure vs Detected Failure"
        ]
        .sort_values(
            by="absolute_effect",
            ascending=False,
        )
        .head(TOP_DIAGNOSTIC_FEATURES)
    )

    missed_failure_features.to_csv(
        TOP_MISSED_FAILURE_FEATURES_PATH,
        index=False,
    )

    false_alarm_features = (
        diagnostic_df.loc[
            diagnostic_df["comparison"]
            == "False Alarm vs Correct Pass"
        ]
        .sort_values(
            by="absolute_effect",
            ascending=False,
        )
        .head(TOP_DIAGNOSTIC_FEATURES)
    )

    false_alarm_features.to_csv(
        TOP_FALSE_ALARM_FEATURES_PATH,
        index=False,
    )

    if selected_model_name == (
        "Baseline Balanced Random Forest"
    ):
        x_selected_development = (
            x_development_baseline
        )

        x_selected_test = (
            x_test_baseline
        )

    else:
        x_selected_development = (
            x_development_augmented
        )

        x_selected_test = (
            x_test_augmented
        )

    (
        final_model,
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_final_model(
        model=models[selected_model_name],
        x_development=x_selected_development,
        y_development=y_development,
        x_test=x_selected_test,
        y_test=y_test,
        threshold=selected_threshold,
    )

    test_prediction_table = (
        build_test_prediction_table(
            test_df=test_df,
            probabilities=test_probabilities,
            predictions=test_predictions,
            threshold=selected_threshold,
            model_name=selected_model_name,
        )
    )

    test_prediction_table.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    test_metric_row = {
        "model": selected_model_name,
        "threshold": selected_threshold,
        **test_metrics,
    }

    pd.DataFrame(
        [test_metric_row]
    ).to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    joblib.dump(
        final_model,
        BEST_MODEL_PATH,
    )

    save_model_comparison_plot(
        comparison_df
    )

    save_false_alarm_plot(
        comparison_df
    )

    save_sensor_effect_plot(
        diagnostics_df=diagnostic_df,
        comparison_name=(
            "Missed Failure vs Detected Failure"
        ),
        output_filename=(
            "missed_vs_detected_sensor_effects.png"
        ),
    )

    save_sensor_effect_plot(
        diagnostics_df=diagnostic_df,
        comparison_name=(
            "False Alarm vs Correct Pass"
        ),
        output_filename=(
            "false_alarm_vs_correct_pass_effects.png"
        ),
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Feature-level error diagnostics "
            "and augmented modelling"
        ),
        "development_rows": int(
            len(development_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "raw_sensor_count": int(
            len(sensor_columns)
        ),
        "augmented_feature_count": int(
            x_development_augmented.shape[1]
        ),
        "selected_model": (
            selected_model_name
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "development_comparison": (
            comparison_df.to_dict(
                orient="records"
            )
        ),
        "held_out_test_metrics": (
            test_metrics
        ),
        "diagnostic_model": (
            diagnostic_model_name
        ),
        "diagnostic_warning": (
            "Sensor error diagnostics were performed using "
            "development out-of-fold predictions. The held-out "
            "test set was not used for feature selection."
        ),
        "sensor_group_warning": (
            "Sensor zones are analytical index groups only. "
            "They do not represent verified semiconductor "
            "process stages."
        ),
        "top_missed_failure_patterns": (
            missed_failure_features
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
        "top_false_alarm_patterns": (
            false_alarm_features
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
    }

    save_summary(
        summary
    )

    print("\n" + "=" * 118)
    print(
        "HEVEMIND FEATURE-LEVEL ERROR DIAGNOSTICS"
    )
    print("=" * 118)

    print("\nDevelopment model comparison:")

    display_columns = [
        "model",
        "feature_set",
        "roc_auc",
        "pr_auc",
        "threshold",
        "recall_fail",
        "precision_fail",
        "f1_fail",
        "balanced_accuracy",
        "false_positive",
        "false_negative",
        "false_alarms_per_detected_failure",
        "operational_cost",
    ]

    print(
        comparison_df[
            display_columns
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nSelected model:")

    print(
        f"Model:                        "
        f"{selected_model_name}"
    )

    print(
        f"Operational threshold:        "
        f"{selected_threshold:.4f}"
    )

    print("\nHeld-out test performance:")

    print(
        f"ROC-AUC:                      "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                       "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:                  "
        f"{test_metrics['brier_score']:.4f}"
    )

    print(
        f"Failure recall:               "
        f"{test_metrics['recall_fail']:.4f}"
    )

    print(
        f"Failure precision:            "
        f"{test_metrics['precision_fail']:.4f}"
    )

    print(
        f"Failure F1:                   "
        f"{test_metrics['f1_fail']:.4f}"
    )

    print(
        f"Balanced accuracy:            "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"True Pass:                    "
        f"{test_metrics['true_negative']}"
    )

    print(
        f"False Alarm:                  "
        f"{test_metrics['false_positive']}"
    )

    print(
        f"Missed Failure:               "
        f"{test_metrics['false_negative']}"
    )

    print(
        f"Detected Failure:             "
        f"{test_metrics['true_positive']}"
    )

    print(
        f"False alarms per detection:   "
        f"{test_metrics['false_alarms_per_detected_failure']:.4f}"
    )

    print(
        f"Weighted operational cost:    "
        f"{test_metrics['operational_cost']:.4f}"
    )

    print("\nSaved outputs:")

    print(
        f"Diagnostic sensor table:      "
        f"{DIAGNOSTIC_SENSOR_PATH}"
    )

    print(
        f"Model comparison:             "
        f"{MODEL_COMPARISON_PATH}"
    )

    print(
        f"Test predictions:             "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"Selected pipeline:            "
        f"{BEST_MODEL_PATH}"
    )

    print(
        f"Report directory:             "
        f"{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()