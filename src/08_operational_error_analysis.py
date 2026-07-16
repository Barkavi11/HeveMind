from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

TEST_DATA_PATH = SPLITS_DIR / "test.parquet"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

OPERATIONAL_MODEL_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "operational"
    / "best_operational_pipeline.joblib"
)

REPORTS_DIR = ROOT_DIR / "reports"

OPERATIONAL_PREDICTIONS_PATH = (
    REPORTS_DIR
    / "operational_model_selection"
    / "tables"
    / "selected_model_test_predictions.csv"
)

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

OUTPUT_DIR = REPORTS_DIR / "operational_error_analysis"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

MASTER_ERROR_TABLE_PATH = (
    TABLES_DIR / "test_error_analysis_master.csv"
)

ERROR_TYPE_SUMMARY_PATH = (
    TABLES_DIR / "error_type_summary.csv"
)

QUALITY_SEGMENT_METRICS_PATH = (
    TABLES_DIR / "quality_segment_metrics.csv"
)

MISSINGNESS_SEGMENT_METRICS_PATH = (
    TABLES_DIR / "missingness_segment_metrics.csv"
)

ANOMALY_SEGMENT_METRICS_PATH = (
    TABLES_DIR / "anomaly_segment_metrics.csv"
)

CONFIDENCE_SEGMENT_METRICS_PATH = (
    TABLES_DIR / "confidence_segment_metrics.csv"
)

TIME_SEGMENT_METRICS_PATH = (
    TABLES_DIR / "time_segment_metrics.csv"
)

TOP_FALSE_NEGATIVES_PATH = (
    TABLES_DIR / "missed_failures.csv"
)

TOP_FALSE_POSITIVES_PATH = (
    TABLES_DIR / "highest_confidence_false_alarms.csv"
)

PERMUTATION_IMPORTANCE_PATH = (
    TABLES_DIR / "permutation_importance_diagnostic.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR / "operational_error_analysis_summary.json"
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

PERMUTATION_REPEATS = 20
TOP_FEATURES_TO_SAVE = 50


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
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {TEST_DATA_PATH}"
        )

    dataframe = pd.read_parquet(
        TEST_DATA_PATH
    )

    dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
        dataframe[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    return dataframe


def load_operational_predictions() -> pd.DataFrame:
    if not OPERATIONAL_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "Operational prediction file was not found.\n"
            "Run src/05_operational_model_selection.py first."
        )

    predictions = pd.read_csv(
        OPERATIONAL_PREDICTIONS_PATH
    )

    predictions[TIMESTAMP_COLUMN] = pd.to_datetime(
        predictions[TIMESTAMP_COLUMN],
        errors="coerce",
    )

    return predictions


def load_quality_scores() -> pd.DataFrame:
    if not QUALITY_SCORES_PATH.exists():
        raise FileNotFoundError(
            "Record-quality file was not found.\n"
            "Correct and run src/07_quality_imputation_anomaly.py first."
        )

    quality_df = pd.read_csv(
        QUALITY_SCORES_PATH
    )

    quality_df = quality_df.loc[
        quality_df["dataset_split"] == "test"
    ].copy()

    return quality_df


def load_anomaly_scores() -> pd.DataFrame:
    if not ANOMALY_SCORES_PATH.exists():
        raise FileNotFoundError(
            "Anomaly-score file was not found.\n"
            "Correct and run src/07_quality_imputation_anomaly.py first."
        )

    anomaly_df = pd.read_csv(
        ANOMALY_SCORES_PATH
    )

    anomaly_df = anomaly_df.loc[
        anomaly_df["dataset_split"] == "test"
    ].copy()

    return anomaly_df


def load_operational_model() -> Any:
    if not OPERATIONAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Operational model not found: {OPERATIONAL_MODEL_PATH}"
        )

    return joblib.load(
        OPERATIONAL_MODEL_PATH
    )


# ============================================================
# SENSOR UTILITIES
# ============================================================
def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    columns = [
        column
        for column in dataframe.columns
        if column.startswith(SENSOR_PREFIX)
    ]

    if not columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return columns


# ============================================================
# MASTER ERROR TABLE
# ============================================================
def build_master_error_table(
    test_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    anomaly_df: pd.DataFrame,
) -> pd.DataFrame:
    sensor_columns = get_sensor_columns(
        test_df
    )

    base_columns = [
        ID_COLUMN,
        TIMESTAMP_COLUMN,
        TARGET_COLUMN,
        STATUS_COLUMN,
    ]

    master = test_df[
        base_columns
    ].copy()

    master["missing_sensor_count"] = (
        test_df[sensor_columns]
        .isna()
        .sum(axis=1)
        .to_numpy()
    )

    master["missing_sensor_rate"] = (
        master["missing_sensor_count"]
        / len(sensor_columns)
    )

    prediction_columns = [
        ID_COLUMN,
        "fail_probability",
        "predicted_target",
        "predicted_status",
        "decision_threshold",
        "model_name",
        "prediction_correct",
        "error_type",
    ]

    available_prediction_columns = [
        column
        for column in prediction_columns
        if column in predictions_df.columns
    ]

    master = master.merge(
        predictions_df[
            available_prediction_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    quality_columns = [
        ID_COLUMN,
        "mean_observed_sensor_reliability",
        "completeness_score",
        "data_quality_score",
        "data_quality_band",
    ]

    available_quality_columns = [
        column
        for column in quality_columns
        if column in quality_df.columns
    ]

    master = master.merge(
        quality_df[
            available_quality_columns
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    master = master.merge(
        anomaly_df[
            [
                ID_COLUMN,
                "anomaly_score",
            ]
        ],
        on=ID_COLUMN,
        how="left",
        validate="one_to_one",
    )

    if "error_type" not in master.columns:
        master["error_type"] = np.select(
            [
                (
                    (master[TARGET_COLUMN] == 1)
                    & (master["predicted_target"] == 0)
                ),
                (
                    (master[TARGET_COLUMN] == 0)
                    & (master["predicted_target"] == 1)
                ),
            ],
            [
                "Missed Failure",
                "False Alarm",
            ],
            default="Correct",
        )

    master["absolute_threshold_distance"] = (
        master["fail_probability"]
        - master["decision_threshold"]
    ).abs()

    master["prediction_confidence"] = np.where(
        master["predicted_target"] == 1,
        master["fail_probability"],
        1.0 - master["fail_probability"],
    )

    master["probability_band"] = pd.cut(
        master["fail_probability"],
        bins=[
            -0.001,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            1.00,
        ],
        labels=[
            "0-10%",
            "10-25%",
            "25-50%",
            "50-75%",
            "75-90%",
            "90-100%",
        ],
    )

    master["missingness_quartile"] = pd.qcut(
        master["missing_sensor_rate"],
        q=4,
        duplicates="drop",
    )

    master["anomaly_quartile"] = pd.qcut(
        master["anomaly_score"],
        q=4,
        labels=[
            "Q1 Lowest",
            "Q2",
            "Q3",
            "Q4 Highest",
        ],
        duplicates="drop",
    )

    master["confidence_band"] = pd.cut(
        master["prediction_confidence"],
        bins=[
            -0.001,
            0.60,
            0.75,
            0.90,
            1.00,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Very High",
        ],
    )

    master["production_date"] = (
        master[TIMESTAMP_COLUMN].dt.date
    )

    master["production_hour"] = (
        master[TIMESTAMP_COLUMN].dt.hour
    )

    master["production_day"] = (
        master[TIMESTAMP_COLUMN].dt.day_name()
    )

    master["shift"] = pd.cut(
        master["production_hour"],
        bins=[
            -1,
            7,
            15,
            23,
        ],
        labels=[
            "Night",
            "Morning",
            "Evening",
        ],
    )

    return master


# ============================================================
# METRIC FUNCTIONS
# ============================================================
def calculate_segment_metrics(
    dataframe: pd.DataFrame,
    segment_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for segment_value, segment_df in dataframe.groupby(
        segment_column,
        observed=False,
        dropna=False,
    ):
        if segment_df.empty:
            continue

        y_true = segment_df[
            TARGET_COLUMN
        ].astype(int)

        y_pred = segment_df[
            "predicted_target"
        ].astype(int)

        probabilities = segment_df[
            "fail_probability"
        ].astype(float)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        row: dict[str, Any] = {
            "segment_variable": segment_column,
            "segment_value": str(segment_value),
            "records": int(len(segment_df)),
            "actual_failures": int(
                (y_true == 1).sum()
            ),
            "predicted_failures": int(
                (y_pred == 1).sum()
            ),
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "failure_recall": float(
                recall_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "failure_precision": float(
                precision_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "failure_f1": float(
                f1_score(
                    y_true,
                    y_pred,
                    zero_division=0,
                )
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(
                    y_true,
                    y_pred,
                )
            ),
            "mean_fail_probability": float(
                probabilities.mean()
            ),
            "mean_data_quality": float(
                segment_df[
                    "data_quality_score"
                ].mean()
            ),
            "mean_anomaly_score": float(
                segment_df[
                    "anomaly_score"
                ].mean()
            ),
        }

        if y_true.nunique() == 2:
            row["roc_auc"] = float(
                roc_auc_score(
                    y_true,
                    probabilities,
                )
            )

            row["pr_auc"] = float(
                average_precision_score(
                    y_true,
                    probabilities,
                )
            )

        else:
            row["roc_auc"] = np.nan
            row["pr_auc"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# ERROR SUMMARIES
# ============================================================
def build_error_type_summary(
    master_df: pd.DataFrame,
) -> pd.DataFrame:
    return (
        master_df
        .groupby(
            "error_type",
            dropna=False,
        )
        .agg(
            records=(
                ID_COLUMN,
                "size",
            ),
            mean_failure_probability=(
                "fail_probability",
                "mean",
            ),
            median_failure_probability=(
                "fail_probability",
                "median",
            ),
            mean_data_quality=(
                "data_quality_score",
                "mean",
            ),
            mean_missingness=(
                "missing_sensor_rate",
                "mean",
            ),
            mean_anomaly_score=(
                "anomaly_score",
                "mean",
            ),
            mean_prediction_confidence=(
                "prediction_confidence",
                "mean",
            ),
        )
        .reset_index()
    )


# ============================================================
# DIAGNOSTIC PERMUTATION IMPORTANCE
# ============================================================
def calculate_permutation_importance(
    model: Any,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Diagnostic only.

    Do not use this test-set importance table to select features or
    retrain the model. It is intended only to understand behaviour
    after final test evaluation.
    """
    sensor_columns = get_sensor_columns(
        test_df
    )

    x_test = test_df[
        sensor_columns
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    LOGGER.info(
        "Calculating diagnostic permutation importance"
    )

    result = permutation_importance(
        estimator=model,
        X=x_test,
        y=y_test,
        scoring="average_precision",
        n_repeats=PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    importance_df = pd.DataFrame(
        {
            "feature": sensor_columns,
            "importance_mean": (
                result.importances_mean
            ),
            "importance_std": (
                result.importances_std
            ),
        }
    )

    importance_df["importance_positive"] = (
        importance_df["importance_mean"] > 0
    )

    return (
        importance_df
        .sort_values(
            by="importance_mean",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_error_distribution_plot(
    master_df: pd.DataFrame,
) -> None:
    counts = (
        master_df["error_type"]
        .value_counts()
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    counts.plot(
        kind="bar",
        ax=axis,
    )

    axis.set_title(
        "Operational Prediction Outcomes"
    )

    axis.set_xlabel(
        "Outcome"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.tick_params(
        axis="x",
        rotation=0,
    )

    for index, value in enumerate(
        counts.values
    ):
        axis.text(
            index,
            value,
            str(int(value)),
            ha="center",
            va="bottom",
        )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "prediction_error_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_probability_by_error_plot(
    master_df: pd.DataFrame,
) -> None:
    groups = [
        group["fail_probability"].dropna()
        for _, group in master_df.groupby(
            "error_type"
        )
    ]

    labels = [
        str(name)
        for name, _ in master_df.groupby(
            "error_type"
        )
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.boxplot(
        groups,
        tick_labels=labels,
    )

    axis.set_title(
        "Failure Probability by Prediction Outcome"
    )

    axis.set_xlabel(
        "Prediction Outcome"
    )

    axis.set_ylabel(
        "Predicted Failure Probability"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "failure_probability_by_error_type.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_quality_by_error_plot(
    master_df: pd.DataFrame,
) -> None:
    groups = [
        group["data_quality_score"].dropna()
        for _, group in master_df.groupby(
            "error_type"
        )
    ]

    labels = [
        str(name)
        for name, _ in master_df.groupby(
            "error_type"
        )
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.boxplot(
        groups,
        tick_labels=labels,
    )

    axis.set_title(
        "Data Quality by Prediction Outcome"
    )

    axis.set_xlabel(
        "Prediction Outcome"
    )

    axis.set_ylabel(
        "Data Quality Score"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "data_quality_by_error_type.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_anomaly_by_error_plot(
    master_df: pd.DataFrame,
) -> None:
    groups = [
        group["anomaly_score"].dropna()
        for _, group in master_df.groupby(
            "error_type"
        )
    ]

    labels = [
        str(name)
        for name, _ in master_df.groupby(
            "error_type"
        )
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.boxplot(
        groups,
        tick_labels=labels,
    )

    axis.set_title(
        "Anomaly Score by Prediction Outcome"
    )

    axis.set_xlabel(
        "Prediction Outcome"
    )

    axis.set_ylabel(
        "Anomaly Score"
    )

    axis.tick_params(
        axis="x",
        rotation=20,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "anomaly_score_by_error_type.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_top_importance_plot(
    importance_df: pd.DataFrame,
) -> None:
    plot_df = (
        importance_df
        .head(25)
        .sort_values(
            by="importance_mean",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.barh(
        plot_df["feature"],
        plot_df["importance_mean"],
        xerr=plot_df["importance_std"],
    )

    axis.set_title(
        "Diagnostic Test-Set Permutation Importance"
    )

    axis.set_xlabel(
        "Decrease in PR-AUC"
    )

    axis.set_ylabel(
        "Sensor"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "diagnostic_permutation_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# SUMMARY INTERPRETATION
# ============================================================
def build_summary(
    master_df: pd.DataFrame,
    quality_metrics: pd.DataFrame,
    missingness_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
    confidence_metrics: pd.DataFrame,
    importance_df: pd.DataFrame,
) -> dict[str, Any]:
    missed_failures = master_df.loc[
        master_df["error_type"]
        == "Missed Failure"
    ]

    false_alarms = master_df.loc[
        master_df["error_type"]
        == "False Alarm"
    ]

    correct_predictions = master_df.loc[
        master_df["error_type"]
        == "Correct"
    ]

    return {
        "project": "HeveMind",
        "stage": "Operational error analysis",
        "test_records": int(
            len(master_df)
        ),
        "actual_failures": int(
            (master_df[TARGET_COLUMN] == 1).sum()
        ),
        "detected_failures": int(
            (
                (master_df[TARGET_COLUMN] == 1)
                & (
                    master_df[
                        "predicted_target"
                    ] == 1
                )
            ).sum()
        ),
        "missed_failures": int(
            len(missed_failures)
        ),
        "false_alarms": int(
            len(false_alarms)
        ),
        "mean_quality": {
            "correct": float(
                correct_predictions[
                    "data_quality_score"
                ].mean()
            ),
            "missed_failure": float(
                missed_failures[
                    "data_quality_score"
                ].mean()
            ),
            "false_alarm": float(
                false_alarms[
                    "data_quality_score"
                ].mean()
            ),
        },
        "mean_anomaly": {
            "correct": float(
                correct_predictions[
                    "anomaly_score"
                ].mean()
            ),
            "missed_failure": float(
                missed_failures[
                    "anomaly_score"
                ].mean()
            ),
            "false_alarm": float(
                false_alarms[
                    "anomaly_score"
                ].mean()
            ),
        },
        "mean_missingness": {
            "correct": float(
                correct_predictions[
                    "missing_sensor_rate"
                ].mean()
            ),
            "missed_failure": float(
                missed_failures[
                    "missing_sensor_rate"
                ].mean()
            ),
            "false_alarm": float(
                false_alarms[
                    "missing_sensor_rate"
                ].mean()
            ),
        },
        "quality_segment_metrics": (
            quality_metrics.to_dict(
                orient="records"
            )
        ),
        "missingness_segment_metrics": (
            missingness_metrics.to_dict(
                orient="records"
            )
        ),
        "anomaly_segment_metrics": (
            anomaly_metrics.to_dict(
                orient="records"
            )
        ),
        "confidence_segment_metrics": (
            confidence_metrics.to_dict(
                orient="records"
            )
        ),
        "top_diagnostic_features": (
            importance_df
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
        "important_warning": (
            "Permutation importance was calculated on the held-out "
            "test set for diagnostic interpretation only. It must not "
            "be used for feature selection or model retraining."
        ),
    }


# ============================================================
# CONSOLE OUTPUT
# ============================================================
def print_console_summary(
    master_df: pd.DataFrame,
    error_summary: pd.DataFrame,
    quality_metrics: pd.DataFrame,
    missingness_metrics: pd.DataFrame,
    anomaly_metrics: pd.DataFrame,
) -> None:
    print("\n" + "=" * 110)
    print("HEVEMIND OPERATIONAL ERROR ANALYSIS")
    print("=" * 110)

    print("\nPrediction outcome summary:")

    print(
        error_summary
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\nPerformance by data-quality band:")

    print(
        quality_metrics[
            [
                "segment_value",
                "records",
                "actual_failures",
                "failure_recall",
                "failure_precision",
                "balanced_accuracy",
                "false_positive",
                "false_negative",
            ]
        ]
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\nPerformance by missingness quartile:")

    print(
        missingness_metrics[
            [
                "segment_value",
                "records",
                "actual_failures",
                "failure_recall",
                "failure_precision",
                "false_positive",
                "false_negative",
            ]
        ]
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\nPerformance by anomaly quartile:")

    print(
        anomaly_metrics[
            [
                "segment_value",
                "records",
                "actual_failures",
                "failure_recall",
                "failure_precision",
                "false_positive",
                "false_negative",
            ]
        ]
        .round(4)
        .to_string(
            index=False,
        )
    )

    missed_count = int(
        (
            master_df["error_type"]
            == "Missed Failure"
        ).sum()
    )

    false_alarm_count = int(
        (
            master_df["error_type"]
            == "False Alarm"
        ).sum()
    )

    print("\nCritical errors:")

    print(
        f"Missed failures:              "
        f"{missed_count}"
    )

    print(
        f"False alarms:                 "
        f"{false_alarm_count}"
    )

    print("\nSaved outputs:")

    print(
        f"Master error table:           "
        f"{MASTER_ERROR_TABLE_PATH}"
    )

    print(
        f"Missed failures:              "
        f"{TOP_FALSE_NEGATIVES_PATH}"
    )

    print(
        f"False alarms:                 "
        f"{TOP_FALSE_POSITIVES_PATH}"
    )

    print(
        f"Diagnostic feature analysis:  "
        f"{PERMUTATION_IMPORTANCE_PATH}"
    )

    print(
        f"Report directory:             "
        f"{OUTPUT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading data and model artifacts"
    )

    test_df = load_test_data()
    predictions_df = load_operational_predictions()
    quality_df = load_quality_scores()
    anomaly_df = load_anomaly_scores()
    operational_model = load_operational_model()

    LOGGER.info(
        "Building master error-analysis table"
    )

    master_df = build_master_error_table(
        test_df=test_df,
        predictions_df=predictions_df,
        quality_df=quality_df,
        anomaly_df=anomaly_df,
    )

    master_df.to_csv(
        MASTER_ERROR_TABLE_PATH,
        index=False,
    )

    error_summary = build_error_type_summary(
        master_df
    )

    error_summary.to_csv(
        ERROR_TYPE_SUMMARY_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating performance by data-quality segment"
    )

    quality_metrics = calculate_segment_metrics(
        master_df,
        "data_quality_band",
    )

    quality_metrics.to_csv(
        QUALITY_SEGMENT_METRICS_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating performance by missingness segment"
    )

    missingness_metrics = calculate_segment_metrics(
        master_df,
        "missingness_quartile",
    )

    missingness_metrics.to_csv(
        MISSINGNESS_SEGMENT_METRICS_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating performance by anomaly segment"
    )

    anomaly_metrics = calculate_segment_metrics(
        master_df,
        "anomaly_quartile",
    )

    anomaly_metrics.to_csv(
        ANOMALY_SEGMENT_METRICS_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating performance by confidence segment"
    )

    confidence_metrics = calculate_segment_metrics(
        master_df,
        "confidence_band",
    )

    confidence_metrics.to_csv(
        CONFIDENCE_SEGMENT_METRICS_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating performance by production shift"
    )

    time_metrics = calculate_segment_metrics(
        master_df,
        "shift",
    )

    time_metrics.to_csv(
        TIME_SEGMENT_METRICS_PATH,
        index=False,
    )

    missed_failures = (
        master_df.loc[
            master_df["error_type"]
            == "Missed Failure"
        ]
        .sort_values(
            by="fail_probability",
            ascending=True,
        )
    )

    missed_failures.to_csv(
        TOP_FALSE_NEGATIVES_PATH,
        index=False,
    )

    false_alarms = (
        master_df.loc[
            master_df["error_type"]
            == "False Alarm"
        ]
        .sort_values(
            by="fail_probability",
            ascending=False,
        )
    )

    false_alarms.to_csv(
        TOP_FALSE_POSITIVES_PATH,
        index=False,
    )

    importance_df = calculate_permutation_importance(
        model=operational_model,
        test_df=test_df,
    )

    importance_df.head(
        TOP_FEATURES_TO_SAVE
    ).to_csv(
        PERMUTATION_IMPORTANCE_PATH,
        index=False,
    )

    LOGGER.info(
        "Generating diagnostic figures"
    )

    save_error_distribution_plot(
        master_df
    )

    save_probability_by_error_plot(
        master_df
    )

    save_quality_by_error_plot(
        master_df
    )

    save_anomaly_by_error_plot(
        master_df
    )

    save_top_importance_plot(
        importance_df
    )

    summary = build_summary(
        master_df=master_df,
        quality_metrics=quality_metrics,
        missingness_metrics=missingness_metrics,
        anomaly_metrics=anomaly_metrics,
        confidence_metrics=confidence_metrics,
        importance_df=importance_df,
    )

    with SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )

    print_console_summary(
        master_df=master_df,
        error_summary=error_summary,
        quality_metrics=quality_metrics,
        missingness_metrics=missingness_metrics,
        anomaly_metrics=anomaly_metrics,
    )


if __name__ == "__main__":
    main()