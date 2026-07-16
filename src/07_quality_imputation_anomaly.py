from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.ensemble import (
    ExtraTreesRegressor,
    IsolationForest,
)

from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.impute import IterativeImputer, SimpleImputer
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


# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

LOGGER = logging.getLogger(__name__)


# ============================================================
# PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VALIDATION_PATH = SPLITS_DIR / "validation.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
QUALITY_ARTIFACTS_DIR = ARTIFACTS_DIR / "quality_engine"

REPORTS_DIR = ROOT_DIR / "reports"
QUALITY_REPORTS_DIR = REPORTS_DIR / "quality_engine"
TABLES_DIR = QUALITY_REPORTS_DIR / "tables"
FIGURES_DIR = QUALITY_REPORTS_DIR / "figures"

SENSOR_RELIABILITY_PATH = (
    TABLES_DIR / "sensor_reliability_profile.csv"
)

RECORD_QUALITY_PATH = (
    TABLES_DIR / "record_quality_scores.csv"
)

ANOMALY_RESULTS_PATH = (
    TABLES_DIR / "anomaly_scores.csv"
)

IMPUTATION_RESULTS_PATH = (
    TABLES_DIR / "imputation_model_comparison.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR / "quality_model_test_predictions.csv"
)

SUMMARY_PATH = (
    QUALITY_REPORTS_DIR / "quality_engine_summary.json"
)

ITERATIVE_IMPUTER_PATH = (
    QUALITY_ARTIFACTS_DIR / "iterative_imputer.joblib"
)

ANOMALY_MODEL_PATH = (
    QUALITY_ARTIFACTS_DIR / "isolation_forest.joblib"
)

QUALITY_MODEL_PATH = (
    QUALITY_ARTIFACTS_DIR / "quality_aware_model.joblib"
)


# ============================================================
# CONFIGURATION
# ============================================================
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
ID_COLUMN = "wafer_id"
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

ITERATIVE_MAX_ITER = 10
ITERATIVE_SAMPLE_POSTERIOR = False

ISOLATION_FOREST_CONTAMINATION = 0.05


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        QUALITY_ARTIFACTS_DIR,
        QUALITY_REPORTS_DIR,
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
def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}"
        )

    dataframe = pd.read_parquet(path)

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)
    test_df = load_split(TEST_PATH)

    development_df = pd.concat(
        [train_df, validation_df],
        axis=0,
        ignore_index=True,
    )

    development_df = development_df.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return development_df, test_df


# ============================================================
# FEATURE UTILITIES
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


def split_features_target(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    return (
        dataframe[sensor_columns].copy(),
        dataframe[TARGET_COLUMN].astype(int).copy(),
    )


# ============================================================
# SENSOR RELIABILITY
# ============================================================
def calculate_sensor_reliability_profile(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Creates a statistical reliability proxy for each sensor.

    This is not physical hardware reliability. It estimates trustworthiness
    from completeness, stability, and outlier burden.
    """
    rows: list[dict[str, Any]] = []

    for column in sensor_columns:
        series = dataframe[column]

        missing_rate = float(
            series.isna().mean()
        )

        observed = series.dropna()

        if observed.empty:
            rows.append(
                {
                    "sensor": column,
                    "missing_rate": 1.0,
                    "outlier_rate": 1.0,
                    "relative_variability": np.nan,
                    "stability_score": 0.0,
                    "completeness_score": 0.0,
                    "outlier_score": 0.0,
                    "sensor_reliability_score": 0.0,
                }
            )
            continue

        median = float(
            observed.median()
        )

        q1 = float(
            observed.quantile(0.25)
        )

        q3 = float(
            observed.quantile(0.75)
        )

        iqr = q3 - q1

        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        outlier_rate = float(
            (
                (observed < lower_bound)
                | (observed > upper_bound)
            ).mean()
        )

        mean_absolute_value = float(
            np.mean(np.abs(observed))
        )

        relative_variability = float(
            observed.std(ddof=1)
            / (mean_absolute_value + 1e-8)
        )

        completeness_score = float(
            1.0 - missing_rate
        )

        outlier_score = float(
            max(0.0, 1.0 - outlier_rate)
        )

        stability_score = float(
            1.0 / (1.0 + relative_variability)
        )

        reliability_score = (
            0.50 * completeness_score
            + 0.25 * outlier_score
            + 0.25 * stability_score
        )

        rows.append(
            {
                "sensor": column,
                "missing_rate": missing_rate,
                "outlier_rate": outlier_rate,
                "relative_variability": relative_variability,
                "stability_score": stability_score,
                "completeness_score": completeness_score,
                "outlier_score": outlier_score,
                "sensor_reliability_score": float(
                    np.clip(
                        reliability_score,
                        0.0,
                        1.0,
                    )
                ),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="sensor_reliability_score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# RECORD QUALITY SCORE
# ============================================================
def calculate_record_quality_scores(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    reliability_profile: pd.DataFrame,
) -> pd.DataFrame:
    reliability_map = (
        reliability_profile
        .set_index("sensor")[
            "sensor_reliability_score"
        ]
        .to_dict()
    )

    output = dataframe[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    sensor_data = dataframe[
        sensor_columns
    ]

    missing_mask = sensor_data.isna()

    output["missing_sensor_count"] = (
        missing_mask.sum(axis=1)
    )

    output["missing_sensor_rate"] = (
        output["missing_sensor_count"]
        / len(sensor_columns)
    )

    observed_count = (
        len(sensor_columns)
        - output["missing_sensor_count"]
    )

    output["observed_sensor_count"] = (
        observed_count
    )

    sensor_reliability_vector = np.array(
        [
            reliability_map[column]
            for column in sensor_columns
        ],
        dtype=float,
    )

    observed_mask_array = (
        (~missing_mask)
        .to_numpy(dtype=float)
    )

    weighted_reliability_sum = (
        observed_mask_array
        * sensor_reliability_vector
    ).sum(axis=1)

    reliability_denominator = np.maximum(
        observed_mask_array.sum(axis=1),
        1.0,
    )

    output["mean_observed_sensor_reliability"] = (
        weighted_reliability_sum
        / reliability_denominator
    )

    output["completeness_score"] = (
        1.0
        - output["missing_sensor_rate"]
    )

    output["data_quality_score"] = (
        0.60
        * output["completeness_score"]
        + 0.40
        * output[
            "mean_observed_sensor_reliability"
        ]
    )

    output["data_quality_score"] = (
        output["data_quality_score"]
        .clip(0.0, 1.0)
    )

    output["data_quality_band"] = pd.cut(
        output["data_quality_score"],
        bins=[
            -0.001,
            0.60,
            0.75,
            0.90,
            1.00,
        ],
        labels=[
            "Poor",
            "Review",
            "Acceptable",
            "High",
        ],
    )

    return output


# ============================================================
# IMPUTATION PIPELINES
# ============================================================
def build_median_pipeline() -> Pipeline:
    return Pipeline(
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
                    n_estimators=600,
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


def build_iterative_pipeline() -> Pipeline:
    iterative_estimator = ExtraTreesRegressor(
        n_estimators=30,
        max_depth=10,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    iterative_imputer = IterativeImputer(
        estimator=iterative_estimator,
        max_iter=ITERATIVE_MAX_ITER,
        initial_strategy="median",
        skip_complete=True,
        sample_posterior=ITERATIVE_SAMPLE_POSTERIOR,
        random_state=RANDOM_STATE,
        verbose=0,
    )

    return Pipeline(
        steps=[
            (
                "imputer",
                iterative_imputer,
            ),
            (
                "model",
                BalancedRandomForestClassifier(
                    n_estimators=600,
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


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================
def generate_oof_probabilities(
    pipeline: Pipeline,
    x_data: pd.DataFrame,
    y_data: pd.Series,
) -> np.ndarray:
    cross_validator = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    return cross_val_predict(
        estimator=pipeline,
        X=x_data,
        y=y_data,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


# ============================================================
# THRESHOLD SELECTION
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

    balanced_accuracy = (
        balanced_accuracy_score(
            y_true,
            predictions,
        )
    )

    operational_cost = (
        MISSED_FAILURE_COST * fn
        + FALSE_ALARM_COST * fp
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
        "operational_cost": float(
            operational_cost
        ),
    }


def select_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []

    for threshold in np.arange(
        THRESHOLD_MINIMUM,
        THRESHOLD_MAXIMUM + THRESHOLD_STEP,
        THRESHOLD_STEP,
    ):
        rows.append(
            calculate_threshold_metrics(
                y_true=y_true.to_numpy(),
                probabilities=probabilities,
                threshold=float(threshold),
            )
        )

    threshold_df = pd.DataFrame(rows)

    eligible = threshold_df[
        threshold_df["recall_fail"]
        >= MINIMUM_FAILURE_RECALL
    ].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
            by=[
                "operational_cost",
                "false_positive",
                "precision_fail",
                "balanced_accuracy",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        ).iloc[0]

    else:
        selected = threshold_df.sort_values(
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

    return selected.to_dict()


# ============================================================
# IMPUTATION MODEL BENCHMARK
# ============================================================
def benchmark_imputation_models(
    x_development: pd.DataFrame,
    y_development: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, Pipeline],
    dict[str, float],
]:
    models = {
        "Median Imputation": build_median_pipeline(),
        "Iterative Imputation": build_iterative_pipeline(),
    }

    fitted_templates: dict[str, Pipeline] = {}
    threshold_mapping: dict[str, float] = {}

    rows: list[dict[str, Any]] = []

    for model_name, pipeline in models.items():
        LOGGER.info(
            "Evaluating model: %s",
            model_name,
        )

        probabilities = generate_oof_probabilities(
            pipeline=pipeline,
            x_data=x_development,
            y_data=y_development,
        )

        threshold_result = select_threshold(
            y_true=y_development,
            probabilities=probabilities,
        )

        threshold = float(
            threshold_result["threshold"]
        )

        rows.append(
            {
                "model": model_name,
                "roc_auc": float(
                    roc_auc_score(
                        y_development,
                        probabilities,
                    )
                ),
                "pr_auc": float(
                    average_precision_score(
                        y_development,
                        probabilities,
                    )
                ),
                "brier_score": float(
                    brier_score_loss(
                        y_development,
                        probabilities,
                    )
                ),
                **threshold_result,
            }
        )

        fitted_templates[
            model_name
        ] = pipeline

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

    return (
        comparison_df,
        fitted_templates,
        threshold_mapping,
    )


# ============================================================
# ANOMALY MODEL
# ============================================================
def fit_anomaly_model(
    x_development: pd.DataFrame,
    y_development: pd.Series,
) -> tuple[
    Pipeline,
    np.ndarray,
]:
    """
    Trains Isolation Forest only on passing records.
    """
    pass_data = x_development.loc[
        y_development == 0
    ].copy()

    anomaly_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
            (
                "model",
                IsolationForest(
                    n_estimators=500,
                    contamination=(
                        ISOLATION_FOREST_CONTAMINATION
                    ),
                    max_samples="auto",
                    n_jobs=-1,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    anomaly_pipeline.fit(
        pass_data
    )

    development_scores = (
        -anomaly_pipeline.decision_function(
            x_development
        )
    )

    return (
        anomaly_pipeline,
        development_scores,
    )


def normalize_scores(
    values: np.ndarray,
) -> np.ndarray:
    minimum = float(
        np.min(values)
    )

    maximum = float(
        np.max(values)
    )

    if maximum - minimum < 1e-12:
        return np.zeros_like(
            values,
            dtype=float,
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# FINAL MODEL EVALUATION
# ============================================================
def evaluate_final_model(
    pipeline: Pipeline,
    threshold: float,
    x_development: pd.DataFrame,
    y_development: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[
    Pipeline,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    pipeline.fit(
        x_development,
        y_development,
    )

    probabilities = pipeline.predict_proba(
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
        pipeline,
        metrics,
        probabilities,
        predictions,
    )


# ============================================================
# PLOTS
# ============================================================
def save_sensor_reliability_plot(
    reliability_df: pd.DataFrame,
) -> None:
    plot_df = (
        reliability_df
        .sort_values(
            by="sensor_reliability_score",
            ascending=True,
        )
        .head(30)
    )

    figure, axis = plt.subplots(
        figsize=(10, 9)
    )

    axis.barh(
        plot_df["sensor"],
        plot_df["sensor_reliability_score"],
    )

    axis.set_title(
        "Lowest-Reliability Sensor Proxies"
    )

    axis.set_xlabel(
        "Reliability Score"
    )

    axis.set_ylabel(
        "Sensor"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "lowest_sensor_reliability.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_quality_distribution_plot(
    quality_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        quality_df["data_quality_score"],
        bins=30,
    )

    axis.set_title(
        "Record-Level Data Quality Score Distribution"
    )

    axis.set_xlabel(
        "Data Quality Score"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "data_quality_score_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_anomaly_by_class_plot(
    anomaly_df: pd.DataFrame,
) -> None:
    pass_scores = anomaly_df.loc[
        anomaly_df[TARGET_COLUMN] == 0,
        "anomaly_score",
    ]

    fail_scores = anomaly_df.loc[
        anomaly_df[TARGET_COLUMN] == 1,
        "anomaly_score",
    ]

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    axis.hist(
        pass_scores,
        bins=30,
        alpha=0.6,
        label="Pass",
    )

    axis.hist(
        fail_scores,
        bins=30,
        alpha=0.6,
        label="Fail",
    )

    axis.set_title(
        "Anomaly Score Distribution by Outcome"
    )

    axis.set_xlabel(
        "Normalised Anomaly Score"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "anomaly_score_by_class.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# SAVE SUMMARY
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
        "Loading development and test data"
    )

    development_df, test_df = load_datasets()

    sensor_columns = get_sensor_columns(
        development_df
    )

    x_development, y_development = (
        split_features_target(
            development_df,
            sensor_columns,
        )
    )

    x_test, y_test = split_features_target(
        test_df,
        sensor_columns,
    )

    LOGGER.info(
        "Calculating sensor reliability profiles"
    )

    reliability_df = (
        calculate_sensor_reliability_profile(
            development_df,
            sensor_columns,
        )
    )

    reliability_df.to_csv(
        SENSOR_RELIABILITY_PATH,
        index=False,
    )

    LOGGER.info(
        "Calculating record-level data quality scores"
    )

    development_quality_df = (
        calculate_record_quality_scores(
            development_df,
            sensor_columns,
            reliability_df,
        )
    )

    test_quality_df = (
        calculate_record_quality_scores(
            test_df,
            sensor_columns,
            reliability_df,
        )
    )

    quality_output = pd.concat(
        [
            development_quality_df.assign(
                dataset_split="development"
            ),
            test_quality_df.assign(
                dataset_split="test"
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    quality_output.to_csv(
        RECORD_QUALITY_PATH,
        index=False,
    )

    LOGGER.info(
        "Benchmarking median and iterative imputation"
    )

    (
        comparison_df,
        model_templates,
        threshold_mapping,
    ) = benchmark_imputation_models(
        x_development=x_development,
        y_development=y_development,
    )

    comparison_df.to_csv(
        IMPUTATION_RESULTS_PATH,
        index=False,
    )

    best_model_name = str(
        comparison_df.iloc[0]["model"]
    )

    best_threshold = float(
        threshold_mapping[
            best_model_name
        ]
    )

    LOGGER.info(
        "Selected imputation model: %s",
        best_model_name,
    )

    (
        final_model,
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_final_model(
        pipeline=model_templates[
            best_model_name
        ],
        threshold=best_threshold,
        x_development=x_development,
        y_development=y_development,
        x_test=x_test,
        y_test=y_test,
    )

    LOGGER.info(
        "Fitting anomaly-detection model"
    )

    (
        anomaly_model,
        development_anomaly_scores,
    ) = fit_anomaly_model(
        x_development=x_development,
        y_development=y_development,
    )

    test_anomaly_scores = (
        -anomaly_model.decision_function(
            x_test
        )
    )

    combined_anomaly_scores = np.concatenate(
        [
            development_anomaly_scores,
            test_anomaly_scores,
        ]
    )

    combined_normalized_scores = normalize_scores(
        combined_anomaly_scores
    )

    development_normalized = (
        combined_normalized_scores[
            : len(development_anomaly_scores)
        ]
    )

    test_normalized = (
        combined_normalized_scores[
            len(development_anomaly_scores) :
        ]
    )

    anomaly_output = pd.concat(
        [
            development_df[
                [
                    ID_COLUMN,
                    TARGET_COLUMN,
                    STATUS_COLUMN,
                ]
            ].assign(
                anomaly_score=development_normalized,
                dataset_split="development",
            ),
            test_df[
                [
                    ID_COLUMN,
                    TARGET_COLUMN,
                    STATUS_COLUMN,
                ]
            ].assign(
                anomaly_score=test_normalized,
                dataset_split="test",
            ),
        ],
        axis=0,
        ignore_index=True,
    )

    anomaly_output.to_csv(
        ANOMALY_RESULTS_PATH,
        index=False,
    )

    test_prediction_output = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    test_prediction_output[
        "fail_probability"
    ] = test_probabilities

    test_prediction_output[
        "predicted_target"
    ] = test_predictions

    test_prediction_output[
        "decision_threshold"
    ] = best_threshold

    test_prediction_output[
        "model_name"
    ] = best_model_name

    test_prediction_output[
        "data_quality_score"
    ] = test_quality_df[
        "data_quality_score"
    ].to_numpy()

    test_prediction_output[
        "data_quality_band"
    ] = test_quality_df[
        "data_quality_band"
    ].astype(str).to_numpy()

    test_prediction_output[
        "anomaly_score"
    ] = test_normalized

    test_prediction_output[
        "error_type"
    ] = np.select(
        [
            (
                (test_prediction_output[TARGET_COLUMN] == 1)
                & (
                    test_prediction_output[
                        "predicted_target"
                    ] == 0
                )
            ),
            (
                (test_prediction_output[TARGET_COLUMN] == 0)
                & (
                    test_prediction_output[
                        "predicted_target"
                    ] == 1
                )
            ),
        ],
        [
            "Missed Failure",
            "False Alarm",
        ],
        default="Correct",
    )

    test_prediction_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    joblib.dump(
        final_model,
        QUALITY_MODEL_PATH,
    )

    joblib.dump(
        anomaly_model,
        ANOMALY_MODEL_PATH,
    )

    if (
        best_model_name
        == "Iterative Imputation"
    ):
        joblib.dump(
            final_model.named_steps[
                "imputer"
            ],
            ITERATIVE_IMPUTER_PATH,
        )

    save_sensor_reliability_plot(
        reliability_df
    )

    save_quality_distribution_plot(
        quality_output
    )

    save_anomaly_by_class_plot(
        anomaly_output
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Sensor reliability, data quality, "
            "iterative imputation and anomaly detection"
        ),
        "development_rows": int(
            len(development_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "sensor_count": int(
            len(sensor_columns)
        ),
        "selected_model": (
            best_model_name
        ),
        "selected_threshold": (
            best_threshold
        ),
        "test_metrics": (
            test_metrics
        ),
        "mean_development_data_quality": float(
            development_quality_df[
                "data_quality_score"
            ].mean()
        ),
        "mean_test_data_quality": float(
            test_quality_df[
                "data_quality_score"
            ].mean()
        ),
        "mean_pass_anomaly_score": float(
            anomaly_output.loc[
                anomaly_output[
                    TARGET_COLUMN
                ] == 0,
                "anomaly_score",
            ].mean()
        ),
        "mean_fail_anomaly_score": float(
            anomaly_output.loc[
                anomaly_output[
                    TARGET_COLUMN
                ] == 1,
                "anomaly_score",
            ].mean()
        ),
        "model_comparison": (
            comparison_df.to_dict(
                orient="records"
            )
        ),
        "sensor_reliability_definition": {
            "description": (
                "Statistical proxy only; not physical "
                "equipment reliability"
            ),
            "weights": {
                "completeness": 0.50,
                "outlier_burden": 0.25,
                "stability": 0.25,
            },
        },
        "record_quality_definition": {
            "weights": {
                "completeness": 0.60,
                "mean_sensor_reliability": 0.40,
            }
        },
    }

    save_summary(
        summary
    )

    print("\n" + "=" * 100)
    print("HEVEMIND QUALITY, IMPUTATION AND ANOMALY ENGINE")
    print("=" * 100)

    print("\nImputation model comparison:")

    print(
        comparison_df[
            [
                "model",
                "roc_auc",
                "pr_auc",
                "threshold",
                "recall_fail",
                "precision_fail",
                "f1_fail",
                "balanced_accuracy",
                "false_positive",
                "false_negative",
                "operational_cost",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nSelected model:")

    print(
        f"Model:                     "
        f"{best_model_name}"
    )

    print(
        f"Decision threshold:        "
        f"{best_threshold:.4f}"
    )

    print("\nHeld-out test performance:")

    print(
        f"ROC-AUC:                   "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                    "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Failure recall:            "
        f"{test_metrics['recall_fail']:.4f}"
    )

    print(
        f"Failure precision:         "
        f"{test_metrics['precision_fail']:.4f}"
    )

    print(
        f"Failure F1:                "
        f"{test_metrics['f1_fail']:.4f}"
    )

    print(
        f"Balanced accuracy:         "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print(
        f"False alarms:              "
        f"{test_metrics['false_positive']}"
    )

    print(
        f"Missed failures:           "
        f"{test_metrics['false_negative']}"
    )

    print("\nData quality:")

    print(
        f"Mean development score:    "
        f"{development_quality_df['data_quality_score'].mean():.4f}"
    )

    print(
        f"Mean test score:           "
        f"{test_quality_df['data_quality_score'].mean():.4f}"
    )

    print("\nAnomaly analysis:")

    print(
        f"Mean pass anomaly score:   "
        f"{summary['mean_pass_anomaly_score']:.4f}"
    )

    print(
        f"Mean fail anomaly score:   "
        f"{summary['mean_fail_anomaly_score']:.4f}"
    )

    print("\nSaved outputs:")

    print(
        f"Reliability profile:       "
        f"{SENSOR_RELIABILITY_PATH}"
    )

    print(
        f"Record quality scores:     "
        f"{RECORD_QUALITY_PATH}"
    )

    print(
        f"Anomaly scores:            "
        f"{ANOMALY_RESULTS_PATH}"
    )

    print(
        f"Model comparison:          "
        f"{IMPUTATION_RESULTS_PATH}"
    )

    print(
        f"Final model:               "
        f"{QUALITY_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()