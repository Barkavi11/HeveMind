from __future__ import annotations

import json
import logging
import time
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from imblearn.ensemble import (
    BalancedRandomForestClassifier,
    EasyEnsembleClassifier,
)
from lightgbm import LGBMClassifier
from scipy.stats import bootstrap
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
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

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ADVANCED_MODELS_DIR = MODELS_DIR / "advanced"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = ROOT_DIR / "reports"
ADVANCED_REPORTS_DIR = REPORTS_DIR / "advanced_models"
TABLES_DIR = ADVANCED_REPORTS_DIR / "tables"
FIGURES_DIR = ADVANCED_REPORTS_DIR / "figures"

MODEL_COMPARISON_PATH = TABLES_DIR / "advanced_model_comparison.csv"
CV_FOLD_RESULTS_PATH = TABLES_DIR / "cv_fold_results.csv"
OOF_PREDICTIONS_PATH = TABLES_DIR / "out_of_fold_predictions.csv"
TEST_PREDICTIONS_PATH = TABLES_DIR / "advanced_test_predictions.csv"
THRESHOLD_REPORT_PATH = TABLES_DIR / "threshold_analysis.csv"
TEST_CONFIDENCE_INTERVALS_PATH = (
    TABLES_DIR / "test_metric_confidence_intervals.csv"
)

BEST_PIPELINE_PATH = (
    ADVANCED_MODELS_DIR / "best_advanced_pipeline.joblib"
)
BEST_MODEL_PATH = (
    ADVANCED_MODELS_DIR / "best_advanced_model.joblib"
)
BEST_MODEL_SUMMARY_PATH = (
    ADVANCED_REPORTS_DIR / "best_advanced_model_summary.json"
)
FEATURE_COLUMNS_PATH = (
    METADATA_DIR / "advanced_feature_columns.json"
)


# ============================================================
# DATA CONFIGURATION
# ============================================================
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
ID_COLUMN = "wafer_id"
TIMESTAMP_COLUMN = "timestamp"
SENSOR_PREFIX = "sensor_"

RANDOM_STATE = 42

CV_SPLITS = 5
CV_REPEATS = 3

MINIMUM_FAILURE_RECALL = 0.75
DEFAULT_THRESHOLD = 0.50

BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_CONFIDENCE_LEVEL = 0.95

PRIMARY_SELECTION_METRIC = "cv_pr_auc_mean"


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    """
    Create all required output directories.
    """
    for directory in [
        ADVANCED_MODELS_DIR,
        METADATA_DIR,
        ADVANCED_REPORTS_DIR,
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
    """
    Load one Parquet split.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset split was not found: {path}\n"
            "Run src/02_data_audit.py first."
        )

    dataframe = pd.read_parquet(path)

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combine train and validation into the development dataset.

    The test set remains untouched until final evaluation.
    """
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
# FEATURE PREPARATION
# ============================================================
def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    """
    Return all sensor feature columns.
    """
    feature_columns = [
        column
        for column in dataframe.columns
        if column.startswith(SENSOR_PREFIX)
    ]

    if not feature_columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return feature_columns


def separate_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate predictors and binary target.
    """
    missing_columns = [
        column
        for column in feature_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Expected feature columns are missing: "
            f"{missing_columns[:10]}"
        )

    x_data = dataframe[feature_columns].copy()
    y_data = dataframe[TARGET_COLUMN].astype(int).copy()

    return x_data, y_data


# ============================================================
# IMBALANCE PARAMETERS
# ============================================================
def calculate_scale_pos_weight(
    target: pd.Series,
) -> float:
    """
    Calculate negative-to-positive ratio for boosting models.
    """
    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "The development dataset contains no failure examples."
        )

    return float(
        negative_count / positive_count
    )


# ============================================================
# PIPELINE BUILDERS
# ============================================================
def build_scaled_preprocessor() -> Pipeline:
    """
    Preprocessing for scale-sensitive models.
    """
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
                "variance_filter",
                VarianceThreshold(
                    threshold=0.0,
                ),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
        ]
    )


def build_tree_preprocessor() -> Pipeline:
    """
    Preprocessing for tree-based models.
    """
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
                "variance_filter",
                VarianceThreshold(
                    threshold=0.0,
                ),
            ),
        ]
    )


def build_model_registry(
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    """
    Create all advanced model pipelines.
    """
    models: dict[str, Pipeline] = {
        "Regularized Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_scaled_preprocessor(),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=0.25,
                        penalty="l2",
                        solver="liblinear",
                        class_weight="balanced",
                        max_iter=5000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Balanced Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
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
        ),
        "Easy Ensemble": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "model",
                    EasyEnsembleClassifier(
                        n_estimators=20,
                        sampling_strategy="auto",
                        replacement=False,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=500,
                        learning_rate=0.03,
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
        ),
        "LightGBM": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "model",
                    LGBMClassifier(
                        objective="binary",
                        n_estimators=500,
                        learning_rate=0.03,
                        num_leaves=15,
                        max_depth=-1,
                        min_child_samples=20,
                        subsample=0.85,
                        colsample_bytree=0.70,
                        reg_alpha=0.10,
                        reg_lambda=2.0,
                        class_weight="balanced",
                        verbosity=-1,
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# CROSS-VALIDATION
# ============================================================
def build_cross_validator() -> RepeatedStratifiedKFold:
    """
    Create repeated stratified cross-validation.
    """
    return RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )


def get_scoring_metrics() -> dict[str, str]:
    """
    Define cross-validation scoring metrics.
    """
    return {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "recall_fail": "recall",
        "precision_fail": "precision",
        "f1_fail": "f1",
        "balanced_accuracy": "balanced_accuracy",
        "neg_brier": "neg_brier_score",
    }


def benchmark_models(
    models: dict[str, Pipeline],
    x_development: pd.DataFrame,
    y_development: pd.Series,
    cross_validator: RepeatedStratifiedKFold,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Benchmark models using repeated stratified cross-validation.
    """
    comparison_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    scoring = get_scoring_metrics()

    for model_name, pipeline in models.items():
        LOGGER.info(
            "Cross-validating model: %s",
            model_name,
        )

        start_time = time.perf_counter()

        results = cross_validate(
            estimator=pipeline,
            X=x_development,
            y=y_development,
            scoring=scoring,
            cv=cross_validator,
            n_jobs=-1,
            return_train_score=False,
            error_score="raise",
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        metric_mapping = {
            "roc_auc": "test_roc_auc",
            "pr_auc": "test_pr_auc",
            "recall_fail": "test_recall_fail",
            "precision_fail": "test_precision_fail",
            "f1_fail": "test_f1_fail",
            "balanced_accuracy": (
                "test_balanced_accuracy"
            ),
            "brier_score": "test_neg_brier",
        }

        summary: dict[str, Any] = {
            "model": model_name,
            "cv_folds": (
                CV_SPLITS * CV_REPEATS
            ),
            "elapsed_seconds": elapsed_seconds,
        }

        for output_name, result_key in metric_mapping.items():
            values = np.asarray(
                results[result_key],
                dtype=float,
            )

            if output_name == "brier_score":
                values = -values

            summary[f"cv_{output_name}_mean"] = float(
                np.mean(values)
            )
            summary[f"cv_{output_name}_std"] = float(
                np.std(values, ddof=1)
            )

            for fold_index, value in enumerate(
                values,
                start=1,
            ):
                fold_rows.append(
                    {
                        "model": model_name,
                        "metric": output_name,
                        "fold": fold_index,
                        "value": float(value),
                    }
                )

        comparison_rows.append(summary)

    comparison_df = pd.DataFrame(
        comparison_rows
    ).sort_values(
        by=PRIMARY_SELECTION_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    fold_results_df = pd.DataFrame(
        fold_rows
    )

    return comparison_df, fold_results_df


# ============================================================
# OUT-OF-FOLD PREDICTIONS
# ============================================================
def generate_oof_probabilities(
    model: Pipeline,
    x_development: pd.DataFrame,
    y_development: pd.Series,
) -> np.ndarray:
    """
    Generate one probability for each development record using
    non-repeated stratified cross-validation.

    A non-repeated five-fold procedure is used here so each row receives
    exactly one out-of-fold probability.
    """
    LOGGER.info(
        "Generating out-of-fold probabilities for threshold selection"
    )

    oof_cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS,
        n_repeats=1,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        estimator=clone(model),
        X=x_development,
        y=y_development,
        cv=oof_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return probabilities


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================
def build_threshold_analysis(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    """
    Evaluate thresholds from 0.01 to 0.99.
    """
    rows: list[dict[str, Any]] = []

    thresholds = np.linspace(
        0.01,
        0.99,
        99,
    )

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).ravel()

        precision = precision_score(
            y_true,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
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

        rows.append(
            {
                "threshold": float(threshold),
                "precision_fail": float(precision),
                "recall_fail": float(recall),
                "f1_fail": float(f1),
                "balanced_accuracy": float(
                    balanced_accuracy
                ),
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
                "predicted_failures": int(
                    predictions.sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def select_operational_threshold(
    threshold_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Select the highest-precision threshold that maintains the required
    minimum failure recall.

    If no threshold achieves the recall requirement, use the threshold
    with the highest recall and then highest precision.
    """
    eligible = threshold_df[
        threshold_df["recall_fail"]
        >= MINIMUM_FAILURE_RECALL
    ].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
            by=[
                "precision_fail",
                "f1_fail",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "Highest precision while maintaining "
            f"failure recall >= {MINIMUM_FAILURE_RECALL:.2f}"
        )

    else:
        selected = threshold_df.sort_values(
            by=[
                "recall_fail",
                "precision_fail",
                "f1_fail",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "Recall requirement was not achieved; selected the "
            "highest available recall"
        )

    return {
        "threshold": float(selected["threshold"]),
        "precision_fail": float(
            selected["precision_fail"]
        ),
        "recall_fail": float(
            selected["recall_fail"]
        ),
        "f1_fail": float(
            selected["f1_fail"]
        ),
        "balanced_accuracy": float(
            selected["balanced_accuracy"]
        ),
        "selection_reason": selection_reason,
    }


# ============================================================
# METRICS
# ============================================================
def calculate_binary_metrics(
    y_true: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, Any]:
    """
    Calculate probability and threshold-based metrics.
    """
    y_array = np.asarray(y_true)
    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        f"{prefix}_roc_auc": float(
            roc_auc_score(
                y_array,
                probabilities,
            )
        ),
        f"{prefix}_pr_auc": float(
            average_precision_score(
                y_array,
                probabilities,
            )
        ),
        f"{prefix}_brier_score": float(
            brier_score_loss(
                y_array,
                probabilities,
            )
        ),
        f"{prefix}_recall_fail": float(
            recall_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_precision_fail": float(
            precision_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_f1_fail": float(
            f1_score(
                y_array,
                predictions,
                zero_division=0,
            )
        ),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(
                y_array,
                predictions,
            )
        ),
        f"{prefix}_true_negative": int(tn),
        f"{prefix}_false_positive": int(fp),
        f"{prefix}_false_negative": int(fn),
        f"{prefix}_true_positive": int(tp),
        f"{prefix}_predicted_failures": int(
            predictions.sum()
        ),
    }


# ============================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
def bootstrap_metric_interval(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
    metric_name: str,
) -> tuple[float, float, float]:
    """
    Calculate bootstrap confidence intervals.

    Stratified resampling is used manually because the positive class is
    small and ordinary resampling may occasionally omit failures.
    """
    random_generator = np.random.default_rng(
        RANDOM_STATE
    )

    positive_indices = np.where(
        y_true == 1
    )[0]

    negative_indices = np.where(
        y_true == 0
    )[0]

    bootstrap_values: list[float] = []

    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled_positive = random_generator.choice(
            positive_indices,
            size=len(positive_indices),
            replace=True,
        )

        sampled_negative = random_generator.choice(
            negative_indices,
            size=len(negative_indices),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                sampled_positive,
                sampled_negative,
            ]
        )

        sampled_true = y_true[
            sampled_indices
        ]

        sampled_probabilities = probabilities[
            sampled_indices
        ]

        sampled_predictions = (
            sampled_probabilities >= threshold
        ).astype(int)

        if metric_name == "roc_auc":
            value = roc_auc_score(
                sampled_true,
                sampled_probabilities,
            )

        elif metric_name == "pr_auc":
            value = average_precision_score(
                sampled_true,
                sampled_probabilities,
            )

        elif metric_name == "recall_fail":
            value = recall_score(
                sampled_true,
                sampled_predictions,
                zero_division=0,
            )

        elif metric_name == "precision_fail":
            value = precision_score(
                sampled_true,
                sampled_predictions,
                zero_division=0,
            )

        elif metric_name == "f1_fail":
            value = f1_score(
                sampled_true,
                sampled_predictions,
                zero_division=0,
            )

        elif metric_name == "balanced_accuracy":
            value = balanced_accuracy_score(
                sampled_true,
                sampled_predictions,
            )

        else:
            raise ValueError(
                f"Unsupported bootstrap metric: {metric_name}"
            )

        bootstrap_values.append(
            float(value)
        )

    alpha = (
        1.0 - BOOTSTRAP_CONFIDENCE_LEVEL
    ) / 2.0

    lower_percentile = 100 * alpha
    upper_percentile = 100 * (
        1.0 - alpha
    )

    point_metrics = calculate_binary_metrics(
        y_true=y_true,
        probabilities=probabilities,
        threshold=threshold,
        prefix="point",
    )

    point_key_mapping = {
        "roc_auc": "point_roc_auc",
        "pr_auc": "point_pr_auc",
        "recall_fail": "point_recall_fail",
        "precision_fail": "point_precision_fail",
        "f1_fail": "point_f1_fail",
        "balanced_accuracy": (
            "point_balanced_accuracy"
        ),
    }

    point_estimate = float(
        point_metrics[
            point_key_mapping[metric_name]
        ]
    )

    lower_bound = float(
        np.percentile(
            bootstrap_values,
            lower_percentile,
        )
    )

    upper_bound = float(
        np.percentile(
            bootstrap_values,
            upper_percentile,
        )
    )

    return (
        point_estimate,
        lower_bound,
        upper_bound,
    )


def build_confidence_interval_report(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Build confidence intervals for the primary test metrics.
    """
    metric_names = [
        "roc_auc",
        "pr_auc",
        "recall_fail",
        "precision_fail",
        "f1_fail",
        "balanced_accuracy",
    ]

    rows = []

    for metric_name in metric_names:
        LOGGER.info(
            "Bootstrapping confidence interval: %s",
            metric_name,
        )

        estimate, lower, upper = (
            bootstrap_metric_interval(
                y_true=y_true,
                probabilities=probabilities,
                threshold=threshold,
                metric_name=metric_name,
            )
        )

        rows.append(
            {
                "metric": metric_name,
                "estimate": estimate,
                "ci_lower": lower,
                "ci_upper": upper,
                "confidence_level": (
                    BOOTSTRAP_CONFIDENCE_LEVEL
                ),
                "bootstrap_resamples": (
                    BOOTSTRAP_RESAMPLES
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# FINAL MODEL
# ============================================================
def fit_final_model(
    pipeline: Pipeline,
    x_development: pd.DataFrame,
    y_development: pd.Series,
) -> Pipeline:
    """
    Fit the selected pipeline using all development data.
    """
    final_pipeline = clone(pipeline)

    final_pipeline.fit(
        x_development,
        y_development,
    )

    return final_pipeline


# ============================================================
# PREDICTION TABLES
# ============================================================
def build_oof_prediction_table(
    development_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """
    Build the development out-of-fold prediction table.
    """
    output = development_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    output["fail_probability"] = probabilities
    output["predicted_target"] = (
        probabilities >= threshold
    ).astype(int)

    output["predicted_status"] = np.where(
        output["predicted_target"] == 1,
        "Fail",
        "Pass",
    )

    output["decision_threshold"] = threshold
    output["model_name"] = model_name
    output["prediction_source"] = "Out-of-fold"

    return output


def build_test_prediction_table(
    test_df: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    model_name: str,
) -> pd.DataFrame:
    """
    Build the final test prediction table.
    """
    output = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    output["fail_probability"] = probabilities
    output["predicted_target"] = (
        probabilities >= threshold
    ).astype(int)

    output["predicted_status"] = np.where(
        output["predicted_target"] == 1,
        "Fail",
        "Pass",
    )

    output["decision_threshold"] = threshold
    output["model_name"] = model_name
    output["prediction_source"] = "Held-out test"

    output["prediction_correct"] = (
        output["predicted_target"]
        == output[TARGET_COLUMN]
    )

    output["error_type"] = np.select(
        [
            (
                (output[TARGET_COLUMN] == 1)
                & (output["predicted_target"] == 0)
            ),
            (
                (output[TARGET_COLUMN] == 0)
                & (output["predicted_target"] == 1)
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
# VISUALISATIONS
# ============================================================
def save_cv_metric_plot(
    comparison_df: pd.DataFrame,
) -> None:
    """
    Save PR-AUC comparison chart.
    """
    plot_df = comparison_df.sort_values(
        by="cv_pr_auc_mean",
        ascending=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.barh(
        plot_df["model"],
        plot_df["cv_pr_auc_mean"],
        xerr=plot_df["cv_pr_auc_std"],
    )

    axis.set_title(
        "Repeated Cross-Validation PR-AUC"
    )
    axis.set_xlabel(
        "Mean PR-AUC"
    )
    axis.set_ylabel(
        "Model"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "cv_pr_auc_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_threshold_tradeoff_plot(
    threshold_df: pd.DataFrame,
    selected_threshold: float,
) -> None:
    """
    Plot precision, recall and F1 across thresholds.
    """
    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.plot(
        threshold_df["threshold"],
        threshold_df["precision_fail"],
        label="Failure precision",
    )

    axis.plot(
        threshold_df["threshold"],
        threshold_df["recall_fail"],
        label="Failure recall",
    )

    axis.plot(
        threshold_df["threshold"],
        threshold_df["f1_fail"],
        label="Failure F1",
    )

    axis.axvline(
        selected_threshold,
        linestyle="--",
        label=(
            f"Selected threshold "
            f"({selected_threshold:.2f})"
        ),
    )

    axis.axhline(
        MINIMUM_FAILURE_RECALL,
        linestyle=":",
        label=(
            f"Minimum recall "
            f"({MINIMUM_FAILURE_RECALL:.2f})"
        ),
    )

    axis.set_title(
        "Operational Threshold Trade-Off"
    )
    axis.set_xlabel(
        "Decision Threshold"
    )
    axis.set_ylabel(
        "Metric Value"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "threshold_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_roc_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    """
    Save the final test ROC curve.
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(
        y_true,
        probabilities,
    )

    roc_auc = roc_auc_score(
        y_true,
        probabilities,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"{model_name} (AUC={roc_auc:.3f})",
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    axis.set_title(
        "Held-Out Test ROC Curve"
    )
    axis.set_xlabel(
        "False Positive Rate"
    )
    axis.set_ylabel(
        "True Positive Rate"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "test_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_precision_recall_curve(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    """
    Save the final test precision-recall curve.
    """
    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    baseline = float(
        np.mean(y_true)
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.plot(
        recall,
        precision,
        label=f"{model_name} (AP={pr_auc:.3f})",
    )

    axis.axhline(
        baseline,
        linestyle="--",
        label=(
            f"Failure prevalence "
            f"({baseline:.3f})"
        ),
    )

    axis.set_title(
        "Held-Out Test Precision-Recall Curve"
    )
    axis.set_xlabel(
        "Recall"
    )
    axis.set_ylabel(
        "Precision"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "test_precision_recall_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
) -> None:
    """
    Save the final test confusion matrix.
    """
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(
        figsize=(6, 5)
    )

    image = axis.imshow(matrix)

    axis.set_title(
        f"Test Confusion Matrix: {model_name}"
    )
    axis.set_xlabel(
        "Predicted Class"
    )
    axis.set_ylabel(
        "Actual Class"
    )

    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])

    axis.set_xticklabels(
        ["Pass", "Fail"]
    )
    axis.set_yticklabels(
        ["Pass", "Fail"]
    )

    for row_index in range(
        matrix.shape[0]
    ):
        for column_index in range(
            matrix.shape[1]
        ):
            axis.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_calibration_plot(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    """
    Save the final test calibration plot.
    """
    fraction_positive, mean_probability = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=8,
            strategy="quantile",
        )
    )

    figure, axis = plt.subplots(
        figsize=(7, 6)
    )

    axis.plot(
        mean_probability,
        fraction_positive,
        marker="o",
        label=model_name,
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.set_title(
        "Held-Out Test Probability Calibration"
    )
    axis.set_xlabel(
        "Mean Predicted Failure Probability"
    )
    axis.set_ylabel(
        "Observed Failure Rate"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "test_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# SAVE METADATA
# ============================================================
def save_feature_metadata(
    feature_columns: list[str],
) -> None:
    """
    Save feature names used by the final pipeline.
    """
    payload = {
        "feature_count": len(feature_columns),
        "features": feature_columns,
    }

    with FEATURE_COLUMNS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            payload,
            file,
            indent=4,
        )


def save_summary(
    summary: dict[str, Any],
) -> None:
    """
    Save the advanced-model summary.
    """
    with BEST_MODEL_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            default=str,
        )


# ============================================================
# CONSOLE OUTPUT
# ============================================================
def print_console_summary(
    comparison_df: pd.DataFrame,
    best_model_name: str,
    threshold_result: dict[str, Any],
    test_metrics: dict[str, Any],
    confidence_intervals: pd.DataFrame,
) -> None:
    """
    Print a readable terminal summary.
    """
    print("\n" + "=" * 104)
    print("HEVEMIND ADVANCED MODEL BENCHMARK")
    print("=" * 104)

    display_columns = [
        "model",
        "cv_roc_auc_mean",
        "cv_roc_auc_std",
        "cv_pr_auc_mean",
        "cv_pr_auc_std",
        "cv_recall_fail_mean",
        "cv_precision_fail_mean",
        "cv_f1_fail_mean",
        "cv_balanced_accuracy_mean",
    ]

    print(
        comparison_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\n" + "=" * 104)
    print("SELECTED ADVANCED MODEL")
    print("=" * 104)

    print(
        f"Model:                       "
        f"{best_model_name}"
    )
    print(
        f"Operational threshold:       "
        f"{threshold_result['threshold']:.4f}"
    )
    print(
        f"Threshold selection:         "
        f"{threshold_result['selection_reason']}"
    )

    print("\nHeld-out test performance:")

    print(
        f"ROC-AUC:                     "
        f"{test_metrics['test_roc_auc']:.4f}"
    )
    print(
        f"PR-AUC:                      "
        f"{test_metrics['test_pr_auc']:.4f}"
    )
    print(
        f"Brier score:                 "
        f"{test_metrics['test_brier_score']:.4f}"
    )
    print(
        f"Failure recall:              "
        f"{test_metrics['test_recall_fail']:.4f}"
    )
    print(
        f"Failure precision:           "
        f"{test_metrics['test_precision_fail']:.4f}"
    )
    print(
        f"Failure F1:                  "
        f"{test_metrics['test_f1_fail']:.4f}"
    )
    print(
        f"Balanced accuracy:           "
        f"{test_metrics['test_balanced_accuracy']:.4f}"
    )

    print("\nHeld-out test confusion matrix:")

    print(
        f"True Pass:                   "
        f"{test_metrics['test_true_negative']}"
    )
    print(
        f"False Alarm:                 "
        f"{test_metrics['test_false_positive']}"
    )
    print(
        f"Missed Failure:              "
        f"{test_metrics['test_false_negative']}"
    )
    print(
        f"Detected Failure:            "
        f"{test_metrics['test_true_positive']}"
    )

    print(
        "\nBootstrap confidence intervals:"
    )

    print(
        confidence_intervals
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\nSaved outputs:")

    print(
        f"Best pipeline:               "
        f"{BEST_PIPELINE_PATH}"
    )
    print(
        f"Model comparison:            "
        f"{MODEL_COMPARISON_PATH}"
    )
    print(
        f"Test predictions:            "
        f"{TEST_PREDICTIONS_PATH}"
    )
    print(
        f"Advanced report folder:      "
        f"{ADVANCED_REPORTS_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading development and test datasets"
    )

    development_df, test_df = load_datasets()

    feature_columns = get_sensor_columns(
        development_df
    )

    x_development, y_development = (
        separate_features_and_target(
            development_df,
            feature_columns,
        )
    )

    x_test, y_test = (
        separate_features_and_target(
            test_df,
            feature_columns,
        )
    )

    scale_pos_weight = (
        calculate_scale_pos_weight(
            y_development
        )
    )

    LOGGER.info(
        "Development rows: %s",
        len(development_df),
    )

    LOGGER.info(
        "Test rows: %s",
        len(test_df),
    )

    LOGGER.info(
        "Sensor features: %s",
        len(feature_columns),
    )

    LOGGER.info(
        "Scale positive weight: %.4f",
        scale_pos_weight,
    )

    models = build_model_registry(
        scale_pos_weight=scale_pos_weight
    )

    cross_validator = build_cross_validator()

    comparison_df, fold_results_df = (
        benchmark_models(
            models=models,
            x_development=x_development,
            y_development=y_development,
            cross_validator=cross_validator,
        )
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    fold_results_df.to_csv(
        CV_FOLD_RESULTS_PATH,
        index=False,
    )

    best_model_name = str(
        comparison_df.iloc[0]["model"]
    )

    best_pipeline_template = models[
        best_model_name
    ]

    LOGGER.info(
        "Best cross-validation model: %s",
        best_model_name,
    )

    oof_probabilities = (
        generate_oof_probabilities(
            model=best_pipeline_template,
            x_development=x_development,
            y_development=y_development,
        )
    )

    threshold_df = build_threshold_analysis(
        y_true=y_development,
        probabilities=oof_probabilities,
    )

    threshold_df.to_csv(
        THRESHOLD_REPORT_PATH,
        index=False,
    )

    threshold_result = (
        select_operational_threshold(
            threshold_df
        )
    )

    selected_threshold = float(
        threshold_result["threshold"]
    )

    oof_prediction_table = (
        build_oof_prediction_table(
            development_df=development_df,
            probabilities=oof_probabilities,
            threshold=selected_threshold,
            model_name=best_model_name,
        )
    )

    oof_prediction_table.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    LOGGER.info(
        "Fitting final model on all development data"
    )

    final_pipeline = fit_final_model(
        pipeline=best_pipeline_template,
        x_development=x_development,
        y_development=y_development,
    )

    test_probabilities = (
        final_pipeline.predict_proba(
            x_test
        )[:, 1]
    )

    test_predictions = (
        test_probabilities
        >= selected_threshold
    ).astype(int)

    test_metrics = calculate_binary_metrics(
        y_true=y_test,
        probabilities=test_probabilities,
        threshold=selected_threshold,
        prefix="test",
    )

    test_metrics[
        "classification_report"
    ] = classification_report(
        y_test,
        test_predictions,
        labels=[0, 1],
        target_names=[
            "Pass",
            "Fail",
        ],
        zero_division=0,
        output_dict=True,
    )

    test_prediction_table = (
        build_test_prediction_table(
            test_df=test_df,
            probabilities=test_probabilities,
            threshold=selected_threshold,
            model_name=best_model_name,
        )
    )

    test_prediction_table.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    confidence_intervals = (
        build_confidence_interval_report(
            y_true=y_test.to_numpy(),
            probabilities=test_probabilities,
            threshold=selected_threshold,
        )
    )

    confidence_intervals.to_csv(
        TEST_CONFIDENCE_INTERVALS_PATH,
        index=False,
    )

    joblib.dump(
        final_pipeline,
        BEST_PIPELINE_PATH,
    )

    joblib.dump(
        final_pipeline.named_steps["model"],
        BEST_MODEL_PATH,
    )

    save_feature_metadata(
        feature_columns
    )

    summary = {
        "project": "HeveMind",
        "model_stage": "Advanced benchmark",
        "best_model": best_model_name,
        "model_selection_metric": (
            PRIMARY_SELECTION_METRIC
        ),
        "development_rows": int(
            len(development_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "feature_count": int(
            len(feature_columns)
        ),
        "failure_prevalence_development": float(
            y_development.mean()
        ),
        "scale_pos_weight": scale_pos_weight,
        "cross_validation": {
            "splits": CV_SPLITS,
            "repeats": CV_REPEATS,
            "total_folds": (
                CV_SPLITS * CV_REPEATS
            ),
        },
        "threshold_policy": {
            "minimum_failure_recall": (
                MINIMUM_FAILURE_RECALL
            ),
            **threshold_result,
        },
        "cross_validation_results": (
            comparison_df.to_dict(
                orient="records"
            )
        ),
        "test_results": test_metrics,
        "bootstrap_confidence_intervals": (
            confidence_intervals.to_dict(
                orient="records"
            )
        ),
    }

    save_summary(summary)

    save_cv_metric_plot(
        comparison_df
    )

    save_threshold_tradeoff_plot(
        threshold_df=threshold_df,
        selected_threshold=selected_threshold,
    )

    save_test_roc_curve(
        y_true=y_test,
        probabilities=test_probabilities,
        model_name=best_model_name,
    )

    save_test_precision_recall_curve(
        y_true=y_test,
        probabilities=test_probabilities,
        model_name=best_model_name,
    )

    save_test_confusion_matrix(
        y_true=y_test,
        predictions=test_predictions,
        model_name=best_model_name,
    )

    save_test_calibration_plot(
        y_true=y_test,
        probabilities=test_probabilities,
        model_name=best_model_name,
    )

    print_console_summary(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        threshold_result=threshold_result,
        test_metrics=test_metrics,
        confidence_intervals=confidence_intervals,
    )


if __name__ == "__main__":
    main()