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
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
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
    StratifiedKFold,
    cross_val_predict,
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
OPERATIONAL_MODELS_DIR = MODELS_DIR / "operational"
METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = ROOT_DIR / "reports"
OPERATIONAL_REPORTS_DIR = REPORTS_DIR / "operational_model_selection"
TABLES_DIR = OPERATIONAL_REPORTS_DIR / "tables"
FIGURES_DIR = OPERATIONAL_REPORTS_DIR / "figures"

MODEL_COMPARISON_PATH = (
    TABLES_DIR / "operational_model_comparison.csv"
)

ALL_THRESHOLD_RESULTS_PATH = (
    TABLES_DIR / "all_model_threshold_analysis.csv"
)

OOF_PREDICTIONS_PATH = (
    TABLES_DIR / "all_model_oof_predictions.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR / "selected_model_test_predictions.csv"
)

TEST_METRICS_PATH = (
    TABLES_DIR / "selected_model_test_metrics.csv"
)

BEST_PIPELINE_PATH = (
    OPERATIONAL_MODELS_DIR
    / "best_operational_pipeline.joblib"
)

BEST_MODEL_PATH = (
    OPERATIONAL_MODELS_DIR
    / "best_operational_model.joblib"
)

SUMMARY_PATH = (
    OPERATIONAL_REPORTS_DIR
    / "operational_model_summary.json"
)

FEATURE_METADATA_PATH = (
    METADATA_DIR
    / "operational_feature_columns.json"
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
CV_REPEATS = 3

MINIMUM_FAILURE_RECALL = 0.75

MISSED_FAILURE_COST = 10.0
FALSE_ALARM_COST = 1.0

MINIMUM_THRESHOLD = 0.005
MAXIMUM_THRESHOLD = 0.995
THRESHOLD_STEP = 0.005


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        OPERATIONAL_MODELS_DIR,
        METADATA_DIR,
        OPERATIONAL_REPORTS_DIR,
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
            f"Required split not found: {path}\n"
            "Run src/02_data_audit.py first."
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


def split_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    x_data = dataframe[feature_columns].copy()
    y_data = dataframe[TARGET_COLUMN].astype(int).copy()

    return x_data, y_data


def calculate_scale_pos_weight(
    target: pd.Series,
) -> float:
    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "No failure samples were found."
        )

    return float(
        negative_count / positive_count
    )


# ============================================================
# PREPROCESSING
# ============================================================
def build_scaled_preprocessor() -> Pipeline:
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


# ============================================================
# MODEL REGISTRY
# ============================================================
def build_model_registry(
    scale_pos_weight: float,
) -> dict[str, Pipeline]:
    return {
        "Baseline Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_tree_preprocessor(),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=600,
                        class_weight="balanced",
                        max_depth=None,
                        min_samples_split=5,
                        min_samples_leaf=2,
                        max_features="sqrt",
                        n_jobs=-1,
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

    alert_count = int(
        predictions.sum()
    )

    alert_rate = float(
        alert_count / len(predictions)
    )

    false_alarms_per_detected_failure = (
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
        "alert_count": alert_count,
        "alert_rate": alert_rate,
        "false_alarms_per_detected_failure": (
            false_alarms_per_detected_failure
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
    thresholds = np.arange(
        MINIMUM_THRESHOLD,
        MAXIMUM_THRESHOLD + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    rows = []

    y_array = y_true.to_numpy()

    for threshold in thresholds:
        metrics = calculate_threshold_metrics(
            y_true=y_array,
            probabilities=probabilities,
            threshold=float(threshold),
        )

        metrics["model"] = model_name
        rows.append(metrics)

    return pd.DataFrame(rows)


def select_model_threshold(
    threshold_table: pd.DataFrame,
) -> dict[str, Any]:
    eligible = threshold_table[
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
            "Minimum operational cost among thresholds "
            f"with failure recall >= {MINIMUM_FAILURE_RECALL:.2f}"
        )

    else:
        selected = threshold_table.sort_values(
            by=[
                "recall_fail",
                "operational_cost",
                "false_positive",
                "precision_fail",
            ],
            ascending=[
                False,
                True,
                True,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "No threshold achieved the required recall; "
            "selected highest-recall alternative"
        )

    result = selected.to_dict()
    result["selection_reason"] = selection_reason

    return result


# ============================================================
# MODEL-LEVEL EVALUATION
# ============================================================
def evaluate_model_oof(
    model_name: str,
    probabilities: np.ndarray,
    y_true: pd.Series,
    threshold_result: dict[str, Any],
) -> dict[str, Any]:
    threshold = float(
        threshold_result["threshold"]
    )

    metrics = calculate_threshold_metrics(
        y_true=y_true.to_numpy(),
        probabilities=probabilities,
        threshold=threshold,
    )

    return {
        "model": model_name,
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "brier_score": float(
            brier_score_loss(
                y_true,
                probabilities,
            )
        ),
        **metrics,
        "selection_reason": (
            threshold_result["selection_reason"]
        ),
    }


# ============================================================
# MODEL SELECTION
# ============================================================
def select_best_operational_model(
    comparison_df: pd.DataFrame,
) -> pd.Series:
    eligible = comparison_df[
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

    return selected


# ============================================================
# FINAL TEST EVALUATION
# ============================================================
def evaluate_selected_model_on_test(
    pipeline: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
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
            "classification_report": (
                classification_report(
                    y_test,
                    predictions,
                    labels=[0, 1],
                    target_names=[
                        "Pass",
                        "Fail",
                    ],
                    zero_division=0,
                    output_dict=True,
                )
            ),
        }
    )

    return metrics, probabilities, predictions


# ============================================================
# OUTPUT TABLES
# ============================================================
def build_oof_prediction_table(
    development_df: pd.DataFrame,
    model_probabilities: dict[str, np.ndarray],
    selected_thresholds: dict[str, float],
) -> pd.DataFrame:
    output = development_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    for model_name, probabilities in model_probabilities.items():
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        threshold = selected_thresholds[
            model_name
        ]

        output[
            f"{safe_name}_fail_probability"
        ] = probabilities

        output[
            f"{safe_name}_prediction"
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

    output["prediction_correct"] = (
        output[TARGET_COLUMN]
        == output["predicted_target"]
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
def save_operational_cost_plot(
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
        "Development Operational Cost by Model"
    )
    axis.set_xlabel(
        "Weighted Operational Cost"
    )
    axis.set_ylabel(
        "Model"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "operational_cost_comparison.png",
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
        "Development False Alarms by Model"
    )
    axis.set_xlabel(
        "False Alarms"
    )
    axis.set_ylabel(
        "Model"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "false_alarm_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_precision_recall_operational_plot(
    comparison_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    for _, row in comparison_df.iterrows():
        axis.scatter(
            row["recall_fail"],
            row["precision_fail"],
            s=90,
        )

        axis.annotate(
            row["model"],
            (
                row["recall_fail"],
                row["precision_fail"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    axis.axvline(
        MINIMUM_FAILURE_RECALL,
        linestyle="--",
        label=(
            f"Minimum recall "
            f"{MINIMUM_FAILURE_RECALL:.2f}"
        ),
    )

    axis.set_title(
        "Operational Precision-Recall Trade-Off"
    )
    axis.set_xlabel(
        "Failure Recall"
    )
    axis.set_ylabel(
        "Failure Precision"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "operational_precision_recall_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_selected_test_confusion_matrix(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
) -> None:
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
        f"Selected Model Test Confusion Matrix\n{model_name}"
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
        FIGURES_DIR
        / "selected_model_test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_selected_test_curves(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
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
        "Selected Model Held-Out Test ROC Curve"
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
        FIGURES_DIR / "selected_model_test_roc_curve.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)

    precision, recall, _ = precision_recall_curve(
        y_true,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_true,
        probabilities,
    )

    baseline = float(
        y_true.mean()
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
        "Selected Model Held-Out Test Precision-Recall Curve"
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
        FIGURES_DIR
        / "selected_model_test_precision_recall_curve.png",
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
    payload = {
        "feature_count": len(feature_columns),
        "features": feature_columns,
    }

    with FEATURE_METADATA_PATH.open(
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
    with SUMMARY_PATH.open(
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
    selected_model_name: str,
    selected_threshold: float,
    test_metrics: dict[str, Any],
) -> None:
    print("\n" + "=" * 120)
    print("HEVEMIND OPERATIONAL MODEL SELECTION")
    print("=" * 120)

    display_columns = [
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
        "true_positive",
        "false_alarms_per_detected_failure",
        "operational_cost",
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

    print("\n" + "=" * 120)
    print("SELECTED OPERATIONAL MODEL")
    print("=" * 120)

    print(
        f"Model:                          "
        f"{selected_model_name}"
    )

    print(
        f"Decision threshold:             "
        f"{selected_threshold:.4f}"
    )

    print("\nHeld-out test performance:")

    print(
        f"ROC-AUC:                        "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                         "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:                    "
        f"{test_metrics['brier_score']:.4f}"
    )

    print(
        f"Failure recall:                 "
        f"{test_metrics['recall_fail']:.4f}"
    )

    print(
        f"Failure precision:              "
        f"{test_metrics['precision_fail']:.4f}"
    )

    print(
        f"Failure F1:                     "
        f"{test_metrics['f1_fail']:.4f}"
    )

    print(
        f"Balanced accuracy:              "
        f"{test_metrics['balanced_accuracy']:.4f}"
    )

    print("\nHeld-out test confusion matrix:")

    print(
        f"True Pass:                      "
        f"{test_metrics['true_negative']}"
    )

    print(
        f"False Alarm:                    "
        f"{test_metrics['false_positive']}"
    )

    print(
        f"Missed Failure:                 "
        f"{test_metrics['false_negative']}"
    )

    print(
        f"Detected Failure:               "
        f"{test_metrics['true_positive']}"
    )

    print(
        f"False alarms per detection:     "
        f"{test_metrics['false_alarms_per_detected_failure']:.4f}"
    )

    print(
        f"Weighted operational cost:      "
        f"{test_metrics['operational_cost']:.4f}"
    )

    print("\nSaved outputs:")

    print(
        f"Best pipeline:                  "
        f"{BEST_PIPELINE_PATH}"
    )

    print(
        f"Model comparison:               "
        f"{MODEL_COMPARISON_PATH}"
    )

    print(
        f"Test predictions:               "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"Operational report folder:      "
        f"{OPERATIONAL_REPORTS_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading development and test datasets"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    feature_columns = get_sensor_columns(
        development_df
    )

    x_development, y_development = (
        split_features_and_target(
            development_df,
            feature_columns,
        )
    )

    x_test, y_test = (
        split_features_and_target(
            test_df,
            feature_columns,
        )
    )

    scale_pos_weight = calculate_scale_pos_weight(
        y_development
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
        "Features: %s",
        len(feature_columns),
    )

    LOGGER.info(
        "Scale positive weight: %.4f",
        scale_pos_weight,
    )

    models = build_model_registry(
        scale_pos_weight=scale_pos_weight
    )

    model_probabilities: dict[str, np.ndarray] = {}
    selected_thresholds: dict[str, float] = {}

    comparison_rows: list[dict[str, Any]] = []
    all_threshold_tables: list[pd.DataFrame] = []

    for model_name, pipeline in models.items():
        LOGGER.info(
            "Generating out-of-fold probabilities: %s",
            model_name,
        )

        start_time = time.perf_counter()

        probabilities = generate_oof_probabilities(
            model=pipeline,
            x_data=x_development,
            y_data=y_development,
        )

        elapsed_seconds = (
            time.perf_counter() - start_time
        )

        threshold_table = build_threshold_table(
            model_name=model_name,
            y_true=y_development,
            probabilities=probabilities,
        )

        threshold_result = select_model_threshold(
            threshold_table
        )

        operational_result = evaluate_model_oof(
            model_name=model_name,
            probabilities=probabilities,
            y_true=y_development,
            threshold_result=threshold_result,
        )

        operational_result[
            "elapsed_seconds"
        ] = elapsed_seconds

        comparison_rows.append(
            operational_result
        )

        all_threshold_tables.append(
            threshold_table
        )

        model_probabilities[
            model_name
        ] = probabilities

        selected_thresholds[
            model_name
        ] = float(
            threshold_result["threshold"]
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    selected_row = select_best_operational_model(
        comparison_df
    )

    selected_model_name = str(
        selected_row["model"]
    )

    selected_threshold = float(
        selected_row["threshold"]
    )

    comparison_df["selected_model"] = (
        comparison_df["model"]
        == selected_model_name
    )

    comparison_df = comparison_df.sort_values(
        by=[
            "selected_model",
            "operational_cost",
            "false_positive",
        ],
        ascending=[
            False,
            True,
            True,
        ],
    ).reset_index(drop=True)

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    all_threshold_df = pd.concat(
        all_threshold_tables,
        axis=0,
        ignore_index=True,
    )

    all_threshold_df.to_csv(
        ALL_THRESHOLD_RESULTS_PATH,
        index=False,
    )

    oof_prediction_table = build_oof_prediction_table(
        development_df=development_df,
        model_probabilities=model_probabilities,
        selected_thresholds=selected_thresholds,
    )

    oof_prediction_table.to_csv(
        OOF_PREDICTIONS_PATH,
        index=False,
    )

    LOGGER.info(
        "Selected operational model: %s",
        selected_model_name,
    )

    final_pipeline = clone(
        models[selected_model_name]
    )

    final_pipeline.fit(
        x_development,
        y_development,
    )

    (
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_selected_model_on_test(
        pipeline=final_pipeline,
        x_test=x_test,
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

    test_metrics_table = pd.DataFrame(
        [
            {
                key: value
                for key, value in test_metrics.items()
                if key != "classification_report"
            }
        ]
    )

    test_metrics_table.to_csv(
        TEST_METRICS_PATH,
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

    save_operational_cost_plot(
        comparison_df
    )

    save_false_alarm_plot(
        comparison_df
    )

    save_precision_recall_operational_plot(
        comparison_df
    )

    save_selected_test_confusion_matrix(
        y_true=y_test,
        predictions=test_predictions,
        model_name=selected_model_name,
    )

    save_selected_test_curves(
        y_true=y_test,
        probabilities=test_probabilities,
        model_name=selected_model_name,
    )

    summary = {
        "project": "HeveMind",
        "stage": "Operational model selection",
        "selection_policy": {
            "minimum_failure_recall": (
                MINIMUM_FAILURE_RECALL
            ),
            "missed_failure_cost": (
                MISSED_FAILURE_COST
            ),
            "false_alarm_cost": (
                FALSE_ALARM_COST
            ),
            "model_selection_priority": [
                "minimum recall requirement",
                "minimum operational cost",
                "minimum false alarms",
                "maximum precision",
                "maximum balanced accuracy",
                "maximum PR-AUC",
            ],
        },
        "development_rows": int(
            len(development_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "feature_count": int(
            len(feature_columns)
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
        "held_out_test_results": (
            test_metrics
        ),
    }

    save_summary(summary)

    print_console_summary(
        comparison_df=comparison_df,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()