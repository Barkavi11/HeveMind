from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_class_weight


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

ARTIFACTS_DIR = ROOT_DIR / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
PREPROCESSORS_DIR = ARTIFACTS_DIR / "preprocessors"

REPORTS_DIR = ROOT_DIR / "reports"
MODEL_REPORTS_DIR = REPORTS_DIR / "baseline_models"
TABLES_DIR = MODEL_REPORTS_DIR / "tables"
FIGURES_DIR = MODEL_REPORTS_DIR / "figures"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VALIDATION_PATH = SPLITS_DIR / "validation.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

MODEL_COMPARISON_PATH = TABLES_DIR / "model_comparison.csv"
VALIDATION_PREDICTIONS_PATH = TABLES_DIR / "validation_predictions.csv"
TEST_PREDICTIONS_PATH = TABLES_DIR / "test_predictions.csv"
BEST_MODEL_SUMMARY_PATH = MODEL_REPORTS_DIR / "best_model_summary.json"

BEST_MODEL_PATH = MODELS_DIR / "best_baseline_model.joblib"
BEST_PIPELINE_PATH = MODELS_DIR / "best_baseline_pipeline.joblib"
FEATURE_COLUMNS_PATH = PREPROCESSORS_DIR / "feature_columns.json"


# ============================================================
# CONFIGURATION
# ============================================================
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
ID_COLUMN = "wafer_id"
TIMESTAMP_COLUMN = "timestamp"

SENSOR_PREFIX = "sensor_"

RANDOM_STATE = 42
DECISION_THRESHOLD = 0.50

PRIMARY_SELECTION_METRIC = "validation_pr_auc"


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        MODELS_DIR,
        PREPROCESSORS_DIR,
        MODEL_REPORTS_DIR,
        TABLES_DIR,
        FIGURES_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset split not found: {path}\n"
            "Run src/02_data_audit.py first."
        )

    dataframe = pd.read_parquet(path)

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_all_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = load_split(TRAIN_PATH)
    validation_df = load_split(VALIDATION_PATH)
    test_df = load_split(TEST_PATH)

    return train_df, validation_df, test_df


# ============================================================
# FEATURE PREPARATION
# ============================================================
def get_sensor_columns(dataframe: pd.DataFrame) -> list[str]:
    sensor_columns = [
        column
        for column in dataframe.columns
        if column.startswith(SENSOR_PREFIX)
    ]

    if not sensor_columns:
        raise ValueError("No sensor feature columns were found.")

    return sensor_columns


def separate_features_and_target(
    dataframe: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    missing_features = [
        feature
        for feature in feature_columns
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing expected features: {missing_features[:10]}"
        )

    features = dataframe[feature_columns].copy()
    target = dataframe[TARGET_COLUMN].astype(int).copy()

    return features, target


# ============================================================
# CLASS IMBALANCE
# ============================================================
def calculate_class_weights(target: pd.Series) -> dict[int, float]:
    classes = np.array([0, 1])

    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=target,
    )

    return {
        int(class_label): float(weight)
        for class_label, weight in zip(classes, weights)
    }


# ============================================================
# PREPROCESSING
# ============================================================
def build_numeric_preprocessor(
    feature_columns: list[str],
    use_scaling: bool,
) -> ColumnTransformer:
    steps: list[tuple[str, BaseEstimator]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
            ),
        )
    ]

    if use_scaling:
        steps.append(
            (
                "scaler",
                RobustScaler(),
            )
        )

    numeric_pipeline = Pipeline(steps=steps)

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                feature_columns,
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ============================================================
# MODELS
# ============================================================
def build_model_registry(
    feature_columns: list[str],
    class_weights: dict[int, float],
) -> dict[str, Pipeline]:
    logistic_preprocessor = build_numeric_preprocessor(
        feature_columns=feature_columns,
        use_scaling=True,
    )

    tree_preprocessor = build_numeric_preprocessor(
        feature_columns=feature_columns,
        use_scaling=False,
    )

    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                (
                    "preprocessor",
                    logistic_preprocessor,
                ),
                (
                    "model",
                    LogisticRegression(
                        class_weight=class_weights,
                        max_iter=5000,
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight=class_weights,
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
        "Extra Trees": Pipeline(
            steps=[
                (
                    "preprocessor",
                    tree_preprocessor,
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight=class_weights,
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
    }

    return models


# ============================================================
# METRICS
# ============================================================
def calculate_metrics(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)

    return {
        f"{prefix}_roc_auc": float(
            roc_auc_score(y_true, probabilities)
        ),
        f"{prefix}_pr_auc": float(
            average_precision_score(y_true, probabilities)
        ),
        f"{prefix}_recall_fail": float(
            recall_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        f"{prefix}_precision_fail": float(
            precision_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        f"{prefix}_f1_fail": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        f"{prefix}_balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        f"{prefix}_predicted_failures": int(
            predictions.sum()
        ),
    }


def calculate_confusion_values(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
    prefix: str,
) -> dict[str, int]:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    return {
        f"{prefix}_true_negative": int(tn),
        f"{prefix}_false_positive": int(fp),
        f"{prefix}_false_negative": int(fn),
        f"{prefix}_true_positive": int(tp),
    }


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================
def find_best_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(
        y_true,
        probabilities,
    )

    if len(thresholds) == 0:
        return {
            "threshold": 0.50,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
        }

    precision = precision[:-1]
    recall = recall[:-1]

    f1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    best_index = int(np.argmax(f1_scores))

    return {
        "threshold": float(thresholds[best_index]),
        "precision": float(precision[best_index]),
        "recall": float(recall[best_index]),
        "f1": float(f1_scores[best_index]),
    }


# ============================================================
# MODEL TRAINING
# ============================================================
def train_and_evaluate_models(
    models: dict[str, Pipeline],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_validation: pd.DataFrame,
    y_validation: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, Pipeline],
    dict[str, np.ndarray],
    dict[str, float],
]:
    result_rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    validation_probabilities: dict[str, np.ndarray] = {}
    optimized_thresholds: dict[str, float] = {}

    for model_name, pipeline in models.items():
        LOGGER.info("Training model: %s", model_name)

        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(
            x_validation
        )[:, 1]

        threshold_result = find_best_threshold(
            y_true=y_validation,
            probabilities=probabilities,
        )

        optimized_threshold = threshold_result["threshold"]

        result = {
            "model": model_name,
            "default_threshold": DECISION_THRESHOLD,
            "optimized_threshold": optimized_threshold,
            "optimized_threshold_precision": (
                threshold_result["precision"]
            ),
            "optimized_threshold_recall": (
                threshold_result["recall"]
            ),
            "optimized_threshold_f1": threshold_result["f1"],
        }

        result.update(
            calculate_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                threshold=DECISION_THRESHOLD,
                prefix="validation_default",
            )
        )

        result.update(
            calculate_metrics(
                y_true=y_validation,
                probabilities=probabilities,
                threshold=optimized_threshold,
                prefix="validation",
            )
        )

        result.update(
            calculate_confusion_values(
                y_true=y_validation,
                probabilities=probabilities,
                threshold=optimized_threshold,
                prefix="validation",
            )
        )

        result_rows.append(result)

        fitted_models[model_name] = pipeline
        validation_probabilities[model_name] = probabilities
        optimized_thresholds[model_name] = optimized_threshold

    comparison_df = pd.DataFrame(result_rows)

    comparison_df = comparison_df.sort_values(
        by=PRIMARY_SELECTION_METRIC,
        ascending=False,
    ).reset_index(drop=True)

    return (
        comparison_df,
        fitted_models,
        validation_probabilities,
        optimized_thresholds,
    )


# ============================================================
# TEST EVALUATION
# ============================================================
def evaluate_best_model_on_test(
    model_name: str,
    pipeline: Pipeline,
    threshold: float,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, Any] = {
        "model": model_name,
        "decision_threshold": threshold,
    }

    metrics.update(
        calculate_metrics(
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
            prefix="test",
        )
    )

    metrics.update(
        calculate_confusion_values(
            y_true=y_test,
            probabilities=probabilities,
            threshold=threshold,
            prefix="test",
        )
    )

    metrics["classification_report"] = classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Pass", "Fail"],
        zero_division=0,
        output_dict=True,
    )

    return metrics, probabilities, predictions


# ============================================================
# PREDICTION TABLES
# ============================================================
def build_prediction_table(
    source_df: pd.DataFrame,
    probabilities_by_model: dict[str, np.ndarray],
    thresholds_by_model: dict[str, float],
) -> pd.DataFrame:
    output = source_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    for model_name, probabilities in probabilities_by_model.items():
        safe_name = (
            model_name
            .lower()
            .replace(" ", "_")
            .replace("-", "_")
        )

        threshold = thresholds_by_model[model_name]

        output[f"{safe_name}_fail_probability"] = probabilities
        output[f"{safe_name}_prediction"] = (
            probabilities >= threshold
        ).astype(int)

    return output


# ============================================================
# VISUALISATIONS
# ============================================================
def save_roc_curve_plot(
    y_true: pd.Series,
    probabilities_by_model: dict[str, np.ndarray],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))

    for model_name, probabilities in probabilities_by_model.items():
        false_positive_rate, true_positive_rate, _ = roc_curve(
            y_true,
            probabilities,
        )

        auc_value = roc_auc_score(
            y_true,
            probabilities,
        )

        axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{model_name} (AUC={auc_value:.3f})",
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    axis.set_title("Validation ROC Curves")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "validation_roc_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_precision_recall_plot(
    y_true: pd.Series,
    probabilities_by_model: dict[str, np.ndarray],
) -> None:
    figure, axis = plt.subplots(figsize=(8, 6))

    baseline = float(y_true.mean())

    for model_name, probabilities in probabilities_by_model.items():
        precision, recall, _ = precision_recall_curve(
            y_true,
            probabilities,
        )

        pr_auc = average_precision_score(
            y_true,
            probabilities,
        )

        axis.plot(
            recall,
            precision,
            label=f"{model_name} (AP={pr_auc:.3f})",
        )

    axis.axhline(
        baseline,
        linestyle="--",
        label=f"Failure prevalence ({baseline:.3f})",
    )

    axis.set_title("Validation Precision-Recall Curves")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "validation_precision_recall_curves.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_confusion_matrix_plot(
    y_true: pd.Series,
    predictions: np.ndarray,
    model_name: str,
) -> None:
    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    figure, axis = plt.subplots(figsize=(6, 5))

    image = axis.imshow(matrix)

    axis.set_title(f"Test Confusion Matrix: {model_name}")
    axis.set_xlabel("Predicted Class")
    axis.set_ylabel("Actual Class")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["Pass", "Fail"])
    axis.set_yticklabels(["Pass", "Fail"])

    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )

    figure.colorbar(image, ax=axis)
    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR / "best_model_test_confusion_matrix.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_calibration_plot(
    y_true: pd.Series,
    probabilities: np.ndarray,
    model_name: str,
) -> None:
    fraction_of_positives, mean_predicted_value = (
        calibration_curve(
            y_true,
            probabilities,
            n_bins=8,
            strategy="quantile",
        )
    )

    figure, axis = plt.subplots(figsize=(7, 6))

    axis.plot(
        mean_predicted_value,
        fraction_of_positives,
        marker="o",
        label=model_name,
    )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.set_title("Test Probability Calibration")
    axis.set_xlabel("Mean Predicted Failure Probability")
    axis.set_ylabel("Observed Failure Rate")
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "best_model_test_calibration.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# SAVE ARTIFACTS
# ============================================================
def save_feature_columns(feature_columns: list[str]) -> None:
    with FEATURE_COLUMNS_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "feature_count": len(feature_columns),
                "features": feature_columns,
            },
            file,
            indent=4,
        )


def save_best_model_summary(summary: dict[str, Any]) -> None:
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
    best_threshold: float,
    test_metrics: dict[str, Any],
) -> None:
    print("\n" + "=" * 88)
    print("HEVEMIND BASELINE MODEL COMPARISON")
    print("=" * 88)

    display_columns = [
        "model",
        "validation_roc_auc",
        "validation_pr_auc",
        "validation_recall_fail",
        "validation_precision_fail",
        "validation_f1_fail",
        "validation_balanced_accuracy",
        "optimized_threshold",
    ]

    print(
        comparison_df[display_columns]
        .round(4)
        .to_string(index=False)
    )

    print("\n" + "=" * 88)
    print("BEST BASELINE MODEL")
    print("=" * 88)

    print(f"Model:                {best_model_name}")
    print(f"Decision threshold:   {best_threshold:.4f}")
    print(f"Test ROC-AUC:         {test_metrics['test_roc_auc']:.4f}")
    print(f"Test PR-AUC:          {test_metrics['test_pr_auc']:.4f}")
    print(
        f"Test fail recall:     "
        f"{test_metrics['test_recall_fail']:.4f}"
    )
    print(
        f"Test fail precision:  "
        f"{test_metrics['test_precision_fail']:.4f}"
    )
    print(f"Test fail F1:         {test_metrics['test_f1_fail']:.4f}")
    print(
        f"Balanced accuracy:    "
        f"{test_metrics['test_balanced_accuracy']:.4f}"
    )

    print("\nTest confusion matrix values:")
    print(
        f"True Pass:            "
        f"{test_metrics['test_true_negative']}"
    )
    print(
        f"False Alarm:          "
        f"{test_metrics['test_false_positive']}"
    )
    print(
        f"Missed Failure:       "
        f"{test_metrics['test_false_negative']}"
    )
    print(
        f"Detected Failure:     "
        f"{test_metrics['test_true_positive']}"
    )

    print("\nSaved outputs:")
    print(f"Best pipeline:        {BEST_PIPELINE_PATH}")
    print(f"Model comparison:     {MODEL_COMPARISON_PATH}")
    print(f"Model report folder:  {MODEL_REPORTS_DIR}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info("Loading dataset splits")
    train_df, validation_df, test_df = load_all_splits()

    feature_columns = get_sensor_columns(train_df)

    x_train, y_train = separate_features_and_target(
        train_df,
        feature_columns,
    )

    x_validation, y_validation = separate_features_and_target(
        validation_df,
        feature_columns,
    )

    x_test, y_test = separate_features_and_target(
        test_df,
        feature_columns,
    )

    class_weights = calculate_class_weights(y_train)

    LOGGER.info("Calculated class weights: %s", class_weights)

    models = build_model_registry(
        feature_columns=feature_columns,
        class_weights=class_weights,
    )

    (
        comparison_df,
        fitted_models,
        validation_probabilities,
        optimized_thresholds,
    ) = train_and_evaluate_models(
        models=models,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
    )

    comparison_df.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    best_model_name = str(
        comparison_df.iloc[0]["model"]
    )

    best_pipeline = fitted_models[best_model_name]
    best_threshold = optimized_thresholds[best_model_name]

    LOGGER.info(
        "Best validation model: %s",
        best_model_name,
    )

    test_metrics, test_probabilities, test_predictions = (
        evaluate_best_model_on_test(
            model_name=best_model_name,
            pipeline=best_pipeline,
            threshold=best_threshold,
            x_test=x_test,
            y_test=y_test,
        )
    )

    validation_prediction_table = build_prediction_table(
        source_df=validation_df,
        probabilities_by_model=validation_probabilities,
        thresholds_by_model=optimized_thresholds,
    )

    validation_prediction_table.to_csv(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
    )

    test_prediction_table = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    test_prediction_table["fail_probability"] = test_probabilities
    test_prediction_table["predicted_target"] = test_predictions
    test_prediction_table["predicted_status"] = np.where(
        test_predictions == 1,
        "Fail",
        "Pass",
    )
    test_prediction_table["decision_threshold"] = best_threshold
    test_prediction_table["model_name"] = best_model_name

    test_prediction_table.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    joblib.dump(
        best_pipeline,
        BEST_PIPELINE_PATH,
    )

    model_component = best_pipeline.named_steps["model"]

    joblib.dump(
        model_component,
        BEST_MODEL_PATH,
    )

    save_feature_columns(feature_columns)

    complete_summary = {
        "best_model": best_model_name,
        "selection_metric": PRIMARY_SELECTION_METRIC,
        "optimized_validation_threshold": best_threshold,
        "class_weights": class_weights,
        "number_of_features": len(feature_columns),
        "validation_results": comparison_df.to_dict(
            orient="records"
        ),
        "test_results": test_metrics,
    }

    save_best_model_summary(complete_summary)

    save_roc_curve_plot(
        y_true=y_validation,
        probabilities_by_model=validation_probabilities,
    )

    save_precision_recall_plot(
        y_true=y_validation,
        probabilities_by_model=validation_probabilities,
    )

    save_confusion_matrix_plot(
        y_true=y_test,
        predictions=test_predictions,
        model_name=best_model_name,
    )

    save_calibration_plot(
        y_true=y_test,
        probabilities=test_probabilities,
        model_name=best_model_name,
    )

    print_console_summary(
        comparison_df=comparison_df,
        best_model_name=best_model_name,
        best_threshold=best_threshold,
        test_metrics=test_metrics,
    )


if __name__ == "__main__":
    main()