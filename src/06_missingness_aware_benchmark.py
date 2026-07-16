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

from catboost import CatBoostClassifier
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
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
MISSINGNESS_MODELS_DIR = MODELS_DIR / "missingness_aware"

REPORTS_DIR = ROOT_DIR / "reports"
MISSINGNESS_REPORTS_DIR = REPORTS_DIR / "missingness_aware"
TABLES_DIR = MISSINGNESS_REPORTS_DIR / "tables"
FIGURES_DIR = MISSINGNESS_REPORTS_DIR / "figures"

MISSINGNESS_FEATURE_REPORT_PATH = (
    TABLES_DIR / "missingness_feature_association.csv"
)

ROW_MISSINGNESS_REPORT_PATH = (
    TABLES_DIR / "row_missingness_association.csv"
)

MODEL_COMPARISON_PATH = (
    TABLES_DIR / "missingness_model_comparison.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    TABLES_DIR / "missingness_threshold_analysis.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR / "best_missingness_model_test_predictions.csv"
)

STRESS_TEST_RESULTS_PATH = (
    TABLES_DIR / "missingness_stress_test_results.csv"
)

BEST_PIPELINE_PATH = (
    MISSINGNESS_MODELS_DIR / "best_missingness_pipeline.joblib"
)

SUMMARY_PATH = (
    MISSINGNESS_REPORTS_DIR / "missingness_benchmark_summary.json"
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

STRESS_TEST_MISSING_RATES = [
    0.00,
    0.05,
    0.10,
    0.20,
    0.30,
    0.40,
]


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        MISSINGNESS_MODELS_DIR,
        MISSINGNESS_REPORTS_DIR,
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


def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
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


# ============================================================
# FEATURE PREPARATION
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


def split_features_target(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    x_data = dataframe[sensor_columns].copy()
    y_data = dataframe[TARGET_COLUMN].astype(int).copy()

    return x_data, y_data


def calculate_scale_pos_weight(
    target: pd.Series,
) -> float:
    negative_count = int((target == 0).sum())
    positive_count = int((target == 1).sum())

    if positive_count == 0:
        raise ValueError(
            "No failure cases were found."
        )

    return float(
        negative_count / positive_count
    )


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================
def analyse_missingness_association(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Measures whether each sensor's missingness indicator is associated
    with the pass/fail outcome.
    """
    rows: list[dict[str, Any]] = []

    target = dataframe[TARGET_COLUMN].astype(int)

    for column in sensor_columns:
        missing_mask = dataframe[column].isna().astype(int)

        missing_count = int(missing_mask.sum())
        observed_count = int((missing_mask == 0).sum())

        failure_rate_when_missing = (
            float(
                target[missing_mask == 1].mean()
            )
            if missing_count > 0
            else np.nan
        )

        failure_rate_when_observed = (
            float(
                target[missing_mask == 0].mean()
            )
            if observed_count > 0
            else np.nan
        )

        failure_rate_difference = (
            failure_rate_when_missing
            - failure_rate_when_observed
            if (
                pd.notna(failure_rate_when_missing)
                and pd.notna(failure_rate_when_observed)
            )
            else np.nan
        )

        rows.append(
            {
                "sensor": column,
                "missing_count": missing_count,
                "missing_rate": float(
                    missing_mask.mean()
                ),
                "failure_rate_when_missing": (
                    failure_rate_when_missing
                ),
                "failure_rate_when_observed": (
                    failure_rate_when_observed
                ),
                "failure_rate_difference": (
                    failure_rate_difference
                ),
            }
        )

    report = pd.DataFrame(rows)

    informative_columns = report.loc[
        report["missing_count"] > 0,
        "sensor",
    ].tolist()

    if informative_columns:
        missing_matrix = (
            dataframe[informative_columns]
            .isna()
            .astype(int)
        )

        mutual_information = mutual_info_classif(
            missing_matrix,
            target,
            discrete_features=True,
            random_state=RANDOM_STATE,
        )

        mi_mapping = dict(
            zip(
                informative_columns,
                mutual_information,
            )
        )

        report["missingness_mutual_information"] = (
            report["sensor"]
            .map(mi_mapping)
            .fillna(0.0)
        )

    else:
        report["missingness_mutual_information"] = 0.0

    return report.sort_values(
        by=[
            "missingness_mutual_information",
            "failure_rate_difference",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(drop=True)


def analyse_row_missingness(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    output = dataframe[
        [
            ID_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    output["missing_sensor_count"] = (
        dataframe[sensor_columns]
        .isna()
        .sum(axis=1)
    )

    output["missing_sensor_rate"] = (
        output["missing_sensor_count"]
        / len(sensor_columns)
    )

    output["missingness_band"] = pd.cut(
        output["missing_sensor_rate"],
        bins=[
            -0.001,
            0.01,
            0.03,
            0.05,
            0.10,
            1.00,
        ],
        labels=[
            "0-1%",
            "1-3%",
            "3-5%",
            "5-10%",
            ">10%",
        ],
    )

    band_summary = (
        output
        .groupby(
            "missingness_band",
            observed=False,
        )
        .agg(
            records=(TARGET_COLUMN, "size"),
            failures=(TARGET_COLUMN, "sum"),
            failure_rate=(TARGET_COLUMN, "mean"),
            average_missing_count=(
                "missing_sensor_count",
                "mean",
            ),
            average_missing_rate=(
                "missing_sensor_rate",
                "mean",
            ),
        )
        .reset_index()
    )

    return band_summary


# ============================================================
# FEATURE AUGMENTATION
# ============================================================
def add_missingness_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adds per-feature missingness masks and row-level data-quality
    indicators.
    """
    output = dataframe.copy()

    sensor_columns = list(output.columns)

    missing_mask = output.isna().astype(np.int8)

    missing_mask.columns = [
        f"{column}_missing"
        for column in sensor_columns
    ]

    output["row_missing_count"] = (
        output.isna().sum(axis=1)
    )

    output["row_missing_rate"] = (
        output["row_missing_count"]
        / len(sensor_columns)
    )

    output["row_observed_count"] = (
        len(sensor_columns)
        - output["row_missing_count"]
    )

    output = pd.concat(
        [
            output,
            missing_mask,
        ],
        axis=1,
    )

    return output


# ============================================================
# CUSTOM TRANSFORMER
# ============================================================
class MissingnessFeatureTransformer(
    BaseEstimator
):
    """
    Adds missingness masks before imputation.

    This transformer preserves feature names when given a DataFrame.
    """

    def fit(
        self,
        x_data: pd.DataFrame,
        y_data: pd.Series | None = None,
    ) -> "MissingnessFeatureTransformer":
        self.input_features_ = list(x_data.columns)
        return self

    def transform(
        self,
        x_data: pd.DataFrame,
    ) -> pd.DataFrame:
        if not isinstance(x_data, pd.DataFrame):
            x_data = pd.DataFrame(
                x_data,
                columns=self.input_features_,
            )

        return add_missingness_features(
            x_data
        )

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> np.ndarray:
        features = (
            list(input_features)
            if input_features is not None
            else self.input_features_
        )

        output_features = (
            features
            + [
                "row_missing_count",
                "row_missing_rate",
                "row_observed_count",
            ]
            + [
                f"{feature}_missing"
                for feature in features
            ]
        )

        return np.asarray(
            output_features,
            dtype=object,
        )


# ============================================================
# MODEL REGISTRY
# ============================================================
def build_model_registry(
    scale_pos_weight: float,
) -> dict[str, Pipeline | ClassifierMixin]:
    """
    Native-missingness models receive NaN values directly.

    Imputation-aware models receive explicit missingness masks.
    """
    return {
        "Native XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=600,
            learning_rate=0.025,
            max_depth=4,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.70,
            gamma=0.10,
            reg_alpha=0.10,
            reg_lambda=2.0,
            scale_pos_weight=scale_pos_weight,
            missing=np.nan,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Native CatBoost": CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=5,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            verbose=False,
            random_seed=RANDOM_STATE,
            allow_writing_files=False,
        ),
        "Native HistGradientBoosting": (
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                min_samples_leaf=20,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )
        ),
        "XGBoost with Missingness Mask": Pipeline(
            steps=[
                (
                    "missingness_features",
                    MissingnessFeatureTransformer(),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "model",
                    XGBClassifier(
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_estimators=600,
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
        ),
        "Logistic with Missingness Mask": Pipeline(
            steps=[
                (
                    "missingness_features",
                    MissingnessFeatureTransformer(),
                ),
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                    ),
                ),
                (
                    "scaler",
                    RobustScaler(),
                ),
                (
                    "model",
                    __import__(
                        "sklearn.linear_model",
                        fromlist=["LogisticRegression"],
                    ).LogisticRegression(
                        C=0.25,
                        class_weight="balanced",
                        solver="liblinear",
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
    model: BaseEstimator,
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

    thresholds = np.arange(
        THRESHOLD_MINIMUM,
        THRESHOLD_MAXIMUM + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    for threshold in thresholds:
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

        reason = (
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

        reason = (
            "No threshold achieved minimum recall"
        )

    result = selected.to_dict()
    result["selection_reason"] = reason

    return result


# ============================================================
# MODEL BENCHMARKING
# ============================================================
def benchmark_models(
    models: dict[str, BaseEstimator],
    x_development: pd.DataFrame,
    y_development: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, float],
]:
    comparison_rows: list[dict[str, Any]] = []
    threshold_tables: list[pd.DataFrame] = []

    probability_mapping: dict[str, np.ndarray] = {}
    threshold_mapping: dict[str, float] = {}

    for model_name, model in models.items():
        LOGGER.info(
            "Evaluating missingness-aware model: %s",
            model_name,
        )

        start_time = time.perf_counter()

        probabilities = generate_oof_probabilities(
            model=model,
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

        selected_threshold = select_operational_threshold(
            threshold_table
        )

        threshold = float(
            selected_threshold["threshold"]
        )

        operational_metrics = calculate_threshold_metrics(
            y_true=y_development.to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )

        comparison_rows.append(
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
                **operational_metrics,
                "elapsed_seconds": elapsed_seconds,
                "selection_reason": (
                    selected_threshold[
                        "selection_reason"
                    ]
                ),
            }
        )

        threshold_tables.append(
            threshold_table
        )

        probability_mapping[model_name] = probabilities
        threshold_mapping[model_name] = threshold

    comparison_df = pd.DataFrame(
        comparison_rows
    )

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


# ============================================================
# FINAL MODEL SELECTION
# ============================================================
def select_best_model(
    comparison_df: pd.DataFrame,
) -> str:
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

    return str(
        selected["model"]
    )


# ============================================================
# TEST EVALUATION
# ============================================================
def evaluate_test_set(
    model: BaseEstimator,
    x_development: pd.DataFrame,
    y_development: pd.Series,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> tuple[
    BaseEstimator,
    dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    final_model = clone(model)

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
# MISSINGNESS STRESS TEST
# ============================================================
def inject_additional_missingness(
    x_data: pd.DataFrame,
    missing_rate: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Randomly masks a fraction of currently observed sensor values.

    Existing missing values remain unchanged.
    """
    if missing_rate <= 0:
        return x_data.copy()

    rng = np.random.default_rng(
        random_state
    )

    output = x_data.copy()

    observed_mask = output.notna().to_numpy()

    random_values = rng.random(
        size=output.shape
    )

    additional_mask = (
        observed_mask
        & (random_values < missing_rate)
    )

    output_array = output.to_numpy(
        dtype=float,
        copy=True,
    )

    output_array[additional_mask] = np.nan

    return pd.DataFrame(
        output_array,
        columns=output.columns,
        index=output.index,
    )


def run_missingness_stress_test(
    final_model: BaseEstimator,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for missing_rate in STRESS_TEST_MISSING_RATES:
        LOGGER.info(
            "Running missingness stress test: %.0f%%",
            missing_rate * 100,
        )

        stressed_x = inject_additional_missingness(
            x_data=x_test,
            missing_rate=missing_rate,
            random_state=(
                RANDOM_STATE
                + int(missing_rate * 1000)
            ),
        )

        probabilities = final_model.predict_proba(
            stressed_x
        )[:, 1]

        metrics = calculate_threshold_metrics(
            y_true=y_test.to_numpy(),
            probabilities=probabilities,
            threshold=threshold,
        )

        rows.append(
            {
                "additional_missing_rate": (
                    missing_rate
                ),
                "average_total_missing_rate": float(
                    stressed_x.isna().mean().mean()
                ),
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
                **metrics,
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PREDICTION TABLE
# ============================================================
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

    sensor_columns = get_sensor_columns(
        test_df
    )

    output["missing_sensor_count"] = (
        test_df[sensor_columns]
        .isna()
        .sum(axis=1)
    )

    output["missing_sensor_rate"] = (
        output["missing_sensor_count"]
        / len(sensor_columns)
    )

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
def save_missingness_signal_plot(
    report: pd.DataFrame,
) -> None:
    top_features = (
        report
        .head(25)
        .sort_values(
            by="missingness_mutual_information",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        top_features["sensor"],
        top_features[
            "missingness_mutual_information"
        ],
    )

    axis.set_title(
        "Top Missingness Patterns Associated with Failure"
    )
    axis.set_xlabel(
        "Mutual Information"
    )
    axis.set_ylabel(
        "Sensor Missingness Indicator"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "top_missingness_failure_associations.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


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
        "Missingness-Aware Model Operational Cost"
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
        / "missingness_model_operational_cost.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_stress_test_plot(
    stress_test_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.plot(
        stress_test_df[
            "additional_missing_rate"
        ] * 100,
        stress_test_df["pr_auc"],
        marker="o",
        label="PR-AUC",
    )

    axis.plot(
        stress_test_df[
            "additional_missing_rate"
        ] * 100,
        stress_test_df["recall_fail"],
        marker="o",
        label="Failure recall",
    )

    axis.plot(
        stress_test_df[
            "additional_missing_rate"
        ] * 100,
        stress_test_df["precision_fail"],
        marker="o",
        label="Failure precision",
    )

    axis.set_title(
        "Model Robustness Under Additional Missingness"
    )
    axis.set_xlabel(
        "Additional Artificial Missingness (%)"
    )
    axis.set_ylabel(
        "Metric Value"
    )
    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "missingness_stress_test.png",
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
# CONSOLE OUTPUT
# ============================================================
def print_console_summary(
    comparison_df: pd.DataFrame,
    selected_model_name: str,
    selected_threshold: float,
    test_metrics: dict[str, Any],
    stress_test_df: pd.DataFrame,
) -> None:
    print("\n" + "=" * 120)
    print("HEVEMIND MISSINGNESS-AWARE MODEL BENCHMARK")
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
    print("SELECTED MISSINGNESS-AWARE MODEL")
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

    print("\nMissingness stress-test summary:")

    print(
        stress_test_df[
            [
                "additional_missing_rate",
                "average_total_missing_rate",
                "pr_auc",
                "recall_fail",
                "precision_fail",
                "false_positive",
                "false_negative",
            ]
        ]
        .round(4)
        .to_string(
            index=False,
        )
    )

    print("\nSaved outputs:")

    print(
        f"Best model:                     "
        f"{BEST_PIPELINE_PATH}"
    )

    print(
        f"Model comparison:               "
        f"{MODEL_COMPARISON_PATH}"
    )

    print(
        f"Stress test:                    "
        f"{STRESS_TEST_RESULTS_PATH}"
    )

    print(
        f"Report directory:               "
        f"{MISSINGNESS_REPORTS_DIR}"
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
        "Sensor features: %s",
        len(sensor_columns),
    )

    LOGGER.info(
        "Analysing sensor missingness associations"
    )

    missingness_report = (
        analyse_missingness_association(
            development_df,
            sensor_columns,
        )
    )

    missingness_report.to_csv(
        MISSINGNESS_FEATURE_REPORT_PATH,
        index=False,
    )

    row_missingness_report = analyse_row_missingness(
        development_df,
        sensor_columns,
    )

    row_missingness_report.to_csv(
        ROW_MISSINGNESS_REPORT_PATH,
        index=False,
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
        x_development=x_development,
        y_development=y_development,
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

    selected_threshold = threshold_mapping[
        selected_model_name
    ]

    LOGGER.info(
        "Selected model: %s",
        selected_model_name,
    )

    (
        final_model,
        test_metrics,
        test_probabilities,
        test_predictions,
    ) = evaluate_test_set(
        model=models[selected_model_name],
        x_development=x_development,
        y_development=y_development,
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

    stress_test_df = run_missingness_stress_test(
        final_model=final_model,
        x_test=x_test,
        y_test=y_test,
        threshold=selected_threshold,
    )

    stress_test_df.to_csv(
        STRESS_TEST_RESULTS_PATH,
        index=False,
    )

    joblib.dump(
        final_model,
        BEST_PIPELINE_PATH,
    )

    save_missingness_signal_plot(
        missingness_report
    )

    save_model_comparison_plot(
        comparison_df
    )

    save_stress_test_plot(
        stress_test_df
    )

    summary = {
        "project": "HeveMind",
        "stage": "Missingness-aware benchmark",
        "development_rows": int(
            len(development_df)
        ),
        "test_rows": int(
            len(test_df)
        ),
        "sensor_features": int(
            len(sensor_columns)
        ),
        "failure_prevalence": float(
            y_development.mean()
        ),
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
        },
        "selected_model": (
            selected_model_name
        ),
        "selected_threshold": float(
            selected_threshold
        ),
        "development_results": (
            comparison_df.to_dict(
                orient="records"
            )
        ),
        "held_out_test_results": (
            test_metrics
        ),
        "missingness_stress_test": (
            stress_test_df.to_dict(
                orient="records"
            )
        ),
        "top_missingness_signals": (
            missingness_report
            .head(20)
            .to_dict(
                orient="records"
            )
        ),
    }

    save_summary(summary)

    print_console_summary(
        comparison_df=comparison_df,
        selected_model_name=selected_model_name,
        selected_threshold=selected_threshold,
        test_metrics=test_metrics,
        stress_test_df=stress_test_df,
    )


if __name__ == "__main__":
    main()