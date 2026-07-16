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
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


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
CALIBRATION_ARTIFACTS_DIR = (
    ARTIFACTS_DIR
    / "models"
    / "calibrated_decision"
)

METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = ROOT_DIR / "reports"
CALIBRATION_REPORTS_DIR = (
    REPORTS_DIR
    / "calibration_decision_policy"
)

TABLES_DIR = CALIBRATION_REPORTS_DIR / "tables"
FIGURES_DIR = CALIBRATION_REPORTS_DIR / "figures"

BASE_MODEL_PATH = (
    CALIBRATION_ARTIFACTS_DIR
    / "balanced_random_forest.joblib"
)

CALIBRATOR_PATH = (
    CALIBRATION_ARTIFACTS_DIR
    / "probability_calibrator.joblib"
)

COMPLETE_SYSTEM_PATH = (
    CALIBRATION_ARTIFACTS_DIR
    / "calibrated_decision_system.joblib"
)

FEATURE_COLUMNS_PATH = (
    METADATA_DIR
    / "calibrated_model_feature_columns.json"
)

CALIBRATION_COMPARISON_PATH = (
    TABLES_DIR
    / "calibration_method_comparison.csv"
)

DEVELOPMENT_OOF_PATH = (
    TABLES_DIR
    / "development_calibrated_oof_predictions.csv"
)

DECISION_THRESHOLD_ANALYSIS_PATH = (
    TABLES_DIR
    / "three_level_threshold_analysis.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR
    / "calibrated_test_predictions.csv"
)

TEST_DECISION_SUMMARY_PATH = (
    TABLES_DIR
    / "test_decision_summary.csv"
)

TEST_METRICS_PATH = (
    TABLES_DIR
    / "calibrated_test_metrics.csv"
)

SUMMARY_PATH = (
    CALIBRATION_REPORTS_DIR
    / "calibration_decision_summary.json"
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

CALIBRATION_BINS = 10

LOWER_THRESHOLD_MINIMUM = 0.01
LOWER_THRESHOLD_MAXIMUM = 0.40
UPPER_THRESHOLD_MINIMUM = 0.10
UPPER_THRESHOLD_MAXIMUM = 0.90
THRESHOLD_STEP = 0.01

# Operational constraints
MINIMUM_LOW_RISK_NPV = 0.98
MAXIMUM_LOW_RISK_FAILURE_LEAKAGE = 0.10
MINIMUM_HIGH_RISK_FAILURE_CAPTURE = 0.25
MAXIMUM_REVIEW_RATE = 0.65

# Decision-policy cost assumptions
LOW_RISK_FAILURE_COST = 20.0
REVIEW_PASS_COST = 1.0
REVIEW_FAILURE_COST = 0.5
HIGH_RISK_PASS_COST = 3.0
HIGH_RISK_FAILURE_COST = 0.0


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        CALIBRATION_ARTIFACTS_DIR,
        METADATA_DIR,
        CALIBRATION_REPORTS_DIR,
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
            f"Required dataset split was not found: {path}"
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
# MODEL
# ============================================================
def build_balanced_random_forest_pipeline() -> Pipeline:
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


# ============================================================
# OUT-OF-FOLD PROBABILITIES
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
        estimator=clone(pipeline),
        X=x_data,
        y=y_data,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]


# ============================================================
# CALIBRATORS
# ============================================================
class IdentityCalibrator:
    """
    Leaves raw model probabilities unchanged.
    """

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> "IdentityCalibrator":
        return self

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        return np.clip(
            np.asarray(probabilities, dtype=float),
            0.0,
            1.0,
        )


class SigmoidCalibrator:
    """
    Platt-style sigmoid calibration using logistic regression.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(
            solver="lbfgs",
            random_state=RANDOM_STATE,
        )

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> "SigmoidCalibrator":
        self.model.fit(
            probabilities.reshape(-1, 1),
            target,
        )

        return self

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        return self.model.predict_proba(
            probabilities.reshape(-1, 1)
        )[:, 1]


class IsotonicCalibrator:
    """
    Non-parametric monotonic calibration.
    """

    def __init__(self) -> None:
        self.model = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip",
        )

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> "IsotonicCalibrator":
        self.model.fit(
            probabilities,
            target,
        )

        return self

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        return np.clip(
            self.model.predict(probabilities),
            0.0,
            1.0,
        )


def build_calibrator_registry() -> dict[str, Any]:
    return {
        "Uncalibrated": IdentityCalibrator(),
        "Sigmoid": SigmoidCalibrator(),
        "Isotonic": IsotonicCalibrator(),
    }


# ============================================================
# CALIBRATION METRICS
# ============================================================
def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = CALIBRATION_BINS,
) -> float:
    """
    Calculates expected calibration error using equal-width bins.
    """
    bin_edges = np.linspace(
        0.0,
        1.0,
        number_of_bins + 1,
    )

    bin_indices = np.digitize(
        probabilities,
        bin_edges[1:-1],
        right=True,
    )

    total_records = len(y_true)
    calibration_error = 0.0

    for bin_index in range(number_of_bins):
        mask = bin_indices == bin_index

        if not np.any(mask):
            continue

        observed_rate = float(
            np.mean(y_true[mask])
        )

        mean_probability = float(
            np.mean(probabilities[mask])
        )

        bin_weight = float(
            np.sum(mask) / total_records
        )

        calibration_error += (
            bin_weight
            * abs(
                observed_rate
                - mean_probability
            )
        )

    return float(calibration_error)


def calculate_probability_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    clipped_probabilities = np.clip(
        probabilities,
        1e-8,
        1.0 - 1e-8,
    )

    return {
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
        "log_loss": float(
            log_loss(
                y_true,
                clipped_probabilities,
                labels=[0, 1],
            )
        ),
        "expected_calibration_error": (
            expected_calibration_error(
                y_true,
                probabilities,
            )
        ),
        "mean_predicted_probability": float(
            np.mean(probabilities)
        ),
        "observed_failure_rate": float(
            np.mean(y_true)
        ),
    }


def benchmark_calibrators(
    raw_oof_probabilities: np.ndarray,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    comparison_rows: list[dict[str, Any]] = []
    fitted_calibrators: dict[str, Any] = {}
    calibrated_probabilities: dict[str, np.ndarray] = {}

    y_array = target.to_numpy(dtype=int)

    for calibrator_name, calibrator in (
        build_calibrator_registry().items()
    ):
        LOGGER.info(
            "Evaluating calibration method: %s",
            calibrator_name,
        )

        fitted_calibrator = calibrator.fit(
            raw_oof_probabilities,
            y_array,
        )

        probabilities = fitted_calibrator.predict(
            raw_oof_probabilities
        )

        metrics = calculate_probability_metrics(
            y_true=y_array,
            probabilities=probabilities,
        )

        comparison_rows.append(
            {
                "calibration_method": calibrator_name,
                **metrics,
            }
        )

        fitted_calibrators[
            calibrator_name
        ] = fitted_calibrator

        calibrated_probabilities[
            calibrator_name
        ] = probabilities

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_df = comparison_df.sort_values(
        by=[
            "brier_score",
            "expected_calibration_error",
            "log_loss",
        ],
        ascending=[
            True,
            True,
            True,
        ],
    ).reset_index(drop=True)

    return (
        comparison_df,
        fitted_calibrators,
        calibrated_probabilities,
    )


# ============================================================
# THREE-LEVEL DECISION POLICY
# ============================================================
def assign_decision_class(
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    return np.select(
        [
            probabilities < lower_threshold,
            probabilities >= upper_threshold,
        ],
        [
            "Low Risk",
            "High Risk",
        ],
        default="Engineering Review",
    )


def calculate_decision_policy_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, Any]:
    decisions = assign_decision_class(
        probabilities=probabilities,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    low_mask = decisions == "Low Risk"
    review_mask = decisions == "Engineering Review"
    high_mask = decisions == "High Risk"

    total_records = len(y_true)
    total_failures = int(
        np.sum(y_true == 1)
    )

    low_records = int(
        np.sum(low_mask)
    )

    review_records = int(
        np.sum(review_mask)
    )

    high_records = int(
        np.sum(high_mask)
    )

    low_failures = int(
        np.sum(
            (y_true == 1)
            & low_mask
        )
    )

    low_passes = int(
        np.sum(
            (y_true == 0)
            & low_mask
        )
    )

    review_failures = int(
        np.sum(
            (y_true == 1)
            & review_mask
        )
    )

    review_passes = int(
        np.sum(
            (y_true == 0)
            & review_mask
        )
    )

    high_failures = int(
        np.sum(
            (y_true == 1)
            & high_mask
        )
    )

    high_passes = int(
        np.sum(
            (y_true == 0)
            & high_mask
        )
    )

    low_risk_npv = (
        float(
            low_passes / low_records
        )
        if low_records > 0
        else np.nan
    )

    high_risk_precision = (
        float(
            high_failures / high_records
        )
        if high_records > 0
        else 0.0
    )

    high_risk_failure_capture = (
        float(
            high_failures / total_failures
        )
        if total_failures > 0
        else 0.0
    )

    review_failure_capture = (
        float(
            review_failures / total_failures
        )
        if total_failures > 0
        else 0.0
    )

    review_plus_high_failure_capture = (
        float(
            (
                review_failures
                + high_failures
            )
            / total_failures
        )
        if total_failures > 0
        else 0.0
    )

    low_risk_failure_leakage = (
        float(
            low_failures / total_failures
        )
        if total_failures > 0
        else 0.0
    )

    review_rate = float(
        review_records / total_records
    )

    automation_coverage = float(
        (
            low_records
            + high_records
        )
        / total_records
    )

    operational_cost = (
        LOW_RISK_FAILURE_COST
        * low_failures
        + REVIEW_PASS_COST
        * review_passes
        + REVIEW_FAILURE_COST
        * review_failures
        + HIGH_RISK_PASS_COST
        * high_passes
        + HIGH_RISK_FAILURE_COST
        * high_failures
    )

    return {
        "lower_threshold": float(
            lower_threshold
        ),
        "upper_threshold": float(
            upper_threshold
        ),
        "total_records": int(
            total_records
        ),
        "total_failures": int(
            total_failures
        ),
        "low_risk_records": low_records,
        "review_records": review_records,
        "high_risk_records": high_records,
        "low_risk_rate": float(
            low_records / total_records
        ),
        "review_rate": review_rate,
        "high_risk_rate": float(
            high_records / total_records
        ),
        "automation_coverage": automation_coverage,
        "low_risk_failures": low_failures,
        "low_risk_passes": low_passes,
        "review_failures": review_failures,
        "review_passes": review_passes,
        "high_risk_failures": high_failures,
        "high_risk_passes": high_passes,
        "low_risk_npv": low_risk_npv,
        "high_risk_precision": (
            high_risk_precision
        ),
        "high_risk_failure_capture": (
            high_risk_failure_capture
        ),
        "review_failure_capture": (
            review_failure_capture
        ),
        "review_plus_high_failure_capture": (
            review_plus_high_failure_capture
        ),
        "low_risk_failure_leakage": (
            low_risk_failure_leakage
        ),
        "operational_cost": float(
            operational_cost
        ),
    }


def build_decision_threshold_analysis(
    target: pd.Series,
    calibrated_probabilities: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    lower_thresholds = np.arange(
        LOWER_THRESHOLD_MINIMUM,
        LOWER_THRESHOLD_MAXIMUM
        + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    upper_thresholds = np.arange(
        UPPER_THRESHOLD_MINIMUM,
        UPPER_THRESHOLD_MAXIMUM
        + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    y_array = target.to_numpy(dtype=int)

    for lower_threshold in lower_thresholds:
        for upper_threshold in upper_thresholds:
            if (
                upper_threshold
                <= lower_threshold
            ):
                continue

            metrics = (
                calculate_decision_policy_metrics(
                    y_true=y_array,
                    probabilities=calibrated_probabilities,
                    lower_threshold=float(
                        lower_threshold
                    ),
                    upper_threshold=float(
                        upper_threshold
                    ),
                )
            )

            rows.append(metrics)

    return pd.DataFrame(rows)


def select_decision_thresholds(
    threshold_df: pd.DataFrame,
) -> dict[str, Any]:
    eligible = threshold_df.loc[
        (
            threshold_df["low_risk_npv"]
            >= MINIMUM_LOW_RISK_NPV
        )
        & (
            threshold_df[
                "low_risk_failure_leakage"
            ]
            <= MAXIMUM_LOW_RISK_FAILURE_LEAKAGE
        )
        & (
            threshold_df[
                "high_risk_failure_capture"
            ]
            >= MINIMUM_HIGH_RISK_FAILURE_CAPTURE
        )
        & (
            threshold_df["review_rate"]
            <= MAXIMUM_REVIEW_RATE
        )
    ].copy()

    if not eligible.empty:
        selected = eligible.sort_values(
            by=[
                "operational_cost",
                "review_rate",
                "high_risk_precision",
                "automation_coverage",
            ],
            ascending=[
                True,
                True,
                False,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "Minimum operational cost among policies meeting "
            "all safety and review-burden constraints"
        )

    else:
        LOGGER.warning(
            "No threshold pair satisfied all operational constraints. "
            "Using the safest available fallback policy."
        )

        fallback = threshold_df.loc[
            threshold_df[
                "low_risk_failure_leakage"
            ]
            <= MAXIMUM_LOW_RISK_FAILURE_LEAKAGE
        ].copy()

        if fallback.empty:
            fallback = threshold_df.copy()

        selected = fallback.sort_values(
            by=[
                "low_risk_failure_leakage",
                "operational_cost",
                "review_rate",
                "high_risk_precision",
            ],
            ascending=[
                True,
                True,
                True,
                False,
            ],
        ).iloc[0]

        selection_reason = (
            "Fallback policy prioritising minimum failure leakage "
            "followed by operational cost"
        )

    result = selected.to_dict()
    result["selection_reason"] = selection_reason

    return result


# ============================================================
# TEST OUTPUTS
# ============================================================
def build_test_prediction_table(
    test_df: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
    calibration_method: str,
) -> pd.DataFrame:
    output = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    decisions = assign_decision_class(
        probabilities=calibrated_probabilities,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    output["raw_failure_probability"] = (
        raw_probabilities
    )

    output[
        "calibrated_failure_probability"
    ] = calibrated_probabilities

    output["calibration_method"] = (
        calibration_method
    )

    output["lower_threshold"] = (
        lower_threshold
    )

    output["upper_threshold"] = (
        upper_threshold
    )

    output["decision"] = decisions

    output["requires_engineer_review"] = (
        output["decision"]
        == "Engineering Review"
    )

    output["automatic_action"] = np.select(
        [
            output["decision"] == "Low Risk",
            output["decision"] == "High Risk",
        ],
        [
            "Continue processing",
            "Hold and inspect",
        ],
        default="Engineer assessment required",
    )

    output["decision_outcome"] = np.select(
        [
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output["decision"]
                    == "Low Risk"
                )
            ),
            (
                (output[TARGET_COLUMN] == 0)
                & (
                    output["decision"]
                    == "High Risk"
                )
            ),
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output["decision"]
                    == "High Risk"
                )
            ),
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output["decision"]
                    == "Engineering Review"
                )
            ),
        ],
        [
            "Unsafe Low-Risk Failure",
            "High-Risk False Alarm",
            "Correct High-Risk Failure",
            "Failure Routed to Review",
        ],
        default="Acceptable Routing",
    )

    return output


def build_decision_summary(
    prediction_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for decision, group in prediction_df.groupby(
        "decision",
        observed=False,
    ):
        records = len(group)
        failures = int(
            group[TARGET_COLUMN].sum()
        )

        rows.append(
            {
                "decision": decision,
                "records": records,
                "record_rate": float(
                    records
                    / len(prediction_df)
                ),
                "actual_failures": failures,
                "actual_passes": int(
                    records - failures
                ),
                "observed_failure_rate": float(
                    failures / records
                )
                if records > 0
                else np.nan,
                "mean_calibrated_probability": float(
                    group[
                        "calibrated_failure_probability"
                    ].mean()
                ),
            }
        )

    order_mapping = {
        "Low Risk": 0,
        "Engineering Review": 1,
        "High Risk": 2,
    }

    summary_df = pd.DataFrame(rows)

    summary_df["order"] = (
        summary_df["decision"]
        .map(order_mapping)
    )

    return (
        summary_df
        .sort_values("order")
        .drop(columns="order")
        .reset_index(drop=True)
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_calibration_comparison_plot(
    target: pd.Series,
    probability_mapping: dict[str, np.ndarray],
) -> None:
    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    for method_name, probabilities in (
        probability_mapping.items()
    ):
        observed_fraction, mean_probability = (
            calibration_curve(
                target,
                probabilities,
                n_bins=CALIBRATION_BINS,
                strategy="quantile",
            )
        )

        axis.plot(
            mean_probability,
            observed_fraction,
            marker="o",
            label=method_name,
        )

    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect calibration",
    )

    axis.set_title(
        "Development Out-of-Fold Calibration Comparison"
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
        FIGURES_DIR
        / "development_calibration_comparison.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_probability_distribution_plot(
    target: pd.Series,
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> None:
    pass_probabilities = probabilities[
        target.to_numpy() == 0
    ]

    fail_probabilities = probabilities[
        target.to_numpy() == 1
    ]

    figure, axis = plt.subplots(
        figsize=(10, 6)
    )

    axis.hist(
        pass_probabilities,
        bins=30,
        alpha=0.6,
        label="Pass",
    )

    axis.hist(
        fail_probabilities,
        bins=30,
        alpha=0.6,
        label="Fail",
    )

    axis.axvline(
        lower_threshold,
        linestyle="--",
        label=(
            f"Low/Review threshold "
            f"({lower_threshold:.3f})"
        ),
    )

    axis.axvline(
        upper_threshold,
        linestyle=":",
        label=(
            f"Review/High threshold "
            f"({upper_threshold:.3f})"
        ),
    )

    axis.set_title(
        "Calibrated Failure Probability Distribution"
    )

    axis.set_xlabel(
        "Calibrated Failure Probability"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "calibrated_probability_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_policy_tradeoff_plot(
    threshold_df: pd.DataFrame,
    selected_policy: dict[str, Any],
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    scatter = axis.scatter(
        threshold_df["review_rate"],
        threshold_df["high_risk_precision"],
        c=threshold_df["operational_cost"],
        alpha=0.5,
    )

    axis.scatter(
        selected_policy["review_rate"],
        selected_policy["high_risk_precision"],
        marker="X",
        s=180,
        label="Selected policy",
    )

    axis.set_title(
        "Three-Level Policy Trade-Off"
    )

    axis.set_xlabel(
        "Engineering Review Rate"
    )

    axis.set_ylabel(
        "High-Risk Precision"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="Operational Cost",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "decision_policy_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_decision_plot(
    decision_summary: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    positions = np.arange(
        len(decision_summary)
    )

    axis.bar(
        positions,
        decision_summary["actual_passes"],
        label="Pass",
    )

    axis.bar(
        positions,
        decision_summary["actual_failures"],
        bottom=decision_summary["actual_passes"],
        label="Fail",
    )

    axis.set_xticks(positions)

    axis.set_xticklabels(
        decision_summary["decision"],
        rotation=0,
    )

    axis.set_title(
        "Held-Out Test Outcomes by Decision Class"
    )

    axis.set_xlabel(
        "HeveMind Decision"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "test_decision_outcomes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


# ============================================================
# ARTIFACT WRAPPER
# ============================================================
class CalibratedDecisionSystem:
    """
    Serializable inference wrapper containing:

    1. Base prediction model.
    2. Probability calibrator.
    3. Three-level decision thresholds.
    """

    def __init__(
        self,
        base_model: Pipeline,
        calibrator: Any,
        calibration_method: str,
        lower_threshold: float,
        upper_threshold: float,
        feature_columns: list[str],
    ) -> None:
        self.base_model = base_model
        self.calibrator = calibrator
        self.calibration_method = calibration_method
        self.lower_threshold = lower_threshold
        self.upper_threshold = upper_threshold
        self.feature_columns = feature_columns

    def predict_failure_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        features = dataframe[
            self.feature_columns
        ].copy()

        raw_probabilities = (
            self.base_model.predict_proba(
                features
            )[:, 1]
        )

        return self.calibrator.predict(
            raw_probabilities
        )

    def predict_decision(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        calibrated_probabilities = (
            self.predict_failure_probability(
                dataframe
            )
        )

        decisions = assign_decision_class(
            probabilities=calibrated_probabilities,
            lower_threshold=self.lower_threshold,
            upper_threshold=self.upper_threshold,
        )

        return pd.DataFrame(
            {
                "calibrated_failure_probability": (
                    calibrated_probabilities
                ),
                "decision": decisions,
            },
            index=dataframe.index,
        )


# ============================================================
# SAVE UTILITIES
# ============================================================
def save_feature_columns(
    feature_columns: list[str],
) -> None:
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
    calibration_df: pd.DataFrame,
    selected_calibration_method: str,
    selected_policy: dict[str, Any],
    test_probability_metrics: dict[str, float],
    test_policy_metrics: dict[str, Any],
    decision_summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 118)
    print(
        "HEVEMIND PROBABILITY CALIBRATION "
        "AND THREE-LEVEL DECISION POLICY"
    )
    print("=" * 118)

    print("\nDevelopment calibration comparison:")

    print(
        calibration_df[
            [
                "calibration_method",
                "roc_auc",
                "pr_auc",
                "brier_score",
                "log_loss",
                "expected_calibration_error",
                "mean_predicted_probability",
                "observed_failure_rate",
            ]
        ]
        .round(4)
        .to_string(index=False)
    )

    print("\nSelected calibration method:")

    print(
        f"Method:                         "
        f"{selected_calibration_method}"
    )

    print("\nSelected development decision policy:")

    print(
        f"Lower threshold:                "
        f"{selected_policy['lower_threshold']:.4f}"
    )

    print(
        f"Upper threshold:                "
        f"{selected_policy['upper_threshold']:.4f}"
    )

    print(
        f"Selection reason:               "
        f"{selected_policy['selection_reason']}"
    )

    print(
        f"Development review rate:        "
        f"{selected_policy['review_rate']:.4f}"
    )

    print(
        f"Development automation coverage:"
        f" {selected_policy['automation_coverage']:.4f}"
    )

    print(
        f"Development low-risk NPV:       "
        f"{selected_policy['low_risk_npv']:.4f}"
    )

    print(
        f"Development high-risk precision:"
        f" {selected_policy['high_risk_precision']:.4f}"
    )

    print(
        f"Development failure leakage:    "
        f"{selected_policy['low_risk_failure_leakage']:.4f}"
    )

    print("\nHeld-out test probability performance:")

    print(
        f"ROC-AUC:                        "
        f"{test_probability_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                         "
        f"{test_probability_metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:                    "
        f"{test_probability_metrics['brier_score']:.4f}"
    )

    print(
        f"Log loss:                       "
        f"{test_probability_metrics['log_loss']:.4f}"
    )

    print(
        f"Calibration error:              "
        f"{test_probability_metrics['expected_calibration_error']:.4f}"
    )

    print("\nHeld-out test decision policy:")

    print(
        f"Low-risk records:               "
        f"{test_policy_metrics['low_risk_records']}"
    )

    print(
        f"Engineering-review records:     "
        f"{test_policy_metrics['review_records']}"
    )

    print(
        f"High-risk records:              "
        f"{test_policy_metrics['high_risk_records']}"
    )

    print(
        f"Review rate:                    "
        f"{test_policy_metrics['review_rate']:.4f}"
    )

    print(
        f"Automation coverage:            "
        f"{test_policy_metrics['automation_coverage']:.4f}"
    )

    print(
        f"Low-risk failures:              "
        f"{test_policy_metrics['low_risk_failures']}"
    )

    print(
        f"Failures routed to review:      "
        f"{test_policy_metrics['review_failures']}"
    )

    print(
        f"High-risk failures:             "
        f"{test_policy_metrics['high_risk_failures']}"
    )

    print(
        f"High-risk false alarms:         "
        f"{test_policy_metrics['high_risk_passes']}"
    )

    print(
        f"Low-risk NPV:                   "
        f"{test_policy_metrics['low_risk_npv']:.4f}"
    )

    print(
        f"High-risk precision:            "
        f"{test_policy_metrics['high_risk_precision']:.4f}"
    )

    print("\nTest decision summary:")

    print(
        decision_summary
        .round(4)
        .to_string(index=False)
    )

    print("\nSaved outputs:")

    print(
        f"Complete decision system:       "
        f"{COMPLETE_SYSTEM_PATH}"
    )

    print(
        f"Calibrator:                     "
        f"{CALIBRATOR_PATH}"
    )

    print(
        f"Test predictions:               "
        f"{TEST_PREDICTIONS_PATH}"
    )

    print(
        f"Decision policy report:         "
        f"{CALIBRATION_REPORTS_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading development and held-out test data"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    sensor_columns = get_sensor_columns(
        development_df
    )

    x_development = development_df[
        sensor_columns
    ].copy()

    y_development = development_df[
        TARGET_COLUMN
    ].astype(int)

    x_test = test_df[
        sensor_columns
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    base_pipeline = (
        build_balanced_random_forest_pipeline()
    )

    LOGGER.info(
        "Generating development out-of-fold raw probabilities"
    )

    raw_oof_probabilities = (
        generate_oof_probabilities(
            pipeline=base_pipeline,
            x_data=x_development,
            y_data=y_development,
        )
    )

    (
        calibration_df,
        fitted_calibrators,
        calibrated_probability_mapping,
    ) = benchmark_calibrators(
        raw_oof_probabilities=raw_oof_probabilities,
        target=y_development,
    )

    calibration_df.to_csv(
        CALIBRATION_COMPARISON_PATH,
        index=False,
    )

    selected_calibration_method = str(
        calibration_df.iloc[0][
            "calibration_method"
        ]
    )

    selected_calibrator = (
        fitted_calibrators[
            selected_calibration_method
        ]
    )

    selected_oof_probabilities = (
        calibrated_probability_mapping[
            selected_calibration_method
        ]
    )

    LOGGER.info(
        "Selected calibration method: %s",
        selected_calibration_method,
    )

    LOGGER.info(
        "Optimising three-level decision thresholds"
    )

    threshold_df = (
        build_decision_threshold_analysis(
            target=y_development,
            calibrated_probabilities=(
                selected_oof_probabilities
            ),
        )
    )

    threshold_df.to_csv(
        DECISION_THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    selected_policy = (
        select_decision_thresholds(
            threshold_df
        )
    )

    lower_threshold = float(
        selected_policy[
            "lower_threshold"
        ]
    )

    upper_threshold = float(
        selected_policy[
            "upper_threshold"
        ]
    )

    development_oof_output = (
        development_df[
            [
                ID_COLUMN,
                TIMESTAMP_COLUMN,
                TARGET_COLUMN,
                STATUS_COLUMN,
            ]
        ]
        .copy()
    )

    development_oof_output[
        "raw_failure_probability"
    ] = raw_oof_probabilities

    for (
        method_name,
        probabilities,
    ) in calibrated_probability_mapping.items():
        safe_name = (
            method_name
            .lower()
            .replace(" ", "_")
        )

        development_oof_output[
            f"{safe_name}_failure_probability"
        ] = probabilities

    development_oof_output[
        "selected_calibration_method"
    ] = selected_calibration_method

    development_oof_output[
        "decision"
    ] = assign_decision_class(
        probabilities=selected_oof_probabilities,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    development_oof_output.to_csv(
        DEVELOPMENT_OOF_PATH,
        index=False,
    )

    LOGGER.info(
        "Fitting final base model on all development data"
    )

    final_base_model = clone(
        base_pipeline
    )

    final_base_model.fit(
        x_development,
        y_development,
    )

    raw_test_probabilities = (
        final_base_model.predict_proba(
            x_test
        )[:, 1]
    )

    calibrated_test_probabilities = (
        selected_calibrator.predict(
            raw_test_probabilities
        )
    )

    test_probability_metrics = (
        calculate_probability_metrics(
            y_true=y_test.to_numpy(),
            probabilities=(
                calibrated_test_probabilities
            ),
        )
    )

    test_policy_metrics = (
        calculate_decision_policy_metrics(
            y_true=y_test.to_numpy(),
            probabilities=(
                calibrated_test_probabilities
            ),
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
    )

    test_prediction_df = (
        build_test_prediction_table(
            test_df=test_df,
            raw_probabilities=(
                raw_test_probabilities
            ),
            calibrated_probabilities=(
                calibrated_test_probabilities
            ),
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            calibration_method=(
                selected_calibration_method
            ),
        )
    )

    test_prediction_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    decision_summary = build_decision_summary(
        test_prediction_df
    )

    decision_summary.to_csv(
        TEST_DECISION_SUMMARY_PATH,
        index=False,
    )

    test_metrics_row = {
        "calibration_method": (
            selected_calibration_method
        ),
        "lower_threshold": (
            lower_threshold
        ),
        "upper_threshold": (
            upper_threshold
        ),
        **test_probability_metrics,
        **test_policy_metrics,
    }

    pd.DataFrame(
        [test_metrics_row]
    ).to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    complete_system = (
        CalibratedDecisionSystem(
            base_model=final_base_model,
            calibrator=selected_calibrator,
            calibration_method=(
                selected_calibration_method
            ),
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
            feature_columns=sensor_columns,
        )
    )

    joblib.dump(
        final_base_model,
        BASE_MODEL_PATH,
    )

    joblib.dump(
        selected_calibrator,
        CALIBRATOR_PATH,
    )

    joblib.dump(
        complete_system,
        COMPLETE_SYSTEM_PATH,
    )

    save_feature_columns(
        sensor_columns
    )

    save_calibration_comparison_plot(
        target=y_development,
        probability_mapping=(
            calibrated_probability_mapping
        ),
    )

    save_probability_distribution_plot(
        target=y_development,
        probabilities=(
            selected_oof_probabilities
        ),
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    save_policy_tradeoff_plot(
        threshold_df=threshold_df,
        selected_policy=selected_policy,
    )

    save_test_decision_plot(
        decision_summary
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Probability calibration and "
            "three-level engineering decision policy"
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
        "selected_calibration_method": (
            selected_calibration_method
        ),
        "calibration_comparison": (
            calibration_df.to_dict(
                orient="records"
            )
        ),
        "selected_policy": (
            selected_policy
        ),
        "held_out_test_probability_metrics": (
            test_probability_metrics
        ),
        "held_out_test_policy_metrics": (
            test_policy_metrics
        ),
        "held_out_test_decision_summary": (
            decision_summary.to_dict(
                orient="records"
            )
        ),
        "decision_definitions": {
            "Low Risk": (
                "Calibrated probability below the lower threshold. "
                "Continue processing with standard monitoring."
            ),
            "Engineering Review": (
                "Probability lies between the lower and upper "
                "thresholds. Human engineering assessment is required."
            ),
            "High Risk": (
                "Calibrated probability is at or above the upper "
                "threshold. Hold the record for inspection."
            ),
        },
        "cost_assumptions": {
            "low_risk_failure_cost": (
                LOW_RISK_FAILURE_COST
            ),
            "review_pass_cost": (
                REVIEW_PASS_COST
            ),
            "review_failure_cost": (
                REVIEW_FAILURE_COST
            ),
            "high_risk_pass_cost": (
                HIGH_RISK_PASS_COST
            ),
            "high_risk_failure_cost": (
                HIGH_RISK_FAILURE_COST
            ),
            "warning": (
                "These are analytical decision costs, not verified "
                "Infineon or semiconductor-fab financial values."
            ),
        },
    }

    save_summary(
        summary
    )

    print_console_summary(
        calibration_df=calibration_df,
        selected_calibration_method=(
            selected_calibration_method
        ),
        selected_policy=selected_policy,
        test_probability_metrics=(
            test_probability_metrics
        ),
        test_policy_metrics=(
            test_policy_metrics
        ),
        decision_summary=decision_summary,
    )


if __name__ == "__main__":
    main()