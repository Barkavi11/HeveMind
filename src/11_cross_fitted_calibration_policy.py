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
    log_loss,
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

MODEL_ARTIFACTS_DIR = (
    ARTIFACTS_DIR
    / "models"
    / "cross_fitted_calibration"
)

METADATA_DIR = ARTIFACTS_DIR / "metadata"

REPORTS_DIR = ROOT_DIR / "reports"

OUTPUT_DIR = (
    REPORTS_DIR
    / "cross_fitted_calibration_policy"
)

TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

BASE_MODEL_PATH = (
    MODEL_ARTIFACTS_DIR
    / "balanced_random_forest.joblib"
)

CALIBRATOR_PATH = (
    MODEL_ARTIFACTS_DIR
    / "cross_fitted_calibrator.joblib"
)

DECISION_SYSTEM_PATH = (
    MODEL_ARTIFACTS_DIR
    / "cross_fitted_decision_system.joblib"
)

FEATURE_COLUMNS_PATH = (
    METADATA_DIR
    / "cross_fitted_feature_columns.json"
)

CALIBRATION_COMPARISON_PATH = (
    TABLES_DIR
    / "cross_fitted_calibration_comparison.csv"
)

DEVELOPMENT_PREDICTIONS_PATH = (
    TABLES_DIR
    / "cross_fitted_development_predictions.csv"
)

THRESHOLD_ANALYSIS_PATH = (
    TABLES_DIR
    / "cross_fitted_policy_threshold_analysis.csv"
)

TEST_PREDICTIONS_PATH = (
    TABLES_DIR
    / "cross_fitted_test_predictions.csv"
)

TEST_DECISION_SUMMARY_PATH = (
    TABLES_DIR
    / "cross_fitted_test_decision_summary.csv"
)

TEST_METRICS_PATH = (
    TABLES_DIR
    / "cross_fitted_test_metrics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "cross_fitted_calibration_summary.json"
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

BASE_MODEL_CV_SPLITS = 5
CALIBRATION_CV_SPLITS = 5
CALIBRATION_BINS = 10

PROBABILITY_EPSILON = 1e-6

LOWER_THRESHOLD_MINIMUM = 0.005
LOWER_THRESHOLD_MAXIMUM = 0.25

UPPER_THRESHOLD_MINIMUM = 0.05
UPPER_THRESHOLD_MAXIMUM = 0.60

THRESHOLD_STEP = 0.005

# Operational targets
MAXIMUM_LOW_RISK_FAILURE_LEAKAGE = 0.10
MINIMUM_LOW_RISK_NPV = 0.98
MAXIMUM_REVIEW_RATE = 0.50
MINIMUM_AUTOMATION_COVERAGE = 0.50
MINIMUM_HIGH_RISK_FAILURE_CAPTURE = 0.20
MINIMUM_REVIEW_PLUS_HIGH_CAPTURE = 0.90

# Analytical decision costs
LOW_RISK_FAILURE_COST = 20.0
LOW_RISK_PASS_COST = 0.0

REVIEW_FAILURE_COST = 0.5
REVIEW_PASS_COST = 1.0

HIGH_RISK_FAILURE_COST = 0.0
HIGH_RISK_PASS_COST = 3.0


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        MODEL_ARTIFACTS_DIR,
        METADATA_DIR,
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
# BASE MODEL
# ============================================================
def build_base_pipeline() -> Pipeline:
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


def generate_base_oof_probabilities(
    pipeline: Pipeline,
    features: pd.DataFrame,
    target: pd.Series,
) -> np.ndarray:
    cross_validator = StratifiedKFold(
        n_splits=BASE_MODEL_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    probabilities = cross_val_predict(
        estimator=clone(pipeline),
        X=features,
        y=target,
        cv=cross_validator,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    return probabilities


# ============================================================
# CALIBRATORS
# ============================================================
class IdentityCalibrator:
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
            np.asarray(
                probabilities,
                dtype=float,
            ),
            0.0,
            1.0,
        )


class SigmoidCalibrator:
    def __init__(self) -> None:
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
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


class BetaCalibrator:
    """
    Beta calibration using logistic regression on:

        log(p)
        log(1 - p)

    This is often more flexible than sigmoid calibration while
    remaining smoother than isotonic regression.
    """

    def __init__(self) -> None:
        self.model = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=RANDOM_STATE,
        )

    @staticmethod
    def _transform(
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.clip(
            np.asarray(
                probabilities,
                dtype=float,
            ),
            PROBABILITY_EPSILON,
            1.0 - PROBABILITY_EPSILON,
        )

        return np.column_stack(
            [
                np.log(probabilities),
                np.log1p(-probabilities),
            ]
        )

    def fit(
        self,
        probabilities: np.ndarray,
        target: np.ndarray,
    ) -> "BetaCalibrator":
        transformed = self._transform(
            probabilities
        )

        self.model.fit(
            transformed,
            target,
        )

        return self

    def predict(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        transformed = self._transform(
            probabilities
        )

        return self.model.predict_proba(
            transformed
        )[:, 1]


class IsotonicCalibrator:
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
            self.model.predict(
                probabilities
            ),
            0.0,
            1.0,
        )


def build_calibrator(
    method_name: str,
) -> Any:
    registry = {
        "Uncalibrated": IdentityCalibrator,
        "Sigmoid": SigmoidCalibrator,
        "Beta": BetaCalibrator,
        "Isotonic": IsotonicCalibrator,
    }

    if method_name not in registry:
        raise ValueError(
            f"Unsupported calibration method: {method_name}"
        )

    return registry[method_name]()


def get_calibration_methods() -> list[str]:
    return [
        "Uncalibrated",
        "Sigmoid",
        "Beta",
        "Isotonic",
    ]


# ============================================================
# CROSS-FITTED CALIBRATION
# ============================================================
def generate_cross_fitted_calibrated_probabilities(
    raw_oof_probabilities: np.ndarray,
    target: pd.Series,
    method_name: str,
) -> np.ndarray:
    """
    Cross-fit the calibration layer.

    Each development row is calibrated using a calibrator fitted only
    on other development rows.
    """
    target_array = target.to_numpy(
        dtype=int
    )

    calibrated_probabilities = np.zeros(
        len(target_array),
        dtype=float,
    )

    cross_validator = StratifiedKFold(
        n_splits=CALIBRATION_CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE + 100,
    )

    for fold_number, (
        calibration_train_indices,
        calibration_validation_indices,
    ) in enumerate(
        cross_validator.split(
            raw_oof_probabilities,
            target_array,
        ),
        start=1,
    ):
        LOGGER.info(
            "Cross-fitting %s calibrator: fold %s/%s",
            method_name,
            fold_number,
            CALIBRATION_CV_SPLITS,
        )

        calibrator = build_calibrator(
            method_name
        )

        calibrator.fit(
            raw_oof_probabilities[
                calibration_train_indices
            ],
            target_array[
                calibration_train_indices
            ],
        )

        calibrated_probabilities[
            calibration_validation_indices
        ] = calibrator.predict(
            raw_oof_probabilities[
                calibration_validation_indices
            ]
        )

    return np.clip(
        calibrated_probabilities,
        0.0,
        1.0,
    )


# ============================================================
# CALIBRATION METRICS
# ============================================================
def expected_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = CALIBRATION_BINS,
) -> float:
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

    calibration_error = 0.0
    number_of_records = len(target)

    for bin_index in range(
        number_of_bins
    ):
        bin_mask = (
            bin_indices == bin_index
        )

        if not np.any(bin_mask):
            continue

        observed_failure_rate = float(
            target[bin_mask].mean()
        )

        mean_probability = float(
            probabilities[bin_mask].mean()
        )

        bin_weight = float(
            bin_mask.sum()
            / number_of_records
        )

        calibration_error += (
            bin_weight
            * abs(
                observed_failure_rate
                - mean_probability
            )
        )

    return float(
        calibration_error
    )


def maximum_calibration_error(
    target: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = CALIBRATION_BINS,
) -> float:
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

    errors: list[float] = []

    for bin_index in range(
        number_of_bins
    ):
        bin_mask = (
            bin_indices == bin_index
        )

        if not np.any(bin_mask):
            continue

        observed_failure_rate = float(
            target[bin_mask].mean()
        )

        mean_probability = float(
            probabilities[bin_mask].mean()
        )

        errors.append(
            abs(
                observed_failure_rate
                - mean_probability
            )
        )

    return (
        float(max(errors))
        if errors
        else 0.0
    )


def calculate_probability_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float]:
    clipped_probabilities = np.clip(
        probabilities,
        PROBABILITY_EPSILON,
        1.0 - PROBABILITY_EPSILON,
    )

    return {
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
        "log_loss": float(
            log_loss(
                target,
                clipped_probabilities,
                labels=[0, 1],
            )
        ),
        "expected_calibration_error": float(
            expected_calibration_error(
                target,
                probabilities,
            )
        ),
        "maximum_calibration_error": float(
            maximum_calibration_error(
                target,
                probabilities,
            )
        ),
        "mean_predicted_probability": float(
            probabilities.mean()
        ),
        "observed_failure_rate": float(
            target.mean()
        ),
    }


def benchmark_cross_fitted_calibrators(
    raw_oof_probabilities: np.ndarray,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    dict[str, np.ndarray],
]:
    target_array = target.to_numpy(
        dtype=int
    )

    rows: list[dict[str, Any]] = []
    probability_mapping: dict[
        str,
        np.ndarray,
    ] = {}

    for method_name in get_calibration_methods():
        LOGGER.info(
            "Evaluating cross-fitted calibration method: %s",
            method_name,
        )

        if method_name == "Uncalibrated":
            calibrated_probabilities = (
                raw_oof_probabilities.copy()
            )

        else:
            calibrated_probabilities = (
                generate_cross_fitted_calibrated_probabilities(
                    raw_oof_probabilities=(
                        raw_oof_probabilities
                    ),
                    target=target,
                    method_name=method_name,
                )
            )

        metrics = calculate_probability_metrics(
            target=target_array,
            probabilities=calibrated_probabilities,
        )

        rows.append(
            {
                "calibration_method": (
                    method_name
                ),
                **metrics,
            }
        )

        probability_mapping[
            method_name
        ] = calibrated_probabilities

    comparison_df = pd.DataFrame(
        rows
    )

    comparison_df["ranking_score"] = (
        comparison_df["brier_score"]
        + comparison_df[
            "expected_calibration_error"
        ]
        + 0.25
        * comparison_df[
            "maximum_calibration_error"
        ]
    )

    comparison_df = comparison_df.sort_values(
        by=[
            "ranking_score",
            "brier_score",
            "expected_calibration_error",
            "log_loss",
        ],
        ascending=True,
    ).reset_index(drop=True)

    return (
        comparison_df,
        probability_mapping,
    )


# ============================================================
# DECISION POLICY
# ============================================================
def assign_decision(
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    return np.select(
        [
            probabilities
            < lower_threshold,
            probabilities
            >= upper_threshold,
        ],
        [
            "Low Risk",
            "High Risk",
        ],
        default="Engineering Review",
    )


def calculate_policy_metrics(
    target: np.ndarray,
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> dict[str, Any]:
    decisions = assign_decision(
        probabilities=probabilities,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    low_mask = (
        decisions == "Low Risk"
    )

    review_mask = (
        decisions == "Engineering Review"
    )

    high_mask = (
        decisions == "High Risk"
    )

    total_records = len(target)

    total_failures = int(
        np.sum(target == 1)
    )

    total_passes = int(
        np.sum(target == 0)
    )

    low_records = int(
        low_mask.sum()
    )

    review_records = int(
        review_mask.sum()
    )

    high_records = int(
        high_mask.sum()
    )

    low_failures = int(
        np.sum(
            low_mask
            & (target == 1)
        )
    )

    low_passes = int(
        np.sum(
            low_mask
            & (target == 0)
        )
    )

    review_failures = int(
        np.sum(
            review_mask
            & (target == 1)
        )
    )

    review_passes = int(
        np.sum(
            review_mask
            & (target == 0)
        )
    )

    high_failures = int(
        np.sum(
            high_mask
            & (target == 1)
        )
    )

    high_passes = int(
        np.sum(
            high_mask
            & (target == 0)
        )
    )

    low_risk_npv = (
        low_passes / low_records
        if low_records > 0
        else np.nan
    )

    high_risk_precision = (
        high_failures / high_records
        if high_records > 0
        else 0.0
    )

    low_risk_failure_leakage = (
        low_failures / total_failures
        if total_failures > 0
        else 0.0
    )

    high_risk_failure_capture = (
        high_failures / total_failures
        if total_failures > 0
        else 0.0
    )

    review_failure_capture = (
        review_failures / total_failures
        if total_failures > 0
        else 0.0
    )

    review_plus_high_capture = (
        (
            review_failures
            + high_failures
        )
        / total_failures
        if total_failures > 0
        else 0.0
    )

    review_rate = (
        review_records / total_records
    )

    automation_coverage = (
        (
            low_records
            + high_records
        )
        / total_records
    )

    high_risk_false_alarm_rate = (
        high_passes / total_passes
        if total_passes > 0
        else 0.0
    )

    operational_cost = (
        LOW_RISK_FAILURE_COST
        * low_failures
        + LOW_RISK_PASS_COST
        * low_passes
        + REVIEW_FAILURE_COST
        * review_failures
        + REVIEW_PASS_COST
        * review_passes
        + HIGH_RISK_FAILURE_COST
        * high_failures
        + HIGH_RISK_PASS_COST
        * high_passes
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
        "total_passes": int(
            total_passes
        ),
        "low_risk_records": int(
            low_records
        ),
        "review_records": int(
            review_records
        ),
        "high_risk_records": int(
            high_records
        ),
        "low_risk_rate": float(
            low_records / total_records
        ),
        "review_rate": float(
            review_rate
        ),
        "high_risk_rate": float(
            high_records / total_records
        ),
        "automation_coverage": float(
            automation_coverage
        ),
        "low_risk_failures": int(
            low_failures
        ),
        "low_risk_passes": int(
            low_passes
        ),
        "review_failures": int(
            review_failures
        ),
        "review_passes": int(
            review_passes
        ),
        "high_risk_failures": int(
            high_failures
        ),
        "high_risk_passes": int(
            high_passes
        ),
        "low_risk_npv": float(
            low_risk_npv
        )
        if np.isfinite(low_risk_npv)
        else np.nan,
        "high_risk_precision": float(
            high_risk_precision
        ),
        "low_risk_failure_leakage": float(
            low_risk_failure_leakage
        ),
        "high_risk_failure_capture": float(
            high_risk_failure_capture
        ),
        "review_failure_capture": float(
            review_failure_capture
        ),
        "review_plus_high_failure_capture": float(
            review_plus_high_capture
        ),
        "high_risk_false_alarm_rate": float(
            high_risk_false_alarm_rate
        ),
        "operational_cost": float(
            operational_cost
        ),
    }


def build_threshold_analysis(
    target: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    target_array = target.to_numpy(
        dtype=int
    )

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

    for lower_threshold in lower_thresholds:
        for upper_threshold in upper_thresholds:
            if (
                upper_threshold
                <= lower_threshold
            ):
                continue

            metrics = calculate_policy_metrics(
                target=target_array,
                probabilities=probabilities,
                lower_threshold=float(
                    lower_threshold
                ),
                upper_threshold=float(
                    upper_threshold
                ),
            )

            rows.append(
                metrics
            )

    return pd.DataFrame(
        rows
    )


def select_policy(
    threshold_df: pd.DataFrame,
) -> dict[str, Any]:
    strict_eligible = threshold_df.loc[
        (
            threshold_df[
                "low_risk_failure_leakage"
            ]
            <= MAXIMUM_LOW_RISK_FAILURE_LEAKAGE
        )
        & (
            threshold_df[
                "low_risk_npv"
            ]
            >= MINIMUM_LOW_RISK_NPV
        )
        & (
            threshold_df[
                "review_rate"
            ]
            <= MAXIMUM_REVIEW_RATE
        )
        & (
            threshold_df[
                "automation_coverage"
            ]
            >= MINIMUM_AUTOMATION_COVERAGE
        )
        & (
            threshold_df[
                "high_risk_failure_capture"
            ]
            >= MINIMUM_HIGH_RISK_FAILURE_CAPTURE
        )
        & (
            threshold_df[
                "review_plus_high_failure_capture"
            ]
            >= MINIMUM_REVIEW_PLUS_HIGH_CAPTURE
        )
    ].copy()

    if not strict_eligible.empty:
        selected = strict_eligible.sort_values(
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
            "Policy satisfies all operational constraints "
            "and minimises weighted operational cost"
        )

    else:
        LOGGER.warning(
            "No policy satisfied all operational constraints. "
            "Applying tiered fallback selection."
        )

        safe_candidates = threshold_df.loc[
            (
                threshold_df[
                    "low_risk_failure_leakage"
                ]
                <= MAXIMUM_LOW_RISK_FAILURE_LEAKAGE
            )
            & (
                threshold_df[
                    "review_plus_high_failure_capture"
                ]
                >= MINIMUM_REVIEW_PLUS_HIGH_CAPTURE
            )
        ].copy()

        if not safe_candidates.empty:
            selected = safe_candidates.sort_values(
                by=[
                    "review_rate",
                    "operational_cost",
                    "high_risk_failure_capture",
                    "high_risk_precision",
                ],
                ascending=[
                    True,
                    True,
                    False,
                    False,
                ],
            ).iloc[0]

            selection_reason = (
                "Fallback policy preserves failure-routing safety "
                "while minimising engineering-review burden"
            )

        else:
            selected = threshold_df.sort_values(
                by=[
                    "low_risk_failure_leakage",
                    "review_plus_high_failure_capture",
                    "operational_cost",
                    "review_rate",
                ],
                ascending=[
                    True,
                    False,
                    True,
                    True,
                ],
            ).iloc[0]

            selection_reason = (
                "Emergency fallback prioritising minimum unsafe "
                "low-risk leakage"
            )

    result = selected.to_dict()
    result["selection_reason"] = (
        selection_reason
    )

    return result


# ============================================================
# FINAL CALIBRATOR AND SYSTEM
# ============================================================
class CrossFittedDecisionSystem:
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
        self.calibration_method = (
            calibration_method
        )
        self.lower_threshold = (
            lower_threshold
        )
        self.upper_threshold = (
            upper_threshold
        )
        self.feature_columns = (
            feature_columns
        )

    def predict_raw_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        features = dataframe[
            self.feature_columns
        ].copy()

        return self.base_model.predict_proba(
            features
        )[:, 1]

    def predict_calibrated_probability(
        self,
        dataframe: pd.DataFrame,
    ) -> np.ndarray:
        raw_probabilities = (
            self.predict_raw_probability(
                dataframe
            )
        )

        return self.calibrator.predict(
            raw_probabilities
        )

    def predict_decision(
        self,
        dataframe: pd.DataFrame,
    ) -> pd.DataFrame:
        raw_probabilities = (
            self.predict_raw_probability(
                dataframe
            )
        )

        calibrated_probabilities = (
            self.calibrator.predict(
                raw_probabilities
            )
        )

        decisions = assign_decision(
            probabilities=(
                calibrated_probabilities
            ),
            lower_threshold=(
                self.lower_threshold
            ),
            upper_threshold=(
                self.upper_threshold
            ),
        )

        actions = np.select(
            [
                decisions == "Low Risk",
                decisions == "High Risk",
            ],
            [
                "Continue processing",
                "Hold and inspect",
            ],
            default=(
                "Engineering assessment required"
            ),
        )

        return pd.DataFrame(
            {
                "raw_failure_probability": (
                    raw_probabilities
                ),
                "calibrated_failure_probability": (
                    calibrated_probabilities
                ),
                "decision": decisions,
                "recommended_action": actions,
            },
            index=dataframe.index,
        )


# ============================================================
# OUTPUT TABLES
# ============================================================
def build_development_prediction_table(
    development_df: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_mapping: dict[str, np.ndarray],
    selected_method: str,
    selected_policy: dict[str, Any],
) -> pd.DataFrame:
    output = development_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    output[
        "raw_failure_probability"
    ] = raw_probabilities

    for method_name, probabilities in (
        calibrated_mapping.items()
    ):
        safe_name = (
            method_name
            .lower()
            .replace(" ", "_")
        )

        output[
            f"{safe_name}_cross_fitted_probability"
        ] = probabilities

    selected_probabilities = (
        calibrated_mapping[
            selected_method
        ]
    )

    output[
        "selected_calibration_method"
    ] = selected_method

    output[
        "selected_calibrated_probability"
    ] = selected_probabilities

    output["decision"] = assign_decision(
        probabilities=selected_probabilities,
        lower_threshold=float(
            selected_policy[
                "lower_threshold"
            ]
        ),
        upper_threshold=float(
            selected_policy[
                "upper_threshold"
            ]
        ),
    )

    return output


def build_test_prediction_table(
    test_df: pd.DataFrame,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
    selected_method: str,
    lower_threshold: float,
    upper_threshold: float,
) -> pd.DataFrame:
    output = test_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    decisions = assign_decision(
        probabilities=calibrated_probabilities,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
    )

    output[
        "raw_failure_probability"
    ] = raw_probabilities

    output[
        "calibrated_failure_probability"
    ] = calibrated_probabilities

    output[
        "calibration_method"
    ] = selected_method

    output[
        "lower_threshold"
    ] = lower_threshold

    output[
        "upper_threshold"
    ] = upper_threshold

    output[
        "decision"
    ] = decisions

    output[
        "recommended_action"
    ] = np.select(
        [
            decisions == "Low Risk",
            decisions == "High Risk",
        ],
        [
            "Continue processing",
            "Hold and inspect",
        ],
        default=(
            "Engineering assessment required"
        ),
    )

    output[
        "decision_outcome"
    ] = np.select(
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

    for decision_name in [
        "Low Risk",
        "Engineering Review",
        "High Risk",
    ]:
        group = prediction_df.loc[
            prediction_df["decision"]
            == decision_name
        ]

        record_count = len(group)

        failure_count = int(
            group[TARGET_COLUMN].sum()
        )

        pass_count = int(
            record_count - failure_count
        )

        rows.append(
            {
                "decision": decision_name,
                "records": int(
                    record_count
                ),
                "record_rate": float(
                    record_count
                    / len(prediction_df)
                ),
                "actual_failures": int(
                    failure_count
                ),
                "actual_passes": int(
                    pass_count
                ),
                "observed_failure_rate": (
                    float(
                        failure_count
                        / record_count
                    )
                    if record_count > 0
                    else np.nan
                ),
                "mean_calibrated_probability": (
                    float(
                        group[
                            "calibrated_failure_probability"
                        ].mean()
                    )
                    if record_count > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_cross_fitted_calibration_plot(
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
        "Cross-Fitted Development Calibration"
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
        / "cross_fitted_calibration_comparison.png",
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
            f"{lower_threshold:.3f}"
        ),
    )

    axis.axvline(
        upper_threshold,
        linestyle=":",
        label=(
            f"Review/High threshold "
            f"{upper_threshold:.3f}"
        ),
    )

    axis.set_title(
        "Cross-Fitted Calibrated Probability Distribution"
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
        / "cross_fitted_probability_distribution.png",
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
        threshold_df[
            "high_risk_precision"
        ],
        c=threshold_df[
            "operational_cost"
        ],
        alpha=0.5,
    )

    axis.scatter(
        selected_policy[
            "review_rate"
        ],
        selected_policy[
            "high_risk_precision"
        ],
        marker="X",
        s=180,
        label="Selected policy",
    )

    axis.axvline(
        MAXIMUM_REVIEW_RATE,
        linestyle="--",
        label=(
            f"Maximum target review rate "
            f"{MAXIMUM_REVIEW_RATE:.0%}"
        ),
    )

    axis.set_title(
        "Cross-Fitted Decision Policy Trade-Off"
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
        label="Weighted Operational Cost",
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "cross_fitted_policy_tradeoff.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_test_decision_outcomes_plot(
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
        decision_summary[
            "actual_passes"
        ],
        label="Pass",
    )

    axis.bar(
        positions,
        decision_summary[
            "actual_failures"
        ],
        bottom=decision_summary[
            "actual_passes"
        ],
        label="Fail",
    )

    axis.set_xticks(
        positions
    )

    axis.set_xticklabels(
        decision_summary[
            "decision"
        ]
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
        / "cross_fitted_test_decision_outcomes.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


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
                "feature_count": len(
                    feature_columns
                ),
                "features": (
                    feature_columns
                ),
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
    selected_method: str,
    selected_policy: dict[str, Any],
    test_probability_metrics: dict[str, float],
    test_policy_metrics: dict[str, Any],
    decision_summary: pd.DataFrame,
) -> None:
    print("\n" + "=" * 122)
    print(
        "HEVEMIND CROSS-FITTED CALIBRATION "
        "AND OPERATIONAL DECISION POLICY"
    )
    print("=" * 122)

    print(
        "\nCross-fitted development calibration comparison:"
    )

    display_columns = [
        "calibration_method",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
        "maximum_calibration_error",
        "mean_predicted_probability",
        "observed_failure_rate",
        "ranking_score",
    ]

    print(
        calibration_df[
            display_columns
        ]
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nSelected calibration method:")

    print(
        f"Method:                          "
        f"{selected_method}"
    )

    print("\nSelected development policy:")

    print(
        f"Lower threshold:                 "
        f"{selected_policy['lower_threshold']:.4f}"
    )

    print(
        f"Upper threshold:                 "
        f"{selected_policy['upper_threshold']:.4f}"
    )

    print(
        f"Selection reason:                "
        f"{selected_policy['selection_reason']}"
    )

    print(
        f"Development review rate:         "
        f"{selected_policy['review_rate']:.4f}"
    )

    print(
        f"Development automation coverage: "
        f"{selected_policy['automation_coverage']:.4f}"
    )

    print(
        f"Development low-risk NPV:        "
        f"{selected_policy['low_risk_npv']:.4f}"
    )

    print(
        f"Development failure leakage:     "
        f"{selected_policy['low_risk_failure_leakage']:.4f}"
    )

    print(
        f"Development high-risk precision: "
        f"{selected_policy['high_risk_precision']:.4f}"
    )

    print(
        f"Development high-risk capture:   "
        f"{selected_policy['high_risk_failure_capture']:.4f}"
    )

    print("\nHeld-out test probability metrics:")

    print(
        f"ROC-AUC:                         "
        f"{test_probability_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                          "
        f"{test_probability_metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:                     "
        f"{test_probability_metrics['brier_score']:.4f}"
    )

    print(
        f"Log loss:                        "
        f"{test_probability_metrics['log_loss']:.4f}"
    )

    print(
        f"Expected calibration error:      "
        f"{test_probability_metrics['expected_calibration_error']:.4f}"
    )

    print(
        f"Maximum calibration error:       "
        f"{test_probability_metrics['maximum_calibration_error']:.4f}"
    )

    print("\nHeld-out test decision policy:")

    print(
        f"Low-risk records:                "
        f"{test_policy_metrics['low_risk_records']}"
    )

    print(
        f"Engineering-review records:      "
        f"{test_policy_metrics['review_records']}"
    )

    print(
        f"High-risk records:               "
        f"{test_policy_metrics['high_risk_records']}"
    )

    print(
        f"Review rate:                     "
        f"{test_policy_metrics['review_rate']:.4f}"
    )

    print(
        f"Automation coverage:             "
        f"{test_policy_metrics['automation_coverage']:.4f}"
    )

    print(
        f"Low-risk failures:               "
        f"{test_policy_metrics['low_risk_failures']}"
    )

    print(
        f"Failures routed to review:       "
        f"{test_policy_metrics['review_failures']}"
    )

    print(
        f"High-risk failures:              "
        f"{test_policy_metrics['high_risk_failures']}"
    )

    print(
        f"High-risk false alarms:          "
        f"{test_policy_metrics['high_risk_passes']}"
    )

    print(
        f"Low-risk NPV:                    "
        f"{test_policy_metrics['low_risk_npv']:.4f}"
    )

    print(
        f"High-risk precision:             "
        f"{test_policy_metrics['high_risk_precision']:.4f}"
    )

    print(
        f"Review plus high-risk capture:   "
        f"{test_policy_metrics['review_plus_high_failure_capture']:.4f}"
    )

    print("\nHeld-out test decision summary:")

    print(
        decision_summary
        .round(4)
        .to_string(
            index=False
        )
    )

    print("\nSaved outputs:")

    print(
        f"Decision system:                 "
        f"{DECISION_SYSTEM_PATH}"
    )

    print(
        f"Base model:                      "
        f"{BASE_MODEL_PATH}"
    )

    print(
        f"Calibrator:                      "
        f"{CALIBRATOR_PATH}"
    )

    print(
        f"Test predictions:                "
        f"{TEST_PREDICTIONS_PATH}"
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
        "Loading development and test datasets"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    feature_columns = get_sensor_columns(
        development_df
    )

    x_development = development_df[
        feature_columns
    ].copy()

    y_development = development_df[
        TARGET_COLUMN
    ].astype(int)

    x_test = test_df[
        feature_columns
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].astype(int)

    base_pipeline = build_base_pipeline()

    LOGGER.info(
        "Generating base-model out-of-fold probabilities"
    )

    raw_oof_probabilities = (
        generate_base_oof_probabilities(
            pipeline=base_pipeline,
            features=x_development,
            target=y_development,
        )
    )

    (
        calibration_df,
        cross_fitted_probability_mapping,
    ) = benchmark_cross_fitted_calibrators(
        raw_oof_probabilities=(
            raw_oof_probabilities
        ),
        target=y_development,
    )

    calibration_df.to_csv(
        CALIBRATION_COMPARISON_PATH,
        index=False,
    )

    selected_method = str(
        calibration_df.iloc[0][
            "calibration_method"
        ]
    )

    selected_development_probabilities = (
        cross_fitted_probability_mapping[
            selected_method
        ]
    )

    LOGGER.info(
        "Selected calibration method: %s",
        selected_method,
    )

    LOGGER.info(
        "Optimising three-level policy using cross-fitted probabilities"
    )

    threshold_df = build_threshold_analysis(
        target=y_development,
        probabilities=(
            selected_development_probabilities
        ),
    )

    threshold_df.to_csv(
        THRESHOLD_ANALYSIS_PATH,
        index=False,
    )

    selected_policy = select_policy(
        threshold_df
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

    development_prediction_df = (
        build_development_prediction_table(
            development_df=development_df,
            raw_probabilities=(
                raw_oof_probabilities
            ),
            calibrated_mapping=(
                cross_fitted_probability_mapping
            ),
            selected_method=(
                selected_method
            ),
            selected_policy=(
                selected_policy
            ),
        )
    )

    development_prediction_df.to_csv(
        DEVELOPMENT_PREDICTIONS_PATH,
        index=False,
    )

    LOGGER.info(
        "Fitting deployment calibrator on all development OOF probabilities"
    )

    final_calibrator = build_calibrator(
        selected_method
    )

    final_calibrator.fit(
        raw_oof_probabilities,
        y_development.to_numpy(
            dtype=int
        ),
    )

    LOGGER.info(
        "Fitting final base model on all development records"
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
        final_calibrator.predict(
            raw_test_probabilities
        )
    )

    test_probability_metrics = (
        calculate_probability_metrics(
            target=y_test.to_numpy(
                dtype=int
            ),
            probabilities=(
                calibrated_test_probabilities
            ),
        )
    )

    test_policy_metrics = (
        calculate_policy_metrics(
            target=y_test.to_numpy(
                dtype=int
            ),
            probabilities=(
                calibrated_test_probabilities
            ),
            lower_threshold=(
                lower_threshold
            ),
            upper_threshold=(
                upper_threshold
            ),
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
            selected_method=(
                selected_method
            ),
            lower_threshold=(
                lower_threshold
            ),
            upper_threshold=(
                upper_threshold
            ),
        )
    )

    test_prediction_df.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    decision_summary = (
        build_decision_summary(
            test_prediction_df
        )
    )

    decision_summary.to_csv(
        TEST_DECISION_SUMMARY_PATH,
        index=False,
    )

    pd.DataFrame(
        [
            {
                "selected_calibration_method": (
                    selected_method
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
        ]
    ).to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    decision_system = (
        CrossFittedDecisionSystem(
            base_model=final_base_model,
            calibrator=final_calibrator,
            calibration_method=(
                selected_method
            ),
            lower_threshold=(
                lower_threshold
            ),
            upper_threshold=(
                upper_threshold
            ),
            feature_columns=(
                feature_columns
            ),
        )
    )

    joblib.dump(
        final_base_model,
        BASE_MODEL_PATH,
    )

    joblib.dump(
        final_calibrator,
        CALIBRATOR_PATH,
    )

    joblib.dump(
        decision_system,
        DECISION_SYSTEM_PATH,
    )

    save_feature_columns(
        feature_columns
    )

    save_cross_fitted_calibration_plot(
        target=y_development,
        probability_mapping=(
            cross_fitted_probability_mapping
        ),
    )

    save_probability_distribution_plot(
        target=y_development,
        probabilities=(
            selected_development_probabilities
        ),
        lower_threshold=(
            lower_threshold
        ),
        upper_threshold=(
            upper_threshold
        ),
    )

    save_policy_tradeoff_plot(
        threshold_df=threshold_df,
        selected_policy=(
            selected_policy
        ),
    )

    save_test_decision_outcomes_plot(
        decision_summary
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Cross-fitted probability calibration "
            "and three-level decision policy"
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
        "base_model": (
            "Balanced Random Forest"
        ),
        "selected_calibration_method": (
            selected_method
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
        "methodology": {
            "base_probability_generation": (
                "Five-fold out-of-fold prediction"
            ),
            "calibration_evaluation": (
                "Five-fold cross-fitted calibration"
            ),
            "final_calibrator_training": (
                "Selected calibrator fitted on all development "
                "out-of-fold raw probabilities"
            ),
            "test_usage": (
                "Held-out test set used only once after model, "
                "calibrator and thresholds were selected"
            ),
        },
        "decision_definitions": {
            "Low Risk": (
                "Calibrated failure probability below the lower "
                "threshold. Continue processing with standard monitoring."
            ),
            "Engineering Review": (
                "Calibrated probability lies between the two "
                "thresholds. Manual engineering assessment is required."
            ),
            "High Risk": (
                "Calibrated probability meets or exceeds the upper "
                "threshold. Hold and inspect before continuing."
            ),
        },
        "cost_assumption_warning": (
            "Operational costs are analytical weights and are not "
            "verified semiconductor-fab financial values."
        ),
    }

    save_summary(
        summary
    )

    print_console_summary(
        calibration_df=calibration_df,
        selected_method=selected_method,
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