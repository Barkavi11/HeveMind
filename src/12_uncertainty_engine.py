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
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


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

CALIBRATED_MODEL_DIR = (
    ARTIFACTS_DIR
    / "models"
    / "cross_fitted_calibration"
)

BASE_MODEL_PATH = (
    CALIBRATED_MODEL_DIR
    / "balanced_random_forest.joblib"
)

CALIBRATOR_PATH = (
    CALIBRATED_MODEL_DIR
    / "cross_fitted_calibrator.joblib"
)

CALIBRATION_REPORT_DIR = (
    ROOT_DIR
    / "reports"
    / "cross_fitted_calibration_policy"
)

CALIBRATION_SUMMARY_PATH = (
    CALIBRATION_REPORT_DIR
    / "cross_fitted_calibration_summary.json"
)

UNCERTAINTY_ARTIFACT_DIR = (
    ARTIFACTS_DIR
    / "uncertainty_engine"
)

UNCERTAINTY_REPORT_DIR = (
    ROOT_DIR
    / "reports"
    / "uncertainty_engine"
)

TABLES_DIR = UNCERTAINTY_REPORT_DIR / "tables"
FIGURES_DIR = UNCERTAINTY_REPORT_DIR / "figures"

UNCERTAINTY_PREPROCESSOR_PATH = (
    UNCERTAINTY_ARTIFACT_DIR
    / "uncertainty_preprocessor.joblib"
)

ISOLATION_FOREST_PATH = (
    UNCERTAINTY_ARTIFACT_DIR
    / "uncertainty_isolation_forest.joblib"
)

NEAREST_NEIGHBOUR_PATH = (
    UNCERTAINTY_ARTIFACT_DIR
    / "uncertainty_nearest_neighbours.joblib"
)

REFERENCE_METADATA_PATH = (
    UNCERTAINTY_ARTIFACT_DIR
    / "uncertainty_reference_metadata.json"
)

DEVELOPMENT_UNCERTAINTY_PATH = (
    TABLES_DIR
    / "development_uncertainty_scores.csv"
)

TEST_UNCERTAINTY_PATH = (
    TABLES_DIR
    / "test_uncertainty_scores.csv"
)

TEST_DECISION_SUMMARY_PATH = (
    TABLES_DIR
    / "uncertainty_adjusted_decision_summary.csv"
)

UNCERTAINTY_BAND_SUMMARY_PATH = (
    TABLES_DIR
    / "uncertainty_band_performance.csv"
)

CONFIDENCE_BAND_SUMMARY_PATH = (
    TABLES_DIR
    / "confidence_band_performance.csv"
)

ABSTENTION_SUMMARY_PATH = (
    TABLES_DIR
    / "abstention_summary.csv"
)

SUMMARY_PATH = (
    UNCERTAINTY_REPORT_DIR
    / "uncertainty_engine_summary.json"
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

N_NEIGHBOURS = 5

ISOLATION_FOREST_ESTIMATORS = 500
ISOLATION_FOREST_CONTAMINATION = "auto"

LOW_CONFIDENCE_QUANTILE = 0.10
VERY_HIGH_UNCERTAINTY_QUANTILE = 0.95
EXTREME_OOD_QUANTILE = 0.975

TREE_INTERVAL_LOWER_QUANTILE = 0.05
TREE_INTERVAL_UPPER_QUANTILE = 0.95

# Confidence composition
ENSEMBLE_AGREEMENT_WEIGHT = 0.35
FOLD_STABILITY_WEIGHT = 0.20
DATA_CONFIDENCE_WEIGHT = 0.30
DECISION_MARGIN_WEIGHT = 0.15

# Data-confidence composition
COMPLETENESS_WEIGHT = 0.40
OOD_CONFIDENCE_WEIGHT = 0.35
NEIGHBOUR_CONFIDENCE_WEIGHT = 0.25


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        UNCERTAINTY_ARTIFACT_DIR,
        UNCERTAINTY_REPORT_DIR,
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
    train_df = load_split(
        TRAIN_PATH
    )

    validation_df = load_split(
        VALIDATION_PATH
    )

    test_df = load_split(
        TEST_PATH
    )

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


def load_calibration_artifacts() -> tuple[Any, Any, dict[str, Any]]:
    if not BASE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "The calibrated base model was not found. "
            "Run src/11_cross_fitted_calibration_policy.py first."
        )

    if not CALIBRATOR_PATH.exists():
        raise FileNotFoundError(
            "The Beta calibrator was not found. "
            "Run src/11_cross_fitted_calibration_policy.py first."
        )

    if not CALIBRATION_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "The cross-fitted calibration summary was not found."
        )

    base_model = joblib.load(
        BASE_MODEL_PATH
    )

    calibrator = joblib.load(
        CALIBRATOR_PATH
    )

    with CALIBRATION_SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        summary = json.load(
            file
        )

    return (
        base_model,
        calibrator,
        summary,
    )


def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            SENSOR_PREFIX
        )
    ]

    if not columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return columns


# ============================================================
# BASE MODEL CONSTRUCTION
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


# ============================================================
# CALIBRATION UTILITIES
# ============================================================
def calibrate_probabilities(
    calibrator: Any,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    probabilities = calibrator.predict(
        np.asarray(
            raw_probabilities,
            dtype=float,
        )
    )

    return np.clip(
        probabilities,
        0.0,
        1.0,
    )


def assign_original_decision(
    calibrated_probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    return np.select(
        [
            calibrated_probabilities
            < lower_threshold,
            calibrated_probabilities
            >= upper_threshold,
        ],
        [
            "Low Risk",
            "High Risk",
        ],
        default="Engineering Review",
    )


# ============================================================
# TREE-LEVEL PREDICTIONS
# ============================================================
def extract_tree_probabilities(
    fitted_pipeline: Pipeline,
    features: pd.DataFrame,
) -> np.ndarray:
    """
    Return one failure probability per tree and record.

    Shape:
        number_of_records x number_of_trees
    """
    if "imputer" not in fitted_pipeline.named_steps:
        raise ValueError(
            "The fitted pipeline does not contain an imputer."
        )

    if "model" not in fitted_pipeline.named_steps:
        raise ValueError(
            "The fitted pipeline does not contain a model."
        )

    imputer = fitted_pipeline.named_steps[
        "imputer"
    ]

    forest = fitted_pipeline.named_steps[
        "model"
    ]

    transformed_features = imputer.transform(
        features
    )

    estimators = getattr(
        forest,
        "estimators_",
        None,
    )

    if estimators is None:
        raise ValueError(
            "The fitted model does not expose individual estimators."
        )

    tree_probabilities = np.column_stack(
        [
            estimator.predict_proba(
                transformed_features
            )[:, 1]
            for estimator in estimators
        ]
    )

    return tree_probabilities


def calculate_tree_uncertainty(
    tree_probabilities: np.ndarray,
    calibrator: Any,
) -> dict[str, np.ndarray]:
    raw_mean = np.mean(
        tree_probabilities,
        axis=1,
    )

    raw_standard_deviation = np.std(
        tree_probabilities,
        axis=1,
        ddof=1,
    )

    raw_lower = np.quantile(
        tree_probabilities,
        TREE_INTERVAL_LOWER_QUANTILE,
        axis=1,
    )

    raw_upper = np.quantile(
        tree_probabilities,
        TREE_INTERVAL_UPPER_QUANTILE,
        axis=1,
    )

    calibrated_mean = calibrate_probabilities(
        calibrator,
        raw_mean,
    )

    calibrated_lower = calibrate_probabilities(
        calibrator,
        raw_lower,
    )

    calibrated_upper = calibrate_probabilities(
        calibrator,
        raw_upper,
    )

    interval_width = (
        calibrated_upper
        - calibrated_lower
    )

    return {
        "raw_tree_mean": raw_mean,
        "raw_tree_standard_deviation": (
            raw_standard_deviation
        ),
        "raw_tree_lower": raw_lower,
        "raw_tree_upper": raw_upper,
        "calibrated_tree_mean": (
            calibrated_mean
        ),
        "calibrated_tree_lower": (
            calibrated_lower
        ),
        "calibrated_tree_upper": (
            calibrated_upper
        ),
        "calibrated_interval_width": (
            interval_width
        ),
    }


# ============================================================
# CROSS-FITTED DEVELOPMENT UNCERTAINTY
# ============================================================
def generate_cross_fitted_uncertainty(
    features: pd.DataFrame,
    target: pd.Series,
    test_features: pd.DataFrame,
    calibrator: Any,
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
]:
    """
    Generate uncertainty indicators without fitting on the row being
    evaluated.

    Also returns calibrated predictions for the test set from each fold
    model to quantify model-to-model disagreement.
    """
    number_of_records = len(
        features
    )

    raw_oof_probability = np.zeros(
        number_of_records,
        dtype=float,
    )

    tree_standard_deviation = np.zeros(
        number_of_records,
        dtype=float,
    )

    interval_lower = np.zeros(
        number_of_records,
        dtype=float,
    )

    interval_upper = np.zeros(
        number_of_records,
        dtype=float,
    )

    interval_width = np.zeros(
        number_of_records,
        dtype=float,
    )

    cross_validator = StratifiedKFold(
        n_splits=CV_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    test_fold_probabilities: list[
        np.ndarray
    ] = []

    for fold_number, (
        training_indices,
        validation_indices,
    ) in enumerate(
        cross_validator.split(
            features,
            target,
        ),
        start=1,
    ):
        LOGGER.info(
            "Generating cross-fitted uncertainty: fold %s/%s",
            fold_number,
            CV_SPLITS,
        )

        fold_pipeline = build_base_pipeline()

        fold_pipeline.fit(
            features.iloc[
                training_indices
            ],
            target.iloc[
                training_indices
            ],
        )

        validation_features = features.iloc[
            validation_indices
        ]

        validation_tree_probabilities = (
            extract_tree_probabilities(
                fitted_pipeline=fold_pipeline,
                features=validation_features,
            )
        )

        validation_uncertainty = (
            calculate_tree_uncertainty(
                tree_probabilities=(
                    validation_tree_probabilities
                ),
                calibrator=calibrator,
            )
        )

        raw_oof_probability[
            validation_indices
        ] = validation_uncertainty[
            "raw_tree_mean"
        ]

        tree_standard_deviation[
            validation_indices
        ] = validation_uncertainty[
            "raw_tree_standard_deviation"
        ]

        interval_lower[
            validation_indices
        ] = validation_uncertainty[
            "calibrated_tree_lower"
        ]

        interval_upper[
            validation_indices
        ] = validation_uncertainty[
            "calibrated_tree_upper"
        ]

        interval_width[
            validation_indices
        ] = validation_uncertainty[
            "calibrated_interval_width"
        ]

        raw_test_probability = (
            fold_pipeline.predict_proba(
                test_features
            )[:, 1]
        )

        calibrated_test_probability = (
            calibrate_probabilities(
                calibrator=calibrator,
                raw_probabilities=(
                    raw_test_probability
                ),
            )
        )

        test_fold_probabilities.append(
            calibrated_test_probability
        )

    calibrated_oof_probability = (
        calibrate_probabilities(
            calibrator=calibrator,
            raw_probabilities=(
                raw_oof_probability
            ),
        )
    )

    test_fold_probability_matrix = (
        np.column_stack(
            test_fold_probabilities
        )
    )

    results = {
        "raw_probability": (
            raw_oof_probability
        ),
        "calibrated_probability": (
            calibrated_oof_probability
        ),
        "tree_standard_deviation": (
            tree_standard_deviation
        ),
        "interval_lower": (
            interval_lower
        ),
        "interval_upper": (
            interval_upper
        ),
        "interval_width": (
            interval_width
        ),
    }

    return (
        results,
        test_fold_probability_matrix,
    )


# ============================================================
# OOD AND DATA CONFIDENCE ENGINE
# ============================================================
def build_uncertainty_preprocessor() -> Pipeline:
    return Pipeline(
        steps=[
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
        ]
    )


def fit_distribution_models(
    development_features: pd.DataFrame,
) -> tuple[
    Pipeline,
    IsolationForest,
    NearestNeighbors,
    np.ndarray,
    np.ndarray,
]:
    preprocessor = (
        build_uncertainty_preprocessor()
    )

    transformed_development = (
        preprocessor.fit_transform(
            development_features
        )
    )

    isolation_forest = IsolationForest(
        n_estimators=(
            ISOLATION_FOREST_ESTIMATORS
        ),
        contamination=(
            ISOLATION_FOREST_CONTAMINATION
        ),
        max_samples="auto",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    isolation_forest.fit(
        transformed_development
    )

    # Higher values indicate greater abnormality.
    development_ood_raw = (
        -isolation_forest.decision_function(
            transformed_development
        )
    )

    nearest_neighbours = NearestNeighbors(
        n_neighbors=(
            N_NEIGHBOURS + 1
        ),
        metric="euclidean",
        n_jobs=-1,
    )

    nearest_neighbours.fit(
        transformed_development
    )

    development_distances, _ = (
        nearest_neighbours.kneighbors(
            transformed_development
        )
    )

    # Exclude the first neighbour because it is the row itself.
    development_neighbour_distance = (
        development_distances[:, 1:]
        .mean(axis=1)
    )

    return (
        preprocessor,
        isolation_forest,
        nearest_neighbours,
        development_ood_raw,
        development_neighbour_distance,
    )


def score_distribution_shift(
    features: pd.DataFrame,
    preprocessor: Pipeline,
    isolation_forest: IsolationForest,
    nearest_neighbours: NearestNeighbors,
) -> tuple[np.ndarray, np.ndarray]:
    transformed_features = (
        preprocessor.transform(
            features
        )
    )

    ood_raw = (
        -isolation_forest.decision_function(
            transformed_features
        )
    )

    distances, _ = (
        nearest_neighbours.kneighbors(
            transformed_features,
            n_neighbors=N_NEIGHBOURS,
        )
    )

    mean_neighbour_distance = (
        distances.mean(axis=1)
    )

    return (
        ood_raw,
        mean_neighbour_distance,
    )


# ============================================================
# NORMALISATION UTILITIES
# ============================================================
def percentile_normalize(
    values: np.ndarray,
    reference_values: np.ndarray,
) -> np.ndarray:
    """
    Convert values into an empirical percentile relative to the
    development reference distribution.
    """
    reference_sorted = np.sort(
        np.asarray(
            reference_values,
            dtype=float,
        )
    )

    ranks = np.searchsorted(
        reference_sorted,
        np.asarray(
            values,
            dtype=float,
        ),
        side="right",
    )

    percentiles = (
        ranks
        / max(
            len(reference_sorted),
            1,
        )
    )

    return np.clip(
        percentiles,
        0.0,
        1.0,
    )


def upper_reference_normalize(
    values: np.ndarray,
    reference_values: np.ndarray,
    upper_quantile: float = 0.95,
) -> np.ndarray:
    reference_scale = float(
        np.quantile(
            reference_values,
            upper_quantile,
        )
    )

    if (
        not np.isfinite(
            reference_scale
        )
        or reference_scale <= 1e-12
    ):
        return np.zeros_like(
            values,
            dtype=float,
        )

    return np.clip(
        values / reference_scale,
        0.0,
        1.0,
    )


# ============================================================
# CONFIDENCE ENGINE
# ============================================================
def calculate_decision_margin(
    probabilities: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
) -> np.ndarray:
    """
    Measures distance from the nearest operational decision boundary.

    Values near zero represent borderline decisions.
    """
    distance_to_lower = np.abs(
        probabilities
        - lower_threshold
    )

    distance_to_upper = np.abs(
        probabilities
        - upper_threshold
    )

    nearest_distance = np.minimum(
        distance_to_lower,
        distance_to_upper,
    )

    maximum_possible_distance = max(
        lower_threshold,
        1.0 - upper_threshold,
        upper_threshold - lower_threshold,
        1e-8,
    )

    return np.clip(
        nearest_distance
        / maximum_possible_distance,
        0.0,
        1.0,
    )


def calculate_confidence_components(
    calibrated_probabilities: np.ndarray,
    tree_standard_deviation: np.ndarray,
    fold_standard_deviation: np.ndarray,
    missing_rate: np.ndarray,
    ood_percentile: np.ndarray,
    neighbour_percentile: np.ndarray,
    lower_threshold: float,
    upper_threshold: float,
    tree_reference: np.ndarray,
    fold_reference: np.ndarray,
) -> dict[str, np.ndarray]:
    normalized_tree_uncertainty = (
        upper_reference_normalize(
            values=tree_standard_deviation,
            reference_values=tree_reference,
        )
    )

    normalized_fold_uncertainty = (
        upper_reference_normalize(
            values=fold_standard_deviation,
            reference_values=fold_reference,
        )
    )

    ensemble_agreement = (
        1.0
        - normalized_tree_uncertainty
    )

    fold_stability = (
        1.0
        - normalized_fold_uncertainty
    )

    completeness_confidence = (
        1.0
        - np.clip(
            missing_rate,
            0.0,
            1.0,
        )
    )

    ood_confidence = (
        1.0
        - ood_percentile
    )

    neighbour_confidence = (
        1.0
        - neighbour_percentile
    )

    data_confidence = (
        COMPLETENESS_WEIGHT
        * completeness_confidence
        + OOD_CONFIDENCE_WEIGHT
        * ood_confidence
        + NEIGHBOUR_CONFIDENCE_WEIGHT
        * neighbour_confidence
    )

    decision_margin = (
        calculate_decision_margin(
            probabilities=(
                calibrated_probabilities
            ),
            lower_threshold=(
                lower_threshold
            ),
            upper_threshold=(
                upper_threshold
            ),
        )
    )

    prediction_confidence = (
        ENSEMBLE_AGREEMENT_WEIGHT
        * ensemble_agreement
        + FOLD_STABILITY_WEIGHT
        * fold_stability
        + DATA_CONFIDENCE_WEIGHT
        * data_confidence
        + DECISION_MARGIN_WEIGHT
        * decision_margin
    )

    prediction_confidence = np.clip(
        prediction_confidence,
        0.0,
        1.0,
    )

    combined_uncertainty = (
        1.0
        - prediction_confidence
    )

    return {
        "normalized_tree_uncertainty": (
            normalized_tree_uncertainty
        ),
        "normalized_fold_uncertainty": (
            normalized_fold_uncertainty
        ),
        "ensemble_agreement": (
            ensemble_agreement
        ),
        "fold_stability": (
            fold_stability
        ),
        "completeness_confidence": (
            completeness_confidence
        ),
        "ood_confidence": (
            ood_confidence
        ),
        "neighbour_confidence": (
            neighbour_confidence
        ),
        "data_confidence": (
            data_confidence
        ),
        "decision_margin": (
            decision_margin
        ),
        "prediction_confidence": (
            prediction_confidence
        ),
        "combined_uncertainty": (
            combined_uncertainty
        ),
    }


# ============================================================
# UNCERTAINTY BANDS AND ABSTENTION
# ============================================================
def assign_uncertainty_band(
    uncertainty: np.ndarray,
    moderate_threshold: float,
    high_threshold: float,
) -> np.ndarray:
    return np.select(
        [
            uncertainty < moderate_threshold,
            uncertainty >= high_threshold,
        ],
        [
            "Low",
            "High",
        ],
        default="Moderate",
    )


def assign_confidence_band(
    confidence: np.ndarray,
) -> np.ndarray:
    return np.select(
        [
            confidence >= 0.80,
            confidence >= 0.60,
            confidence >= 0.40,
        ],
        [
            "Very High",
            "High",
            "Moderate",
        ],
        default="Low",
    )


def apply_abstention_policy(
    original_decisions: np.ndarray,
    confidence: np.ndarray,
    combined_uncertainty: np.ndarray,
    ood_percentile: np.ndarray,
    low_confidence_threshold: float,
    high_uncertainty_threshold: float,
    extreme_ood_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Route unreliable automatic decisions to Insufficient Evidence.

    Engineering Review records remain unchanged because they are
    already assigned for human assessment.
    """
    original_decisions = np.asarray(
        original_decisions,
        dtype=object,
    )

    confidence = np.asarray(
        confidence,
        dtype=float,
    )

    combined_uncertainty = np.asarray(
        combined_uncertainty,
        dtype=float,
    )

    ood_percentile = np.asarray(
        ood_percentile,
        dtype=float,
    )

    expected_length = len(
        original_decisions
    )

    arrays_to_validate = {
        "confidence": confidence,
        "combined_uncertainty": combined_uncertainty,
        "ood_percentile": ood_percentile,
    }

    for array_name, array_values in arrays_to_validate.items():
        if len(array_values) != expected_length:
            raise ValueError(
                f"{array_name} has {len(array_values)} rows, "
                f"but original_decisions has {expected_length} rows."
            )

    low_confidence_mask = (
        confidence
        < low_confidence_threshold
    )

    high_uncertainty_mask = (
        combined_uncertainty
        >= high_uncertainty_threshold
    )

    extreme_ood_mask = (
        ood_percentile
        >= extreme_ood_threshold
    )

    automatic_decision_mask = np.isin(
        original_decisions,
        [
            "Low Risk",
            "High Risk",
        ],
    )

    low_confidence_high_uncertainty_mask = (
        low_confidence_mask
        & high_uncertainty_mask
    )

    abstain_mask = (
        automatic_decision_mask
        & (
            extreme_ood_mask
            | low_confidence_high_uncertainty_mask
        )
    )

    adjusted_decisions = (
        original_decisions
        .astype(object)
        .copy()
    )

    adjusted_decisions[
        abstain_mask
    ] = "Insufficient Evidence"

    abstention_reason = np.full(
        expected_length,
        "No abstention",
        dtype=object,
    )

    extreme_ood_abstention_mask = (
        abstain_mask
        & extreme_ood_mask
    )

    confidence_uncertainty_abstention_mask = (
        abstain_mask
        & ~extreme_ood_mask
        & low_confidence_high_uncertainty_mask
    )

    abstention_reason[
        extreme_ood_abstention_mask
    ] = (
        "Extreme out-of-distribution evidence"
    )

    abstention_reason[
        confidence_uncertainty_abstention_mask
    ] = (
        "Low confidence and high uncertainty"
    )

    return (
        adjusted_decisions,
        abstention_reason,
    )

# ============================================================
# FINAL TEST UNCERTAINTY
# ============================================================
def calculate_final_test_uncertainty(
    final_base_model: Pipeline,
    calibrator: Any,
    test_features: pd.DataFrame,
    test_fold_probability_matrix: np.ndarray,
) -> dict[str, np.ndarray]:
    tree_probabilities = (
        extract_tree_probabilities(
            fitted_pipeline=final_base_model,
            features=test_features,
        )
    )

    tree_results = (
        calculate_tree_uncertainty(
            tree_probabilities=(
                tree_probabilities
            ),
            calibrator=calibrator,
        )
    )

    raw_probability = (
        final_base_model.predict_proba(
            test_features
        )[:, 1]
    )

    calibrated_probability = (
        calibrate_probabilities(
            calibrator=calibrator,
            raw_probabilities=(
                raw_probability
            ),
        )
    )

    fold_probability_mean = (
        np.mean(
            test_fold_probability_matrix,
            axis=1,
        )
    )

    fold_probability_standard_deviation = (
        np.std(
            test_fold_probability_matrix,
            axis=1,
            ddof=1,
        )
    )

    fold_probability_lower = np.quantile(
        test_fold_probability_matrix,
        TREE_INTERVAL_LOWER_QUANTILE,
        axis=1,
    )

    fold_probability_upper = np.quantile(
        test_fold_probability_matrix,
        TREE_INTERVAL_UPPER_QUANTILE,
        axis=1,
    )

    return {
        "raw_probability": (
            raw_probability
        ),
        "calibrated_probability": (
            calibrated_probability
        ),
        "tree_standard_deviation": (
            tree_results[
                "raw_tree_standard_deviation"
            ]
        ),
        "tree_interval_lower": (
            tree_results[
                "calibrated_tree_lower"
            ]
        ),
        "tree_interval_upper": (
            tree_results[
                "calibrated_tree_upper"
            ]
        ),
        "tree_interval_width": (
            tree_results[
                "calibrated_interval_width"
            ]
        ),
        "fold_probability_mean": (
            fold_probability_mean
        ),
        "fold_standard_deviation": (
            fold_probability_standard_deviation
        ),
        "fold_interval_lower": (
            fold_probability_lower
        ),
        "fold_interval_upper": (
            fold_probability_upper
        ),
    }


# ============================================================
# OUTPUT TABLES
# ============================================================
def build_uncertainty_output(
    source_df: pd.DataFrame,
    calibrated_probability: np.ndarray,
    original_decision: np.ndarray,
    adjusted_decision: np.ndarray,
    abstention_reason: np.ndarray,
    tree_standard_deviation: np.ndarray,
    fold_standard_deviation: np.ndarray,
    interval_lower: np.ndarray,
    interval_upper: np.ndarray,
    interval_width: np.ndarray,
    missing_rate: np.ndarray,
    ood_raw: np.ndarray,
    ood_percentile: np.ndarray,
    neighbour_distance: np.ndarray,
    neighbour_percentile: np.ndarray,
    confidence_components: dict[str, np.ndarray],
    uncertainty_band: np.ndarray,
    confidence_band: np.ndarray,
) -> pd.DataFrame:
    output = source_df[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    output[
        "calibrated_failure_probability"
    ] = calibrated_probability

    output[
        "probability_interval_lower"
    ] = interval_lower

    output[
        "probability_interval_upper"
    ] = interval_upper

    output[
        "probability_interval_width"
    ] = interval_width

    output[
        "tree_probability_standard_deviation"
    ] = tree_standard_deviation

    output[
        "fold_probability_standard_deviation"
    ] = fold_standard_deviation

    output[
        "missing_sensor_rate"
    ] = missing_rate

    output[
        "ood_raw_score"
    ] = ood_raw

    output[
        "ood_percentile"
    ] = ood_percentile

    output[
        "neighbour_distance"
    ] = neighbour_distance

    output[
        "neighbour_distance_percentile"
    ] = neighbour_percentile

    for (
        component_name,
        component_values,
    ) in confidence_components.items():
        output[
            component_name
        ] = component_values

    output[
        "uncertainty_band"
    ] = uncertainty_band

    output[
        "confidence_band"
    ] = confidence_band

    output[
        "original_decision"
    ] = original_decision

    output[
        "uncertainty_adjusted_decision"
    ] = adjusted_decision

    output[
        "abstention_reason"
    ] = abstention_reason

    output[
        "requires_human_review"
    ] = np.isin(
        adjusted_decision,
        [
            "Engineering Review",
            "Insufficient Evidence",
        ],
    )

    output[
        "recommended_action"
    ] = np.select(
        [
            adjusted_decision
            == "Low Risk",
            adjusted_decision
            == "High Risk",
            adjusted_decision
            == "Insufficient Evidence",
        ],
        [
            "Continue processing with standard monitoring",
            "Hold and inspect",
            "Repeat measurements and obtain engineering review",
        ],
        default="Engineering assessment required",
    )

    output[
        "routing_outcome"
    ] = np.select(
        [
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output[
                        "uncertainty_adjusted_decision"
                    ]
                    == "Low Risk"
                )
            ),
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output[
                        "uncertainty_adjusted_decision"
                    ]
                    == "High Risk"
                )
            ),
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output[
                        "uncertainty_adjusted_decision"
                    ]
                    == "Engineering Review"
                )
            ),
            (
                (output[TARGET_COLUMN] == 1)
                & (
                    output[
                        "uncertainty_adjusted_decision"
                    ]
                    == "Insufficient Evidence"
                )
            ),
            (
                (output[TARGET_COLUMN] == 0)
                & (
                    output[
                        "uncertainty_adjusted_decision"
                    ]
                    == "High Risk"
                )
            ),
        ],
        [
            "Unsafe Low-Risk Failure",
            "Correct High-Risk Failure",
            "Failure Routed to Review",
            "Failure Captured by Abstention",
            "High-Risk False Alarm",
        ],
        default="Acceptable Routing",
    )

    return output


def build_decision_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    decision_order = [
        "Low Risk",
        "Engineering Review",
        "High Risk",
        "Insufficient Evidence",
    ]

    rows: list[dict[str, Any]] = []

    for decision in decision_order:
        group = dataframe.loc[
            dataframe[
                "uncertainty_adjusted_decision"
            ]
            == decision
        ]

        records = int(
            len(group)
        )

        failures = int(
            group[TARGET_COLUMN].sum()
        )

        passes = int(
            records - failures
        )

        rows.append(
            {
                "decision": decision,
                "records": records,
                "record_rate": float(
                    records
                    / len(dataframe)
                ),
                "actual_failures": failures,
                "actual_passes": passes,
                "observed_failure_rate": (
                    float(
                        failures / records
                    )
                    if records > 0
                    else np.nan
                ),
                "mean_failure_probability": (
                    float(
                        group[
                            "calibrated_failure_probability"
                        ].mean()
                    )
                    if records > 0
                    else np.nan
                ),
                "mean_prediction_confidence": (
                    float(
                        group[
                            "prediction_confidence"
                        ].mean()
                    )
                    if records > 0
                    else np.nan
                ),
                "mean_combined_uncertainty": (
                    float(
                        group[
                            "combined_uncertainty"
                        ].mean()
                    )
                    if records > 0
                    else np.nan
                ),
                "mean_ood_percentile": (
                    float(
                        group[
                            "ood_percentile"
                        ].mean()
                    )
                    if records > 0
                    else np.nan
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_band_performance(
    dataframe: pd.DataFrame,
    band_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for band, group in dataframe.groupby(
        band_column,
        observed=False,
        dropna=False,
    ):
        records = int(
            len(group)
        )

        failures = int(
            group[TARGET_COLUMN].sum()
        )

        unsafe_low_risk = int(
            (
                (
                    group[TARGET_COLUMN]
                    == 1
                )
                & (
                    group[
                        "uncertainty_adjusted_decision"
                    ]
                    == "Low Risk"
                )
            ).sum()
        )

        high_risk_failures = int(
            (
                (
                    group[TARGET_COLUMN]
                    == 1
                )
                & (
                    group[
                        "uncertainty_adjusted_decision"
                    ]
                    == "High Risk"
                )
            ).sum()
        )

        review_or_abstention_failures = int(
            (
                (
                    group[TARGET_COLUMN]
                    == 1
                )
                & (
                    group[
                        "uncertainty_adjusted_decision"
                    ].isin(
                        [
                            "Engineering Review",
                            "Insufficient Evidence",
                        ]
                    )
                )
            ).sum()
        )

        rows.append(
            {
                "band_variable": (
                    band_column
                ),
                "band": str(
                    band
                ),
                "records": records,
                "actual_failures": failures,
                "failure_rate": (
                    float(
                        failures / records
                    )
                    if records > 0
                    else np.nan
                ),
                "unsafe_low_risk_failures": (
                    unsafe_low_risk
                ),
                "high_risk_failures": (
                    high_risk_failures
                ),
                "review_or_abstention_failures": (
                    review_or_abstention_failures
                ),
                "mean_confidence": float(
                    group[
                        "prediction_confidence"
                    ].mean()
                ),
                "mean_uncertainty": float(
                    group[
                        "combined_uncertainty"
                    ].mean()
                ),
                "mean_interval_width": float(
                    group[
                        "probability_interval_width"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PROBABILITY METRICS
# ============================================================
def calculate_probability_metrics(
    target: pd.Series,
    probabilities: np.ndarray,
) -> dict[str, float]:
    clipped = np.clip(
        probabilities,
        1e-8,
        1.0 - 1e-8,
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
                clipped,
                labels=[0, 1],
            )
        ),
    }


# ============================================================
# VISUALISATIONS
# ============================================================
def save_confidence_distribution_plot(
    test_output: pd.DataFrame,
) -> None:
    pass_values = test_output.loc[
        test_output[TARGET_COLUMN] == 0,
        "prediction_confidence",
    ]

    failure_values = test_output.loc[
        test_output[TARGET_COLUMN] == 1,
        "prediction_confidence",
    ]

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.hist(
        pass_values,
        bins=25,
        alpha=0.6,
        label="Pass",
    )

    axis.hist(
        failure_values,
        bins=25,
        alpha=0.6,
        label="Fail",
    )

    axis.set_title(
        "Prediction Confidence by Actual Outcome"
    )

    axis.set_xlabel(
        "Prediction Confidence"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "prediction_confidence_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_uncertainty_probability_plot(
    test_output: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    scatter = axis.scatter(
        test_output[
            "calibrated_failure_probability"
        ],
        test_output[
            "combined_uncertainty"
        ],
        c=test_output[
            TARGET_COLUMN
        ],
        alpha=0.7,
    )

    axis.set_title(
        "Failure Probability versus Combined Uncertainty"
    )

    axis.set_xlabel(
        "Calibrated Failure Probability"
    )

    axis.set_ylabel(
        "Combined Uncertainty"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="Actual Outcome",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "probability_vs_uncertainty.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_ood_confidence_plot(
    test_output: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    scatter = axis.scatter(
        test_output[
            "ood_percentile"
        ],
        test_output[
            "prediction_confidence"
        ],
        c=test_output[
            "calibrated_failure_probability"
        ],
        alpha=0.7,
    )

    axis.set_title(
        "Out-of-Distribution Evidence versus Confidence"
    )

    axis.set_xlabel(
        "OOD Percentile"
    )

    axis.set_ylabel(
        "Prediction Confidence"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="Calibrated Failure Probability",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "ood_vs_prediction_confidence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


def save_adjusted_decision_plot(
    decision_summary: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(10, 6)
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
        ],
        rotation=10,
    )

    axis.set_title(
        "Uncertainty-Adjusted Test Decisions"
    )

    axis.set_xlabel(
        "Decision"
    )

    axis.set_ylabel(
        "Number of Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "uncertainty_adjusted_decisions.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(figure)


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
    test_metrics: dict[str, float],
    test_output: pd.DataFrame,
    decision_summary: pd.DataFrame,
    thresholds: dict[str, float],
) -> None:
    print("\n" + "=" * 120)
    print(
        "HEVEMIND UNCERTAINTY AND CONFIDENCE ENGINE"
    )
    print("=" * 120)

    print("\nHeld-out test probability performance:")

    print(
        f"ROC-AUC:                         "
        f"{test_metrics['roc_auc']:.4f}"
    )

    print(
        f"PR-AUC:                          "
        f"{test_metrics['pr_auc']:.4f}"
    )

    print(
        f"Brier score:                     "
        f"{test_metrics['brier_score']:.4f}"
    )

    print(
        f"Log loss:                        "
        f"{test_metrics['log_loss']:.4f}"
    )

    print("\nUncertainty reference thresholds:")

    print(
        f"Low-confidence threshold:        "
        f"{thresholds['low_confidence_threshold']:.4f}"
    )

    print(
        f"High-uncertainty threshold:      "
        f"{thresholds['high_uncertainty_threshold']:.4f}"
    )

    print(
        f"Extreme-OOD threshold:           "
        f"{thresholds['extreme_ood_threshold']:.4f}"
    )

    print("\nHeld-out test uncertainty:")

    print(
        f"Mean prediction confidence:      "
        f"{test_output['prediction_confidence'].mean():.4f}"
    )

    print(
        f"Mean combined uncertainty:       "
        f"{test_output['combined_uncertainty'].mean():.4f}"
    )

    print(
        f"Mean data confidence:            "
        f"{test_output['data_confidence'].mean():.4f}"
    )

    print(
        f"Mean probability interval width: "
        f"{test_output['probability_interval_width'].mean():.4f}"
    )

    print(
        f"Insufficient-evidence records:   "
        f"{(test_output['uncertainty_adjusted_decision'] == 'Insufficient Evidence').sum()}"
    )

    unsafe_low_risk = int(
        (
            (
                test_output[TARGET_COLUMN]
                == 1
            )
            & (
                test_output[
                    "uncertainty_adjusted_decision"
                ]
                == "Low Risk"
            )
        ).sum()
    )

    captured_failures = int(
        (
            (
                test_output[TARGET_COLUMN]
                == 1
            )
            & (
                test_output[
                    "uncertainty_adjusted_decision"
                ]
                != "Low Risk"
            )
        ).sum()
    )

    total_failures = int(
        test_output[
            TARGET_COLUMN
        ].sum()
    )

    print("\nSafety routing:")

    print(
        f"Actual failures:                 "
        f"{total_failures}"
    )

    print(
        f"Failures safely flagged:         "
        f"{captured_failures}"
    )

    print(
        f"Unsafe Low-Risk failures:        "
        f"{unsafe_low_risk}"
    )

    print(
        f"Failure safety rate:             "
        f"{captured_failures / max(total_failures, 1):.4f}"
    )

    print("\nDecision summary:")
    print(
        decision_summary[
            [
                "decision",
                "records",
                "actual_failures",
                "actual_passes",
                "mean_failure_probability",
                "mean_prediction_confidence",
                "mean_combined_uncertainty",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")

    print(
        f"Development uncertainty table:   "
        f"{DEVELOPMENT_UNCERTAINTY_PATH}"
    )

    print(
        f"Test uncertainty table:          "
        f"{TEST_UNCERTAINTY_PATH}"
    )

    print(
        f"Decision summary:                "
        f"{TEST_DECISION_SUMMARY_PATH}"
    )

    print(
        f"Uncertainty bands:               "
        f"{UNCERTAINTY_BAND_SUMMARY_PATH}"
    )

    print(
        f"Confidence bands:                "
        f"{CONFIDENCE_BAND_SUMMARY_PATH}"
    )

    print(
        f"Abstention summary:              "
        f"{ABSTENTION_SUMMARY_PATH}"
    )

    print(
        f"Report directory:                "
        f"{UNCERTAINTY_REPORT_DIR}"
    )

# ============================================================
# BETA CALIBRATOR
# ============================================================
class BetaCalibrator:
    """
    Beta calibration using logistic regression on:

        log(p)
        log(1 - p)

    This class is included here so that the calibrator artifact saved
    by script 11 can be loaded correctly.
    """

    PROBABILITY_EPSILON = 1e-6

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression

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
            BetaCalibrator.PROBABILITY_EPSILON,
            1.0 - BetaCalibrator.PROBABILITY_EPSILON,
        )

        return np.column_stack(
            [
                np.log(
                    probabilities
                ),
                np.log1p(
                    -probabilities
                ),
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

    (
        base_model,
        calibrator,
        calibration_summary,
    ) = load_calibration_artifacts()

    sensor_columns = get_sensor_columns(
        development_df
    )

    X_development = development_df[
        sensor_columns
    ]

    y_development = development_df[
        TARGET_COLUMN
    ]

    X_test = test_df[
        sensor_columns
    ]

    y_test = test_df[
        TARGET_COLUMN
    ]

    selected_policy = calibration_summary[
        "selected_policy"
    ]

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

    LOGGER.info(
        "Generating cross-fitted uncertainty estimates"
    )

    (
        development_uncertainty,
        test_fold_probability_matrix,
    ) = generate_cross_fitted_uncertainty(
        features=X_development,
        target=y_development,
        test_features=X_test,
        calibrator=calibrator,
    )

    LOGGER.info(
        "Training final deployment model"
    )

    final_model = build_base_pipeline()

    final_model.fit(
        X_development,
        y_development,
    )

    LOGGER.info(
        "Building distribution reference models"
    )

    (
        preprocessor,
        isolation_forest,
        nearest_neighbours,
        development_ood_raw,
        development_neighbour_distance,
    ) = fit_distribution_models(
        X_development
    )

    development_ood_percentile = (
        percentile_normalize(
            development_ood_raw,
            development_ood_raw,
        )
    )

    development_neighbour_percentile = (
        percentile_normalize(
            development_neighbour_distance,
            development_neighbour_distance,
        )
    )

    (
        test_ood_raw,
        test_neighbour_distance,
    ) = score_distribution_shift(
        X_test,
        preprocessor,
        isolation_forest,
        nearest_neighbours,
    )

    test_ood_percentile = (
        percentile_normalize(
            test_ood_raw,
            development_ood_raw,
        )
    )

    test_neighbour_percentile = (
        percentile_normalize(
            test_neighbour_distance,
            development_neighbour_distance,
        )
    )

    LOGGER.info(
        "Calculating uncertainty"
    )

    test_uncertainty = (
        calculate_final_test_uncertainty(
            final_model,
            calibrator,
            X_test,
            test_fold_probability_matrix,
        )
    )

    fold_std_reference = np.std(
        test_fold_probability_matrix,
        axis=1,
    )

    confidence = calculate_confidence_components(
        calibrated_probabilities=test_uncertainty[
            "calibrated_probability"
        ],
        tree_standard_deviation=test_uncertainty[
            "tree_standard_deviation"
        ],
        fold_standard_deviation=test_uncertainty[
            "fold_standard_deviation"
        ],
        missing_rate=X_test.isna().mean(axis=1).values,
        ood_percentile=test_ood_percentile,
        neighbour_percentile=test_neighbour_percentile,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        tree_reference=development_uncertainty[
            "tree_standard_deviation"
        ],
        fold_reference=fold_std_reference,
    )

    development_missing_rate = (
    X_development
    .isna()
    .mean(axis=1)
    .to_numpy(dtype=float)
)

    development_fold_standard_deviation = (
    development_uncertainty[
        "tree_standard_deviation"
    ].copy()
)

    development_confidence = (
        calculate_confidence_components(
            calibrated_probabilities=(
                development_uncertainty[
                    "calibrated_probability"
                ]
            ),
            tree_standard_deviation=(
                development_uncertainty[
                    "tree_standard_deviation"
                ]
            ),
            fold_standard_deviation=(
                development_fold_standard_deviation
            ),
            missing_rate=(
                development_missing_rate
            ),
            ood_percentile=(
                development_ood_percentile
            ),
            neighbour_percentile=(
                development_neighbour_percentile
            ),
            lower_threshold=(
                lower_threshold
            ),
            upper_threshold=(
                upper_threshold
            ),
            tree_reference=(
                development_uncertainty[
                    "tree_standard_deviation"
                ]
            ),
            fold_reference=(
                development_fold_standard_deviation
            ),
        )
    )


    low_confidence_threshold = float(
    np.quantile(
        development_confidence[
            "prediction_confidence"
        ],
        LOW_CONFIDENCE_QUANTILE,
    )
)

    high_uncertainty_threshold = float(
        np.quantile(
        development_confidence[
            "combined_uncertainty"
        ],
        VERY_HIGH_UNCERTAINTY_QUANTILE,
    )
)

    extreme_ood_threshold = float(
        np.quantile(
            development_ood_percentile,
            EXTREME_OOD_QUANTILE,
        )
    )

    LOGGER.info(
        "Building development uncertainty output"
    )

    development_original_decisions = (
        assign_original_decision(
            calibrated_probabilities=(
                development_uncertainty[
                    "calibrated_probability"
                ]
            ),
            lower_threshold=lower_threshold,
            upper_threshold=upper_threshold,
        )
    )

    (
        development_adjusted_decisions,
        development_abstention_reasons,
    ) = apply_abstention_policy(
        original_decisions=(
            development_original_decisions
        ),
        confidence=(
            development_confidence[
                "prediction_confidence"
            ]
        ),
        combined_uncertainty=(
            development_confidence[
                "combined_uncertainty"
            ]
        ),
        ood_percentile=(
            development_ood_percentile
        ),
        low_confidence_threshold=(
            low_confidence_threshold
        ),
        high_uncertainty_threshold=(
            high_uncertainty_threshold
        ),
        extreme_ood_threshold=(
            extreme_ood_threshold
        ),
    )

    development_uncertainty_band = (
        assign_uncertainty_band(
            uncertainty=(
                development_confidence[
                    "combined_uncertainty"
                ]
            ),
            moderate_threshold=float(
                np.quantile(
                    development_confidence[
                        "combined_uncertainty"
                    ],
                    0.50,
                )
            ),
            high_threshold=(
                high_uncertainty_threshold
            ),
        )
    )

    development_confidence_band = (
        assign_confidence_band(
            development_confidence[
                "prediction_confidence"
            ]
        )
    )

    development_output = (
        build_uncertainty_output(
            source_df=development_df,
            calibrated_probability=(
                development_uncertainty[
                    "calibrated_probability"
                ]
            ),
            original_decision=(
                development_original_decisions
            ),
            adjusted_decision=(
                development_adjusted_decisions
            ),
            abstention_reason=(
                development_abstention_reasons
            ),
            tree_standard_deviation=(
                development_uncertainty[
                    "tree_standard_deviation"
                ]
            ),
            fold_standard_deviation=(
                development_fold_standard_deviation
            ),
            interval_lower=(
                development_uncertainty[
                    "interval_lower"
                ]
            ),
            interval_upper=(
                development_uncertainty[
                    "interval_upper"
                ]
            ),
            interval_width=(
                development_uncertainty[
                    "interval_width"
                ]
            ),
            missing_rate=(
                development_missing_rate
            ),
            ood_raw=(
                development_ood_raw
            ),
            ood_percentile=(
                development_ood_percentile
            ),
            neighbour_distance=(
                development_neighbour_distance
            ),
            neighbour_percentile=(
                development_neighbour_percentile
            ),
            confidence_components=(
                development_confidence
            ),
            uncertainty_band=(
                development_uncertainty_band
            ),
            confidence_band=(
                development_confidence_band
            ),
        )
    )

    development_output.to_csv(
        DEVELOPMENT_UNCERTAINTY_PATH,
        index=False,
    )

    LOGGER.info(
        "Saved development uncertainty scores: %s",
        DEVELOPMENT_UNCERTAINTY_PATH,
    )

    original_decisions = assign_original_decision(
        test_uncertainty[
            "calibrated_probability"
        ],
        lower_threshold,
        upper_threshold,
    )

    (
        adjusted_decisions,
        abstention_reason,
    ) = apply_abstention_policy(
        original_decisions,
        confidence["prediction_confidence"],
        confidence["combined_uncertainty"],
        test_ood_percentile,
        low_confidence_threshold,
        high_uncertainty_threshold,
        extreme_ood_threshold,
    )

    uncertainty_band = assign_uncertainty_band(
        confidence["combined_uncertainty"],
        np.quantile(
            confidence["combined_uncertainty"],
            0.33,
        ),
        np.quantile(
            confidence["combined_uncertainty"],
            0.66,
        ),
    )

    confidence_band = assign_confidence_band(
        confidence["prediction_confidence"]
    )

    test_output = build_uncertainty_output(
        source_df=test_df,
        calibrated_probability=test_uncertainty[
            "calibrated_probability"
        ],
        original_decision=original_decisions,
        adjusted_decision=adjusted_decisions,
        abstention_reason=abstention_reason,
        tree_standard_deviation=test_uncertainty[
            "tree_standard_deviation"
        ],
        fold_standard_deviation=test_uncertainty[
            "fold_standard_deviation"
        ],
        interval_lower=test_uncertainty[
            "fold_interval_lower"
        ],
        interval_upper=test_uncertainty[
            "fold_interval_upper"
        ],
        interval_width=(
            test_uncertainty[
                "fold_interval_upper"
        ]
        - test_uncertainty[
            "fold_interval_lower"
        ]
    ),
        missing_rate=X_test.isna().mean(axis=1).values,
        ood_raw=test_ood_raw,
        ood_percentile=test_ood_percentile,
        neighbour_distance=test_neighbour_distance,
        neighbour_percentile=test_neighbour_percentile,
        confidence_components=confidence,
        uncertainty_band=uncertainty_band,
        confidence_band=confidence_band,
    )

    decision_summary = build_decision_summary(
        test_output
    )

    uncertainty_band_summary = (
        build_band_performance(
            test_output,
            "uncertainty_band",
        )
    )

    confidence_band_summary = (
        build_band_performance(
            test_output,
            "confidence_band",
        )
    )

    metrics = calculate_probability_metrics(
        y_test,
        test_uncertainty[
            "calibrated_probability"
        ],
    )

    test_output.to_csv(
        TEST_UNCERTAINTY_PATH,
        index=False,
    )

    decision_summary.to_csv(
        TEST_DECISION_SUMMARY_PATH,
        index=False,
    )

    uncertainty_band_summary.to_csv(
        UNCERTAINTY_BAND_SUMMARY_PATH,
        index=False,
    )

    confidence_band_summary.to_csv(
        CONFIDENCE_BAND_SUMMARY_PATH,
        index=False,
    )

    save_confidence_distribution_plot(
        test_output
    )

    save_uncertainty_probability_plot(
        test_output
    )

    save_ood_confidence_plot(
        test_output
    )

    save_adjusted_decision_plot(
        decision_summary
    )

    joblib.dump(
        preprocessor,
        UNCERTAINTY_PREPROCESSOR_PATH,
    )

    joblib.dump(
        isolation_forest,
        ISOLATION_FOREST_PATH,
    )

    joblib.dump(
        nearest_neighbours,
        NEAREST_NEIGHBOUR_PATH,
    )

    save_json(
        REFERENCE_METADATA_PATH,
        {
            "low_confidence_threshold": float(
                low_confidence_threshold
            ),
            "high_uncertainty_threshold": float(
                high_uncertainty_threshold
            ),
            "extreme_ood_threshold": float(
                extreme_ood_threshold
            ),
        },
    )

    save_json(
        SUMMARY_PATH,
        {
            "metrics": metrics,
            "decision_summary": decision_summary.to_dict(
                orient="records"
            ),
        },
    )

    print_console_summary(
        metrics,
        test_output,
        decision_summary,
        {
            "low_confidence_threshold": low_confidence_threshold,
            "high_uncertainty_threshold": high_uncertainty_threshold,
            "extreme_ood_threshold": extreme_ood_threshold,
        },
    )


if __name__ == "__main__":
    main()