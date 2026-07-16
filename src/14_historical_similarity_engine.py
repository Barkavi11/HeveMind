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

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
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

REPORTS_DIR = ROOT_DIR / "reports"

UNCERTAINTY_RESULTS_PATH = (
    REPORTS_DIR
    / "uncertainty_engine"
    / "tables"
    / "test_uncertainty_scores.csv"
)

EXPLAINABILITY_SUMMARY_PATH = (
    REPORTS_DIR
    / "explainability_engine"
    / "tables"
    / "wafer_explanation_summary.csv"
)

GLOBAL_SHAP_PATH = (
    REPORTS_DIR
    / "explainability_engine"
    / "tables"
    / "global_shap_importance.csv"
)

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

SIMILARITY_ARTIFACT_DIR = (
    ARTIFACTS_DIR
    / "historical_similarity_engine"
)

SIMILARITY_REPORT_DIR = (
    REPORTS_DIR
    / "historical_similarity_engine"
)

TABLES_DIR = SIMILARITY_REPORT_DIR / "tables"
FIGURES_DIR = SIMILARITY_REPORT_DIR / "figures"
LOCAL_FIGURES_DIR = FIGURES_DIR / "wafer_similarity"

PREPROCESSOR_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "similarity_preprocessor.joblib"
)

PCA_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "similarity_pca.joblib"
)

NEIGHBOUR_MODEL_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "historical_neighbour_model.joblib"
)

REFERENCE_EMBEDDINGS_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "development_reference_embeddings.npy"
)

REFERENCE_METADATA_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "historical_reference_metadata.csv"
)

FEATURE_METADATA_PATH = (
    SIMILARITY_ARTIFACT_DIR
    / "similarity_feature_metadata.json"
)

NEIGHBOUR_RESULTS_PATH = (
    TABLES_DIR
    / "historical_neighbour_results.csv"
)

WAFER_SUMMARY_PATH = (
    TABLES_DIR
    / "historical_similarity_summary.csv"
)

DECISION_GROUP_SUMMARY_PATH = (
    TABLES_DIR
    / "similarity_by_decision_group.csv"
)

ACTUAL_OUTCOME_SUMMARY_PATH = (
    TABLES_DIR
    / "similarity_by_actual_outcome.csv"
)

FAILURE_RETRIEVAL_SUMMARY_PATH = (
    TABLES_DIR
    / "failure_retrieval_performance.csv"
)

SUMMARY_PATH = (
    SIMILARITY_REPORT_DIR
    / "historical_similarity_engine_summary.json"
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

TOP_K_NEIGHBOURS = 10
MINIMUM_NEIGHBOURS = 3

MAX_PCA_COMPONENTS = 100
PCA_VARIANCE_TARGET = 0.95

TOP_SHAP_FEATURES_FOR_CONTEXT = 30

MAX_LOCAL_PLOTS = 25

SIMILARITY_EPSILON = 1e-12


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        SIMILARITY_ARTIFACT_DIR,
        SIMILARITY_REPORT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        LOCAL_FIGURES_DIR,
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

    test_df = test_df.reset_index(drop=True)

    return development_df, test_df


def load_uncertainty_results() -> pd.DataFrame:
    if not UNCERTAINTY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Uncertainty results were not found. "
            "Run src/12_uncertainty_engine.py first."
        )

    dataframe = pd.read_csv(
        UNCERTAINTY_RESULTS_PATH
    )

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        "calibrated_failure_probability",
        "prediction_confidence",
        "combined_uncertainty",
        "ood_percentile",
        "uncertainty_adjusted_decision",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Uncertainty-results file is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_explainability_summary() -> pd.DataFrame:
    if not EXPLAINABILITY_SUMMARY_PATH.exists():
        LOGGER.warning(
            "Wafer explanation summary was not found. "
            "Similarity analysis will continue without SHAP text."
        )

        return pd.DataFrame()

    return pd.read_csv(
        EXPLAINABILITY_SUMMARY_PATH
    )


def load_global_shap_features() -> list[str]:
    if not GLOBAL_SHAP_PATH.exists():
        LOGGER.warning(
            "Global SHAP table was not found. "
            "All available sensors will be used."
        )

        return []

    shap_df = pd.read_csv(
        GLOBAL_SHAP_PATH
    )

    if "feature" not in shap_df.columns:
        return []

    selected_features = (
        shap_df["feature"]
        .astype(str)
        .loc[
            lambda values: values.str.startswith(
                SENSOR_PREFIX
            )
        ]
        .head(
            TOP_SHAP_FEATURES_FOR_CONTEXT
        )
        .tolist()
    )

    return selected_features


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
# VALIDATION
# ============================================================
def validate_unique_ids(
    dataframe: pd.DataFrame,
    dataframe_name: str,
) -> None:
    if ID_COLUMN not in dataframe.columns:
        raise ValueError(
            f"{dataframe_name} does not contain {ID_COLUMN}."
        )

    duplicate_count = int(
        dataframe[ID_COLUMN].duplicated().sum()
    )

    if duplicate_count > 0:
        raise ValueError(
            f"{dataframe_name} contains {duplicate_count} "
            f"duplicate wafer IDs."
        )


# ============================================================
# SIMILARITY REPRESENTATION
# ============================================================
def build_similarity_preprocessor() -> Pipeline:
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
                "scaler",
                RobustScaler(),
            ),
        ]
    )


def determine_pca_components(
    transformed_development: np.ndarray,
) -> int:
    maximum_allowed = min(
        MAX_PCA_COMPONENTS,
        transformed_development.shape[0] - 1,
        transformed_development.shape[1],
    )

    if maximum_allowed < 2:
        raise ValueError(
            "Insufficient data dimensions for PCA."
        )

    exploratory_pca = PCA(
        n_components=maximum_allowed,
        random_state=RANDOM_STATE,
    )

    exploratory_pca.fit(
        transformed_development
    )

    cumulative_variance = np.cumsum(
        exploratory_pca.explained_variance_ratio_
    )

    required_components = int(
        np.searchsorted(
            cumulative_variance,
            PCA_VARIANCE_TARGET,
        )
        + 1
    )

    return max(
        2,
        min(
            required_components,
            maximum_allowed,
        ),
    )


def fit_similarity_representation(
    development_features: pd.DataFrame,
    test_features: pd.DataFrame,
) -> tuple[
    Pipeline,
    PCA,
    np.ndarray,
    np.ndarray,
]:
    preprocessor = build_similarity_preprocessor()

    transformed_development = (
        preprocessor.fit_transform(
            development_features
        )
    )

    transformed_test = preprocessor.transform(
        test_features
    )

    number_of_components = determine_pca_components(
        transformed_development
    )

    LOGGER.info(
        "Fitting PCA representation with %s components",
        number_of_components,
    )

    pca = PCA(
        n_components=number_of_components,
        random_state=RANDOM_STATE,
    )

    development_embeddings = pca.fit_transform(
        transformed_development
    )

    test_embeddings = pca.transform(
        transformed_test
    )

    return (
        preprocessor,
        pca,
        development_embeddings,
        test_embeddings,
    )


# ============================================================
# DISTANCE CALIBRATION
# ============================================================
def estimate_reference_distance_scale(
    development_embeddings: np.ndarray,
) -> float:
    neighbour_count = min(
        TOP_K_NEIGHBOURS + 1,
        len(development_embeddings),
    )

    reference_model = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="euclidean",
        n_jobs=-1,
    )

    reference_model.fit(
        development_embeddings
    )

    distances, _ = reference_model.kneighbors(
        development_embeddings
    )

    if distances.shape[1] > 1:
        non_self_distances = distances[:, 1:]
    else:
        non_self_distances = distances

    positive_distances = non_self_distances[
        non_self_distances > 0
    ]

    if positive_distances.size == 0:
        return 1.0

    distance_scale = float(
        np.median(
            positive_distances
        )
    )

    if (
        not np.isfinite(distance_scale)
        or distance_scale <= SIMILARITY_EPSILON
    ):
        return 1.0

    return distance_scale


def distance_to_similarity(
    distances: np.ndarray,
    distance_scale: float,
) -> np.ndarray:
    """
    Converts Euclidean distance into a bounded similarity score.

    This is an analytical similarity transformation, not a probability.
    """
    safe_scale = max(
        distance_scale,
        SIMILARITY_EPSILON,
    )

    similarity = np.exp(
        -distances / safe_scale
    )

    return np.clip(
        similarity,
        0.0,
        1.0,
    )


# ============================================================
# NEIGHBOUR RETRIEVAL
# ============================================================
def fit_neighbour_model(
    development_embeddings: np.ndarray,
) -> NearestNeighbors:
    neighbour_count = min(
        TOP_K_NEIGHBOURS,
        len(development_embeddings),
    )

    model = NearestNeighbors(
        n_neighbors=neighbour_count,
        metric="euclidean",
        n_jobs=-1,
    )

    model.fit(
        development_embeddings
    )

    return model


def retrieve_historical_neighbours(
    neighbour_model: NearestNeighbors,
    test_embeddings: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    distances, indices = neighbour_model.kneighbors(
        test_embeddings,
        n_neighbors=min(
            TOP_K_NEIGHBOURS,
            neighbour_model.n_samples_fit_,
        ),
    )

    return distances, indices


# ============================================================
# STATISTICAL UTILITIES
# ============================================================
def wilson_interval(
    successes: int,
    total: int,
    confidence_z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan

    proportion = successes / total

    denominator = (
        1.0
        + confidence_z**2 / total
    )

    centre = (
        proportion
        + confidence_z**2 / (2.0 * total)
    ) / denominator

    margin = (
        confidence_z
        * np.sqrt(
            (
                proportion
                * (1.0 - proportion)
                / total
            )
            + (
                confidence_z**2
                / (4.0 * total**2)
            )
        )
        / denominator
    )

    return (
        max(0.0, centre - margin),
        min(1.0, centre + margin),
    )


def calculate_weighted_failure_rate(
    neighbour_targets: np.ndarray,
    similarities: np.ndarray,
) -> float:
    weight_sum = float(
        np.sum(similarities)
    )

    if weight_sum <= SIMILARITY_EPSILON:
        return float(
            np.mean(neighbour_targets)
        )

    return float(
        np.sum(
            neighbour_targets
            * similarities
        )
        / weight_sum
    )


def assign_similarity_strength(
    mean_similarity: float,
) -> str:
    if mean_similarity >= 0.75:
        return "Very Strong"

    if mean_similarity >= 0.55:
        return "Strong"

    if mean_similarity >= 0.35:
        return "Moderate"

    if mean_similarity >= 0.20:
        return "Weak"

    return "Very Weak"


def assign_historical_evidence_level(
    weighted_failure_rate: float,
    failed_neighbour_count: int,
    similarity_strength: str,
) -> str:
    if (
        failed_neighbour_count >= 5
        and weighted_failure_rate >= 0.50
        and similarity_strength in {
            "Very Strong",
            "Strong",
        }
    ):
        return "Strong Failure Evidence"

    if (
        failed_neighbour_count >= 3
        and weighted_failure_rate >= 0.25
    ):
        return "Moderate Failure Evidence"

    if failed_neighbour_count >= 1:
        return "Limited Failure Evidence"

    return "No Retrieved Failure Evidence"


# ============================================================
# RESULT CONSTRUCTION
# ============================================================
def build_similarity_results(
    development_df: pd.DataFrame,
    test_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    explainability_df: pd.DataFrame,
    neighbour_distances: np.ndarray,
    neighbour_indices: np.ndarray,
    distance_scale: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neighbour_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    uncertainty_lookup = uncertainty_df.set_index(
        ID_COLUMN
    )

    explanation_lookup = (
        explainability_df.set_index(
            ID_COLUMN
        )
        if not explainability_df.empty
        and ID_COLUMN in explainability_df.columns
        else None
    )

    for test_position in range(
        len(test_df)
    ):
        query_record = test_df.iloc[
            test_position
        ]

        query_id = query_record[
            ID_COLUMN
        ]

        if query_id not in uncertainty_lookup.index:
            raise KeyError(
                f"Wafer {query_id} is missing from "
                "the uncertainty-results table."
            )

        uncertainty_record = uncertainty_lookup.loc[
            query_id
        ]

        distances = neighbour_distances[
            test_position
        ]

        indices = neighbour_indices[
            test_position
        ]

        similarities = distance_to_similarity(
            distances,
            distance_scale,
        )

        neighbour_records = development_df.iloc[
            indices
        ].copy()

        neighbour_targets = (
            neighbour_records[
                TARGET_COLUMN
            ]
            .astype(int)
            .to_numpy()
        )

        failed_neighbour_count = int(
            neighbour_targets.sum()
        )

        passed_neighbour_count = int(
            len(neighbour_targets)
            - failed_neighbour_count
        )

        raw_failure_rate = float(
            neighbour_targets.mean()
        )

        weighted_failure_rate = (
            calculate_weighted_failure_rate(
                neighbour_targets=neighbour_targets,
                similarities=similarities,
            )
        )

        lower_interval, upper_interval = (
            wilson_interval(
                successes=failed_neighbour_count,
                total=len(neighbour_targets),
            )
        )

        mean_similarity = float(
            similarities.mean()
        )

        maximum_similarity = float(
            similarities.max()
        )

        minimum_similarity = float(
            similarities.min()
        )

        similarity_strength = (
            assign_similarity_strength(
                mean_similarity
            )
        )

        evidence_level = (
            assign_historical_evidence_level(
                weighted_failure_rate=(
                    weighted_failure_rate
                ),
                failed_neighbour_count=(
                    failed_neighbour_count
                ),
                similarity_strength=(
                    similarity_strength
                ),
            )
        )

        failed_similarity_values = similarities[
            neighbour_targets == 1
        ]

        passed_similarity_values = similarities[
            neighbour_targets == 0
        ]

        mean_failed_similarity = (
            float(
                failed_similarity_values.mean()
            )
            if failed_similarity_values.size > 0
            else np.nan
        )

        mean_passed_similarity = (
            float(
                passed_similarity_values.mean()
            )
            if passed_similarity_values.size > 0
            else np.nan
        )

        for rank, (
            historical_index,
            distance,
            similarity,
        ) in enumerate(
            zip(
                indices,
                distances,
                similarities,
                strict=True,
            ),
            start=1,
        ):
            historical_record = development_df.iloc[
                historical_index
            ]

            neighbour_rows.append(
                {
                    "query_wafer_id": query_id,
                    "query_actual_target": int(
                        query_record[
                            TARGET_COLUMN
                        ]
                    ),
                    "query_actual_status": (
                        query_record[
                            STATUS_COLUMN
                        ]
                    ),
                    "query_decision": (
                        uncertainty_record[
                            "uncertainty_adjusted_decision"
                        ]
                    ),
                    "query_failure_probability": float(
                        uncertainty_record[
                            "calibrated_failure_probability"
                        ]
                    ),
                    "query_prediction_confidence": float(
                        uncertainty_record[
                            "prediction_confidence"
                        ]
                    ),
                    "neighbour_rank": rank,
                    "historical_wafer_id": (
                        historical_record[
                            ID_COLUMN
                        ]
                    ),
                    "historical_target": int(
                        historical_record[
                            TARGET_COLUMN
                        ]
                    ),
                    "historical_status": (
                        historical_record[
                            STATUS_COLUMN
                        ]
                    ),
                    "distance": float(
                        distance
                    ),
                    "similarity_score": float(
                        similarity
                    ),
                }
            )

        top_risk_features = ""
        top_protective_features = ""

        if (
            explanation_lookup is not None
            and query_id in explanation_lookup.index
        ):
            explanation_record = (
                explanation_lookup.loc[
                    query_id
                ]
            )

            top_risk_features = str(
                explanation_record.get(
                    "top_risk_features",
                    "",
                )
            )

            top_protective_features = str(
                explanation_record.get(
                    "top_protective_features",
                    "",
                )
            )

        summary_rows.append(
            {
                ID_COLUMN: query_id,
                TARGET_COLUMN: int(
                    query_record[
                        TARGET_COLUMN
                    ]
                ),
                STATUS_COLUMN: (
                    query_record[
                        STATUS_COLUMN
                    ]
                ),
                "decision": (
                    uncertainty_record[
                        "uncertainty_adjusted_decision"
                    ]
                ),
                "calibrated_failure_probability": float(
                    uncertainty_record[
                        "calibrated_failure_probability"
                    ]
                ),
                "prediction_confidence": float(
                    uncertainty_record[
                        "prediction_confidence"
                    ]
                ),
                "combined_uncertainty": float(
                    uncertainty_record[
                        "combined_uncertainty"
                    ]
                ),
                "data_familiarity_percentile": float(
                    1.0
                    - uncertainty_record[
                        "ood_percentile"
                    ]
                ),
                "neighbour_count": int(
                    len(neighbour_targets)
                ),
                "failed_neighbour_count": (
                    failed_neighbour_count
                ),
                "passed_neighbour_count": (
                    passed_neighbour_count
                ),
                "historical_raw_failure_rate": (
                    raw_failure_rate
                ),
                "historical_weighted_failure_rate": (
                    weighted_failure_rate
                ),
                "historical_failure_rate_lower_95": (
                    lower_interval
                ),
                "historical_failure_rate_upper_95": (
                    upper_interval
                ),
                "mean_similarity": mean_similarity,
                "maximum_similarity": (
                    maximum_similarity
                ),
                "minimum_similarity": (
                    minimum_similarity
                ),
                "mean_failed_neighbour_similarity": (
                    mean_failed_similarity
                ),
                "mean_passed_neighbour_similarity": (
                    mean_passed_similarity
                ),
                "similarity_strength": (
                    similarity_strength
                ),
                "historical_evidence_level": (
                    evidence_level
                ),
                "top_risk_features": (
                    top_risk_features
                ),
                "top_protective_features": (
                    top_protective_features
                ),
                "historical_evidence_text": (
                    create_historical_evidence_text(
                        neighbour_count=len(
                            neighbour_targets
                        ),
                        failed_neighbour_count=(
                            failed_neighbour_count
                        ),
                        weighted_failure_rate=(
                            weighted_failure_rate
                        ),
                        mean_similarity=(
                            mean_similarity
                        ),
                        evidence_level=(
                            evidence_level
                        ),
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            neighbour_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


def create_historical_evidence_text(
    neighbour_count: int,
    failed_neighbour_count: int,
    weighted_failure_rate: float,
    mean_similarity: float,
    evidence_level: str,
) -> str:
    return (
        f"The {neighbour_count} nearest development records contained "
        f"{failed_neighbour_count} observed failures. "
        f"The similarity-weighted historical failure rate was "
        f"{weighted_failure_rate:.2%}, and the mean analytical "
        f"similarity score was {mean_similarity:.2%}. "
        f"The resulting evidence category is {evidence_level}. "
        f"This is historical statistical evidence, not proof of a "
        f"physical failure mechanism."
    )


# ============================================================
# GROUP SUMMARIES
# ============================================================
def build_group_summary(
    similarity_df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for group_value, group in similarity_df.groupby(
        group_column,
        observed=False,
        dropna=False,
    ):
        rows.append(
            {
                "group_variable": group_column,
                "group_value": str(
                    group_value
                ),
                "records": int(
                    len(group)
                ),
                "actual_failures": int(
                    group[
                        TARGET_COLUMN
                    ].sum()
                ),
                "mean_model_probability": float(
                    group[
                        "calibrated_failure_probability"
                    ].mean()
                ),
                "mean_historical_failure_rate": float(
                    group[
                        "historical_weighted_failure_rate"
                    ].mean()
                ),
                "mean_similarity": float(
                    group[
                        "mean_similarity"
                    ].mean()
                ),
                "mean_failed_neighbour_count": float(
                    group[
                        "failed_neighbour_count"
                    ].mean()
                ),
                "strong_or_moderate_evidence_rate": float(
                    group[
                        "historical_evidence_level"
                    ]
                    .isin(
                        [
                            "Strong Failure Evidence",
                            "Moderate Failure Evidence",
                        ]
                    )
                    .mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_failure_retrieval_summary(
    similarity_df: pd.DataFrame,
) -> pd.DataFrame:
    thresholds = [
        1,
        2,
        3,
        5,
    ]

    rows: list[dict[str, Any]] = []

    actual_failures = similarity_df.loc[
        similarity_df[
            TARGET_COLUMN
        ]
        == 1
    ]

    actual_passes = similarity_df.loc[
        similarity_df[
            TARGET_COLUMN
        ]
        == 0
    ]

    for minimum_failed_neighbours in thresholds:
        failure_retrieval_rate = float(
            (
                actual_failures[
                    "failed_neighbour_count"
                ]
                >= minimum_failed_neighbours
            ).mean()
        )

        pass_alert_rate = float(
            (
                actual_passes[
                    "failed_neighbour_count"
                ]
                >= minimum_failed_neighbours
            ).mean()
        )

        rows.append(
            {
                "minimum_failed_neighbours": (
                    minimum_failed_neighbours
                ),
                "actual_failure_records": int(
                    len(actual_failures)
                ),
                "actual_pass_records": int(
                    len(actual_passes)
                ),
                "failure_retrieval_rate": (
                    failure_retrieval_rate
                ),
                "pass_alert_rate": (
                    pass_alert_rate
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_historical_vs_model_plot(
    similarity_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 7)
    )

    scatter = axis.scatter(
        similarity_df[
            "calibrated_failure_probability"
        ],
        similarity_df[
            "historical_weighted_failure_rate"
        ],
        c=similarity_df[
            TARGET_COLUMN
        ],
        alpha=0.7,
    )

    axis.set_title(
        "Model Probability versus Historical Failure Evidence"
    )

    axis.set_xlabel(
        "Calibrated Model Failure Probability"
    )

    axis.set_ylabel(
        "Similarity-Weighted Historical Failure Rate"
    )

    figure.colorbar(
        scatter,
        ax=axis,
        label="Actual Outcome",
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "model_probability_vs_historical_evidence.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_similarity_by_outcome_plot(
    similarity_df: pd.DataFrame,
) -> None:
    pass_values = similarity_df.loc[
        similarity_df[
            TARGET_COLUMN
        ]
        == 0,
        "mean_similarity",
    ]

    failure_values = similarity_df.loc[
        similarity_df[
            TARGET_COLUMN
        ]
        == 1,
        "mean_similarity",
    ]

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.boxplot(
        [
            pass_values,
            failure_values,
        ],
        tick_labels=[
            "Pass",
            "Fail",
        ],
    )

    axis.set_title(
        "Historical Similarity by Actual Outcome"
    )

    axis.set_xlabel(
        "Actual Outcome"
    )

    axis.set_ylabel(
        "Mean Similarity to Retrieved Records"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "similarity_by_actual_outcome.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_failed_neighbour_count_plot(
    similarity_df: pd.DataFrame,
) -> None:
    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    maximum_count = int(
        similarity_df[
            "failed_neighbour_count"
        ].max()
    )

    bins = np.arange(
        -0.5,
        maximum_count + 1.5,
        1.0,
    )

    axis.hist(
        similarity_df.loc[
            similarity_df[
                TARGET_COLUMN
            ]
            == 0,
            "failed_neighbour_count",
        ],
        bins=bins,
        alpha=0.6,
        label="Pass",
    )

    axis.hist(
        similarity_df.loc[
            similarity_df[
                TARGET_COLUMN
            ]
            == 1,
            "failed_neighbour_count",
        ],
        bins=bins,
        alpha=0.6,
        label="Fail",
    )

    axis.set_title(
        "Failed Historical Neighbours by Actual Outcome"
    )

    axis.set_xlabel(
        "Number of Failed Neighbours in Top-K"
    )

    axis.set_ylabel(
        "Number of Test Records"
    )

    axis.legend()

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "failed_neighbour_count_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_local_similarity_plot(
    wafer_id: Any,
    neighbour_df: pd.DataFrame,
) -> None:
    plot_df = neighbour_df.sort_values(
        by="neighbour_rank",
        ascending=False,
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    labels = [
        f"{record_id} ({status})"
        for record_id, status in zip(
            plot_df[
                "historical_wafer_id"
            ],
            plot_df[
                "historical_status"
            ],
            strict=True,
        )
    ]

    axis.barh(
        labels,
        plot_df[
            "similarity_score"
        ],
    )

    axis.set_title(
        f"Most Similar Historical Wafers: {wafer_id}"
    )

    axis.set_xlabel(
        "Analytical Similarity Score"
    )

    axis.set_ylabel(
        "Historical Wafer and Outcome"
    )

    axis.set_xlim(
        0.0,
        1.0,
    )

    figure.tight_layout()

    safe_id = str(
        wafer_id
    ).replace(
        "/",
        "_",
    )

    figure.savefig(
        LOCAL_FIGURES_DIR
        / f"{safe_id}_historical_similarity.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


# ============================================================
# LOCAL PLOT SELECTION
# ============================================================
def select_records_for_local_plots(
    similarity_df: pd.DataFrame,
) -> list[Any]:
    priority_mapping = {
        "Insufficient Evidence": 0,
        "High Risk": 1,
        "Engineering Review": 2,
        "Low Risk": 3,
    }

    selection_df = similarity_df.copy()

    selection_df[
        "actual_failure_priority"
    ] = (
        1
        - selection_df[
            TARGET_COLUMN
        ].astype(int)
    )

    selection_df[
        "decision_priority"
    ] = (
        selection_df[
            "decision"
        ]
        .map(
            priority_mapping
        )
        .fillna(
            99
        )
    )

    selection_df = selection_df.sort_values(
        by=[
            "actual_failure_priority",
            "decision_priority",
            "historical_weighted_failure_rate",
            "mean_similarity",
        ],
        ascending=[
            True,
            True,
            False,
            False,
        ],
    )

    return (
        selection_df[
            ID_COLUMN
        ]
        .head(
            MAX_LOCAL_PLOTS
        )
        .tolist()
    )


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
    summary_df: pd.DataFrame,
    failure_retrieval_df: pd.DataFrame,
    pca: PCA,
    distance_scale: float,
    plotted_count: int,
) -> None:
    print("\n" + "=" * 122)
    print(
        "HEVEMIND HISTORICAL SIMILARITY "
        "AND CASE-BASED EVIDENCE ENGINE"
    )
    print("=" * 122)

    print("\nSimilarity representation:")

    print(
        f"PCA components:                  "
        f"{pca.n_components_}"
    )

    print(
        f"Explained variance retained:     "
        f"{pca.explained_variance_ratio_.sum():.4f}"
    )

    print(
        f"Distance calibration scale:      "
        f"{distance_scale:.4f}"
    )

    print(
        f"Historical neighbours per wafer: "
        f"{TOP_K_NEIGHBOURS}"
    )

    print("\nHistorical evidence summary:")

    print(
        f"Test records analysed:           "
        f"{len(summary_df)}"
    )

    print(
        f"Mean historical failure rate:    "
        f"{summary_df['historical_weighted_failure_rate'].mean():.4f}"
    )

    print(
        f"Mean similarity:                 "
        f"{summary_df['mean_similarity'].mean():.4f}"
    )

    print(
        f"Records with failed neighbour:   "
        f"{(summary_df['failed_neighbour_count'] >= 1).sum()}"
    )

    print(
        f"Strong/moderate failure evidence:"
        f" {summary_df['historical_evidence_level'].isin(['Strong Failure Evidence', 'Moderate Failure Evidence']).sum()}"
    )

    print("\nEvidence categories:")

    print(
        summary_df[
            "historical_evidence_level"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nFailure retrieval performance:")

    print(
        failure_retrieval_df
        .round(4)
        .to_string(
            index=False
        )
    )

    print(
        f"\nLocal similarity plots:          "
        f"{plotted_count}"
    )

    print("\nSaved outputs:")

    print(
        f"Neighbour-level results:         "
        f"{NEIGHBOUR_RESULTS_PATH}"
    )

    print(
        f"Wafer similarity summary:        "
        f"{WAFER_SUMMARY_PATH}"
    )

    print(
        f"Failure retrieval report:        "
        f"{FAILURE_RETRIEVAL_SUMMARY_PATH}"
    )

    print(
        f"Report directory:                "
        f"{SIMILARITY_REPORT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading development, test, uncertainty and SHAP data"
    )

    development_df, test_df = (
        load_development_and_test()
    )

    uncertainty_df = load_uncertainty_results()
    explainability_df = (
        load_explainability_summary()
    )

    validate_unique_ids(
        development_df,
        "Development dataset",
    )

    validate_unique_ids(
        test_df,
        "Test dataset",
    )

    sensor_columns = get_sensor_columns(
        development_df
    )

    missing_test_sensors = [
        column
        for column in sensor_columns
        if column not in test_df.columns
    ]

    if missing_test_sensors:
        raise ValueError(
            "The test dataset is missing sensor columns: "
            f"{missing_test_sensors[:10]}"
        )

    global_shap_features = (
        load_global_shap_features()
    )

    LOGGER.info(
        "Using %s raw sensor features for similarity representation",
        len(sensor_columns),
    )

    x_development = development_df[
        sensor_columns
    ].copy()

    x_test = test_df[
        sensor_columns
    ].copy()

    (
        preprocessor,
        pca,
        development_embeddings,
        test_embeddings,
    ) = fit_similarity_representation(
        development_features=x_development,
        test_features=x_test,
    )

    distance_scale = (
        estimate_reference_distance_scale(
            development_embeddings
        )
    )

    LOGGER.info(
        "Fitting historical nearest-neighbour index"
    )

    neighbour_model = fit_neighbour_model(
        development_embeddings
    )

    (
        neighbour_distances,
        neighbour_indices,
    ) = retrieve_historical_neighbours(
        neighbour_model=neighbour_model,
        test_embeddings=test_embeddings,
    )

    LOGGER.info(
        "Building historical evidence tables"
    )

    (
        neighbour_results_df,
        similarity_summary_df,
    ) = build_similarity_results(
        development_df=development_df,
        test_df=test_df,
        uncertainty_df=uncertainty_df,
        explainability_df=explainability_df,
        neighbour_distances=(
            neighbour_distances
        ),
        neighbour_indices=(
            neighbour_indices
        ),
        distance_scale=distance_scale,
    )

    neighbour_results_df.to_csv(
        NEIGHBOUR_RESULTS_PATH,
        index=False,
    )

    similarity_summary_df.to_csv(
        WAFER_SUMMARY_PATH,
        index=False,
    )

    decision_group_summary = (
        build_group_summary(
            similarity_df=(
                similarity_summary_df
            ),
            group_column="decision",
        )
    )

    decision_group_summary.to_csv(
        DECISION_GROUP_SUMMARY_PATH,
        index=False,
    )

    outcome_group_summary = (
        build_group_summary(
            similarity_df=(
                similarity_summary_df
            ),
            group_column=TARGET_COLUMN,
        )
    )

    outcome_group_summary.to_csv(
        ACTUAL_OUTCOME_SUMMARY_PATH,
        index=False,
    )

    failure_retrieval_summary = (
        build_failure_retrieval_summary(
            similarity_summary_df
        )
    )

    failure_retrieval_summary.to_csv(
        FAILURE_RETRIEVAL_SUMMARY_PATH,
        index=False,
    )

    LOGGER.info(
        "Saving similarity artifacts"
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    joblib.dump(
        pca,
        PCA_PATH,
    )

    joblib.dump(
        neighbour_model,
        NEIGHBOUR_MODEL_PATH,
    )

    np.save(
        REFERENCE_EMBEDDINGS_PATH,
        development_embeddings,
    )

    development_df[
        [
            ID_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
            TIMESTAMP_COLUMN,
        ]
    ].to_csv(
        REFERENCE_METADATA_PATH,
        index=False,
    )

    save_json(
        FEATURE_METADATA_PATH,
        {
            "raw_sensor_count": int(
                len(sensor_columns)
            ),
            "raw_sensor_features": (
                sensor_columns
            ),
            "pca_components": int(
                pca.n_components_
            ),
            "explained_variance_retained": float(
                pca.explained_variance_ratio_.sum()
            ),
            "distance_scale": float(
                distance_scale
            ),
            "top_k_neighbours": int(
                TOP_K_NEIGHBOURS
            ),
            "global_shap_context_features": (
                global_shap_features
            ),
            "similarity_definition": (
                "Exponential transformation of Euclidean "
                "distance in a PCA representation fitted only "
                "on development records."
            ),
            "warning": (
                "Similarity scores are analytical proximity "
                "measures, not probabilities and not evidence "
                "of a physical root cause."
            ),
        },
    )

    LOGGER.info(
        "Generating similarity figures"
    )

    save_historical_vs_model_plot(
        similarity_summary_df
    )

    save_similarity_by_outcome_plot(
        similarity_summary_df
    )

    save_failed_neighbour_count_plot(
        similarity_summary_df
    )

    selected_ids = (
        select_records_for_local_plots(
            similarity_summary_df
        )
    )

    plotted_count = 0

    for wafer_id in selected_ids:
        wafer_neighbours = (
            neighbour_results_df.loc[
                neighbour_results_df[
                    "query_wafer_id"
                ]
                == wafer_id
            ]
            .copy()
        )

        if wafer_neighbours.empty:
            continue

        save_local_similarity_plot(
            wafer_id=wafer_id,
            neighbour_df=wafer_neighbours,
        )

        plotted_count += 1

    actual_failure_records = (
        similarity_summary_df.loc[
            similarity_summary_df[
                TARGET_COLUMN
            ]
            == 1
        ]
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "Historical similarity and "
            "case-based evidence engine"
        ),
        "development_reference_records": int(
            len(development_df)
        ),
        "test_records_analysed": int(
            len(test_df)
        ),
        "raw_sensor_count": int(
            len(sensor_columns)
        ),
        "pca_components": int(
            pca.n_components_
        ),
        "explained_variance_retained": float(
            pca.explained_variance_ratio_.sum()
        ),
        "top_k_neighbours": int(
            TOP_K_NEIGHBOURS
        ),
        "distance_scale": float(
            distance_scale
        ),
        "overall_similarity_summary": {
            "mean_similarity": float(
                similarity_summary_df[
                    "mean_similarity"
                ].mean()
            ),
            "mean_weighted_historical_failure_rate": float(
                similarity_summary_df[
                    "historical_weighted_failure_rate"
                ].mean()
            ),
            "records_with_at_least_one_failed_neighbour": int(
                (
                    similarity_summary_df[
                        "failed_neighbour_count"
                    ]
                    >= 1
                ).sum()
            ),
        },
        "actual_failure_similarity_summary": {
            "actual_failures": int(
                len(actual_failure_records)
            ),
            "mean_failed_neighbour_count": float(
                actual_failure_records[
                    "failed_neighbour_count"
                ].mean()
            ),
            "mean_weighted_historical_failure_rate": float(
                actual_failure_records[
                    "historical_weighted_failure_rate"
                ].mean()
            ),
            "failures_with_at_least_one_failed_neighbour": int(
                (
                    actual_failure_records[
                        "failed_neighbour_count"
                    ]
                    >= 1
                ).sum()
            ),
        },
        "failure_retrieval_performance": (
            failure_retrieval_summary.to_dict(
                orient="records"
            )
        ),
        "decision_group_summary": (
            decision_group_summary.to_dict(
                orient="records"
            )
        ),
        "local_plot_count": int(
            plotted_count
        ),
        "methodological_controls": {
            "reference_data": (
                "Development records only"
            ),
            "query_data": (
                "Held-out test records"
            ),
            "target_used_in_distance_calculation": False,
            "test_outcome_used_in_retrieval": False,
            "similarity_representation": (
                "Median imputation, robust scaling and PCA"
            ),
        },
        "important_warning": (
            "Retrieved historical cases provide statistical "
            "precedent only. Similarity does not prove causality, "
            "equipment origin or a physical process mechanism."
        ),
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print_console_summary(
        summary_df=similarity_summary_df,
        failure_retrieval_df=(
            failure_retrieval_summary
        ),
        pca=pca,
        distance_scale=distance_scale,
        plotted_count=plotted_count,
    )


if __name__ == "__main__":
    main()