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
import shap


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

TEST_PATH = SPLITS_DIR / "test.parquet"

ARTIFACTS_DIR = ROOT_DIR / "artifacts"

MODEL_PATH = (
    ARTIFACTS_DIR
    / "models"
    / "cross_fitted_calibration"
    / "balanced_random_forest.joblib"
)

UNCERTAINTY_REPORT_DIR = (
    ROOT_DIR
    / "reports"
    / "uncertainty_engine"
)

UNCERTAINTY_RESULTS_PATH = (
    UNCERTAINTY_REPORT_DIR
    / "tables"
    / "test_uncertainty_scores.csv"
)

EXPLAINABILITY_ARTIFACT_DIR = (
    ARTIFACTS_DIR
    / "explainability_engine"
)

EXPLAINABILITY_REPORT_DIR = (
    ROOT_DIR
    / "reports"
    / "explainability_engine"
)

TABLES_DIR = EXPLAINABILITY_REPORT_DIR / "tables"
FIGURES_DIR = EXPLAINABILITY_REPORT_DIR / "figures"
LOCAL_FIGURES_DIR = FIGURES_DIR / "local_explanations"

GLOBAL_IMPORTANCE_PATH = (
    TABLES_DIR
    / "global_shap_importance.csv"
)

LOCAL_EXPLANATIONS_PATH = (
    TABLES_DIR
    / "local_shap_explanations.csv"
)

WAFER_EXPLANATION_SUMMARY_PATH = (
    TABLES_DIR
    / "wafer_explanation_summary.csv"
)

DECISION_FEATURE_SUMMARY_PATH = (
    TABLES_DIR
    / "decision_group_feature_summary.csv"
)

FAILURE_FEATURE_SUMMARY_PATH = (
    TABLES_DIR
    / "actual_failure_feature_summary.csv"
)

EXPLAINER_PATH = (
    EXPLAINABILITY_ARTIFACT_DIR
    / "tree_shap_explainer.joblib"
)

FEATURE_NAMES_PATH = (
    EXPLAINABILITY_ARTIFACT_DIR
    / "transformed_feature_names.json"
)

SUMMARY_PATH = (
    EXPLAINABILITY_REPORT_DIR
    / "explainability_engine_summary.json"
)


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
TIMESTAMP_COLUMN = "timestamp"
SENSOR_PREFIX = "sensor_"

TOP_GLOBAL_FEATURES = 40
TOP_LOCAL_FEATURES = 10
TOP_DECISION_FEATURES = 20

MAX_LOCAL_PLOTS = 25

RANDOM_STATE = 42


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        EXPLAINABILITY_ARTIFACT_DIR,
        EXPLAINABILITY_REPORT_DIR,
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
def load_test_data() -> pd.DataFrame:
    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test dataset was not found: {TEST_PATH}"
        )

    dataframe = pd.read_parquet(
        TEST_PATH
    )

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


def load_uncertainty_results() -> pd.DataFrame:
    if not UNCERTAINTY_RESULTS_PATH.exists():
        raise FileNotFoundError(
            "Uncertainty results were not found. "
            "Run src/12_uncertainty_engine.py first."
        )

    dataframe = pd.read_csv(
        UNCERTAINTY_RESULTS_PATH
    )

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        "calibrated_failure_probability",
        "uncertainty_adjusted_decision",
        "prediction_confidence",
        "combined_uncertainty",
        "ood_percentile",
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Uncertainty-results table is missing columns: "
            f"{sorted(missing_columns)}"
        )

    return dataframe


def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The Balanced Random Forest model was not found. "
            "Run src/11_cross_fitted_calibration_policy.py first."
        )

    model = joblib.load(
        MODEL_PATH
    )

    required_steps = {
        "imputer",
        "model",
    }

    if not required_steps.issubset(
        model.named_steps
    ):
        raise ValueError(
            "The loaded model pipeline must contain "
            "'imputer' and 'model' steps."
        )

    return model


def get_sensor_columns(
    dataframe: pd.DataFrame,
) -> list[str]:
    sensor_columns = [
        column
        for column in dataframe.columns
        if column.startswith(
            SENSOR_PREFIX
        )
    ]

    if not sensor_columns:
        raise ValueError(
            "No sensor columns were found."
        )

    return sensor_columns


# ============================================================
# FEATURE TRANSFORMATION
# ============================================================
def transform_features(
    pipeline: Any,
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> tuple[np.ndarray, list[str]]:
    imputer = pipeline.named_steps[
        "imputer"
    ]

    sensor_data = dataframe[
        sensor_columns
    ].copy()

    transformed_data = imputer.transform(
        sensor_data
    )

    try:
        transformed_feature_names = (
            imputer.get_feature_names_out(
                sensor_columns
            ).tolist()
        )

    except Exception:
        transformed_feature_names = (
            sensor_columns.copy()
        )

        if transformed_data.shape[1] > len(
            sensor_columns
        ):
            number_of_indicators = (
                transformed_data.shape[1]
                - len(sensor_columns)
            )

            indicator_names = [
                f"missing_indicator_{index + 1:03d}"
                for index in range(
                    number_of_indicators
                )
            ]

            transformed_feature_names.extend(
                indicator_names
            )

    if transformed_data.shape[1] != len(
        transformed_feature_names
    ):
        raise ValueError(
            "Transformed feature count does not match "
            "the generated feature-name count."
        )

    return (
        np.asarray(
            transformed_data,
            dtype=float,
        ),
        transformed_feature_names,
    )


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================
def extract_failure_shap_values(
    shap_output: Any,
) -> np.ndarray:
    """
    Normalise different SHAP output formats into:

        rows x transformed features

    representing the failure class.
    """
    if isinstance(
        shap_output,
        list,
    ):
        if len(shap_output) == 2:
            return np.asarray(
                shap_output[1],
                dtype=float,
            )

        return np.asarray(
            shap_output[-1],
            dtype=float,
        )

    shap_array = np.asarray(
        shap_output
    )

    if shap_array.ndim == 2:
        return shap_array.astype(
            float
        )

    if shap_array.ndim == 3:
        if shap_array.shape[-1] == 2:
            return shap_array[
                :,
                :,
                1,
            ].astype(float)

        if shap_array.shape[0] == 2:
            return shap_array[
                1,
                :,
                :,
            ].astype(float)

    raise ValueError(
        "Unsupported SHAP output shape: "
        f"{shap_array.shape}"
    )


def calculate_shap_values(
    pipeline: Any,
    transformed_data: np.ndarray,
) -> tuple[Any, np.ndarray]:
    forest_model = pipeline.named_steps[
        "model"
    ]

    LOGGER.info(
        "Creating TreeSHAP explainer"
    )

    explainer = shap.TreeExplainer(
        forest_model,
        feature_perturbation=(
            "tree_path_dependent"
        ),
    )

    LOGGER.info(
        "Calculating SHAP values for %s records",
        transformed_data.shape[0],
    )

    raw_shap_values = explainer.shap_values(
        transformed_data,
        check_additivity=False,
    )

    failure_shap_values = (
        extract_failure_shap_values(
            raw_shap_values
        )
    )

    if failure_shap_values.shape != transformed_data.shape:
        raise ValueError(
            "SHAP matrix shape does not match transformed "
            f"feature matrix: {failure_shap_values.shape} "
            f"versus {transformed_data.shape}"
        )

    return (
        explainer,
        failure_shap_values,
    )


# ============================================================
# GLOBAL EXPLANATIONS
# ============================================================
def build_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    mean_absolute_shap = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    mean_signed_shap = np.mean(
        shap_values,
        axis=0,
    )

    positive_contribution_rate = np.mean(
        shap_values > 0,
        axis=0,
    )

    output = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_absolute_shap": (
                mean_absolute_shap
            ),
            "mean_signed_shap": (
                mean_signed_shap
            ),
            "positive_contribution_rate": (
                positive_contribution_rate
            ),
        }
    )

    total_importance = float(
        output[
            "mean_absolute_shap"
        ].sum()
    )

    if total_importance > 0:
        output[
            "relative_importance"
        ] = (
            output[
                "mean_absolute_shap"
            ]
            / total_importance
        )

    else:
        output[
            "relative_importance"
        ] = 0.0

    output[
        "cumulative_importance"
    ] = (
        output.sort_values(
            "relative_importance",
            ascending=False,
        )[
            "relative_importance"
        ]
        .cumsum()
        .reindex(
            output.index
        )
    )

    return (
        output
        .sort_values(
            by="mean_absolute_shap",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# LOCAL EXPLANATIONS
# ============================================================
def build_local_explanations(
    test_df: pd.DataFrame,
    uncertainty_df: pd.DataFrame,
    transformed_data: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    local_rows: list[
        dict[str, Any]
    ] = []

    summary_rows: list[
        dict[str, Any]
    ] = []

    uncertainty_lookup = (
        uncertainty_df
        .set_index(
            ID_COLUMN
        )
    )

    for row_position in range(
        len(test_df)
    ):
        wafer_id = test_df.iloc[
            row_position
        ][ID_COLUMN]

        if wafer_id not in uncertainty_lookup.index:
            raise KeyError(
                f"Wafer {wafer_id} is missing from "
                "the uncertainty output."
            )

        uncertainty_record = (
            uncertainty_lookup.loc[
                wafer_id
            ]
        )

        row_shap = shap_values[
            row_position
        ]

        row_values = transformed_data[
            row_position
        ]

        sorted_indices = np.argsort(
            np.abs(
                row_shap
            )
        )[::-1]

        top_indices = sorted_indices[
            :TOP_LOCAL_FEATURES
        ]

        positive_total = float(
            np.sum(
                row_shap[
                    row_shap > 0
                ]
            )
        )

        negative_total = float(
            np.sum(
                np.abs(
                    row_shap[
                        row_shap < 0
                    ]
                )
            )
        )

        top_risk_features: list[str] = []
        top_protective_features: list[str] = []

        for rank, feature_index in enumerate(
            top_indices,
            start=1,
        ):
            feature_name = feature_names[
                feature_index
            ]

            shap_contribution = float(
                row_shap[
                    feature_index
                ]
            )

            feature_value = float(
                row_values[
                    feature_index
                ]
            )

            direction = (
                "Increases failure risk"
                if shap_contribution > 0
                else "Decreases failure risk"
            )

            if shap_contribution > 0:
                top_risk_features.append(
                    feature_name
                )

            elif shap_contribution < 0:
                top_protective_features.append(
                    feature_name
                )

            local_rows.append(
                {
                    ID_COLUMN: wafer_id,
                    TARGET_COLUMN: int(
                        test_df.iloc[
                            row_position
                        ][TARGET_COLUMN]
                    ),
                    STATUS_COLUMN: (
                        test_df.iloc[
                            row_position
                        ][STATUS_COLUMN]
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
                    "rank": rank,
                    "feature": feature_name,
                    "feature_value": feature_value,
                    "shap_contribution": (
                        shap_contribution
                    ),
                    "absolute_shap_contribution": abs(
                        shap_contribution
                    ),
                    "contribution_direction": (
                        direction
                    ),
                }
            )

        summary_rows.append(
            {
                ID_COLUMN: wafer_id,
                TARGET_COLUMN: int(
                    test_df.iloc[
                        row_position
                    ][TARGET_COLUMN]
                ),
                STATUS_COLUMN: (
                    test_df.iloc[
                        row_position
                    ][STATUS_COLUMN]
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
                "ood_percentile": float(
                    uncertainty_record[
                        "ood_percentile"
                    ]
                ),
                "positive_shap_total": (
                    positive_total
                ),
                "negative_shap_total": (
                    negative_total
                ),
                "top_risk_features": ", ".join(
                    top_risk_features[:5]
                ),
                "top_protective_features": ", ".join(
                    top_protective_features[:5]
                ),
                "explanation_text": (
                    create_explanation_text(
                        decision=str(
                            uncertainty_record[
                                "uncertainty_adjusted_decision"
                            ]
                        ),
                        failure_probability=float(
                            uncertainty_record[
                                "calibrated_failure_probability"
                            ]
                        ),
                        prediction_confidence=float(
                            uncertainty_record[
                                "prediction_confidence"
                            ]
                        ),
                        top_risk_features=(
                            top_risk_features
                        ),
                        top_protective_features=(
                            top_protective_features
                        ),
                    )
                ),
            }
        )

    return (
        pd.DataFrame(
            local_rows
        ),
        pd.DataFrame(
            summary_rows
        ),
    )


def create_explanation_text(
    decision: str,
    failure_probability: float,
    prediction_confidence: float,
    top_risk_features: list[str],
    top_protective_features: list[str],
) -> str:
    risk_text = (
        ", ".join(
            top_risk_features[:3]
        )
        if top_risk_features
        else "no dominant positive sensor contribution"
    )

    protective_text = (
        ", ".join(
            top_protective_features[:3]
        )
        if top_protective_features
        else "no dominant protective sensor contribution"
    )

    return (
        f"The record was assigned to {decision} with a "
        f"calibrated failure probability of "
        f"{failure_probability:.2%}. "
        f"The strongest model contributions increasing risk were "
        f"{risk_text}. Contributions reducing predicted risk were "
        f"{protective_text}. The internal prediction-confidence "
        f"score was {prediction_confidence:.2%}. "
        f"Because SECOM sensors are anonymous, these features "
        f"must be treated as statistical indicators rather than "
        f"confirmed physical root causes."
    )


# ============================================================
# GROUP EXPLANATIONS
# ============================================================
def build_group_feature_summary(
    local_df: pd.DataFrame,
    group_column: str,
) -> pd.DataFrame:
    grouped = (
        local_df
        .groupby(
            [
                group_column,
                "feature",
            ],
            observed=False,
        )
        .agg(
            records=(
                ID_COLUMN,
                "nunique",
            ),
            mean_absolute_shap=(
                "absolute_shap_contribution",
                "mean",
            ),
            mean_signed_shap=(
                "shap_contribution",
                "mean",
            ),
            positive_contribution_rate=(
                "shap_contribution",
                lambda values: float(
                    np.mean(
                        values > 0
                    )
                ),
            ),
        )
        .reset_index()
    )

    grouped[
        "rank_within_group"
    ] = (
        grouped
        .groupby(
            group_column,
            observed=False,
        )[
            "mean_absolute_shap"
        ]
        .rank(
            method="first",
            ascending=False,
        )
    )

    return (
        grouped.loc[
            grouped[
                "rank_within_group"
            ]
            <= TOP_DECISION_FEATURES
        ]
        .sort_values(
            by=[
                group_column,
                "rank_within_group",
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# VISUALISATIONS
# ============================================================
def save_global_importance_plot(
    global_importance_df: pd.DataFrame,
) -> None:
    plot_df = (
        global_importance_df
        .head(
            TOP_GLOBAL_FEATURES
        )
        .sort_values(
            by="mean_absolute_shap",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(10, 12)
    )

    axis.barh(
        plot_df["feature"],
        plot_df["mean_absolute_shap"],
    )

    axis.set_title(
        "Global SHAP Sensor Importance"
    )

    axis.set_xlabel(
        "Mean Absolute SHAP Contribution"
    )

    axis.set_ylabel(
        "Sensor Feature"
    )

    figure.tight_layout()

    figure.savefig(
        FIGURES_DIR
        / "global_shap_importance.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )


def save_shap_summary_plot(
    transformed_data: np.ndarray,
    shap_values: np.ndarray,
    feature_names: list[str],
) -> None:
    plt.figure(
        figsize=(11, 10)
    )

    shap.summary_plot(
        shap_values,
        transformed_data,
        feature_names=feature_names,
        max_display=30,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR
        / "shap_summary_beeswarm.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()


def save_local_contribution_plot(
    wafer_id: Any,
    local_wafer_df: pd.DataFrame,
) -> None:
    plot_df = (
        local_wafer_df
        .head(
            TOP_LOCAL_FEATURES
        )
        .sort_values(
            by="shap_contribution",
            ascending=True,
        )
    )

    figure, axis = plt.subplots(
        figsize=(9, 6)
    )

    axis.barh(
        plot_df["feature"],
        plot_df["shap_contribution"],
    )

    axis.axvline(
        0.0,
        linewidth=1,
    )

    axis.set_title(
        f"Local SHAP Explanation: {wafer_id}"
    )

    axis.set_xlabel(
        "Contribution to Predicted Failure Risk"
    )

    axis.set_ylabel(
        "Sensor Feature"
    )

    figure.tight_layout()

    safe_wafer_id = str(
        wafer_id
    ).replace(
        "/",
        "_",
    )

    figure.savefig(
        LOCAL_FIGURES_DIR
        / f"{safe_wafer_id}_local_shap.png",
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
    uncertainty_df: pd.DataFrame,
) -> list[Any]:
    priority_order = {
        "Insufficient Evidence": 0,
        "High Risk": 1,
        "Engineering Review": 2,
        "Low Risk": 3,
    }

    selection_df = uncertainty_df.copy()

    selection_df[
        "decision_priority"
    ] = (
        selection_df[
            "uncertainty_adjusted_decision"
        ]
        .map(
            priority_order
        )
        .fillna(
            99
        )
    )

    selection_df[
        "actual_failure_priority"
    ] = (
        1
        - selection_df[
            TARGET_COLUMN
        ].astype(
            int
        )
    )

    selection_df = selection_df.sort_values(
        by=[
            "actual_failure_priority",
            "decision_priority",
            "calibrated_failure_probability",
            "combined_uncertainty",
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
    global_importance_df: pd.DataFrame,
    local_df: pd.DataFrame,
    wafer_summary_df: pd.DataFrame,
    plotted_record_count: int,
) -> None:
    print("\n" + "=" * 118)
    print(
        "HEVEMIND EXPLAINABILITY AND SENSOR-ATTRIBUTION ENGINE"
    )
    print("=" * 118)

    print("\nTop global sensor features:")

    print(
        global_importance_df[
            [
                "feature",
                "mean_absolute_shap",
                "mean_signed_shap",
                "relative_importance",
                "positive_contribution_rate",
            ]
        ]
        .head(20)
        .round(6)
        .to_string(
            index=False
        )
    )

    print("\nExplanation coverage:")

    print(
        f"Explained test records:          "
        f"{wafer_summary_df[ID_COLUMN].nunique()}"
    )

    print(
        f"Local contribution rows:         "
        f"{len(local_df)}"
    )

    print(
        f"Local explanation plots:         "
        f"{plotted_record_count}"
    )

    print("\nDecision counts explained:")

    print(
        wafer_summary_df[
            "decision"
        ]
        .value_counts(
            dropna=False
        )
        .to_string()
    )

    print("\nSaved outputs:")

    print(
        f"Global importance:               "
        f"{GLOBAL_IMPORTANCE_PATH}"
    )

    print(
        f"Local explanations:              "
        f"{LOCAL_EXPLANATIONS_PATH}"
    )

    print(
        f"Wafer summaries:                 "
        f"{WAFER_EXPLANATION_SUMMARY_PATH}"
    )

    print(
        f"Decision feature summary:        "
        f"{DECISION_FEATURE_SUMMARY_PATH}"
    )

    print(
        f"Report directory:                "
        f"{EXPLAINABILITY_REPORT_DIR}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    LOGGER.info(
        "Loading model, test data and uncertainty results"
    )

    test_df = load_test_data()
    uncertainty_df = load_uncertainty_results()
    pipeline = load_model()

    sensor_columns = get_sensor_columns(
        test_df
    )

    LOGGER.info(
        "Transforming test sensor measurements"
    )

    (
        transformed_data,
        transformed_feature_names,
    ) = transform_features(
        pipeline=pipeline,
        dataframe=test_df,
        sensor_columns=sensor_columns,
    )

    (
        explainer,
        failure_shap_values,
    ) = calculate_shap_values(
        pipeline=pipeline,
        transformed_data=transformed_data,
    )

    LOGGER.info(
        "Building global SHAP importance table"
    )

    global_importance_df = (
        build_global_importance(
            shap_values=failure_shap_values,
            feature_names=(
                transformed_feature_names
            ),
        )
    )

    global_importance_df.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
    )

    LOGGER.info(
        "Building local wafer explanations"
    )

    (
        local_explanations_df,
        wafer_summary_df,
    ) = build_local_explanations(
        test_df=test_df,
        uncertainty_df=uncertainty_df,
        transformed_data=transformed_data,
        shap_values=failure_shap_values,
        feature_names=(
            transformed_feature_names
        ),
    )

    local_explanations_df.to_csv(
        LOCAL_EXPLANATIONS_PATH,
        index=False,
    )

    wafer_summary_df.to_csv(
        WAFER_EXPLANATION_SUMMARY_PATH,
        index=False,
    )

    decision_feature_summary = (
        build_group_feature_summary(
            local_df=local_explanations_df,
            group_column="decision",
        )
    )

    decision_feature_summary.to_csv(
        DECISION_FEATURE_SUMMARY_PATH,
        index=False,
    )

    failure_local_df = (
        local_explanations_df.loc[
            local_explanations_df[
                TARGET_COLUMN
            ]
            == 1
        ]
        .copy()
    )

    if not failure_local_df.empty:
        failure_feature_summary = (
            failure_local_df
            .groupby(
                "feature"
            )
            .agg(
                failure_records=(
                    ID_COLUMN,
                    "nunique",
                ),
                mean_absolute_shap=(
                    "absolute_shap_contribution",
                    "mean",
                ),
                mean_signed_shap=(
                    "shap_contribution",
                    "mean",
                ),
                positive_contribution_rate=(
                    "shap_contribution",
                    lambda values: float(
                        np.mean(
                            values > 0
                        )
                    ),
                ),
            )
            .reset_index()
            .sort_values(
                by="mean_absolute_shap",
                ascending=False,
            )
        )

    else:
        failure_feature_summary = pd.DataFrame(
            columns=[
                "feature",
                "failure_records",
                "mean_absolute_shap",
                "mean_signed_shap",
                "positive_contribution_rate",
            ]
        )

    failure_feature_summary.to_csv(
        FAILURE_FEATURE_SUMMARY_PATH,
        index=False,
    )

    LOGGER.info(
        "Generating global SHAP figures"
    )

    save_global_importance_plot(
        global_importance_df
    )

    save_shap_summary_plot(
        transformed_data=transformed_data,
        shap_values=failure_shap_values,
        feature_names=(
            transformed_feature_names
        ),
    )

    LOGGER.info(
        "Generating selected local SHAP figures"
    )

    selected_wafer_ids = (
        select_records_for_local_plots(
            uncertainty_df
        )
    )

    plotted_record_count = 0

    for wafer_id in selected_wafer_ids:
        wafer_explanation = (
            local_explanations_df.loc[
                local_explanations_df[
                    ID_COLUMN
                ]
                == wafer_id
            ]
            .sort_values(
                by="rank"
            )
        )

        if wafer_explanation.empty:
            continue

        save_local_contribution_plot(
            wafer_id=wafer_id,
            local_wafer_df=(
                wafer_explanation
            ),
        )

        plotted_record_count += 1

    joblib.dump(
        explainer,
        EXPLAINER_PATH,
    )

    save_json(
        FEATURE_NAMES_PATH,
        {
            "feature_count": len(
                transformed_feature_names
            ),
            "features": (
                transformed_feature_names
            ),
            "warning": (
                "Missing-indicator features are generated by "
                "the fitted imputer. Sensor identities remain "
                "anonymous in SECOM."
            ),
        },
    )

    summary = {
        "project": "HeveMind",
        "stage": (
            "SHAP explainability and "
            "sensor-attribution engine"
        ),
        "test_records_explained": int(
            wafer_summary_df[
                ID_COLUMN
            ].nunique()
        ),
        "transformed_feature_count": int(
            len(
                transformed_feature_names
            )
        ),
        "top_global_features": (
            global_importance_df
            .head(30)
            .to_dict(
                orient="records"
            )
        ),
        "decision_group_top_features": (
            decision_feature_summary
            .to_dict(
                orient="records"
            )
        ),
        "actual_failure_top_features": (
            failure_feature_summary
            .head(30)
            .to_dict(
                orient="records"
            )
        ),
        "local_plot_count": int(
            plotted_record_count
        ),
        "methodology": {
            "explainer": (
                "TreeSHAP applied to the fitted "
                "Balanced Random Forest"
            ),
            "explained_class": (
                "Failure class"
            ),
            "local_feature_count": (
                TOP_LOCAL_FEATURES
            ),
            "global_importance": (
                "Mean absolute SHAP contribution"
            ),
        },
        "important_warning": (
            "SHAP identifies model-attribution patterns, "
            "not confirmed physical causality. Because SECOM "
            "sensor names are anonymous, outputs must be described "
            "as diagnostic sensor hypotheses rather than verified "
            "process root causes."
        ),
    }

    save_json(
        SUMMARY_PATH,
        summary,
    )

    print_console_summary(
        global_importance_df=(
            global_importance_df
        ),
        local_df=(
            local_explanations_df
        ),
        wafer_summary_df=(
            wafer_summary_df
        ),
        plotted_record_count=(
            plotted_record_count
        ),
    )


if __name__ == "__main__":
    main()