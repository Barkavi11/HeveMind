from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split


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

REPORTS_DIR = ROOT_DIR / "reports"
DATA_AUDIT_DIR = REPORTS_DIR / "data_audit"

TABLES_DIR = DATA_AUDIT_DIR / "tables"
FIGURES_DIR = DATA_AUDIT_DIR / "figures"
SPLITS_DIR = PROCESSED_DIR / "splits"

MASTER_CSV_PATH = PROCESSED_DIR / "secom_master.csv"
MASTER_PARQUET_PATH = PROCESSED_DIR / "secom_master.parquet"

AUDITED_PARQUET_PATH = PROCESSED_DIR / "secom_audited.parquet"
AUDITED_CSV_PATH = PROCESSED_DIR / "secom_audited.csv"

AUDIT_SUMMARY_JSON_PATH = DATA_AUDIT_DIR / "audit_summary.json"
FEATURE_MANIFEST_PATH = TABLES_DIR / "feature_manifest.csv"
MISSINGNESS_REPORT_PATH = TABLES_DIR / "missingness_report.csv"
CONSTANT_FEATURES_PATH = TABLES_DIR / "constant_features.csv"
NEAR_CONSTANT_FEATURES_PATH = TABLES_DIR / "near_constant_features.csv"
CORRELATED_PAIRS_PATH = TABLES_DIR / "highly_correlated_feature_pairs.csv"
ROW_QUALITY_PATH = TABLES_DIR / "row_quality_report.csv"
OUTLIER_REPORT_PATH = TABLES_DIR / "outlier_report.csv"
SPLIT_SUMMARY_PATH = TABLES_DIR / "split_summary.csv"


# ============================================================
# CONFIGURATION
# ============================================================
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"
TIMESTAMP_COLUMN = "timestamp"
ID_COLUMN = "wafer_id"

SENSOR_PREFIX = "sensor_"

HIGH_MISSINGNESS_THRESHOLD = 0.60
MODERATE_MISSINGNESS_THRESHOLD = 0.30

NEAR_ZERO_VARIANCE_THRESHOLD = 1e-8
CORRELATION_THRESHOLD = 0.95

OUTLIER_IQR_MULTIPLIER = 1.5

TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_TRAIN = 0.20
RANDOM_STATE = 42


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    """
    Create all output directories required by the audit process.
    """
    for directory in [
        DATA_AUDIT_DIR,
        TABLES_DIR,
        FIGURES_DIR,
        SPLITS_DIR,
    ]:
        directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA LOADING
# ============================================================
def load_master_dataset() -> pd.DataFrame:
    """
    Load the converted SECOM master dataset.

    Parquet is preferred because it preserves datatypes more reliably.
    CSV is used as a fallback.
    """
    if MASTER_PARQUET_PATH.exists():
        LOGGER.info("Loading Parquet dataset: %s", MASTER_PARQUET_PATH)
        dataframe = pd.read_parquet(MASTER_PARQUET_PATH)

    elif MASTER_CSV_PATH.exists():
        LOGGER.info("Loading CSV dataset: %s", MASTER_CSV_PATH)
        dataframe = pd.read_csv(MASTER_CSV_PATH)

    else:
        raise FileNotFoundError(
            "Neither secom_master.parquet nor secom_master.csv was found "
            f"inside {PROCESSED_DIR}"
        )

    if TIMESTAMP_COLUMN in dataframe.columns:
        dataframe[TIMESTAMP_COLUMN] = pd.to_datetime(
            dataframe[TIMESTAMP_COLUMN],
            errors="coerce",
        )

    return dataframe


# ============================================================
# BASIC VALIDATION
# ============================================================
def validate_dataset(dataframe: pd.DataFrame) -> None:
    """
    Validate the expected structure of the master dataset.
    """
    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
        STATUS_COLUMN,
        TIMESTAMP_COLUMN,
    }

    missing_required = required_columns.difference(dataframe.columns)

    if missing_required:
        raise ValueError(
            "The dataset is missing required columns: "
            f"{sorted(missing_required)}"
        )

    sensor_columns = get_sensor_columns(dataframe)

    if not sensor_columns:
        raise ValueError(
            f"No sensor columns beginning with '{SENSOR_PREFIX}' were found."
        )

    unique_targets = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )

    if not unique_targets.issubset({0, 1}):
        raise ValueError(
            "The target column must contain only 0 and 1. "
            f"Observed values: {sorted(unique_targets)}"
        )

    LOGGER.info(
        "Dataset validated: %s rows, %s sensor columns",
        len(dataframe),
        len(sensor_columns),
    )


def get_sensor_columns(dataframe: pd.DataFrame) -> list[str]:
    """
    Return all SECOM sensor feature columns.
    """
    return [
        column
        for column in dataframe.columns
        if column.startswith(SENSOR_PREFIX)
    ]


# ============================================================
# DATASET-LEVEL AUDIT
# ============================================================
def calculate_basic_summary(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> dict[str, Any]:
    """
    Calculate top-level dataset statistics.
    """
    target_counts = (
        dataframe[TARGET_COLUMN]
        .value_counts(dropna=False)
        .sort_index()
        .to_dict()
    )

    status_counts = (
        dataframe[STATUS_COLUMN]
        .value_counts(dropna=False)
        .to_dict()
    )

    timestamp_min = dataframe[TIMESTAMP_COLUMN].min()
    timestamp_max = dataframe[TIMESTAMP_COLUMN].max()

    duplicated_rows = int(dataframe.duplicated().sum())
    duplicated_sensor_rows = int(
        dataframe[sensor_columns].duplicated().sum()
    )

    total_sensor_cells = len(dataframe) * len(sensor_columns)
    total_missing_sensor_values = int(
        dataframe[sensor_columns].isna().sum().sum()
    )

    overall_missing_rate = (
        total_missing_sensor_values / total_sensor_cells
        if total_sensor_cells > 0
        else 0.0
    )

    summary = {
        "rows": int(len(dataframe)),
        "total_columns": int(dataframe.shape[1]),
        "sensor_columns": int(len(sensor_columns)),
        "target_counts": {
            str(key): int(value)
            for key, value in target_counts.items()
        },
        "status_counts": {
            str(key): int(value)
            for key, value in status_counts.items()
        },
        "failure_rate": float(
            dataframe[TARGET_COLUMN].mean()
        ),
        "duplicate_full_rows": duplicated_rows,
        "duplicate_sensor_rows": duplicated_sensor_rows,
        "total_missing_sensor_values": total_missing_sensor_values,
        "overall_sensor_missing_rate": float(overall_missing_rate),
        "timestamp_min": (
            timestamp_min.isoformat()
            if pd.notna(timestamp_min)
            else None
        ),
        "timestamp_max": (
            timestamp_max.isoformat()
            if pd.notna(timestamp_max)
            else None
        ),
    }

    return summary


# ============================================================
# FEATURE MANIFEST
# ============================================================
def build_feature_manifest(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Build a detailed feature-level manifest.

    This table records datatype, missingness, unique values,
    variance, range, and basic distribution statistics.
    """
    rows: list[dict[str, Any]] = []

    for column in sensor_columns:
        series = dataframe[column]

        non_missing = series.dropna()
        unique_count = int(non_missing.nunique())
        missing_count = int(series.isna().sum())
        missing_rate = float(series.isna().mean())

        variance = (
            float(non_missing.var())
            if len(non_missing) > 1
            else np.nan
        )

        row = {
            "feature": column,
            "dtype": str(series.dtype),
            "non_missing_count": int(series.notna().sum()),
            "missing_count": missing_count,
            "missing_rate": missing_rate,
            "unique_count": unique_count,
            "is_constant": bool(unique_count <= 1),
            "variance": variance,
            "minimum": (
                float(non_missing.min())
                if not non_missing.empty
                else np.nan
            ),
            "maximum": (
                float(non_missing.max())
                if not non_missing.empty
                else np.nan
            ),
            "mean": (
                float(non_missing.mean())
                if not non_missing.empty
                else np.nan
            ),
            "median": (
                float(non_missing.median())
                if not non_missing.empty
                else np.nan
            ),
            "standard_deviation": (
                float(non_missing.std())
                if len(non_missing) > 1
                else np.nan
            ),
        }

        rows.append(row)

    manifest = pd.DataFrame(rows)

    manifest["missingness_category"] = pd.cut(
        manifest["missing_rate"],
        bins=[
            -0.001,
            0.00,
            MODERATE_MISSINGNESS_THRESHOLD,
            HIGH_MISSINGNESS_THRESHOLD,
            1.00,
        ],
        labels=[
            "No missingness",
            "Low missingness",
            "Moderate missingness",
            "High missingness",
        ],
    )

    return manifest.sort_values(
        by=["missing_rate", "variance"],
        ascending=[False, True],
    ).reset_index(drop=True)


# ============================================================
# CONSTANT AND NEAR-CONSTANT FEATURES
# ============================================================
def identify_constant_features(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> list[str]:
    """
    Identify features containing one or zero unique non-null values.
    """
    return [
        column
        for column in sensor_columns
        if dataframe[column].nunique(dropna=True) <= 1
    ]


def identify_near_constant_features(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    excluded_features: list[str],
) -> list[str]:
    """
    Identify very low-variance features after temporary median imputation.

    Constant features are excluded because they are reported separately.
    """
    candidate_columns = [
        column
        for column in sensor_columns
        if column not in excluded_features
    ]

    if not candidate_columns:
        return []

    candidate_data = dataframe[candidate_columns].copy()

    candidate_data = candidate_data.fillna(
        candidate_data.median(numeric_only=True)
    )

    selector = VarianceThreshold(
        threshold=NEAR_ZERO_VARIANCE_THRESHOLD
    )

    selector.fit(candidate_data)

    selected_mask = selector.get_support()

    near_constant_features = [
        column
        for column, selected
        in zip(candidate_columns, selected_mask)
        if not selected
    ]

    return near_constant_features


# ============================================================
# MISSINGNESS ANALYSIS
# ============================================================
def build_missingness_report(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Build a feature-level missingness report.
    """
    report = pd.DataFrame(
        {
            "feature": sensor_columns,
            "missing_count": [
                int(dataframe[column].isna().sum())
                for column in sensor_columns
            ],
            "missing_rate": [
                float(dataframe[column].isna().mean())
                for column in sensor_columns
            ],
        }
    )

    report["recommended_action"] = np.select(
        [
            report["missing_rate"] > HIGH_MISSINGNESS_THRESHOLD,
            report["missing_rate"] > MODERATE_MISSINGNESS_THRESHOLD,
            report["missing_rate"] > 0,
        ],
        [
            "Consider dropping",
            "Evaluate carefully before imputation",
            "Median imputation likely acceptable",
        ],
        default="No missing-value treatment required",
    )

    return report.sort_values(
        by="missing_rate",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# CORRELATION ANALYSIS
# ============================================================
def identify_highly_correlated_pairs(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    excluded_features: list[str],
) -> pd.DataFrame:
    """
    Identify absolute Pearson correlations above the configured threshold.

    Median imputation is used only for correlation analysis.
    No fitted values are persisted for modelling.
    """
    candidate_columns = [
        column
        for column in sensor_columns
        if column not in excluded_features
    ]

    if len(candidate_columns) < 2:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    correlation_data = dataframe[candidate_columns].copy()

    correlation_data = correlation_data.fillna(
        correlation_data.median(numeric_only=True)
    )

    correlation_matrix = correlation_data.corr(method="pearson")

    upper_triangle = np.triu(
        np.ones(correlation_matrix.shape),
        k=1,
    ).astype(bool)

    upper_matrix = correlation_matrix.where(upper_triangle)

    correlated_pairs: list[dict[str, Any]] = []

    for feature_2 in upper_matrix.columns:
        matching_features = upper_matrix.index[
            upper_matrix[feature_2].abs() >= CORRELATION_THRESHOLD
        ]

        for feature_1 in matching_features:
            correlation = float(
                upper_matrix.loc[feature_1, feature_2]
            )

            correlated_pairs.append(
                {
                    "feature_1": feature_1,
                    "feature_2": feature_2,
                    "correlation": correlation,
                    "absolute_correlation": abs(correlation),
                }
            )

    if not correlated_pairs:
        return pd.DataFrame(
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
                "absolute_correlation",
            ]
        )

    return (
        pd.DataFrame(correlated_pairs)
        .sort_values(
            by="absolute_correlation",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# ROW-LEVEL QUALITY
# ============================================================
def build_row_quality_report(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
) -> pd.DataFrame:
    """
    Calculate row-level missingness and data completeness.
    """
    report = dataframe[
        [
            ID_COLUMN,
            TIMESTAMP_COLUMN,
            TARGET_COLUMN,
            STATUS_COLUMN,
        ]
    ].copy()

    report["missing_sensor_count"] = (
        dataframe[sensor_columns]
        .isna()
        .sum(axis=1)
        .astype(int)
    )

    report["missing_sensor_rate"] = (
        report["missing_sensor_count"] / len(sensor_columns)
    )

    report["available_sensor_count"] = (
        len(sensor_columns) - report["missing_sensor_count"]
    )

    report["row_quality_flag"] = pd.cut(
        report["missing_sensor_rate"],
        bins=[-0.001, 0.05, 0.15, 0.30, 1.00],
        labels=[
            "Good",
            "Acceptable",
            "Review",
            "Poor",
        ],
    )

    return report.sort_values(
        by="missing_sensor_rate",
        ascending=False,
    ).reset_index(drop=True)


# ============================================================
# OUTLIER ANALYSIS
# ============================================================
def build_outlier_report(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    excluded_features: list[str],
) -> pd.DataFrame:
    """
    Detect univariate outliers using the IQR rule.

    This is an audit report only. Values are not removed or modified.
    """
    candidate_columns = [
        column
        for column in sensor_columns
        if column not in excluded_features
    ]

    report_rows: list[dict[str, Any]] = []

    for column in candidate_columns:
        series = dataframe[column].dropna()

        if series.empty:
            continue

        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1

        lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
        upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr

        outlier_mask = (
            (series < lower_bound)
            | (series > upper_bound)
        )

        outlier_count = int(outlier_mask.sum())
        outlier_rate = float(outlier_count / len(series))

        report_rows.append(
            {
                "feature": column,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "outlier_count": outlier_count,
                "outlier_rate": outlier_rate,
            }
        )

    if not report_rows:
        return pd.DataFrame(
            columns=[
                "feature",
                "q1",
                "q3",
                "iqr",
                "lower_bound",
                "upper_bound",
                "outlier_count",
                "outlier_rate",
            ]
        )

    return (
        pd.DataFrame(report_rows)
        .sort_values(
            by="outlier_rate",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# ============================================================
# AUDITED DATASET
# ============================================================
def build_audited_dataset(
    dataframe: pd.DataFrame,
    constant_features: list[str],
    high_missing_features: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Remove only clearly unusable features:

    1. Constant features.
    2. Features with more than 60% missing values.

    Near-constant and correlated features are retained for now.
    Those decisions should be fitted within the later modelling pipeline.
    """
    features_to_drop = sorted(
        set(constant_features + high_missing_features)
    )

    audited_dataframe = dataframe.drop(
        columns=features_to_drop,
        errors="ignore",
    ).copy()

    return audited_dataframe, features_to_drop


# ============================================================
# DATA SPLITTING
# ============================================================
def create_stratified_splits(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create reproducible stratified train, validation, and test splits.

    Approximate proportions:
    - Train: 64%
    - Validation: 16%
    - Test: 20%

    This split is suitable for baseline modelling. A later experiment
    should also evaluate a chronological split because timestamps exist.
    """
    development_df, test_df = train_test_split(
        dataframe,
        test_size=TEST_SIZE,
        stratify=dataframe[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    train_df, validation_df = train_test_split(
        development_df,
        test_size=VALIDATION_SIZE_WITHIN_TRAIN,
        stratify=development_df[TARGET_COLUMN],
        random_state=RANDOM_STATE,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def build_split_summary(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize the class distribution in each dataset split.
    """
    rows = []

    for split_name, split_df in [
        ("train", train_df),
        ("validation", validation_df),
        ("test", test_df),
    ]:
        total_rows = len(split_df)
        pass_count = int(
            (split_df[TARGET_COLUMN] == 0).sum()
        )
        fail_count = int(
            (split_df[TARGET_COLUMN] == 1).sum()
        )

        rows.append(
            {
                "split": split_name,
                "rows": total_rows,
                "pass_count": pass_count,
                "fail_count": fail_count,
                "fail_rate": (
                    fail_count / total_rows
                    if total_rows > 0
                    else 0.0
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# VISUALISATIONS
# ============================================================
def save_target_distribution_plot(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save the pass/fail class distribution plot.
    """
    counts = (
        dataframe[STATUS_COLUMN]
        .value_counts()
        .reindex(["Pass", "Fail"])
    )

    figure, axis = plt.subplots(figsize=(8, 5))
    counts.plot(kind="bar", ax=axis)

    axis.set_title("SECOM Pass and Fail Distribution")
    axis.set_xlabel("Manufacturing Outcome")
    axis.set_ylabel("Number of Records")
    axis.tick_params(axis="x", rotation=0)

    for index, value in enumerate(counts.values):
        axis.text(
            index,
            value,
            f"{int(value):,}",
            ha="center",
            va="bottom",
        )

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "target_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_missingness_plot(
    missingness_report: pd.DataFrame,
) -> None:
    """
    Save a plot containing the 30 sensors with the highest missingness.
    """
    top_missing = missingness_report.head(30).copy()
    top_missing = top_missing.sort_values(
        by="missing_rate",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 9))

    axis.barh(
        top_missing["feature"],
        top_missing["missing_rate"] * 100,
    )

    axis.set_title("Top 30 Sensors by Missing-Value Rate")
    axis.set_xlabel("Missing Values (%)")
    axis.set_ylabel("Sensor Feature")

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "top_missingness.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_row_missingness_plot(
    row_quality_report: pd.DataFrame,
) -> None:
    """
    Save the distribution of missing sensor counts by row.
    """
    figure, axis = plt.subplots(figsize=(9, 5))

    axis.hist(
        row_quality_report["missing_sensor_count"],
        bins=30,
    )

    axis.set_title("Distribution of Missing Sensor Values per Record")
    axis.set_xlabel("Number of Missing Sensor Values")
    axis.set_ylabel("Number of Records")

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "row_missingness_distribution.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_failure_trend_plot(
    dataframe: pd.DataFrame,
) -> None:
    """
    Save a daily pass/fail trend plot.
    """
    trend_data = dataframe.copy()

    trend_data["production_date"] = (
        trend_data[TIMESTAMP_COLUMN].dt.date
    )

    daily_summary = (
        trend_data
        .groupby("production_date", as_index=False)
        .agg(
            total_records=(TARGET_COLUMN, "size"),
            failures=(TARGET_COLUMN, "sum"),
        )
    )

    daily_summary["failure_rate"] = (
        daily_summary["failures"]
        / daily_summary["total_records"]
    )

    figure, axis = plt.subplots(figsize=(12, 5))

    axis.plot(
        daily_summary["production_date"],
        daily_summary["failure_rate"] * 100,
        marker="o",
    )

    axis.set_title("Daily Failure Rate")
    axis.set_xlabel("Production Date")
    axis.set_ylabel("Failure Rate (%)")
    axis.tick_params(axis="x", rotation=45)

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "daily_failure_rate.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_outlier_plot(
    outlier_report: pd.DataFrame,
) -> None:
    """
    Save the 30 sensors with the highest univariate outlier rates.
    """
    if outlier_report.empty:
        return

    top_outliers = outlier_report.head(30).copy()
    top_outliers = top_outliers.sort_values(
        by="outlier_rate",
        ascending=True,
    )

    figure, axis = plt.subplots(figsize=(10, 9))

    axis.barh(
        top_outliers["feature"],
        top_outliers["outlier_rate"] * 100,
    )

    axis.set_title("Top 30 Sensors by IQR Outlier Rate")
    axis.set_xlabel("Outlier Rate (%)")
    axis.set_ylabel("Sensor Feature")

    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / "top_outlier_rates.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


# ============================================================
# SAVE OUTPUTS
# ============================================================
def save_tables(
    feature_manifest: pd.DataFrame,
    missingness_report: pd.DataFrame,
    constant_features: list[str],
    near_constant_features: list[str],
    correlated_pairs: pd.DataFrame,
    row_quality_report: pd.DataFrame,
    outlier_report: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    """
    Save all audit tables.
    """
    feature_manifest.to_csv(
        FEATURE_MANIFEST_PATH,
        index=False,
    )

    missingness_report.to_csv(
        MISSINGNESS_REPORT_PATH,
        index=False,
    )

    pd.DataFrame(
        {"feature": constant_features}
    ).to_csv(
        CONSTANT_FEATURES_PATH,
        index=False,
    )

    pd.DataFrame(
        {"feature": near_constant_features}
    ).to_csv(
        NEAR_CONSTANT_FEATURES_PATH,
        index=False,
    )

    correlated_pairs.to_csv(
        CORRELATED_PAIRS_PATH,
        index=False,
    )

    row_quality_report.to_csv(
        ROW_QUALITY_PATH,
        index=False,
    )

    outlier_report.to_csv(
        OUTLIER_REPORT_PATH,
        index=False,
    )

    split_summary.to_csv(
        SPLIT_SUMMARY_PATH,
        index=False,
    )


def save_audited_dataset(
    audited_dataframe: pd.DataFrame,
) -> None:
    """
    Save the audited dataset as CSV and Parquet.
    """
    audited_dataframe.to_csv(
        AUDITED_CSV_PATH,
        index=False,
    )

    audited_dataframe.to_parquet(
        AUDITED_PARQUET_PATH,
        index=False,
    )


def save_dataset_splits(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """
    Save train, validation, and test datasets.
    """
    split_mapping = {
        "train": train_df,
        "validation": validation_df,
        "test": test_df,
    }

    for split_name, split_df in split_mapping.items():
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        parquet_path = SPLITS_DIR / f"{split_name}.parquet"

        split_df.to_csv(
            csv_path,
            index=False,
        )

        split_df.to_parquet(
            parquet_path,
            index=False,
        )


def save_audit_summary(
    basic_summary: dict[str, Any],
    constant_features: list[str],
    near_constant_features: list[str],
    high_missing_features: list[str],
    highly_correlated_pairs: pd.DataFrame,
    dropped_features: list[str],
    audited_dataframe: pd.DataFrame,
) -> None:
    """
    Save the complete audit summary as JSON.
    """
    summary = {
        **basic_summary,
        "constant_feature_count": len(constant_features),
        "near_constant_feature_count": len(
            near_constant_features
        ),
        "high_missing_feature_count": len(
            high_missing_features
        ),
        "high_correlation_pair_count": int(
            len(highly_correlated_pairs)
        ),
        "dropped_feature_count": len(dropped_features),
        "dropped_features": dropped_features,
        "audited_rows": int(len(audited_dataframe)),
        "audited_columns": int(
            audited_dataframe.shape[1]
        ),
        "audited_sensor_columns": int(
            len(get_sensor_columns(audited_dataframe))
        ),
        "configuration": {
            "high_missingness_threshold": (
                HIGH_MISSINGNESS_THRESHOLD
            ),
            "moderate_missingness_threshold": (
                MODERATE_MISSINGNESS_THRESHOLD
            ),
            "near_zero_variance_threshold": (
                NEAR_ZERO_VARIANCE_THRESHOLD
            ),
            "correlation_threshold": (
                CORRELATION_THRESHOLD
            ),
            "outlier_iqr_multiplier": (
                OUTLIER_IQR_MULTIPLIER
            ),
            "random_state": RANDOM_STATE,
        },
    }

    with AUDIT_SUMMARY_JSON_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# CONSOLE SUMMARY
# ============================================================
def print_console_summary(
    dataframe: pd.DataFrame,
    sensor_columns: list[str],
    constant_features: list[str],
    near_constant_features: list[str],
    high_missing_features: list[str],
    correlated_pairs: pd.DataFrame,
    dropped_features: list[str],
    audited_dataframe: pd.DataFrame,
    split_summary: pd.DataFrame,
) -> None:
    """
    Print a concise audit summary to the terminal.
    """
    print("\n" + "=" * 76)
    print("HEVEMIND SECOM DATA AUDIT SUMMARY")
    print("=" * 76)

    print(f"Original rows:                 {len(dataframe):,}")
    print(f"Original sensor features:      {len(sensor_columns):,}")
    print(f"Constant sensor features:      {len(constant_features):,}")
    print(
        f"Near-constant sensor features: "
        f"{len(near_constant_features):,}"
    )
    print(
        f"Features with >60% missing:    "
        f"{len(high_missing_features):,}"
    )
    print(
        f"Correlation pairs >= 0.95:     "
        f"{len(correlated_pairs):,}"
    )
    print(f"Features removed in audit:     {len(dropped_features):,}")
    print(
        f"Remaining sensor features:     "
        f"{len(get_sensor_columns(audited_dataframe)):,}"
    )

    print("\nTarget distribution:")
    print(
        dataframe[STATUS_COLUMN]
        .value_counts()
        .to_string()
    )

    print("\nDataset split summary:")
    print(
        split_summary.to_string(index=False)
    )

    print("\nSaved outputs:")
    print(f"Audited CSV:       {AUDITED_CSV_PATH}")
    print(f"Audited Parquet:   {AUDITED_PARQUET_PATH}")
    print(f"Audit report:      {DATA_AUDIT_DIR}")
    print(f"Dataset splits:    {SPLITS_DIR}")


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    create_directories()

    dataframe = load_master_dataset()
    validate_dataset(dataframe)

    sensor_columns = get_sensor_columns(dataframe)

    LOGGER.info("Calculating dataset summary")
    basic_summary = calculate_basic_summary(
        dataframe,
        sensor_columns,
    )

    LOGGER.info("Building feature manifest")
    feature_manifest = build_feature_manifest(
        dataframe,
        sensor_columns,
    )

    LOGGER.info("Identifying constant features")
    constant_features = identify_constant_features(
        dataframe,
        sensor_columns,
    )

    LOGGER.info("Identifying near-constant features")
    near_constant_features = identify_near_constant_features(
        dataframe,
        sensor_columns,
        excluded_features=constant_features,
    )

    LOGGER.info("Building missingness report")
    missingness_report = build_missingness_report(
        dataframe,
        sensor_columns,
    )

    high_missing_features = (
        missingness_report.loc[
            missingness_report["missing_rate"]
            > HIGH_MISSINGNESS_THRESHOLD,
            "feature",
        ]
        .tolist()
    )

    correlation_exclusions = sorted(
        set(
            constant_features
            + high_missing_features
        )
    )

    LOGGER.info("Identifying highly correlated features")
    correlated_pairs = identify_highly_correlated_pairs(
        dataframe,
        sensor_columns,
        excluded_features=correlation_exclusions,
    )

    LOGGER.info("Building row-level quality report")
    row_quality_report = build_row_quality_report(
        dataframe,
        sensor_columns,
    )

    LOGGER.info("Building outlier report")
    outlier_report = build_outlier_report(
        dataframe,
        sensor_columns,
        excluded_features=constant_features,
    )

    LOGGER.info("Building audited dataset")
    audited_dataframe, dropped_features = (
        build_audited_dataset(
            dataframe,
            constant_features=constant_features,
            high_missing_features=high_missing_features,
        )
    )

    LOGGER.info("Creating stratified dataset splits")
    train_df, validation_df, test_df = (
        create_stratified_splits(audited_dataframe)
    )

    split_summary = build_split_summary(
        train_df,
        validation_df,
        test_df,
    )

    LOGGER.info("Saving audit tables")
    save_tables(
        feature_manifest=feature_manifest,
        missingness_report=missingness_report,
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        correlated_pairs=correlated_pairs,
        row_quality_report=row_quality_report,
        outlier_report=outlier_report,
        split_summary=split_summary,
    )

    LOGGER.info("Saving audited dataset")
    save_audited_dataset(audited_dataframe)

    LOGGER.info("Saving dataset splits")
    save_dataset_splits(
        train_df,
        validation_df,
        test_df,
    )

    LOGGER.info("Saving audit summary")
    save_audit_summary(
        basic_summary=basic_summary,
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        high_missing_features=high_missing_features,
        highly_correlated_pairs=correlated_pairs,
        dropped_features=dropped_features,
        audited_dataframe=audited_dataframe,
    )

    LOGGER.info("Generating figures")
    save_target_distribution_plot(dataframe)
    save_missingness_plot(missingness_report)
    save_row_missingness_plot(row_quality_report)
    save_failure_trend_plot(dataframe)
    save_outlier_plot(outlier_report)

    print_console_summary(
        dataframe=dataframe,
        sensor_columns=sensor_columns,
        constant_features=constant_features,
        near_constant_features=near_constant_features,
        high_missing_features=high_missing_features,
        correlated_pairs=correlated_pairs,
        dropped_features=dropped_features,
        audited_dataframe=audited_dataframe,
        split_summary=split_summary,
    )


if __name__ == "__main__":
    main()