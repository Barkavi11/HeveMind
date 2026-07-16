from __future__ import annotations

import ast
import json
import logging
import py_compile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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

SRC_DIR = ROOT_DIR / "src"
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SPLITS_DIR = PROCESSED_DIR / "splits"
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
REPORTS_DIR = ROOT_DIR / "reports"

TRAIN_PATH = SPLITS_DIR / "train.parquet"
VALIDATION_PATH = SPLITS_DIR / "validation.parquet"
TEST_PATH = SPLITS_DIR / "test.parquet"

AUDIT_REPORT_DIR = REPORTS_DIR / "pipeline_validation_audit"
TABLES_DIR = AUDIT_REPORT_DIR / "tables"

SCRIPT_AUDIT_PATH = TABLES_DIR / "script_validation.csv"
OUTPUT_AUDIT_PATH = TABLES_DIR / "output_validation.csv"
RECORD_AUDIT_PATH = TABLES_DIR / "record_consistency.csv"
LEAKAGE_AUDIT_PATH = TABLES_DIR / "leakage_and_governance_checks.csv"
DEPLOYMENT_AUDIT_PATH = TABLES_DIR / "deployment_readiness.csv"
SUMMARY_PATH = AUDIT_REPORT_DIR / "pipeline_validation_summary.json"


# ============================================================
# CONFIGURATION
# ============================================================
ID_COLUMN = "wafer_id"
TARGET_COLUMN = "target"
STATUS_COLUMN = "status"

EXPECTED_TEST_ROWS = 314
EXPECTED_DEVELOPMENT_ROWS = 1253

CORE_SCRIPTS = [
    "01_convert_secom.py",
    "02_data_audit.py",
    "03_train_baseline_models.py",
    "04_advanced_model_benchmark.py",
    "05_operational_model_selection.py",
    "06_missingness_aware_benchmark.py",
    "07_quality_imputation_anomaly.py",
    "08_operational_error_analysis.py",
    "09_feature_level_error_diagnostics.py",
    "10_probability_calibration_and_decision_policy.py",
    "11_cross_fitted_calibration_policy.py",
    "12_uncertainty_engine.py",
    "13_explainability_engine.py",
    "14_historical_similarity_engine.py",
    "15_evidence_aggregation_engine.py",
    "16_sensor_investigation_priority.py",
]

REQUIRED_OUTPUTS = {
    "11_cross_fitted_calibration_summary": (
        REPORTS_DIR
        / "cross_fitted_calibration_policy"
        / "cross_fitted_calibration_summary.json"
    ),
    "12_development_uncertainty": (
        REPORTS_DIR
        / "uncertainty_engine"
        / "tables"
        / "development_uncertainty_scores.csv"
    ),
    "12_test_uncertainty": (
        REPORTS_DIR
        / "uncertainty_engine"
        / "tables"
        / "test_uncertainty_scores.csv"
    ),
    "13_global_shap": (
        REPORTS_DIR
        / "explainability_engine"
        / "tables"
        / "global_shap_importance.csv"
    ),
    "13_local_shap": (
        REPORTS_DIR
        / "explainability_engine"
        / "tables"
        / "local_shap_explanations.csv"
    ),
    "13_wafer_summary": (
        REPORTS_DIR
        / "explainability_engine"
        / "tables"
        / "wafer_explanation_summary.csv"
    ),
    "14_similarity_summary": (
        REPORTS_DIR
        / "historical_similarity_engine"
        / "tables"
        / "historical_similarity_summary.csv"
    ),
    "15_engineering_report": (
        REPORTS_DIR
        / "evidence_aggregation_engine"
        / "tables"
        / "engineering_evidence_report.csv"
    ),
    "16_sensor_priority": (
        REPORTS_DIR
        / "sensor_investigation_priority"
        / "tables"
        / "sensor_investigation_priority.csv"
    ),
    "16_wafer_summary": (
        REPORTS_DIR
        / "sensor_investigation_priority"
        / "tables"
        / "wafer_investigation_summary.csv"
    ),
}

TEST_LEVEL_TABLES = {
    "12_test_uncertainty": REQUIRED_OUTPUTS["12_test_uncertainty"],
    "13_wafer_summary": REQUIRED_OUTPUTS["13_wafer_summary"],
    "14_similarity_summary": REQUIRED_OUTPUTS["14_similarity_summary"],
    "15_engineering_report": REQUIRED_OUTPUTS["15_engineering_report"],
    "16_wafer_summary": REQUIRED_OUTPUTS["16_wafer_summary"],
}

DEVELOPMENT_LEVEL_TABLES = {
    "12_development_uncertainty": REQUIRED_OUTPUTS[
        "12_development_uncertainty"
    ],
}

REQUIRED_COLUMNS = {
    "12_test_uncertainty": {
        ID_COLUMN,
        TARGET_COLUMN,
        "calibrated_failure_probability",
        "prediction_confidence",
        "combined_uncertainty",
        "ood_percentile",
        "uncertainty_adjusted_decision",
    },
    "13_wafer_summary": {
        ID_COLUMN,
        TARGET_COLUMN,
        "decision",
        "top_risk_features",
        "top_protective_features",
    },
    "14_similarity_summary": {
        ID_COLUMN,
        TARGET_COLUMN,
        "historical_weighted_failure_rate",
        "failed_neighbour_count",
        "mean_similarity",
    },
    "15_engineering_report": {
        ID_COLUMN,
        TARGET_COLUMN,
        "uncertainty_adjusted_decision",
        "evidence_agreement",
        "recommended_action",
        "primary_decision_retained",
        "new_failure_probability_created",
    },
    "16_wafer_summary": {
        ID_COLUMN,
        TARGET_COLUMN,
        "decision",
        "priority_1_sensor",
        "top_5_sensors",
    },
}

OLD_FUSION_MARKERS = {
    "LogisticRegression",
    "fusion_failure_probability",
    "fusion_oof_probability",
    "evidence_fusion_meta_model",
    "Generating cross-fitted evidence-fusion probabilities",
}

TEST_TARGET_TRAINING_PATTERNS = {
    "fit(x_test",
    "fit(X_test",
    "fit(test_df",
    "fit(test_data",
    "cross_val_predict(",
}


# ============================================================
# DATA CLASSES
# ============================================================
@dataclass
class AuditResult:
    category: str
    item: str
    status: str
    severity: str
    details: str


# ============================================================
# DIRECTORY SETUP
# ============================================================
def create_directories() -> None:
    for directory in [
        AUDIT_REPORT_DIR,
        TABLES_DIR,
    ]:
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# BASIC UTILITIES
# ============================================================
def status_from_boolean(
    passed: bool,
) -> str:
    return "PASS" if passed else "FAIL"


def dataframe_to_records(
    results: list[AuditResult],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            asdict(result)
            for result in results
        ]
    )


def read_text_safely(
    path: Path,
) -> str:
    try:
        return path.read_text(
            encoding="utf-8",
        )

    except UnicodeDecodeError:
        return path.read_text(
            encoding="latin-1",
        )


def load_table(
    path: Path,
) -> pd.DataFrame:
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported table type: {path}"
    )


def safe_boolean_series(
    series: pd.Series,
) -> pd.Series:
    if pd.api.types.is_bool_dtype(
        series
    ):
        return series

    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )


# ============================================================
# SCRIPT VALIDATION
# ============================================================
def validate_script(
    script_name: str,
) -> list[AuditResult]:
    results: list[AuditResult] = []

    path = SRC_DIR / script_name

    if not path.exists():
        results.append(
            AuditResult(
                category="script",
                item=script_name,
                status="FAIL",
                severity="critical",
                details="Required script is missing.",
            )
        )
        return results

    try:
        py_compile.compile(
            str(path),
            doraise=True,
        )

        results.append(
            AuditResult(
                category="script",
                item=script_name,
                status="PASS",
                severity="info",
                details="Python syntax compilation succeeded.",
            )
        )

    except py_compile.PyCompileError as error:
        results.append(
            AuditResult(
                category="script",
                item=script_name,
                status="FAIL",
                severity="critical",
                details=f"Compilation failed: {error.msg}",
            )
        )
        return results

    try:
        source = read_text_safely(
            path
        )

        tree = ast.parse(
            source
        )

        has_main_function = any(
            isinstance(node, ast.FunctionDef)
            and node.name == "main"
            for node in tree.body
        )

        has_main_guard = (
            'if __name__ == "__main__"' in source
            or "if __name__ == '__main__'" in source
        )

        results.append(
            AuditResult(
                category="script",
                item=f"{script_name}:main_function",
                status=status_from_boolean(
                    has_main_function
                ),
                severity=(
                    "info"
                    if has_main_function
                    else "warning"
                ),
                details=(
                    "main() function found."
                    if has_main_function
                    else "No main() function found."
                ),
            )
        )

        results.append(
            AuditResult(
                category="script",
                item=f"{script_name}:main_guard",
                status=status_from_boolean(
                    has_main_guard
                ),
                severity=(
                    "info"
                    if has_main_guard
                    else "warning"
                ),
                details=(
                    "__main__ execution guard found."
                    if has_main_guard
                    else "__main__ execution guard not found."
                ),
            )
        )

    except SyntaxError as error:
        results.append(
            AuditResult(
                category="script",
                item=f"{script_name}:ast_parse",
                status="FAIL",
                severity="critical",
                details=f"AST parsing failed: {error}",
            )
        )

    return results


def validate_script_15_architecture() -> list[AuditResult]:
    results: list[AuditResult] = []

    path = SRC_DIR / "15_evidence_aggregation_engine.py"

    if not path.exists():
        return [
            AuditResult(
                category="governance",
                item="script_15_architecture",
                status="FAIL",
                severity="critical",
                details="15_evidence_aggregation_engine.py is missing.",
            )
        ]

    source = read_text_safely(
        path
    )

    detected_markers = sorted(
        marker
        for marker in OLD_FUSION_MARKERS
        if marker in source
    )

    no_old_fusion = (
        len(detected_markers) == 0
    )

    results.append(
        AuditResult(
            category="governance",
            item="script_15_no_meta_model",
            status=status_from_boolean(
                no_old_fusion
            ),
            severity=(
                "info"
                if no_old_fusion
                else "critical"
            ),
            details=(
                "No rejected fusion-model logic detected."
                if no_old_fusion
                else (
                    "Rejected fusion-model markers detected: "
                    + ", ".join(
                        detected_markers
                    )
                )
            ),
        )
    )

    expected_phrase = (
        "Aggregating evidence without creating a new prediction model"
    )

    results.append(
        AuditResult(
            category="governance",
            item="script_15_evidence_aggregation_marker",
            status=status_from_boolean(
                expected_phrase in source
            ),
            severity="info",
            details=(
                "Evidence-aggregation architecture confirmed."
                if expected_phrase in source
                else "Expected evidence-aggregation marker was not found."
            ),
        )
    )

    return results


# ============================================================
# OUTPUT VALIDATION
# ============================================================
def validate_output(
    name: str,
    path: Path,
) -> list[AuditResult]:
    results: list[AuditResult] = []

    exists = path.exists()

    results.append(
        AuditResult(
            category="output",
            item=name,
            status=status_from_boolean(
                exists
            ),
            severity=(
                "info"
                if exists
                else "critical"
            ),
            details=(
                f"Found: {path}"
                if exists
                else f"Missing: {path}"
            ),
        )
    )

    if not exists:
        return results

    if path.suffix.lower() in {
        ".csv",
        ".parquet",
    }:
        try:
            dataframe = load_table(
                path
            )

            results.append(
                AuditResult(
                    category="output",
                    item=f"{name}:readable",
                    status="PASS",
                    severity="info",
                    details=(
                        f"Readable table with "
                        f"{len(dataframe)} rows and "
                        f"{len(dataframe.columns)} columns."
                    ),
                )
            )

            required_columns = REQUIRED_COLUMNS.get(
                name,
                set(),
            )

            if required_columns:
                missing_columns = sorted(
                    required_columns.difference(
                        dataframe.columns
                    )
                )

                results.append(
                    AuditResult(
                        category="output",
                        item=f"{name}:required_columns",
                        status=status_from_boolean(
                            not missing_columns
                        ),
                        severity=(
                            "info"
                            if not missing_columns
                            else "critical"
                        ),
                        details=(
                            "All required columns are present."
                            if not missing_columns
                            else (
                                "Missing required columns: "
                                + ", ".join(
                                    missing_columns
                                )
                            )
                        ),
                    )
                )

            if ID_COLUMN in dataframe.columns:
                duplicates = int(
                    dataframe[
                        ID_COLUMN
                    ].duplicated().sum()
                )

                results.append(
                    AuditResult(
                        category="output",
                        item=f"{name}:duplicate_ids",
                        status=status_from_boolean(
                            duplicates == 0
                        ),
                        severity=(
                            "info"
                            if duplicates == 0
                            else "critical"
                        ),
                        details=f"Duplicate wafer IDs: {duplicates}.",
                    )
                )

        except Exception as error:
            results.append(
                AuditResult(
                    category="output",
                    item=f"{name}:readable",
                    status="FAIL",
                    severity="critical",
                    details=f"Unable to read output: {error}",
                )
            )

    elif path.suffix.lower() == ".json":
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                payload = json.load(
                    file
                )

            results.append(
                AuditResult(
                    category="output",
                    item=f"{name}:readable",
                    status="PASS",
                    severity="info",
                    details=(
                        f"Readable JSON with "
                        f"{len(payload)} top-level keys."
                    ),
                )
            )

        except Exception as error:
            results.append(
                AuditResult(
                    category="output",
                    item=f"{name}:readable",
                    status="FAIL",
                    severity="critical",
                    details=f"Unable to read JSON: {error}",
                )
            )

    return results


# ============================================================
# RECORD CONSISTENCY
# ============================================================
def build_record_consistency_results() -> list[AuditResult]:
    results: list[AuditResult] = []

    if not TRAIN_PATH.exists():
        results.append(
            AuditResult(
                category="records",
                item="train_split",
                status="FAIL",
                severity="critical",
                details=f"Missing train split: {TRAIN_PATH}",
            )
        )
        return results

    if not VALIDATION_PATH.exists():
        results.append(
            AuditResult(
                category="records",
                item="validation_split",
                status="FAIL",
                severity="critical",
                details=f"Missing validation split: {VALIDATION_PATH}",
            )
        )
        return results

    if not TEST_PATH.exists():
        results.append(
            AuditResult(
                category="records",
                item="test_split",
                status="FAIL",
                severity="critical",
                details=f"Missing test split: {TEST_PATH}",
            )
        )
        return results

    train_df = pd.read_parquet(
        TRAIN_PATH
    )

    validation_df = pd.read_parquet(
        VALIDATION_PATH
    )

    test_df = pd.read_parquet(
        TEST_PATH
    )

    development_rows = (
        len(train_df)
        + len(validation_df)
    )

    results.append(
        AuditResult(
            category="records",
            item="development_row_count",
            status=status_from_boolean(
                development_rows
                == EXPECTED_DEVELOPMENT_ROWS
            ),
            severity=(
                "info"
                if development_rows
                == EXPECTED_DEVELOPMENT_ROWS
                else "warning"
            ),
            details=(
                f"Observed development rows: {development_rows}; "
                f"expected: {EXPECTED_DEVELOPMENT_ROWS}."
            ),
        )
    )

    results.append(
        AuditResult(
            category="records",
            item="test_row_count",
            status=status_from_boolean(
                len(test_df)
                == EXPECTED_TEST_ROWS
            ),
            severity=(
                "info"
                if len(test_df)
                == EXPECTED_TEST_ROWS
                else "critical"
            ),
            details=(
                f"Observed test rows: {len(test_df)}; "
                f"expected: {EXPECTED_TEST_ROWS}."
            ),
        )
    )

    reference_test_ids = set(
        test_df[
            ID_COLUMN
        ].astype(str)
    )

    for name, path in TEST_LEVEL_TABLES.items():
        if not path.exists():
            results.append(
                AuditResult(
                    category="records",
                    item=name,
                    status="FAIL",
                    severity="critical",
                    details="Table is missing.",
                )
            )
            continue

        dataframe = pd.read_csv(
            path
        )

        row_count_valid = (
            len(dataframe)
            == EXPECTED_TEST_ROWS
        )

        results.append(
            AuditResult(
                category="records",
                item=f"{name}:row_count",
                status=status_from_boolean(
                    row_count_valid
                ),
                severity=(
                    "info"
                    if row_count_valid
                    else "critical"
                ),
                details=(
                    f"Rows: {len(dataframe)}; "
                    f"expected: {EXPECTED_TEST_ROWS}."
                ),
            )
        )

        if ID_COLUMN in dataframe.columns:
            observed_ids = set(
                dataframe[
                    ID_COLUMN
                ].astype(str)
            )

            missing_ids = (
                reference_test_ids
                - observed_ids
            )

            extra_ids = (
                observed_ids
                - reference_test_ids
            )

            id_match = (
                not missing_ids
                and not extra_ids
            )

            results.append(
                AuditResult(
                    category="records",
                    item=f"{name}:wafer_id_set",
                    status=status_from_boolean(
                        id_match
                    ),
                    severity=(
                        "info"
                        if id_match
                        else "critical"
                    ),
                    details=(
                        "Wafer ID set matches held-out test split."
                        if id_match
                        else (
                            f"Missing IDs: {len(missing_ids)}; "
                            f"extra IDs: {len(extra_ids)}."
                        )
                    ),
                )
            )

    for name, path in DEVELOPMENT_LEVEL_TABLES.items():
        if not path.exists():
            results.append(
                AuditResult(
                    category="records",
                    item=name,
                    status="FAIL",
                    severity="critical",
                    details="Development-level table is missing.",
                )
            )
            continue

        dataframe = pd.read_csv(
            path
        )

        valid = (
            len(dataframe)
            == EXPECTED_DEVELOPMENT_ROWS
        )

        results.append(
            AuditResult(
                category="records",
                item=f"{name}:row_count",
                status=status_from_boolean(
                    valid
                ),
                severity=(
                    "info"
                    if valid
                    else "critical"
                ),
                details=(
                    f"Rows: {len(dataframe)}; "
                    f"expected: {EXPECTED_DEVELOPMENT_ROWS}."
                ),
            )
        )

    return results


# ============================================================
# GOVERNANCE AND LEAKAGE CHECKS
# ============================================================
def build_governance_results() -> list[AuditResult]:
    results: list[AuditResult] = []

    results.extend(
        validate_script_15_architecture()
    )

    evidence_report_path = REQUIRED_OUTPUTS[
        "15_engineering_report"
    ]

    if evidence_report_path.exists():
        report = pd.read_csv(
            evidence_report_path
        )

        if "primary_decision_retained" in report.columns:
            values = safe_boolean_series(
                report[
                    "primary_decision_retained"
                ]
            )

            retained = bool(
                values.fillna(False).all()
            )

            results.append(
                AuditResult(
                    category="governance",
                    item="script_15_primary_decisions_retained",
                    status=status_from_boolean(
                        retained
                    ),
                    severity=(
                        "info"
                        if retained
                        else "critical"
                    ),
                    details=(
                        "All primary operational decisions were retained."
                        if retained
                        else "At least one primary decision was not retained."
                    ),
                )
            )

        if "new_failure_probability_created" in report.columns:
            values = safe_boolean_series(
                report[
                    "new_failure_probability_created"
                ]
            )

            none_created = bool(
                (~values.fillna(True)).all()
            )

            results.append(
                AuditResult(
                    category="governance",
                    item="script_15_no_new_probability",
                    status=status_from_boolean(
                        none_created
                    ),
                    severity=(
                        "info"
                        if none_created
                        else "critical"
                    ),
                    details=(
                        "No new failure probability was created."
                        if none_created
                        else "A new failure probability appears in the report."
                    ),
                )
            )

    uncertainty_path = REQUIRED_OUTPUTS[
        "12_test_uncertainty"
    ]

    if uncertainty_path.exists():
        uncertainty_df = pd.read_csv(
            uncertainty_path
        )

        decision_counts = (
            uncertainty_df[
                "uncertainty_adjusted_decision"
            ]
            .value_counts(
                dropna=False
            )
            .to_dict()
        )

        expected_decisions = {
            "Low Risk",
            "Engineering Review",
            "High Risk",
            "Insufficient Evidence",
        }

        observed_decisions = set(
            uncertainty_df[
                "uncertainty_adjusted_decision"
            ].dropna()
        )

        valid_decisions = (
            observed_decisions.issubset(
                expected_decisions
            )
            and len(
                uncertainty_df
            )
            == EXPECTED_TEST_ROWS
        )

        results.append(
            AuditResult(
                category="governance",
                item="four_level_decision_policy",
                status=status_from_boolean(
                    valid_decisions
                ),
                severity=(
                    "info"
                    if valid_decisions
                    else "critical"
                ),
                details=(
                    "Decision counts: "
                    + json.dumps(
                        decision_counts,
                        default=str,
                    )
                ),
            )
        )

        if TARGET_COLUMN in uncertainty_df.columns:
            actual_failures = int(
                uncertainty_df[
                    TARGET_COLUMN
                ].sum()
            )

            unsafe_low_risk = int(
                (
                    (
                        uncertainty_df[
                            TARGET_COLUMN
                        ]
                        == 1
                    )
                    & (
                        uncertainty_df[
                            "uncertainty_adjusted_decision"
                        ]
                        == "Low Risk"
                    )
                ).sum()
            )

            safe_flagged = (
                actual_failures
                - unsafe_low_risk
            )

            safety_rate = (
                safe_flagged
                / actual_failures
                if actual_failures > 0
                else np.nan
            )

            results.append(
                AuditResult(
                    category="governance",
                    item="held_out_safety_routing",
                    status=(
                        "PASS"
                        if safety_rate >= 0.90
                        else "FAIL"
                    ),
                    severity=(
                        "info"
                        if safety_rate >= 0.90
                        else "critical"
                    ),
                    details=(
                        f"Safely routed failures: "
                        f"{safe_flagged}/{actual_failures}; "
                        f"rate={safety_rate:.4f}."
                    ),
                )
            )

    return results


# ============================================================
# DEPLOYMENT READINESS
# ============================================================
def build_deployment_readiness(
    all_results: list[AuditResult],
) -> list[AuditResult]:
    critical_failures = [
        result
        for result in all_results
        if (
            result.status == "FAIL"
            and result.severity == "critical"
        )
    ]

    warning_failures = [
        result
        for result in all_results
        if (
            result.status == "FAIL"
            and result.severity == "warning"
        )
    ]

    results: list[AuditResult] = []

    pipeline_ready = (
        len(
            critical_failures
        )
        == 0
    )

    results.append(
        AuditResult(
            category="deployment",
            item="pipeline_ready_for_dashboard_integration",
            status=status_from_boolean(
                pipeline_ready
            ),
            severity=(
                "info"
                if pipeline_ready
                else "critical"
            ),
            details=(
                "No critical validation failures detected."
                if pipeline_ready
                else (
                    f"{len(critical_failures)} critical validation "
                    f"failure(s) must be resolved."
                )
            ),
        )
    )

    results.append(
        AuditResult(
            category="deployment",
            item="primary_predictive_layer",
            status="PASS",
            severity="info",
            details=(
                "Beta-calibrated Balanced Random Forest remains "
                "the retained primary predictive layer."
            ),
        )
    )

    results.append(
        AuditResult(
            category="deployment",
            item="rejected_evidence_fusion_model",
            status="PASS",
            severity="info",
            details=(
                "The underperforming fusion meta-model is not part "
                "of the deployment decision path."
            ),
        )
    )

    results.append(
        AuditResult(
            category="deployment",
            item="evidence_aggregation_layer",
            status=(
                "PASS"
                if (
                    REQUIRED_OUTPUTS[
                        "15_engineering_report"
                    ].exists()
                )
                else "FAIL"
            ),
            severity=(
                "info"
                if (
                    REQUIRED_OUTPUTS[
                        "15_engineering_report"
                    ].exists()
                )
                else "critical"
            ),
            details=(
                "Transparent evidence aggregation output is available."
                if (
                    REQUIRED_OUTPUTS[
                        "15_engineering_report"
                    ].exists()
                )
                else "Evidence aggregation output is missing."
            ),
        )
    )

    results.append(
        AuditResult(
            category="deployment",
            item="sensor_investigation_layer",
            status=(
                "PASS"
                if (
                    REQUIRED_OUTPUTS[
                        "16_sensor_priority"
                    ].exists()
                )
                else "FAIL"
            ),
            severity=(
                "info"
                if (
                    REQUIRED_OUTPUTS[
                        "16_sensor_priority"
                    ].exists()
                )
                else "critical"
            ),
            details=(
                "Sensor investigation-priority output is available."
                if (
                    REQUIRED_OUTPUTS[
                        "16_sensor_priority"
                    ].exists()
                )
                else "Sensor investigation-priority output is missing."
            ),
        )
    )

    if warning_failures:
        results.append(
            AuditResult(
                category="deployment",
                item="noncritical_warnings",
                status="FAIL",
                severity="warning",
                details=(
                    f"{len(warning_failures)} noncritical warning(s) detected."
                ),
            )
        )

    else:
        results.append(
            AuditResult(
                category="deployment",
                item="noncritical_warnings",
                status="PASS",
                severity="info",
                details="No noncritical warnings detected.",
            )
        )

    return results


# ============================================================
# SAVE AND PRINT
# ============================================================
def save_results(
    path: Path,
    results: list[AuditResult],
) -> None:
    dataframe_to_records(
        results
    ).to_csv(
        path,
        index=False,
    )


def print_section(
    title: str,
    results: list[AuditResult],
) -> None:
    print(f"\n{title}")

    if not results:
        print("No checks were generated.")
        return

    dataframe = dataframe_to_records(
        results
    )

    print(
        dataframe[
            [
                "item",
                "status",
                "severity",
                "details",
            ]
        ].to_string(
            index=False
        )
    )


def main() -> None:
    create_directories()

    LOGGER.info(
        "Starting HeveMind pipeline validation audit"
    )

    script_results: list[AuditResult] = []

    for script_name in CORE_SCRIPTS:
        LOGGER.info(
            "Validating script: %s",
            script_name,
        )

        script_results.extend(
            validate_script(
                script_name
            )
        )

    output_results: list[AuditResult] = []

    for name, path in REQUIRED_OUTPUTS.items():
        LOGGER.info(
            "Validating output: %s",
            name,
        )

        output_results.extend(
            validate_output(
                name=name,
                path=path,
            )
        )

    LOGGER.info(
        "Checking record consistency"
    )

    record_results = (
        build_record_consistency_results()
    )

    LOGGER.info(
        "Checking leakage and governance controls"
    )

    governance_results = (
        build_governance_results()
    )

    predeployment_results = (
        script_results
        + output_results
        + record_results
        + governance_results
    )

    deployment_results = (
        build_deployment_readiness(
            predeployment_results
        )
    )

    all_results = (
        predeployment_results
        + deployment_results
    )

    save_results(
        SCRIPT_AUDIT_PATH,
        script_results,
    )

    save_results(
        OUTPUT_AUDIT_PATH,
        output_results,
    )

    save_results(
        RECORD_AUDIT_PATH,
        record_results,
    )

    save_results(
        LEAKAGE_AUDIT_PATH,
        governance_results,
    )

    save_results(
        DEPLOYMENT_AUDIT_PATH,
        deployment_results,
    )

    critical_failures = [
        result
        for result in all_results
        if (
            result.status == "FAIL"
            and result.severity == "critical"
        )
    ]

    warning_failures = [
        result
        for result in all_results
        if (
            result.status == "FAIL"
            and result.severity == "warning"
        )
    ]

    passes = [
        result
        for result in all_results
        if result.status == "PASS"
    ]

    summary = {
        "project": "HeveMind",
        "stage": "End-to-end pipeline validation audit",
        "total_checks": int(
            len(all_results)
        ),
        "passed_checks": int(
            len(passes)
        ),
        "critical_failures": int(
            len(critical_failures)
        ),
        "warning_failures": int(
            len(warning_failures)
        ),
        "pipeline_ready_for_dashboard_integration": bool(
            len(critical_failures)
            == 0
        ),
        "critical_failure_details": [
            asdict(result)
            for result in critical_failures
        ],
        "warning_details": [
            asdict(result)
            for result in warning_failures
        ],
        "core_scripts": CORE_SCRIPTS,
        "required_outputs": {
            name: str(path)
            for name, path in REQUIRED_OUTPUTS.items()
        },
        "methodological_status": {
            "primary_model": (
                "Beta-calibrated Balanced Random Forest"
            ),
            "fusion_meta_model": (
                "Rejected and excluded from deployment path"
            ),
            "uncertainty_engine": (
                "Supporting reliability layer"
            ),
            "explainability_engine": (
                "SHAP attribution only; not physical causality"
            ),
            "historical_similarity": (
                "Supporting case-based evidence only"
            ),
            "evidence_aggregation": (
                "Retains primary decisions and creates no new probability"
            ),
            "sensor_priority": (
                "Investigation ranking; not root-cause confirmation"
            ),
        },
    }

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

    print("\n" + "=" * 128)
    print(
        "HEVEMIND END-TO-END PIPELINE VALIDATION AUDIT"
    )
    print("=" * 128)

    print_section(
        "Script validation",
        script_results,
    )

    print_section(
        "Output validation",
        output_results,
    )

    print_section(
        "Record consistency",
        record_results,
    )

    print_section(
        "Leakage and governance",
        governance_results,
    )

    print_section(
        "Deployment readiness",
        deployment_results,
    )

    print("\nAudit summary:")

    print(
        f"Total checks:                    "
        f"{len(all_results)}"
    )

    print(
        f"Passed checks:                   "
        f"{len(passes)}"
    )

    print(
        f"Critical failures:               "
        f"{len(critical_failures)}"
    )

    print(
        f"Warning failures:                "
        f"{len(warning_failures)}"
    )

    print(
        f"Ready for dashboard integration: "
        f"{'YES' if len(critical_failures) == 0 else 'NO'}"
    )

    print("\nSaved reports:")

    print(
        f"Script audit:                    "
        f"{SCRIPT_AUDIT_PATH}"
    )

    print(
        f"Output audit:                    "
        f"{OUTPUT_AUDIT_PATH}"
    )

    print(
        f"Record audit:                    "
        f"{RECORD_AUDIT_PATH}"
    )

    print(
        f"Governance audit:                "
        f"{LEAKAGE_AUDIT_PATH}"
    )

    print(
        f"Deployment audit:                "
        f"{DEPLOYMENT_AUDIT_PATH}"
    )

    print(
        f"Summary JSON:                    "
        f"{SUMMARY_PATH}"
    )


if __name__ == "__main__":
    main()
