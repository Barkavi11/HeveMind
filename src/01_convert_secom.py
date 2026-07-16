from pathlib import Path
import re

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================
ROOT_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

SECOM_DATA_PATH = RAW_DIR / "secom.data"
SECOM_LABELS_PATH = RAW_DIR / "secom_labels.data"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD SECOM SENSOR DATA
# ============================================================
def load_sensor_data(path: Path) -> pd.DataFrame:
    """
    Load the space-delimited SECOM manufacturing feature file.

    The original file contains 590 anonymized numerical process
    variables and uses NaN for missing observations.
    """
    if not path.exists():
        raise FileNotFoundError(f"Sensor data file not found: {path}")

    sensor_df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        na_values=["NaN"],
        engine="python",
    )

    sensor_df.columns = [
        f"sensor_{index:03d}"
        for index in range(1, sensor_df.shape[1] + 1)
    ]

    return sensor_df


# ============================================================
# LOAD LABEL AND TIMESTAMP DATA
# ============================================================
def load_label_data(path: Path) -> pd.DataFrame:
    """
    Load SECOM labels safely.

    Expected row format:
    -1 "19/07/2008 11:55:00"
     1 "19/07/2008 13:17:00"
    """
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")

    pattern = re.compile(
        r'^\s*(-?1)\s+"([^"]+)"\s*$'
    )

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            match = pattern.match(line)

            if not match:
                raise ValueError(
                    f"Unable to parse line {line_number}: {line!r}"
                )

            original_label = int(match.group(1))
            timestamp_text = match.group(2)

            records.append(
                {
                    "original_label": original_label,
                    "timestamp": timestamp_text,
                }
            )

    label_df = pd.DataFrame(records)

    label_df["timestamp"] = pd.to_datetime(
        label_df["timestamp"],
        format="%d/%m/%Y %H:%M:%S",
        errors="raise",
    )

    # Original SECOM labels:
    # -1 = pass
    #  1 = fail
    label_df["target"] = label_df["original_label"].map(
        {
            -1: 0,
            1: 1,
        }
    )

    label_df["status"] = label_df["target"].map(
        {
            0: "Pass",
            1: "Fail",
        }
    )

    return label_df


# ============================================================
# BUILD MASTER DATASET
# ============================================================
def build_master_dataset(
    sensor_df: pd.DataFrame,
    label_df: pd.DataFrame,
) -> pd.DataFrame:
    if len(sensor_df) != len(label_df):
        raise ValueError(
            "Sensor and label row counts do not match: "
            f"{len(sensor_df)} sensors versus {len(label_df)} labels."
        )

    master_df = pd.concat(
        [
            label_df.reset_index(drop=True),
            sensor_df.reset_index(drop=True),
        ],
        axis=1,
    )

    master_df.insert(
        0,
        "wafer_id",
        [
            f"WF-{index:06d}"
            for index in range(1, len(master_df) + 1)
        ],
    )

    master_df["production_date"] = master_df["timestamp"].dt.date
    master_df["production_hour"] = master_df["timestamp"].dt.hour
    master_df["production_day"] = master_df["timestamp"].dt.day_name()

    master_df["shift"] = pd.cut(
        master_df["production_hour"],
        bins=[-1, 7, 15, 23],
        labels=["Night", "Morning", "Evening"],
    )

    master_df["missing_sensor_count"] = (
        master_df.filter(regex=r"^sensor_").isna().sum(axis=1)
    )

    return master_df


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================
def print_data_summary(master_df: pd.DataFrame) -> None:
    sensor_columns = [
        column
        for column in master_df.columns
        if column.startswith("sensor_")
    ]

    total_missing = int(
        master_df[sensor_columns].isna().sum().sum()
    )

    constant_columns = [
        column
        for column in sensor_columns
        if master_df[column].nunique(dropna=True) <= 1
    ]

    print("\n" + "=" * 70)
    print("SECOM DATA CONVERSION SUMMARY")
    print("=" * 70)
    print(f"Rows:                    {len(master_df):,}")
    print(f"Sensor features:         {len(sensor_columns):,}")
    print(f"Total missing values:    {total_missing:,}")
    print(f"Constant sensor columns: {len(constant_columns):,}")

    print("\nTarget distribution:")
    print(master_df["status"].value_counts())

    print("\nTarget percentage:")
    print(
        master_df["status"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
        .astype(str)
        .add("%")
    )

    print("\nTimestamp range:")
    print(
        f"{master_df['timestamp'].min()} "
        f"to {master_df['timestamp'].max()}"
    )


# ============================================================
# MAIN
# ============================================================
def main() -> None:
    sensor_df = load_sensor_data(SECOM_DATA_PATH)
    label_df = load_label_data(SECOM_LABELS_PATH)

    master_df = build_master_dataset(
        sensor_df=sensor_df,
        label_df=label_df,
    )

    print_data_summary(master_df)

    csv_path = PROCESSED_DIR / "secom_master.csv"
    parquet_path = PROCESSED_DIR / "secom_master.parquet"

    master_df.to_csv(
        csv_path,
        index=False,
    )

    try:
        master_df.to_parquet(
            parquet_path,
            index=False,
        )
        print(f"\nSaved Parquet file: {parquet_path}")
    except ImportError:
        print(
            "\nParquet was not saved because pyarrow is not installed."
        )
        print("Install it using: pip install pyarrow")

    print(f"Saved CSV file:     {csv_path}")


if __name__ == "__main__":
    main()