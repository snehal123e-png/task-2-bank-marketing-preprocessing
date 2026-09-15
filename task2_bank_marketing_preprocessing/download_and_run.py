from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent

RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW.mkdir(parents=True, exist_ok=True)
PROCESSED.mkdir(parents=True, exist_ok=True)

CSV_PATH = RAW / "bank-full.csv"


def preprocess():

    print("Loading Bank Marketing dataset...")

    df = pd.read_csv(CSV_PATH, sep=";")

    print("\nOriginal Dataset")
    print("----------------")
    print("Rows:", df.shape[0])
    print("Columns:", df.shape[1])

    original_rows = len(df)
    original_columns = len(df.columns)

    # -----------------------------
    # 1. Missing Values
    # -----------------------------

    print("\nMissing Values")
    print("--------------")

    missing = df.isnull().sum()
    print(missing[missing > 0])

    missing_cells = int(df.isnull().sum().sum())

    # Numeric columns
    numeric_columns = [
        "age",
        "balance",
        "day",
        "duration",
        "campaign",
        "pdays",
        "previous"
    ]

    # Convert numeric columns
    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # Handle numerical missing values
    for column in numeric_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].median()
            )

    # Handle categorical missing values
    categorical_columns = df.select_dtypes(
        include="object"
    ).columns

    for column in categorical_columns:

        if df[column].isnull().any():

            df[column] = df[column].fillna(
                df[column].mode()[0]
            )

    # -----------------------------
    # 2. Unknown Values
    # -----------------------------

    print("\nUnknown Values")
    print("--------------")

    unknown_total = 0

    for column in categorical_columns:

        count = (df[column] == "unknown").sum()

        if count > 0:

            print(column, ":", count)

            unknown_total += count

    # -----------------------------
    # 3. Duplicate Removal
    # -----------------------------

    duplicate_count = int(
        df.duplicated().sum()
    )

    print("\nDuplicates:", duplicate_count)

    df = df.drop_duplicates()

    # -----------------------------
    # 4. Feature Engineering
    # -----------------------------

    print("\nCreating engineered features...")

    # Age group
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=[
            "young",
            "adult",
            "middle_age",
            "senior",
            "older"
        ]
    )

    # Balance category
    df["balance_category"] = pd.cut(
        df["balance"],
        bins=[
            -np.inf,
            0,
            1000,
            5000,
            np.inf
        ],
        labels=[
            "negative_zero",
            "low",
            "medium",
            "high"
        ]
    )

    # Campaign intensity
    df["campaign_intensity"] = pd.cut(
        df["campaign"],
        bins=[0, 1, 3, 5, np.inf],
        labels=[
            "single",
            "low",
            "medium",
            "high"
        ],
        include_lowest=True
    )

    # Previous contact
    df["previous_contact"] = (
        df["pdays"] != -1
    ).astype(int)

    # Duration in minutes
    df["contact_duration_minutes"] = (
        df["duration"] / 60
    ).round(2)

    # Has any loan
    df["has_any_loan"] = (
        (df["housing"] == "yes") |
        (df["loan"] == "yes")
    ).astype(int)

    # Credit default
    df["has_credit_default"] = (
        df["default"] == "yes"
    ).astype(int)

    # -----------------------------
    # 5. Outlier Detection
    # -----------------------------

    print("\nOutlier Detection")
    print("-----------------")

    outlier_columns = [
        "age",
        "balance",
        "duration",
        "campaign",
        "previous"
    ]

    outlier_results = []

    for column in outlier_columns:

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = (
            (df[column] < lower) |
            (df[column] > upper)
        ).sum()

        print(
            column,
            ":",
            int(outliers),
            "outliers"
        )

        outlier_results.append([
            column,
            Q1,
            Q3,
            lower,
            upper,
            int(outliers)
        ])

        # Conservative outlier capping
        df[column] = df[column].clip(
            lower,
            upper
        )

    outlier_df = pd.DataFrame(
        outlier_results,
        columns=[
            "column",
            "Q1",
            "Q3",
            "lower_bound",
            "upper_bound",
            "outlier_count"
        ]
    )

    # -----------------------------
    # 6. Target Encoding
    # -----------------------------

    df["target"] = (
        df["y"] == "yes"
    ).astype(int)

    df.drop(
        columns=["y"],
        inplace=True
    )

    # -----------------------------
    # 7. Categorical Encoding
    # -----------------------------

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        dtype=int
    )

    # -----------------------------
    # 8. Final Missing Check
    # -----------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    df = df.fillna(0)

    # -----------------------------
    # 9. Save Clean Dataset
    # -----------------------------

    output_file = (
        PROCESSED /
        "bank_marketing_cleaned.csv"
    )

    df.to_csv(
        output_file,
        index=False
    )

    # -----------------------------
    # 10. Save Summary
    # -----------------------------

    summary = pd.DataFrame({

        "Metric": [
            "Original Rows",
            "Original Columns",
            "Duplicates Removed",
            "Missing Cells",
            "Unknown Values",
            "Final Rows",
            "Final Columns"
        ],

        "Value": [
            original_rows,
            original_columns,
            duplicate_count,
            missing_cells,
            unknown_total,
            df.shape[0],
            df.shape[1]
        ]
    })

    summary.to_csv(
        PROCESSED /
        "preprocessing_summary.csv",
        index=False
    )

    outlier_df.to_csv(
        PROCESSED /
        "outlier_report.csv",
        index=False
    )

    print("\n================================")
    print("PREPROCESSING COMPLETED")
    print("================================")

    print(
        "\nCleaned dataset saved at:"
    )

    print(output_file)

    print("\nFinal Shape:")

    print(df.shape)

    print("\nGenerated files:")

    print(
        "data/processed/bank_marketing_cleaned.csv"
    )

    print(
        "data/processed/preprocessing_summary.csv"
    )

    print(
        "data/processed/outlier_report.csv"
    )


if __name__ == "__main__":

    if not CSV_PATH.exists():

        print(
            "ERROR: bank-full.csv not found!"
        )

        print(
            "Expected location:"
        )

        print(CSV_PATH)

    else:

        preprocess()