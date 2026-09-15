# Task 2 — Data Collection & Preprocessing Report

## 1. Objective
To collect a real-world public dataset containing more than 1,000 records and build a documented preprocessing pipeline covering data quality checks, missing/unknown values, duplicates, data types, outliers, and feature engineering.

## 2. Dataset
The UCI Bank Marketing dataset contains 45,211 instances and 17 columns in the `bank-full.csv` version. It describes direct marketing campaigns conducted by a Portuguese banking institution. The target variable `y` indicates whether a client subscribed to a term deposit.

## 3. Data Quality Checks
The pipeline checks:
- Dataset dimensions
- Data types
- Missing cells
- Duplicate rows
- Categorical `unknown` values
- Numeric outliers

## 4. Missing and Unknown Values
The UCI metadata reports no formal missing cells in the dataset. However, categorical fields can contain the value `unknown`. These values are counted and retained as an explicit category rather than being incorrectly treated as deleted missing rows.

If actual NaN values are encountered, numeric values are imputed with the median and categorical values with the mode.

## 5. Duplicate Handling
Duplicate rows are counted and removed before feature engineering.

## 6. Type Conversion
Numeric columns are explicitly converted to numeric types using safe coercion. String/categorical fields are normalized by trimming whitespace and converting text to lowercase.

## 7. Outlier Handling
The IQR method is used to detect potential outliers in selected numeric variables. For highly skewed numeric variables, values outside the IQR bounds are conservatively capped rather than blindly deleting observations. This preserves the row count while reducing the effect of extreme values.

## 8. Feature Engineering
The following features are created:
- `age_group`
- `balance_category`
- `campaign_intensity`
- `previous_contact`
- `contact_duration_minutes`
- `has_any_loan`
- `has_credit_default`

The target `y` is encoded as `target` where 1 means subscribed and 0 means not subscribed.

## 9. Encoding
Categorical variables are converted using one-hot encoding. This makes the cleaned dataset suitable for downstream analytics and machine learning.

## 10. Outputs
The pipeline produces:
- `bank_marketing_cleaned.csv`
- `preprocessing_summary.csv`
- `outlier_report.csv`

## 11. Reproducibility
Run:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_and_run.py
```

## 12. Dataset Source
UCI Machine Learning Repository — Bank Marketing:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Citation:
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. DOI: 10.24432/C5K306.
