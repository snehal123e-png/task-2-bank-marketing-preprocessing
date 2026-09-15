# Task 2 — Data Collection & Preprocessing

## Project
**Bank Marketing Data Collection & Preprocessing**

This project performs a complete data collection and preprocessing pipeline on the **UCI Bank Marketing dataset**.

- Public source: UCI Machine Learning Repository
- Dataset: Bank Marketing (`bank-full.csv`)
- Samples: 45,211
- Original columns: 17
- Target: `y` (whether the client subscribed to a term deposit)

## Preprocessing performed

1. Dataset collection from the official UCI source
2. Data type inspection and conversion
3. Missing-value and `unknown`-category analysis
4. Duplicate detection and removal
5. IQR-based outlier detection and capping for selected numeric variables
6. Feature engineering
7. Binary target encoding
8. One-hot encoding of categorical variables
9. Standardization of numerical features
10. Export of cleaned/engineered dataset and preprocessing summary

### Engineered features

- `age_group`
- `balance_category`
- `campaign_intensity`
- `previous_contact`
- `contact_duration_minutes`
- `has_any_loan`
- `has_credit_default`

## Folder structure

```text
task2_bank_marketing_preprocessing/
├── data/
│   ├── raw/
│   │   └── bank-full.csv              # downloaded by the script
│   └── processed/
│       ├── bank_marketing_cleaned.csv
│       └── preprocessing_summary.csv
├── notebooks/
│   └── Task_2_Preprocessing.ipynb
├── src/
│   └── preprocessing.py
├── reports/
│   └── preprocessing_report.md
├── download_and_run.py
├── requirements.txt
└── README.md
```

## How to run on Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python download_and_run.py
```

The script downloads the official UCI archive, extracts `bank-full.csv`, runs preprocessing, and creates the processed CSV files.

## Important note

The UCI dataset reports no formal missing cells, but several categorical columns contain the value `unknown`. This project treats those values explicitly and documents them instead of incorrectly claiming that missing cells were deleted.

## Source

UCI Machine Learning Repository — Bank Marketing:
https://archive.ics.uci.edu/dataset/222/bank+marketing

Citation:
Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing. UCI Machine Learning Repository. DOI: 10.24432/C5K306.

## Assignment deliverables

- Cleaned dataset
- Preprocessing source code
- Jupyter Notebook
- Preprocessing documentation
- Reproducible pipeline
