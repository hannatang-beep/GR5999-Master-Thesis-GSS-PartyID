# 📊 GSS Data Processing Pipeline

This project prepares General Social Survey (GSS) data for modeling political party identification (partyID) using LASSO and other machine learning models.

Below is an overview of each Python module and its role in the data pipeline.

## 🧹 Files Overview

### `data_analysis.py`

**Purpose**

- Load the full GSS dataset (`.dta` format)
- Inspect all available variables
- Analyze missing values, data types, and year availability
- Categorize variables into themes (e.g., demographic, economic, religious)

**Use case**

- Early-stage exploratory data analysis
- Used to decide which variables to keep for modeling

### `data_filtering.py`

**Purpose**

- Load the GSS dataset
- Filter observations from 2006–2021 (to cover full election cycles)
- Retain only the selected key variables
- Create `election_cycle` variable based on year
- Export cleaned dataset as `data/gss_2008_2020.csv`

**Use case**

- Prepares full working dataset for downstream model-building

### `generate_gss_2008_2012_partyid3.py`

**Purpose**

- Load the filtered dataset from `data/gss_2008_2020.csv`
- Subset the data to years 2008–2012 (inclusive)
- Create a new 3-category target variable `partyid_3cat`
  - 0 = Democrat (original GSS codes 0–2)
  - 1 = Independent (code 3)
  - 2 = Republican (codes 4–6)
- Standardize all column names to lowercase
- Export result to `data/gss_2008_2012_partyid3.csv`

**Use case**

- Creates the final training dataset for modeling political alignment

## 📂 Folder Structure

```
project_root/
├── data/                          # Contains filtered and processed CSV files
│   ├── gss_2008_2020.csv
│   └── gss_2008_2012_partyid3.csv
├── output/                        # Plots, summaries, intermediate exports
│   └── gss_variable_analysis.csv
├── python scripts/               # All core Python code files
│   ├── data_analysis.py
│   ├── data_filtering.py
│   ├── generate_gss_2008_2012_partyid3.py
│   └── step_multiclass_model.py
├── README.md                     # Project overview (this file)
├── .gitignore                    # Files/folders excluded from Git
└── requirements.txt              # Python package dependencies
```

## ✨ Notes

- `partyid_3cat` is used as the dependent variable in classification models.
- Variables with high missingness (e.g., `abany`, `gunlaw`) are retained for review but may be excluded.
- All data outputs are saved under `/data/` or `/output/` and excluded from GitHub by `.gitignore`.
