📊 GSS Data Processing Pipeline

This project prepares General Social Survey (GSS) data for modeling political party identification (partyID) using LASSO and other machine learning models. Below is an overview of each Python module and its role in the data pipeline.

🧩 Files Overview

data_analysis.py

Purpose:

Load the full GSS dataset (.dta format)

Inspect all available variables

Analyze missing values, data types, and year availability

Categorize variables into themes (e.g., demographic, economic, religious)

Use case:

Early-stage exploratory data analysis

Used to decide which variables to keep for modeling

data_filtering.py

Purpose:

Load the GSS dataset

Filter observations from 2006–2021 (to cover full election cycles)

Retain only the selected key variables

Create election_cycle variable based on year

Export cleaned full dataset to data/gss_2008_2020.csv

Use case:

This is the main full-data preparation script

Output is used for downstream modeling subset extraction

generate_gss_2008_2012_partyid3.py

Purpose:

Load the filtered dataset from data/gss_2008_2020.csv

Subset the data to years 2008–2012 (inclusive)

Create a new 3-category target variable partyid_3cat

0 = Democrat (original GSS codes 0–2)

1 = Independent (code 3)

2 = Republican (codes 4–6)

Standardize all column names to lowercase

Export result to data/gss_2008_2012_partyid3.csv

Use case:

Creates the final training dataset for modeling political alignment

🗂 Folder Structure

project_root/
├── data/                          # Contains filtered and processed CSV files
│   ├── gss_2008_2020.csv
│   └── gss_2008_2012_partyid3.csv
├── data_analysis.py              # Exploratory analysis and variable categorization
├── data_filtering.py             # Full dataset cleaning and filtering
├── generate_gss_2008_2012_partyid3.py  # Final modeling dataset generator
├── step_multiclass_model.py      # (To be added) Machine learning modeling
└── README.md                    # Project overview (this file)

✨ Notes

partyid_3cat is used as the dependent variable in classification models.

Variables like abany and gunlaw are retained temporarily for inspection, and may be removed later due to high missingness.

All final datasets are saved to the /data/ folder for consistency.

