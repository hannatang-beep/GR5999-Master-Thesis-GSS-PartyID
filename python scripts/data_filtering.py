# =============================================================================
# File: data_filtering.py
# 
# Purpose:
# - Load the full GSS dataset and filter for selected election cycles (2006–2021).
# - Retain only selected variables across themes (demographics, policy, religion, etc.).
# - Add a derived variable 'election_cycle' based on year ranges.
# - Export the cleaned full dataset as 'data/gss_2008_2020.csv'.
#
# Notes:
# - This script prepares the master working dataset for all modeling.
# - It includes only variables explicitly listed in the key_variables dictionary.
# - Final filtered data is used for downstream subset creation and modeling.
# =============================================================================


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def initial_data_load_and_filter(file_path):
    """
    Load GSS data and filter for 2008-2020 election cycles
    """
    try:
        # Load the data
        print("Loading GSS data...")
        df = pd.read_stata(file_path, convert_categoricals=False)

        # Print initial information
        print(f"\nOriginal dataset shape: {df.shape}")
        print("\nAvailable columns in the dataset (first 10):")
        print(df.columns.tolist()[:10], "...")

        # Identify year column - we know it's 'year' from the output
        year_col = 'year'

        # Process year data
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce')
        unique_years = sorted(df[year_col].unique())
        print(f"\nUnique years in dataset: {unique_years}")

        # Filter years 2006-2021 for election cycles
        df_filtered = df[df[year_col].between(2006, 2021)].copy()
        print(f"Filtered dataset shape (2006-2021): {df_filtered.shape}")

        # Create election cycle variable
        election_cycles = {
            2008: (2006, 2009),
            2012: (2010, 2013),
            2016: (2014, 2017),
            2020: (2018, 2021)
        }

        # Create election_cycle column before filtering columns
        conditions = [(df_filtered[year_col].between(start, end)) 
                     for cycle, (start, end) in election_cycles.items()]
        choices = list(election_cycles.keys())
        df_filtered['election_cycle'] = np.select(conditions, choices, default=np.nan)

        # Define key variables with correct case
        key_variables = {
            'time': [year_col, 'election_cycle'],
            'dependent': ['partyid'],
            'socioeconomic': ['income', 'educ', 'degree', 'occ', 'prestige', 'sei',
                              'wrkstat', 'hrs1', 'hrs2'],
            'demographic': ['age', 'race', 'sex', 'region', 'size'],
            'attitudinal': ['polviews', 'relig', 'attend', 'reliten'],
            'policy': ['natfare', 'natenvir', 'eqwlth', 'abany', 'gunlaw'],
            'other_predictors': ['strongpid', 'parpartyid', 'parpres', 'parrel']
        }

        # Flatten key variables list
        all_key_vars = [var for category in key_variables.values() for var in category]

        # Check variable availability
        available_vars = [var for var in all_key_vars if var in df_filtered.columns]
        missing_vars = [var for var in all_key_vars if var not in df_filtered.columns]

        print("\nMissing variables:")
        print(missing_vars)

        # Select available variables
        df_filtered = df_filtered[available_vars]
        print(f"\nFinal dataset shape after variable selection: {df_filtered.shape}")

        # Save filtered dataset
        output_file = 'data/gss_2008_2020.csv'
        df_filtered.to_csv(output_file, index=False)
        print(f"\nFiltered dataset saved to {output_file}")

        # Print summary of election cycles
        cycle_counts = df_filtered['election_cycle'].value_counts().sort_index()
        print("\nObservations per election cycle:")
        print(cycle_counts)

        return df_filtered

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nPlease verify that:")
        print("1. The file path is correct")
        print("2. The data folder exists in your project directory")
        print("3. You have read permissions for the file")
        print("\nTraceback for debugging:")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    file_path = "data/gss7222_r4.dta"
    df_filtered = initial_data_load_and_filter(file_path)

    if df_filtered is not None:
        print("\nData processing completed successfully.")
    else:
        print("\nData processing failed.")
