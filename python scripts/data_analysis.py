# =============================================================================
# File: data_analysis.py
# 
# Purpose:
# - Perform variable-level analysis on the full GSS dataset.
# - Output a summary of missing values, data types, value counts, and year coverage.
# - Automatically categorize variables into thematic groups (e.g., demographic, economic).
#
# Notes:
# - Does NOT filter by year or variable availability.
# - Does NOT write out final cleaned dataset for modeling.
# - Used primarily for exploratory analysis and variable selection decisions.
# =============================================================================


import pandas as pd
import numpy as np

def analyze_gss_variables(file_path):
    """
    Analyze and document variables in the GSS dataset.
    Returns a DataFrame with variable information.
    """
    try:
        # Load the data
        print("Loading GSS data for variable analysis...")
        df = pd.read_stata(file_path, convert_categoricals=False)

        # Create a DataFrame to store variable information
        var_info = []

        for col in df.columns:
            non_null = df[col].count()
            null_count = df[col].isnull().sum()
            dtype = df[col].dtype
            unique_values = df[col].nunique()

            # Check if variable exists across years
            years_present = df.groupby('year')[col].count() > 0
            year_coverage = f"{years_present[years_present].index.min()}-{years_present[years_present].index.max()}"

            # Get sample values
            sample_values = df[col].dropna().sample(min(5, unique_values)) if unique_values > 0 else []

            var_info.append({
                'variable': col,
                'dtype': str(dtype),
                'non_null_count': non_null,
                'null_count': null_count,
                'unique_values': unique_values,
                'year_coverage': year_coverage,
                'sample_values': list(sample_values)
            })

        var_df = pd.DataFrame(var_info)
        var_df = var_df.sort_values('non_null_count', ascending=False)
        var_df.to_csv('output/gss_variable_analysis.csv', index=False)
        print("\nVariable analysis saved to 'gss_variable_analysis.csv'")

        print(f"\nTotal variables: {len(var_df)}")
        print(f"Variables with >50% non-null values: {len(var_df[var_df['non_null_count'] > len(df)/2])}")

        return var_df, df

    except Exception as e:
        print(f"\nError in variable analysis: {str(e)}")
        return None, None

def categorize_variables(var_df):
    """
    Categorize variables into different types based on their names and properties.
    """
    categories = {
        'demographic': ['age', 'sex', 'race', 'region', 'size'],
        'economic': ['income', 'work', 'job', 'occ', 'indus', 'emp'],
        'political': ['party', 'polit', 'vote', 'pres', 'strongpid', 'parpartyid', 'parpres', 'parrel'],
        'educational': ['educ', 'degree', 'school'],
        'religious': ['relig', 'church', 'pray', 'god', 'attend', 'reliten'],
        'policy': ['natfare', 'natenvir', 'eqwlth', 'abany', 'gunlaw'],
        'social': ['trust', 'happy', 'life', 'friend']
    }

    def get_category(var_name):
        var_lower = var_name.lower()
        for cat, patterns in categories.items():
            if any(pattern in var_lower for pattern in patterns):
                return cat
        return 'other'

    var_df['category'] = var_df['variable'].apply(get_category)
    return var_df

def initial_data_load_and_filter(file_path, var_df=None):
    """
    Load GSS data and filter for 2006-2021 election cycles using variable analysis results
    """
    try:
        print("\nStarting data filtering process...")
        df = pd.read_stata(file_path, convert_categoricals=False)

        # Use variable analysis to identify available variables
        if var_df is not None:
            print("\nUsing variable analysis to identify key variables...")
            # Get variables with >50% non-null values
            valid_vars = var_df[var_df['non_null_count'] > len(df)/2]['variable'].tolist()
            print(f"Found {len(valid_vars)} variables with >50% completion")

        # Filter years 2006-2021 for election cycles
        df_filtered = df[df['year'].between(2006, 2021)].copy()
        print(f"\nFiltered dataset shape (2006-2021): {df_filtered.shape}")

        # Create election cycle variable
        election_cycles = {
            2008: (2006, 2009),
            2012: (2010, 2013),
            2016: (2014, 2017),
            2020: (2018, 2021)
        }

        conditions = [(df_filtered['year'].between(start, end)) 
                     for cycle, (start, end) in election_cycles.items()]
        choices = list(election_cycles.keys())
        df_filtered['election_cycle'] = np.select(conditions, choices, default=np.nan)

# Removed CSV output to avoid overwriting final cleaned file
# output_file = 'gss_2008_2020.csv'
# df_filtered.to_csv(output_file, index=False)
# print(f"\nFiltered dataset saved to {output_file}")

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

    # First analyze variables
    var_df, raw_data = analyze_gss_variables(file_path)

    if var_df is not None:
        # Categorize variables
        var_df = categorize_variables(var_df)

        # Print category summary
        print("\nVariables by category:")
        print(var_df['category'].value_counts())

        # Now proceed with the filtering using the variable analysis results
        df_filtered = initial_data_load_and_filter(file_path, var_df)

        if df_filtered is not None:
            print("\nData processing completed successfully.")
        else:
            print("\nData processing failed.")
