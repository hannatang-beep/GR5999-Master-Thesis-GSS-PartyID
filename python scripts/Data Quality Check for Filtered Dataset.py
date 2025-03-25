def check_filtered_data_quality(df):
    """
    Perform quality checks on the filtered dataset
    """
    print("\n=== Data Quality Report ===\n")
    
    # 1. Missing value analysis
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)
    print("Missing Values Percentage:")
    print(missing_pct[missing_pct > 0].sort_values(ascending=False))
    
    # 2. Check unique values in key categorical variables
    categorical_vars = ['PARTYID', 'RACE', 'SEX', 'REGION', 'POLVIEWS']
    print("\nUnique values in categorical variables:")
    for var in categorical_vars:
        if var in df.columns:
            print(f"\n{var}:")
            print(df[var].value_counts(dropna=False))
    
    # 3. Summary statistics for numeric variables
    numeric_vars = ['AGE', 'EDUC', 'INCOME', 'PRESTIGE', 'SEI']
    print("\nSummary statistics for numeric variables:")
    print(df[numeric_vars].describe())
    
    # 4. Check distribution of election cycles
    print("\nDistribution across election cycles:")
    print(df['election_cycle'].value_counts().sort_index())
    
    # 5. Basic data validation checks
    print("\nData validation checks:")
    print(f"Age range: {df['AGE'].min()} to {df['AGE'].max()}")
    print(f"Education range: {df['EDUC'].min()} to {df['EDUC'].max()}")
    
    return None

# If running the quality check
if __name__ == "__main__":
    # Load the filtered data
    df = pd.read_csv('gss_2008_2020.csv')
    
    # Run quality check
    check_filtered_data_quality(df)