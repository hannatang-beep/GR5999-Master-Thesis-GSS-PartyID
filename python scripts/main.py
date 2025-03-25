from data_analysis import analyze_gss_variables, categorize_variables
from data_filtering import initial_data_load_and_filter

if __name__ == "__main__":
    file_path = "data/gss7222_r4.dta"
    
    # Run variable analysis first
    print("Step 1: Analyzing variables...")
    var_df, raw_data = analyze_gss_variables(file_path)
    
    if var_df is not None:
        # Categorize variables
        var_df = categorize_variables(var_df)
        print("\nStep 2: Variables categorized")
        
        # Run data filtering
        print("\nStep 3: Filtering data...")
        df_filtered = initial_data_load_and_filter(file_path, var_df)
        
        if df_filtered is not None:
            print("\nAll processing completed successfully.")
        else:
            print("\nData filtering failed.")
    else:
        print("\nVariable analysis failed.")