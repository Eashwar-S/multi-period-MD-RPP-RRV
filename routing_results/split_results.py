import pandas as pd
import glob
import os

def split_xlsx_files():
    # Identify all xlsx files in the current folder
    # Exclude the output files if they already exist
    exclude_files = ['routing_results_part_A.xlsx', 'routing_results_part_B.xlsx', 'split_results.py']
    xlsx_files = [f for f in glob.glob("*.xlsx") if f not in exclude_files]
    
    if not xlsx_files:
        print("No .xlsx files found to process.")
        return

    print(f"Found {len(xlsx_files)} files to combine.")
    
    # Read and concatenate all files
    all_dfs = []
    for file in xlsx_files:
        df = pd.read_excel(file)
        # Ensure it's a DataFrame
        if isinstance(df, pd.DataFrame):
            all_dfs.append(df)
    
    if not all_dfs:
        print("No data found in the files.")
        return

    combined_df = pd.concat(all_dfs, ignore_index=True)
    total_rows = len(combined_df)
    print(f"Total rows combined: {total_rows}")
    
    if total_rows != 72:
        print(f"Warning: Expected 72 rows, but found {total_rows}. Splitting anyway.")

    # Split into two parts of 36 rows each (or half if not 72)
    split_point = 36 if total_rows >= 36 else total_rows // 2
    
    part_a = combined_df.iloc[:split_point]
    part_b = combined_df.iloc[split_point:]
    
    # Save the parts
    part_a_name = "routing_results_part_A.xlsx"
    part_b_name = "routing_results_part_B.xlsx"
    
    part_a.to_excel(part_a_name, index=False)
    part_b.to_excel(part_b_name, index=False)
    
    print(f"Saved {len(part_a)} rows to {part_a_name}")
    print(f"Saved {len(part_b)} rows to {part_b_name}")

if __name__ == "__main__":
    split_xlsx_files()
