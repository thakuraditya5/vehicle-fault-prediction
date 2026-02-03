import pandas as pd
import os

# Use the full path provided by user
FILE_PATH = r"C:\Users\1000684\Downloads\BCS_fault_prediction\9M_Empire_AC_UK07PA6111_bcs.xlsx"

def inspect_data():
    print(f"Reading file: {FILE_PATH}...")
    try:
        # Read only first 1000 rows to be fast
        df = pd.read_excel(FILE_PATH, nrows=1000)
        
        print("\n--- Data Overview ---")
        print(f"Total Columns: {len(df.columns)}")
        print("Columns:", df.columns.tolist())
        
        print("\n--- Target Column Analysis (BCSfault) ---")
        if 'BCSfault' in df.columns:
            print("Target column found!")
            print(df['BCSfault'].value_counts(dropna=False))
            print("\nData Type:", df['BCSfault'].dtype)
        else:
            print("CRITICAL WARNING: 'BCSfault' column NOT found!")

        print("\n--- Data Sample ---")
        print(df.head())
        
        print("\n--- Missing Values ---")
        print(df.isnull().sum()[df.isnull().sum() > 0])

    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_data()
