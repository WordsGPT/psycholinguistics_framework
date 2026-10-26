
import pandas as pd
import sys
import os

# Left join of two CSV files using potentially different column names.
# Usage:
# python combine_csvs.py file1.csv file2.csv key_df1,key_df2 [suffix]

if len(sys.argv) not in (4, 5):
    print(
        "Usage: python combine_csvs.py "
        "<file1.csv> <file2.csv> merge_key1,merge_key2 [<suffix_new_columns>]"
    )
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
merge_keys = sys.argv[3].split(",")
suffix = sys.argv[4] if len(sys.argv) > 4 else "_new"

if len(merge_keys) != 2:
    print(
        "Error: You must provide exactly two merge key names "
        "separated by a comma (e.g., Word,word)"
    )
    sys.exit(1)

key_df1, key_df2 = merge_keys

# Load both CSV files
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Check that the merge keys exist
if key_df1 not in df1.columns:
    print(f"Error: Column '{key_df1}' not found in {file1}")
    sys.exit(1)

if key_df2 not in df2.columns:
    print(f"Error: Column '{key_df2}' not found in {file2}")
    sys.exit(1)

# Get columns from df2 excluding the merge key
df2_columns = [col for col in df2.columns if col != key_df2]
new_cols = [col for col in df2_columns if col not in df1.columns]

# Rename new columns with the specified suffix
df2_renamed = df2[[key_df2] + df2_columns].copy()

rename_map = {col: col + suffix for col in new_cols}
rename_map[key_df2] = key_df1

df2_renamed = df2_renamed.rename(columns=rename_map)

# Perform left join, preserving all rows from df1
df_merged = pd.merge(
    df1,
    df2_renamed,
    on=key_df1,
    how="left"
)

df_merged = df_merged.drop_duplicates()

# Identify columns originating from df2 that were renamed
df2_suffix_cols = [col + suffix for col in new_cols]

# Detect missing or empty values in the new columns
if df2_suffix_cols:
    missing_in_df2_mask = df_merged[df2_suffix_cols].apply(
        lambda col: col.isna()
        | col.astype(str).str.strip().eq("")
    ).any(axis=1)

    rows_with_missing_in_df2 = df_merged[missing_in_df2_mask]

    print(
        f"\nRows from '{file1}' where df2 columns have missing "
        f"or empty values (based on key '{key_df1}'):\n"
    )

    for index, row in rows_with_missing_in_df2.iterrows():
        print(f"Row {index}: {row[key_df1]}")

# Replace missing values with '#N/D'
df_merged = df_merged.fillna("#N/D")

# Create backup filename
file_name, file_ext = os.path.splitext(file1)
backup_file = f"{file_name}_old{file_ext}"

# Rename original file to backup
os.rename(file1, backup_file)
print(f"Original file renamed to: {backup_file}")

# Save merged result as CSV using the original filename
df_merged.to_csv(file1, index=False, encoding="utf-8-sig")

print(f"Merged file saved as: {file1}")
