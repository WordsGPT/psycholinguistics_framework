
import pandas as pd
import sys
import os

# Usage:
# python combine_excels.py file1.xlsx file2.xlsx key_df1,key_df2 [suffix]

if len(sys.argv) not in (4, 5):
    print(
        "Usage: python combine_excels.py "
        "<file1.xlsx> <file2.xlsx> merge_key1,merge_key2 [suffix]"
    )
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
merge_keys = sys.argv[3].split(",")
suffix = sys.argv[4] if len(sys.argv) > 4 else "_new"

if len(merge_keys) != 2:
    print("Error: Provide exactly two merge keys separated by a comma.")
    sys.exit(1)

key_df1, key_df2 = merge_keys

# Load all sheets from the first Excel file
all_sheets = pd.read_excel(file1, sheet_name=None)
sheet_names = list(all_sheets.keys())
first_sheet_name = sheet_names[0]
df1 = all_sheets[first_sheet_name]

# Load the second Excel file
df2 = pd.read_excel(file2)

# Validate merge keys
if key_df1 not in df1.columns:
    sys.exit(f"Error: Column '{key_df1}' not found in {file1}")

if key_df2 not in df2.columns:
    sys.exit(f"Error: Column '{key_df2}' not found in {file2}")

# Identify new columns from df2
df2_columns = [col for col in df2.columns if col != key_df2]
new_cols = [col for col in df2_columns if col not in df1.columns]

# Rename new columns with suffix
df2_renamed = df2[[key_df2] + df2_columns].copy()

rename_map = {col: col + suffix for col in new_cols}
rename_map[key_df2] = key_df1
df2_renamed = df2_renamed.rename(columns=rename_map)

# Compare keys in both directions
keys1 = df1[key_df1]
keys2 = df2[key_df2]

keys1_set = set(keys1.dropna())
keys2_set = set(keys2.dropna())

missing_from_second = df1.loc[
    ~keys1.isin(keys2_set), key_df1
].drop_duplicates()

missing_from_first = df2.loc[
    ~keys2.isin(keys1_set), key_df2
].drop_duplicates()

# Perform left join and track unmatched rows
df_merged = pd.merge(
    df1,
    df2_renamed,
    on=key_df1,
    how="left",
    indicator=True
)

# Count unmatched rows from the first file before deduplication
total_not_found_second = int(
    (df_merged["_merge"] == "left_only").sum()
)

# Remove merge indicator and duplicate rows
df_merged = df_merged.drop(columns=["_merge"])
df_merged = df_merged.drop_duplicates()

# Detect missing or empty values in new columns
df2_suffix_cols = [col + suffix for col in new_cols]

if df2_suffix_cols:
    missing_values_mask = df_merged[df2_suffix_cols].apply(
        lambda col: col.isna() | col.astype(str).str.strip().eq("")
    ).any(axis=1)

    rows_with_missing_values = df_merged[missing_values_mask]

    print(
        "\nRows with missing or empty values in the new columns: "
        f"{len(rows_with_missing_values)}"
    )

    for index, row in rows_with_missing_values.iterrows():
        print(f"Row {index}: {row[key_df1]}")

# Replace missing values with '#N/D'
df_merged = df_merged.fillna("#N/D")

# Create backup filename
file_name, file_ext = os.path.splitext(file1)
backup_file = f"{file_name}_old{file_ext}"

if os.path.exists(backup_file):
    sys.exit(
        f"Error: Backup file already exists: {backup_file}\n"
        "Rename or move the existing backup before running again."
    )

# Back up original Excel file
os.rename(file1, backup_file)

# Save merged result, preserving all sheets
all_sheets[first_sheet_name] = df_merged

with pd.ExcelWriter(file1, engine="openpyxl") as writer:
    for sheet, data in all_sheets.items():
        data.to_excel(writer, sheet_name=sheet, index=False)

# Logs
print(f"\nOriginal file backed up as: {backup_file}")
print(f"Merged file saved as: {file1}")

print("\n========== UNMATCHED KEYS ==========")

print(f"\nFirst 10 keys in {file1} not found in {file2}:")
if missing_from_second.empty:
    print("None")
else:
    for key in missing_from_second.head(10):
        print(key)

print(f"\nFirst 10 keys in {file2} not found in {file1}:")
if missing_from_first.empty:
    print("None")
else:
    for key in missing_from_first.head(10):
        print(key)

# Final summary: keep these prints at the end
print("\n========== FINAL SUMMARY ==========")
print(f"Total rows in the first Excel file: {len(df1)}")
print(
    "Total rows not found in the second Excel file: "
    f"{total_not_found_second}"
)
print(
    "Total rows not found in the first Excel file but yes in the second: "
    f"{len(missing_from_first)}"
)
print("===================================")
