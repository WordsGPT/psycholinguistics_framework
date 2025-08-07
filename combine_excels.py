import pandas as pd
import sys

# Left join of two Excel files using potentially different column names as keys.
# Applies merge to the first sheet of file1.xlsx, keeps other sheets untouched.
# Usage:
# python combine_excels.py file1.xlsx file2.xlsx key_df1,key_df2 [suffix]

if len(sys.argv) != 4 and len(sys.argv) != 5:
    print("Usage: python combine_excels.py <file1.xlsx> <file2.xlsx> merge_key1,merge_key2 [<suffix_new_columns>]")
    sys.exit(1)

file1 = sys.argv[1]
file2 = sys.argv[2]
merge_keys = sys.argv[3].split(",")
suffix = sys.argv[4] if len(sys.argv) > 4 else '_new'

if len(merge_keys) != 2:
    print("Error: You must provide exactly two merge key names separated by a comma (e.g., Word,word)")
    sys.exit(1)

key_df1, key_df2 = merge_keys

# Load all sheets from file1
all_sheets = pd.read_excel(file1, sheet_name=None)
sheet_names = list(all_sheets.keys())
first_sheet_name = sheet_names[0]
df1 = all_sheets[first_sheet_name]

# Load second file (assume just one sheet)
df2 = pd.read_excel(file2)

# Get df2 columns excluding merge key
df2_columns = [col for col in df2.columns if col != key_df2]
new_cols = [col for col in df2_columns if col not in df1.columns]

# Rename df2 columns with suffix
df2_renamed = df2[[key_df2] + df2_columns].copy()
rename_map = {col: col + suffix for col in new_cols}
df2_renamed = df2_renamed.rename(columns={key_df2: key_df1, **rename_map})

# Perform left join (keep all rows from df1)
df_merged = pd.merge(df1, df2_renamed, on=key_df1, how='left')
df_merged = df_merged.drop_duplicates()



# Get only the columns that came from df2 (after renaming)
df2_suffix_cols = [col + suffix for col in new_cols]

# Detect rows where any of the df2 columns are missing or empty
missing_in_df2_mask = df_merged[df2_suffix_cols].map(
    lambda x: pd.isna(x) or (isinstance(x, str) and x.strip() == '')
).any(axis=1)

rows_with_missing_in_df2 = df_merged[missing_in_df2_mask]

print(f"\nRows from '{file1}' where df2 columns have missing or empty values (based on key '{key_df1}'): \n")
for index, row in rows_with_missing_in_df2.iterrows():
    print(f"Row {index}: {row[key_df1]}")




# Fill missing values with '#N/D'
df_merged = df_merged.fillna('#N/D')


# Replace only the first sheet with merged version
all_sheets[first_sheet_name] = df_merged

# Create output filename (same as input file1)
output_file = file1

# Create backup filename by adding _old before the extension
import os
file_name, file_ext = os.path.splitext(file1)
backup_file = f"{file_name}_old{file_ext}"

# Rename original file to backup
os.rename(file1, backup_file)
print(f"Original file renamed to: {backup_file}")

# Save to new Excel with original filename, preserving all sheets
with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet, data in all_sheets.items():
        data.to_excel(writer, sheet_name=sheet, index=False)

print(f"Merged file saved as: {output_file}")
