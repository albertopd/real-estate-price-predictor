import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import numpy as np
import os

# === Load dataset ===
df = pd.read_csv("./data/immoweb_real_estate.csv")

# === Step 1 – Drop columns with too many missing values ===

missing_threshold = 0.5

# Total number of columns before cleaning
initial_col_count = df.shape[1]

# But first, let's try imputing non-empty binary columns (hasAttic, hasBasement, ...)
binary_columns = [col for col in df.columns if "has" in col and not df[col].isna().all()]
for col in binary_columns:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0})
    df[col] = df[col].fillna(0).astype(int)

# And create a new column: hasParking from: parkingCountIndoor and parkingCountOutdoor
df["hasParking"] = (
    (df["parkingCountIndoor"].fillna(0) > 0) |
    (df["parkingCountOutdoor"].fillna(0) > 0)
)
cols_to_drop = ["parkingCountIndoor", "parkingCountOutdoor"]


# Identify other columns to drop (those with >50% missing values)
cols_to_drop = np.union1d(cols_to_drop, df.columns[df.isnull().mean() > missing_threshold])
dropped_col_count = len(cols_to_drop)

# Drop columns
df_cleaned = df.drop(columns=cols_to_drop)

# Count remaining columns
remaining_col_count = df_cleaned.shape[1]

# Display column cleaning summary
print("\n=== Drop columns with a missing threashold = 0.5 ===")
print(f"Initial number of columns: {initial_col_count}")
print(f"Number of columns dropped (>50% missing): {dropped_col_count}")
print(f"Remaining columns: {remaining_col_count}")

print("\nColumns dropped:")
for col in cols_to_drop:
    print(f"  • {col}")

# === Step 2 – Keep only rows with at least 70% non-missing values ===

# But first, let's drop rows without price 
df.dropna(subset=["price"], inplace=True)

min_required = int(df_cleaned.shape[1] * 0.7)  # Minimum non-null values required per row
before_shape = df_cleaned.shape

# Drop rows below threshold
df_cleaned = df_cleaned.dropna(thresh=min_required)
after_shape = df_cleaned.shape

# Display only rows with at least 70% non-missing values
print("\n=== Keep only rows with at least 70% non-missing values ===")
print(f"Threshold: keep rows with ≥70% non-missing values (at least {min_required} non-null columns)")
print(f"Rows before cleaning: {before_shape[0]}")
print(f"Rows after cleaning:  {after_shape[0]}")
print(f"Rows removed:         {before_shape[0] - after_shape[0]}")

# === Step 3 – Handle outliers based on key numerical columns ===
df = df_cleaned.copy()

def handle_outliers_percentile(df, columns, lower=0.25, upper=0.75, use_iqr=False, drop_rows=True):
    """
    Handles outliers in specified columns by either dropping rows outside bounds or clipping (winsorizing) them.

    The bounds can be determined either using raw percentiles (e.g., 0.01 and 0.99) or the IQR (Interquartile Range) method.
    
    Parameters:
        df (pd.DataFrame): Input DataFrame.
        columns (list of str): List of column names to process.
        lower (float): Lower bound for outlier detection. Interpreted as:
                       - a percentile if use_iqr=False
                       - a quantile (e.g., 0.25) for IQR computation if use_iqr=True
        upper (float): Upper bound for outlier detection. Interpreted similarly to `lower`.
        use_iqr (bool): If True, use the IQR method to define outlier thresholds as:
                        [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR]
                        where Q1 and Q3 are computed using `lower` and `upper`.
                        If False, bounds are simple percentiles.
        drop_rows (bool): If True, drop rows with values outside the bounds.
                          If False, clip values to the bounds (winsorization).

    Returns:
        pd.DataFrame: A new DataFrame with outliers handled as specified.
    """
    df_copy = df.copy()

    for col in columns:
        if use_iqr:
            Q1 = df_copy[col].quantile(lower)
            Q3 = df_copy[col].quantile(upper)
            IQR = Q3 - Q1
            low = Q1 - 1.5 * IQR
            high = Q3 + 1.5 * IQR
        else:
            low = df_copy[col].quantile(lower)
            high = df_copy[col].quantile(upper)
        if drop_rows:
            df_copy = df_copy[(df_copy[col] >= low) & (df_copy[col] <= high)]
        else:
            df_copy[col] = df_copy[col].clip(lower=low, upper=high)

    return df_copy

# Apply filters to clip certain extreme outliers
df_clipped_outliers = handle_outliers_percentile(
    df,
    ["buildingConstructionYear", "bedroomCount", "bathroomCount", "toiletCount"],
    lower=0.01,
    upper=0.95, 
    drop_rows=False
)

# Apply filters to remove other extreme outliers
df_no_outliers = handle_outliers_percentile(
    df_clipped_outliers,
    ["price", "habitableSurface"],
    lower=0.25,
    upper=0.75,
    use_iqr=True,
    drop_rows=True
)

# Display final Cleaning Summary without outliers
print("\n=== Final Cleaning Summary (No outliers) ===")
print("Outlier filtering applied on selected columns.")
print(f"Rows before outlier removal: {df.shape[0]}")
print(f"Rows after outlier removal:  {df_no_outliers.shape[0]}")
print(f"Rows removed as outliers:    {df.shape[0] - df_no_outliers.shape[0]}")

# === Export cleaned dataset ===
csv_path = "./data/immoweb_real_estate_cleaned_dataset_alberto.csv"
df_no_outliers.to_csv(csv_path, index=False)
print(f"Cleaned dataset saved to: {csv_path}")

# Save as Excel
excel_path = "./data/immoweb_real_estate_ml_ready_sample10_alberto.xlsx"
df_no_outliers.head(10).to_excel(excel_path, index=False)
print(f"Excel sample file saved to: {excel_path}")