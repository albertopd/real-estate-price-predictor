import os
import shutil
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# 1. Load dataset
df = pd.read_csv("./data/immoweb_real_estate_cleaned_dataset_alberto.csv")
print(f"Dataset shape before cleaning: {df.shape}")

# 2. Drop non-informative columns
df.drop(columns=["Unnamed: 0", "id", "url"], inplace=True)

# 3. Convert booleans (from string to int)
binary_columns = [col for col in df.columns if "has" in col]
""" This is done already on the cleaning step
for col in bool_cols:
    df[col] = df[col].astype(str).str.lower().map({'true': 1, 'false': 0})
    df[col] = df[col].fillna(0).astype(int)
"""

print("=== Binary columns ===")
print(binary_columns)

# 4. Identify categorical columns

# But first, let's transform those that we know how to normalize
# buildingCondition should be transformed as ordinal since these have a natural rank
building_conditions = [['TO_RESTORE', 'TO_RENOVATE', 'TO_BE_DONE_UP', 'GOOD', 'JUST_RENOVATED', 'AS_NEW']]
building_conditions_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=building_conditions)
df['buildingCondition'] = building_conditions_encoder.fit_transform(df[['buildingCondition']])

# epcScore should be transformed as ordinal since these have a natural rank
epc_scores = [['X', 'G', 'G_F', 'G_C', 'F', 'F_D', 'E', 'E_D', 'D', 'D_C', 'C', 'C_B', 'B', 'A', 'A+', 'A++']]
epc_scores_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1, categories=epc_scores)
df['epcScore'] = epc_scores_encoder.fit_transform(df[['epcScore']])

# Locality has high cardinality
# Label encoding can mislead many models (e.g., tree-based ones like XGBoost may handle it better)
# Try different approaches here LabelEncoder, OneHotEncoder or target encoding (mean price per locality)
high_card_cols = ['locality']  
locality_encoder = LabelEncoder()
df['locality'] = locality_encoder.fit_transform(df['locality'])

# Identify the rest of the categorical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
# This is not needed since bol cols were previously converted to binary columns (int)
# categorical_cols = [col for col in categorical_cols if col not in bool_cols]

print("=== Categorical columns ===")
print(categorical_cols)

# 5. Handle missing values for numeric columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [col for col in numeric_cols if col not in binary_columns]
numeric_cols.remove('postCode')  # Exclude postCode from imputing
numeric_cols.remove('price')  # Exclude target from imputing

print("=== Numerical columns ===")
print(numeric_cols)

# 6. Drop rows with missing target
df = df.dropna(subset=['price'])

# 7. Apply preprocessing pipeline

preprocessor = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), numeric_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
], remainder="passthrough")

# Fit and transform
X_prepared = preprocessor.fit_transform(df.drop(columns=["price"]))
target = df["price"].values

# 8. Reconstruct into DataFrame
# Get feature names
num_features = numeric_cols
passed_through_features = ['postCode']  # Excluded from imputing
cat_features = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_cols)
other_features = [col for col in df.columns if col not in (categorical_cols + numeric_cols + ['price', 'postCode'])]

final_columns = list(num_features) + list(cat_features) + passed_through_features + other_features
print("=== Final columns ===")
print(final_columns)

# Combine features and target

df_model = pd.DataFrame(X_prepared, columns=final_columns)
df_model["price"] = target

# 9. Save final dataset (csv)

# Define the folder path
folder_path = "./data/ML"
"""
# Delete the folder if it exists
if os.path.exists(folder_path):
    shutil.rmtree(folder_path)
    print(f"Deleted existing folder: {folder_path}")

# Recreate the folder
os.makedirs(folder_path)
print(f"Recreated folder: {folder_path}")
"""
output_path = "./data/ML/immoweb_real_estate_ml_ready_alberto.csv"
df_model.to_csv(output_path, index=False)
print(f"Cleaned dataset saved to: {output_path}")
print(f"Final shape: {df_model.shape}")

# Save as Excel
excel_path = "./data/ML/immoweb_real_estate_ml_ready_sample10_alberto.xlsx"
df_model.head(10).to_excel(excel_path, index=False)
print(f"Excel sample file saved to: {excel_path}")

# === Apply the preprocessor to the full dataset ===
from sklearn.compose import ColumnTransformer

X_processed = preprocessor.transform(df.drop(columns=["price"]))

# Convert the transformed array into a DataFrame with proper column names
X_transformed_df = pd.DataFrame(X_processed, columns=final_columns)

# Add the target column back if needed
X_transformed_df["price"] = df["price"].reset_index(drop=True)

# === Show top 10 records of ML dataset ===
print("\n=== Top 10 records after preprocessing ===")
print(X_transformed_df.head(10).to_string(index=False))