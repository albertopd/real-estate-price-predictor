from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import root_mean_squared_error

def log_model_metrics(model_name, mae, rmse, r2, csv_path="./data/ML/model_metrics_alberto.csv"):
    new_entry = pd.DataFrame([{
        "model": model_name,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "r2": round(r2, 4)
    }])
    new_entry.to_csv(csv_path, mode="a", index=False, header=False)
    print(f"Metrics logged for: {model_name}")

def display_model_log_and_best_model(csv_path="./data/ML/model_metrics_alberto.csv"):
    """
    Display the model evaluation summary sorted by R² and highlight the best model.
    """
    try:
        df = pd.read_csv(csv_path)
        df = df.drop_duplicates()

        # Extract 'split' info from the model name
        df["split"] = df["model"].str.extract(r"\[(TRAIN|TEST)\]")

        # Sort by split (TRAIN first), then by r2 descending
        df["split"] = pd.Categorical(df["split"], categories=["TRAIN", "TEST"], ordered=True)
        df_sorted = df.sort_values(by=["split", "r2"], ascending=[True, False]).reset_index(drop=True)

        # Add 'best' column with checkmark only on highest r2 per split
        df_sorted["best"] = df_sorted.groupby("split")["r2"].transform(lambda x: x == x.max())
        df_sorted["best"] = df_sorted["best"].replace({True: "✓", False: ""})

        # Display the DataFrame
        print("\n>>> Model Evaluation Summary (sorted by R²):")
        from IPython.display import display
        display(df_sorted)

        # === Get best model on TEST split ===
        test_df = df[df['model'].str.contains(r'\[TEST\]')]
        best_test_row = test_df.loc[test_df['r2'].idxmax()]
        best_model_name = best_test_row['model'].replace(" [TEST]", "")
        print(f"\n👉 Best model based on R²: {best_model_name}")

    except FileNotFoundError:
        print(f"File not found: {csv_path}")
    #except Exception as e:
    #    print(f"An error occurred: {e}")

def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print("-" * 40)
    print(f"{model_name} Evaluation:")
    print(f"  MAE:  {mae:,.2f} €")
    print(f"  RMSE: {rmse:,.2f} €")
    print(f"  R²:   {r2:.4f}")
    print("-" * 40)
    log_model_metrics(model_name, mae, rmse, r2)
    return y_pred, y_true - y_pred

# Load dataset
df = pd.read_csv("./data/ML/immoweb_real_estate_ml_ready_alberto.csv")

# Drop rows with missing values
df = df.dropna()

# Separate features and target
X = df.drop(columns=["price"])
y = df["price"]

# === Remove low variance features (e.g., one-hot dummies rarely activated) ===
selector = VarianceThreshold(threshold=0.01)  # supprime colonnes trop stables
X_reduced_array = selector.fit_transform(X)
X_reduced = pd.DataFrame(X_reduced_array, columns=X.columns[selector.get_support()])
# ==============================================================================

print(f"Initial shape: {X.shape}, Reduced shape: {X_reduced.shape}")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_reduced_array, y, test_size=0.2, random_state=42)

# Train models on all features
lr_model_all = LinearRegression()
rf_model_all = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

lr_model_all.fit(X_train, y_train)
rf_model_all.fit(X_train, y_train)

# Predictions on training data
y_pred_lr_all = lr_model_all.predict(X_train)
y_pred_rf_all = rf_model_all.predict(X_train)

# Evaluate full models
y_pred_lr_all, residuals_lr_all = evaluate_model(y_train, y_pred_lr_all, "Linear Regression (All Features) [TRAIN]")
y_pred_rf_all, residuals_rf_all = evaluate_model(y_train, y_pred_rf_all, "Random Forest (All Features) [TRAIN]")

# Predictions on test data
y_pred_lr_all = lr_model_all.predict(X_test)
y_pred_rf_all = rf_model_all.predict(X_test)

# Evaluate full models
y_pred_lr_all, residuals_lr_all = evaluate_model(y_test, y_pred_lr_all, "Linear Regression (All Features) [TEST]")
y_pred_rf_all, residuals_rf_all = evaluate_model(y_test, y_pred_rf_all, "Random Forest (All Features) [TEST]")

# Select top 10 important features from Random Forest
feature_importances = pd.DataFrame({
    "feature": X_reduced.columns,
    "importance": rf_model_all.feature_importances_
}).sort_values(by="importance", ascending=False)

top_features = feature_importances.head(10)["feature"].tolist()

# Reduced dataset
X_top = X_reduced[top_features]
X_train_top, X_test_top, y_train_top, y_test_top = train_test_split(X_top, y, test_size=0.2, random_state=42)

# Train reduced models
lr_model_top = LinearRegression()
rf_model_top = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

lr_model_top.fit(X_train_top, y_train_top)
rf_model_top.fit(X_train_top, y_train_top)

# Predictions with top features on training data
y_pred_lr_top = lr_model_top.predict(X_train_top)
y_pred_rf_top = rf_model_top.predict(X_train_top)

# Evaluate reduced models
y_pred_lr_top, residuals_lr_top = evaluate_model(y_train_top, y_pred_lr_top, "Linear Regression (Top Features) [TRAIN]")
y_pred_rf_top, residuals_rf_top = evaluate_model(y_train_top, y_pred_rf_top, "Random Forest (Top Features) [TRAIN]")

# Predictions with top features on test data
y_pred_lr_top = lr_model_top.predict(X_test_top)
y_pred_rf_top = rf_model_top.predict(X_test_top)

# Evaluate reduced models
y_pred_lr_top, residuals_lr_top = evaluate_model(y_test_top, y_pred_lr_top, "Linear Regression (Top Features) [TEST]")
y_pred_rf_top, residuals_rf_top = evaluate_model(y_test_top, y_pred_rf_top, "Random Forest (Top Features) [TEST]")

# Display model evaluation summary
display_model_log_and_best_model()