# =========================================
# 1. IMPORT LIBRARIES
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import shap

# =========================================
# 2. LOAD DATASET
# =========================================
df = pd.read_csv("cleaned_datasetcsv.csv")

print("Columns:", df.columns)
print(df.head())

# =========================================
# 3. DEFINE TARGET & FEATURES
# =========================================
TARGET = "Vu"   # Shear strength

X = df.drop(columns=[TARGET])
y = df[TARGET]

# Keep only numerical columns
X = X.select_dtypes(include=["int64", "float64"])

# =========================================
# 4. TRAIN-TEST SPLIT
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 5. DEFINE MODELS
# =========================================
models = {
    "Linear Regression": LinearRegression(),

    "Random Forest": RandomForestRegressor(
        n_estimators=300, random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    ),

    "LightGBM": LGBMRegressor(
        n_estimators=400,
        learning_rate=0.05,
        random_state=42
    )
}

# =========================================
# 6. TRAIN + EVALUATE
# =========================================
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "R2": r2_score(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred)
    })

results_df = pd.DataFrame(results)
print("\nModel Performance:")
print(results_df)

# =========================================
# 7. BEST MODEL (XGBoost)
# =========================================
best_model = models["XGBoost"]

# =========================================
# 8. PREDICTED vs ACTUAL PLOT
# =========================================
y_pred = best_model.predict(X_test)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Shear Strength")
plt.ylabel("Predicted Shear Strength")
plt.title("Predicted vs Actual")
plt.savefig("prediction_plot.png", dpi=300)
plt.show()

# =========================================
# 9. SHAP ANALYSIS (IMPORTANT)
# =========================================
explainer = shap.Explainer(best_model, X_train)
shap_values = explainer(X_train)

# Global importance
shap.summary_plot(shap_values, X_train)

# =========================================
# 10. SAVE RESULTS
# =========================================
results_df.to_csv("model_results.csv", index=False)

print("\n✅ DONE: Model + SHAP complete")
