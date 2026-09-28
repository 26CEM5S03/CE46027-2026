# ============================================================
# ASSIGNMENT: MULTIPLE LINEAR REGRESSION vs KNN REGRESSION
# Dataset: insurance.csv
#
# IMPORTANT:
# This version DOES NOT use scikit-learn.
# It is designed for Spyder environments where sklearn is missing.
#
# Models:
# 1. Multiple Linear Regression - implemented with NumPy
# 2. KNN Regression - implemented with NumPy
#
# K values:
# 3, 5, 7, 9, 11
#
# Metrics:
# MAE, MSE, RMSE, R2, execution time
# ============================================================

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# psutil was already available in your Spyder environment.
# It is used only for resource information.
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# ------------------------------------------------------------
# 1. SETTINGS
# ------------------------------------------------------------
# BOTH FILES MUST BE IN C:\a
#
# C:\a\
#     assignment_regression_C_a.py
#     insurance.csv
#
# The program always reads insurance.csv from C:\a.
# All output graphs/results are also saved inside C:\a\outputs.

BASE_FOLDER = r"C:\a"

DATA_FILE = os.path.join(
    BASE_FOLDER,
    "insurance.csv"
)

OUTPUT_FOLDER = os.path.join(
    BASE_FOLDER,
    "outputs"
)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ------------------------------------------------------------
# 2. CHECK FILES
# ------------------------------------------------------------
if not os.path.exists(BASE_FOLDER):
    raise FileNotFoundError(
        "Folder C:\\a was not found. "
        "Please create C:\\a first."
    )

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(
        "insurance.csv was not found in C:\\a.\n\n"
        "Required folder structure:\n"
        "C:\\a\\assignment_regression_C_a.py\n"
        "C:\\a\\insurance.csv"
    )

print("\nPython script folder: C:\\a")
print("Dataset found:", DATA_FILE)
print("Output folder:", OUTPUT_FOLDER)

# ------------------------------------------------------------
# 3. LOAD DATASET
# ------------------------------------------------------------
print("=" * 70)
print("MULTIPLE LINEAR REGRESSION vs KNN REGRESSION")
print("=" * 70)

print("\nLoading dataset...")

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ------------------------------------------------------------
# 4. TARGET AND FEATURES
# ------------------------------------------------------------
TARGET = "charges"

if TARGET not in df.columns:
    raise ValueError(
        "The target column 'charges' was not found."
    )

# Remove rows with missing values if necessary.
if df.isnull().sum().sum() > 0:
    print("\nMissing values found. Removing incomplete rows.")
    df = df.dropna().reset_index(drop=True)

X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float).values

numeric_features = ["age", "bmi", "children"]
categorical_features = ["sex", "smoker", "region"]

print("\nTarget:", TARGET)
print("Number of observations:", len(df))
print("Number of original features:", X.shape[1])

# ------------------------------------------------------------
# 5. TARGET DISTRIBUTION
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.hist(
    y,
    bins=30,
    edgecolor="black"
)

plt.title("Distribution of Insurance Charges")
plt.xlabel("Charges")
plt.ylabel("Frequency")
plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "target_distribution.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 6. PREPROCESS DATA
# ------------------------------------------------------------
# Convert categorical variables to dummy/one-hot variables.
# drop_first=True avoids redundant dummy variables.

X_processed = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=float
)

# Make sure all columns are numeric.
X_processed = X_processed.astype(float)

feature_names = X_processed.columns.tolist()

X_array = X_processed.values

print("\nProcessed feature matrix shape:")
print(X_array.shape)

print("\nProcessed feature names:")
print(feature_names)

# ------------------------------------------------------------
# 7. TRAIN-TEST SPLIT
# ------------------------------------------------------------
# Reproducible 80/20 split without scikit-learn.

np.random.seed(42)

indices = np.arange(
    len(X_array)
)

np.random.shuffle(indices)

split_index = int(
    0.80 * len(indices)
)

train_indices = indices[:split_index]
test_indices = indices[split_index:]

X_train = X_array[train_indices]
X_test = X_array[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 8. STANDARDIZE FEATURES
# ------------------------------------------------------------
# KNN is distance-based, so scaling is necessary.

train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)

# Avoid division by zero for constant columns.
train_std[train_std == 0] = 1.0

X_train_scaled = (
    X_train - train_mean
) / train_std

X_test_scaled = (
    X_test - train_mean
) / train_std

# ------------------------------------------------------------
# 9. METRIC FUNCTIONS
# ------------------------------------------------------------
def calculate_mae(actual, predicted):
    return np.mean(
        np.abs(actual - predicted)
    )


def calculate_mse(actual, predicted):
    return np.mean(
        (actual - predicted) ** 2
    )


def calculate_rmse(actual, predicted):
    return np.sqrt(
        calculate_mse(actual, predicted)
    )


def calculate_r2(actual, predicted):
    ss_res = np.sum(
        (actual - predicted) ** 2
    )

    ss_tot = np.sum(
        (actual - np.mean(actual)) ** 2
    )

    return 1 - (ss_res / ss_tot)


# ------------------------------------------------------------
# 10. MULTIPLE LINEAR REGRESSION
# ------------------------------------------------------------
# Add intercept column.
#
# beta = pseudoinverse(X) * y
#
# NumPy's pinv is used instead of scikit-learn.

print("\n" + "=" * 70)
print("MULTIPLE LINEAR REGRESSION")
print("=" * 70)

# MLR can work with the processed features.
# Scaling also improves numerical stability.

X_train_mlr = X_train_scaled
X_test_mlr = X_test_scaled

# Add intercept column.
ones_train = np.ones(
    (X_train_mlr.shape[0], 1)
)

ones_test = np.ones(
    (X_test_mlr.shape[0], 1)
)

X_train_mlr = np.hstack(
    (ones_train, X_train_mlr)
)

X_test_mlr = np.hstack(
    (ones_test, X_test_mlr)
)

start_time = time.perf_counter()

beta = np.linalg.pinv(
    X_train_mlr
).dot(y_train)

linear_predictions = (
    X_test_mlr.dot(beta)
)

linear_time = (
    time.perf_counter()
    - start_time
)

linear_mae = calculate_mae(
    y_test,
    linear_predictions
)

linear_mse = calculate_mse(
    y_test,
    linear_predictions
)

linear_rmse = calculate_rmse(
    y_test,
    linear_predictions
)

linear_r2 = calculate_r2(
    y_test,
    linear_predictions
)

print("\nMAE :", round(linear_mae, 4))
print("MSE :", round(linear_mse, 4))
print("RMSE:", round(linear_rmse, 4))
print("R2  :", round(linear_r2, 4))
print(
    "Execution time:",
    round(linear_time, 6),
    "seconds"
)

# ------------------------------------------------------------
# 11. KNN REGRESSION FUNCTION
# ------------------------------------------------------------
def knn_predict(
    X_train_data,
    y_train_data,
    X_test_data,
    k
):
    predictions = []

    for test_point in X_test_data:

        # Euclidean distance
        distances = np.sqrt(
            np.sum(
                (X_train_data - test_point) ** 2,
                axis=1
            )
        )

        # Indices of k nearest samples
        nearest_indices = np.argsort(
            distances
        )[:k]

        # Mean target of nearest neighbours
        prediction = np.mean(
            y_train_data[nearest_indices]
        )

        predictions.append(
            prediction
        )

    return np.array(predictions)


# ------------------------------------------------------------
# 12. RUN KNN FOR DIFFERENT K VALUES
# ------------------------------------------------------------
k_values = [3, 5, 7, 9, 11]

knn_predictions = {}

results = []

# Add MLR result first.
if PSUTIL_AVAILABLE:
    cpu_value = psutil.cpu_percent(interval=0.1)
    memory_value = (
        psutil.virtual_memory().used
        / (1024 ** 2)
    )
else:
    cpu_value = np.nan
    memory_value = np.nan

results.append({
    "Model": "Multiple Linear Regression",
    "Parameter": "None",
    "MAE": linear_mae,
    "MSE": linear_mse,
    "RMSE": linear_rmse,
    "R2 Score": linear_r2,
    "Execution Time (s)": linear_time,
    "CPU (%)": cpu_value,
    "Memory (MB)": memory_value
})

print("\n" + "=" * 70)
print("K-NEAREST NEIGHBOURS REGRESSION")
print("=" * 70)

for k in k_values:

    print("\n" + "-" * 60)
    print("K =", k)
    print("-" * 60)

    if PSUTIL_AVAILABLE:
        cpu_before = psutil.cpu_percent(interval=0.1)
        memory_before = (
            psutil.virtual_memory().used
            / (1024 ** 2)
        )
    else:
        cpu_before = np.nan
        memory_before = np.nan

    start_time = time.perf_counter()

    predictions = knn_predict(
        X_train_scaled,
        y_train,
        X_test_scaled,
        k
    )

    execution_time = (
        time.perf_counter()
        - start_time
    )

    if PSUTIL_AVAILABLE:
        cpu_after = psutil.cpu_percent(interval=0.1)
        memory_after = (
            psutil.virtual_memory().used
            / (1024 ** 2)
        )
    else:
        cpu_after = np.nan
        memory_after = np.nan

    mae = calculate_mae(
        y_test,
        predictions
    )

    mse = calculate_mse(
        y_test,
        predictions
    )

    rmse = calculate_rmse(
        y_test,
        predictions
    )

    r2 = calculate_r2(
        y_test,
        predictions
    )

    print("MAE :", round(mae, 4))
    print("MSE :", round(mse, 4))
    print("RMSE:", round(rmse, 4))
    print("R2  :", round(r2, 4))
    print(
        "Execution time:",
        round(execution_time, 6),
        "seconds"
    )

    if PSUTIL_AVAILABLE:
        print(
            "CPU before:",
            cpu_before,
            "%"
        )
        print(
            "CPU after:",
            cpu_after,
            "%"
        )
        print(
            "Memory before:",
            round(memory_before, 2),
            "MB"
        )
        print(
            "Memory after:",
            round(memory_after, 2),
            "MB"
        )

    knn_predictions[k] = predictions

    results.append({
        "Model": "KNN Regression",
        "Parameter": "K=" + str(k),
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2,
        "Execution Time (s)": execution_time,
        "CPU (%)": cpu_after,
        "Memory (MB)": memory_after
    })

# ------------------------------------------------------------
# 13. FINAL RESULTS TABLE
# ------------------------------------------------------------
results_df = pd.DataFrame(
    results
)

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)

results_df.to_csv(
    os.path.join(
        OUTPUT_FOLDER,
        "regression_results.csv"
    ),
    index=False
)

# ------------------------------------------------------------
# 14. ACTUAL VS PREDICTED
# ------------------------------------------------------------
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    linear_predictions,
    alpha=0.7
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title(
    "Multiple Linear Regression: Actual vs Predicted"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "linear_actual_vs_predicted.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 15. RESIDUAL PLOT
# ------------------------------------------------------------
linear_residuals = (
    y_test - linear_predictions
)

plt.figure(figsize=(8, 6))

plt.scatter(
    linear_predictions,
    linear_residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Charges")
plt.ylabel("Residuals")
plt.title(
    "Multiple Linear Regression - Residual Plot"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "linear_residual_plot.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 16. KNN RMSE VS K
# ------------------------------------------------------------
knn_results = results_df[
    results_df["Model"] == "KNN Regression"
]

plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    knn_results["RMSE"].values,
    marker="o"
)

plt.xlabel("Number of Neighbours (K)")
plt.ylabel("RMSE")
plt.title("KNN Regression: RMSE vs K")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "knn_rmse_vs_k.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 17. KNN R2 VS K
# ------------------------------------------------------------
plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    knn_results["R2 Score"].values,
    marker="o"
)

plt.xlabel("Number of Neighbours (K)")
plt.ylabel("R2 Score")
plt.title("KNN Regression: R2 Score vs K")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "knn_r2_vs_k.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 18. RMSE COMPARISON
# ------------------------------------------------------------
labels = (
    results_df["Model"]
    + " "
    + results_df["Parameter"]
)

plt.figure(figsize=(12, 7))

plt.bar(
    labels,
    results_df["RMSE"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("RMSE")
plt.title("RMSE Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "rmse_comparison.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 19. R2 COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(12, 7))

plt.bar(
    labels,
    results_df["R2 Score"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("R2 Score")
plt.title("R2 Score Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "r2_comparison.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 20. MAE COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(12, 7))

plt.bar(
    labels,
    results_df["MAE"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("MAE")
plt.title("MAE Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "mae_comparison.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 21. EXECUTION TIME COMPARISON
# ------------------------------------------------------------
plt.figure(figsize=(12, 7))

plt.bar(
    labels,
    results_df["Execution Time (s)"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_FOLDER,
        "execution_time_comparison.png"
    ),
    dpi=300
)

plt.show()

# ------------------------------------------------------------
# 22. FINAL SUMMARY
# ------------------------------------------------------------
best_r2 = results_df.loc[
    results_df["R2 Score"].idxmax()
]

lowest_rmse = results_df.loc[
    results_df["RMSE"].idxmin()
]

lowest_mae = results_df.loc[
    results_df["MAE"].idxmin()
]

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)

print(
    "\nHighest R2:",
    best_r2["Model"],
    best_r2["Parameter"],
    "=",
    round(best_r2["R2 Score"], 4)
)

print(
    "Lowest RMSE:",
    lowest_rmse["Model"],
    lowest_rmse["Parameter"],
    "=",
    round(lowest_rmse["RMSE"], 4)
)

print(
    "Lowest MAE:",
    lowest_mae["Model"],
    lowest_mae["Parameter"],
    "=",
    round(lowest_mae["MAE"], 4)
)

# ------------------------------------------------------------
# 23. SAVE SUMMARY
# ------------------------------------------------------------
summary_path = os.path.join(
    OUTPUT_FOLDER,
    "summary.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "REGRESSION MODEL COMPARISON\n"
    )

    file.write("=" * 60 + "\n\n")

    file.write(
        "Dataset: Medical Cost Personal Dataset\n"
    )

    file.write(
        f"Dataset shape: {df.shape}\n"
    )

    file.write(
        f"Training samples: {len(X_train)}\n"
    )

    file.write(
        f"Testing samples: {len(X_test)}\n\n"
    )

    file.write(
        results_df.to_string(
            index=False
        )
    )

    file.write("\n\n")

    file.write(
        f"Highest R2: "
        f"{best_r2['Model']} "
        f"{best_r2['Parameter']} = "
        f"{best_r2['R2 Score']:.4f}\n"
    )

    file.write(
        f"Lowest RMSE: "
        f"{lowest_rmse['Model']} "
        f"{lowest_rmse['Parameter']} = "
        f"{lowest_rmse['RMSE']:.4f}\n"
    )

    file.write(
        f"Lowest MAE: "
        f"{lowest_mae['Model']} "
        f"{lowest_mae['Parameter']} = "
        f"{lowest_mae['MAE']:.4f}\n"
    )

print("\n" + "=" * 70)
print("ASSIGNMENT COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nAll graphs and results are saved in:"
)

print(
    os.path.abspath(
        OUTPUT_FOLDER
    )
)

print("\nGenerated files:")

print("1. target_distribution.png")
print("2. linear_actual_vs_predicted.png")
print("3. linear_residual_plot.png")
print("4. knn_rmse_vs_k.png")
print("5. knn_r2_vs_k.png")
print("6. rmse_comparison.png")
print("7. r2_comparison.png")
print("8. mae_comparison.png")
print("9. execution_time_comparison.png")
print("10. regression_results.csv")
print("11. summary.txt")
