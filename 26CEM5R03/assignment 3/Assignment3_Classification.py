# ============================================================
# ASSIGNMENT-3 : CLASSIFICATION
# SVM AND DECISION TREE COMPARISON
# Dataset: Breast Cancer Wisconsin Diagnostic Dataset
# ============================================================

import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tkinter import Tk
from tkinter.filedialog import askopenfilename

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# 1. SELECT DATASET
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT-3 : CLASSIFICATION")
print("=" * 70)

root = Tk()
root.withdraw()

file_path = askopenfilename(
    title="Select Classification Dataset",
    filetypes=[
        ("Excel files", "*.xlsx *.xls"),
        ("CSV files", "*.csv"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not file_path:
    print("No dataset selected.")
    raise SystemExit

print("\nSelected file:")
print(file_path)


# ============================================================
# 2. CREATE OUTPUT FOLDER
# ============================================================

base_folder = os.path.dirname(file_path)

output_folder = os.path.join(
    base_folder,
    "Assignment3_Results"
)

os.makedirs(output_folder, exist_ok=True)

print("\nResults will be saved in:")
print(output_folder)


# ============================================================
# 3. LOAD DATASET
# ============================================================

print("\n" + "=" * 70)
print("LOADING DATASET")
print("=" * 70)

file_extension = os.path.splitext(file_path)[1].lower()

if file_extension in [".xlsx", ".xls"]:
    df = pd.read_excel(file_path)

elif file_extension == ".csv":
    df = pd.read_csv(file_path)

else:
    print("Unsupported file format.")
    raise SystemExit

print("\nDataset loaded successfully.")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())


# ============================================================
# 4. DATA INSPECTION
# ============================================================

print("\n" + "=" * 70)
print("DATA INSPECTION")
print("=" * 70)

print("\nFirst five records:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ============================================================
# 5. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    df = df.drop_duplicates()
    print("\nDuplicates removed:", duplicate_count)
else:
    print("\nNo duplicate rows found.")


# ============================================================
# 6. CHECK TARGET COLUMN
# ============================================================

target_column = "diagnosis"

if target_column not in df.columns:
    print("\nERROR: 'diagnosis' column not found.")
    print("Available columns:")
    print(df.columns.tolist())
    raise SystemExit

print("\nTarget column:", target_column)

print("\nClass distribution:")
print(df[target_column].value_counts())


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

# ID is an identifier, not a predictive feature.
columns_to_remove = ["id", target_column]

X = df.drop(columns=columns_to_remove)
y = df[target_column]

print("\nNumber of samples:", X.shape[0])
print("Number of features:", X.shape[1])


# ============================================================
# 8. CONVERT FEATURES TO NUMERIC
# ============================================================

X = X.apply(pd.to_numeric, errors="coerce")

if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())


# ============================================================
# 9. ENCODE TARGET
# ============================================================

# B = 0 (Benign)
# M = 1 (Malignant)

y = y.map({
    "B": 0,
    "M": 1
})

if y.isnull().sum() > 0:
    print("\nERROR: Unexpected target values found.")
    raise SystemExit

print("\nTarget encoding:")
print("B = 0 (Benign)")
print("M = 1 (Malignant)")


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

print("\n" + "=" * 70)
print("TRAIN / TEST SPLIT")
print("=" * 70)

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train_raw.shape[0])
print("Testing samples:", X_test_raw.shape[0])


# ============================================================
# 11. STANDARDIZATION
# ============================================================

print("\n" + "=" * 70)
print("FEATURE STANDARDIZATION")
print("=" * 70)

scaler = StandardScaler()

# Fit ONLY on training data to avoid data leakage.
X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

# Separate scaled copy for t-SNE.
X_all_scaled = scaler.transform(X)


# ============================================================
# 12. 2D t-SNE VISUALIZATION
# ============================================================

print("\n" + "=" * 70)
print("2D t-SNE VISUALIZATION")
print("=" * 70)

print("\nRunning t-SNE...")

tsne_start = time.perf_counter()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate="auto",
    init="pca",
    random_state=42
)

X_tsne = tsne.fit_transform(X_all_scaled)

tsne_end = time.perf_counter()
tsne_time = tsne_end - tsne_start

print("t-SNE completed.")
print("t-SNE execution time:",
      round(tsne_time, 4), "seconds")


# Create t-SNE plot
plt.figure(figsize=(10, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="coolwarm",
    alpha=0.75,
    edgecolors="black",
    linewidths=0.3
)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title("2D t-SNE Visualization of Breast Cancer Dataset")

colorbar = plt.colorbar(scatter)
colorbar.set_ticks([0, 1])
colorbar.set_ticklabels(["Benign", "Malignant"])
colorbar.set_label("Diagnosis")

plt.tight_layout()

tsne_file = os.path.join(
    output_folder,
    "tsne_2D_scatter_plot.png"
)

plt.savefig(
    tsne_file,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nt-SNE plot saved:")
print(tsne_file)


# ============================================================
# 13. MODEL EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(
    model,
    model_name,
    X_train_data,
    X_test_data,
    y_train_data,
    y_test_data
):

    print("\n" + "-" * 70)
    print("MODEL:", model_name)
    print("-" * 70)

    # Resource measurements
    try:
        import psutil

        process = psutil.Process(os.getpid())

        memory_before = (
            process.memory_info().rss / (1024 ** 2)
        )

        cpu_before = psutil.cpu_percent(interval=None)

    except ImportError:

        memory_before = np.nan
        cpu_before = np.nan


    # ----------------------------
    # Training time
    # ----------------------------

    train_start = time.perf_counter()

    model.fit(
        X_train_data,
        y_train_data
    )

    train_end = time.perf_counter()

    training_time = train_end - train_start


    # ----------------------------
    # Prediction time
    # ----------------------------

    prediction_start = time.perf_counter()

    y_pred = model.predict(
        X_test_data
    )

    prediction_end = time.perf_counter()

    prediction_time = (
        prediction_end - prediction_start
    )


    # ----------------------------
    # Resource utilization
    # ----------------------------

    try:

        memory_after = (
            process.memory_info().rss / (1024 ** 2)
        )

        cpu_after = psutil.cpu_percent(interval=0.1)

        memory_used = memory_after - memory_before
        cpu_utilization = cpu_after

    except:

        memory_used = np.nan
        cpu_utilization = np.nan


    # ----------------------------
    # Classification metrics
    # ----------------------------

    accuracy = accuracy_score(
        y_test_data,
        y_pred
    )

    precision = precision_score(
        y_test_data,
        y_pred,
        average="binary",
        zero_division=0
    )

    recall = recall_score(
        y_test_data,
        y_pred,
        average="binary",
        zero_division=0
    )

    f1 = f1_score(
        y_test_data,
        y_pred,
        average="binary",
        zero_division=0
    )


    # ----------------------------
    # Confusion matrix
    # ----------------------------

    cm = confusion_matrix(
        y_test_data,
        y_pred
    )

    print("\nAccuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-score :", round(f1, 4))

    print("\nTraining time:",
          round(training_time, 6),
          "seconds")

    print("Prediction time:",
          round(prediction_time, 6),
          "seconds")

    print("Memory change:",
          round(memory_used, 4),
          "MB")

    print("CPU utilization:",
          round(cpu_utilization, 2),
          "%")

    print("\nConfusion Matrix:")
    print(cm)


    # ----------------------------
    # Save confusion matrix
    # ----------------------------

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"]
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title("Confusion Matrix - " + model_name)

    plt.tight_layout()

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("=", "")
        .replace("/", "_")
    )

    cm_file = os.path.join(
        output_folder,
        "confusion_matrix_" + safe_name + ".png"
    )

    plt.savefig(
        cm_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ----------------------------
    # Store results
    # ----------------------------

    results.append({

        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Training_Time_sec": training_time,
        "Prediction_Time_sec": prediction_time,
        "Memory_Change_MB": memory_used,
        "CPU_Utilization_percent": cpu_utilization

    })


# ============================================================
# 14. SVM - LINEAR KERNEL
# ============================================================

print("\n" + "=" * 70)
print("SVM CLASSIFIERS")
print("=" * 70)

svm_linear = SVC(
    kernel="linear",
    C=1.0
)

evaluate_model(
    svm_linear,
    "SVM_Linear",
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 15. SVM - POLYNOMIAL KERNEL
# ============================================================

svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1.0,
    gamma="scale"
)

evaluate_model(
    svm_poly,
    "SVM_Polynomial_Degree3",
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 16. SVM - RBF KERNEL
# ============================================================

svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

evaluate_model(
    svm_rbf,
    "SVM_RBF",
    X_train,
    X_test,
    y_train,
    y_test
)


# ============================================================
# 17. DECISION TREE - VARY max_depth
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE - VARYING max_depth")
print("=" * 70)

max_depth_values = [
    3,
    5,
    10,
    15,
    None
]

for depth in max_depth_values:

    model_name = (
        "DecisionTree_max_depth_" + str(depth)
    )

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_split=2,
        random_state=42
    )

    evaluate_model(
        tree,
        model_name,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 18. DECISION TREE - VARY min_samples_split
# ============================================================

print("\n" + "=" * 70)
print("DECISION TREE - VARYING min_samples_split")
print("=" * 70)

min_samples_values = [
    2,
    5,
    10,
    20
]

for min_split in min_samples_values:

    model_name = (
        "DecisionTree_min_samples_split_"
        + str(min_split)
    )

    tree = DecisionTreeClassifier(
        criterion="gini",
        max_depth=10,
        min_samples_split=min_split,
        random_state=42
    )

    evaluate_model(
        tree,
        model_name,
        X_train,
        X_test,
        y_train,
        y_test
    )


# ============================================================
# 19. FINAL RESULTS TABLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL MODEL COMPARISON")
print("=" * 70)

results_df = pd.DataFrame(results)
results_df = results_df.round(6)

print("\n")
print(results_df.to_string(index=False))


# ============================================================
# 20. SAVE RESULTS TO CSV
# ============================================================

results_file = os.path.join(
    output_folder,
    "model_comparison_results.csv"
)

results_df.to_csv(
    results_file,
    index=False
)

print("\nCSV results saved:")
print(results_file)


# ============================================================
# 21. SAVE RESULTS TO EXCEL
# ============================================================

excel_results_file = os.path.join(
    output_folder,
    "model_comparison_results.xlsx"
)

results_df.to_excel(
    excel_results_file,
    index=False
)

print("\nExcel results saved:")
print(excel_results_file)


# ============================================================
# 22. PERFORMANCE COMPARISON GRAPH
# ============================================================

plt.figure(figsize=(14, 7))

x = np.arange(len(results_df))
width = 0.2

plt.bar(
    x - width * 1.5,
    results_df["Accuracy"],
    width,
    label="Accuracy"
)

plt.bar(
    x - width / 2,
    results_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x + width / 2,
    results_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width * 1.5,
    results_df["F1_Score"],
    width,
    label="F1-score"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=75,
    ha="right"
)

plt.ylabel("Score")
plt.xlabel("Model")
plt.title("Classification Performance Comparison")
plt.ylim(0, 1.05)
plt.legend()

plt.tight_layout()

performance_graph = os.path.join(
    output_folder,
    "classification_performance_comparison.png"
)

plt.savefig(
    performance_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nPerformance graph saved:")
print(performance_graph)


# ============================================================
# 23. TRAINING TIME COMPARISON
# ============================================================

plt.figure(figsize=(14, 7))

plt.bar(
    results_df["Model"],
    results_df["Training_Time_sec"]
)

plt.xticks(
    rotation=75,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Training Time (seconds)")
plt.title("Training Execution Time Comparison")

plt.tight_layout()

time_graph = os.path.join(
    output_folder,
    "training_time_comparison.png"
)

plt.savefig(
    time_graph,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nTraining time graph saved:")
print(time_graph)


# ============================================================
# 24. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("ASSIGNMENT-3 COMPLETED")
print("=" * 70)

print("\nDataset:")
print("Samples :", df.shape[0])
print("Features:", X.shape[1])
print("Classes : B (Benign), M (Malignant)")

print("\nModels evaluated:")
print("1. SVM - Linear kernel")
print("2. SVM - Polynomial kernel")
print("3. SVM - RBF kernel")
print("4. Decision Tree - varying max_depth")
print("5. Decision Tree - varying min_samples_split")

print("\nMetrics:")
print("1. Accuracy")
print("2. Precision")
print("3. Recall")
print("4. F1-score")
print("5. Confusion Matrix")

print("\nComputational measurements:")
print("1. Training execution time")
print("2. Prediction execution time")
print("3. Memory change")
print("4. CPU utilization")

print("\nAll outputs saved in:")
print(output_folder)

print("\n" + "=" * 70)
print("PROGRAM FINISHED")
print("=" * 70)
