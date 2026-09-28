# ============================================================
# ASSIGNMENT 3: CLASSIFICATION
# Dataset: AILM DATASET 1.csv
# Algorithms: SVM and Decision Tree
# Visualization: 2D t-SNE
# ============================================================

import os
import re
import time
import tracemalloc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

FILE_NAME = "AILM DATASET 1.csv"
TARGET = "cp"  # Predict chest-pain category in this dataset
OUTPUT_DIR = "Assignment3_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find the dataset automatically. If it is not found, open a file
# selection window so the user can choose the CSV file manually.
if not os.path.exists(FILE_NAME):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.join(script_dir, FILE_NAME)

    if os.path.exists(candidate):
        FILE_NAME = candidate
    else:
        try:
            import tkinter as tk
            from tkinter import filedialog

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)

            selected_file = filedialog.askopenfilename(
                title="Select AILM DATASET 1.csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
            )

            root.destroy()

            if not selected_file:
                raise FileNotFoundError(
                    "No CSV file was selected. Please run the program again "
                    "and select 'AILM DATASET 1.csv'."
                )

            FILE_NAME = selected_file

        except ImportError:
            raise FileNotFoundError(
                f"Dataset not found: {FILE_NAME}. "
                "Please put the CSV in the same folder as the Python file."
            )

print("\nDataset file being used:")
print(os.path.abspath(FILE_NAME))

df = pd.read_csv(FILE_NAME)

print("=" * 65)
print("ASSIGNMENT 3: CLASSIFICATION")
print("=" * 65)

print("\nDataset shape:", df.shape)
print("\nDataset columns:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

df = df.dropna().copy()

# ------------------------------------------------------------
# 2. SELECT FEATURES AND TARGET
# ------------------------------------------------------------

if TARGET not in df.columns:
    raise ValueError(
        f"Target '{TARGET}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )

X = df.drop(columns=[TARGET])
y = df[TARGET]

X = X.select_dtypes(include=np.number)
valid_rows = X.notna().all(axis=1) & y.notna()
X = X.loc[valid_rows]
y = y.loc[valid_rows]

if y.nunique() < 2:
    raise ValueError("Classification requires at least two target classes.")

print("\nFeature columns:")
print(X.columns.tolist())

print("\nTarget class distribution:")
print(y.value_counts().sort_index())

# ------------------------------------------------------------
# 3. SPLIT DATA INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ------------------------------------------------------------
# 4. GENERATE AND SAVE 2D t-SNE SCATTER PLOT
# ------------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# t-SNE perplexity must be less than the number of samples.
perplexity = min(30, max(1, (len(X_scaled) - 1) // 3))
if len(X_scaled) < 3:
    raise ValueError("At least 3 rows are required to generate a t-SNE plot.")

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=perplexity,
    init="pca",
    learning_rate="auto"
)

X_2d = tsne.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))
class_codes = pd.Categorical(y).codes
plt.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=class_codes,
    cmap="viridis",
    s=25,
    alpha=0.75
)

handles = []
class_values = list(pd.unique(y))
for code, label in enumerate(class_values):
    handles.append(
        plt.Line2D(
            [], [],
            marker="o",
            linestyle="",
            color=plt.cm.viridis(
                code / max(len(class_values) - 1, 1)
            ),
            label=f"Class {label}"
        )
    )

plt.legend(handles=handles, title="Target Classes")
plt.title("2D t-SNE Visualization of the Dataset")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "tsne_scatter_plot.png"),
    dpi=300
)
plt.show()
print("\nt-SNE plot saved successfully.")

# ------------------------------------------------------------
# 5. DEFINE SVM AND DECISION TREE MODELS
# ------------------------------------------------------------

models = {
    "SVM_Linear": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="linear"))
    ]),
    "SVM_Polynomial": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="poly", degree=3))
    ]),
    "SVM_RBF": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf"))
    ])
}

# Compare different Decision Tree max_depth and min_samples_split values.
for depth in [3, 5, None]:
    for min_split in [2, 10, 20]:
        model_name = f"DT_depth{depth}_split{min_split}"
        models[model_name] = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )

# ------------------------------------------------------------
# 6. TRAIN, EVALUATE AND MEASURE RESOURCE USAGE
# ------------------------------------------------------------

results = []
class_labels = sorted(y.unique())

for name, model in models.items():
    print("\n" + "-" * 65)
    print("Model:", name)

    tracemalloc.start()
    start_time = time.perf_counter()
    start_cpu = time.process_time()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    elapsed_time = time.perf_counter() - start_time
    cpu_time = time.process_time() - start_cpu
    current_memory, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    recall = recall_score(
        y_test, y_pred, average="weighted", zero_division=0
    )
    f1 = f1_score(
        y_test, y_pred, average="weighted", zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f1,
        "Execution_Time_sec": elapsed_time,
        "CPU_Time_sec": cpu_time,
        "Peak_Memory_MB": peak_memory / (1024 ** 2)
    })

    print(f"Accuracy:       {accuracy:.4f}")
    print(f"Precision:      {precision:.4f}")
    print(f"Recall:         {recall:.4f}")
    print(f"F1-Score:       {f1:.4f}")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    print(f"CPU time:       {cpu_time:.4f} seconds")
    print(f"Peak memory:    {peak_memory / (1024 ** 2):.2f} MB")

    print("\nClassification Report:")
    print(classification_report(
        y_test, y_pred, labels=class_labels, zero_division=0
    ))

    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    plt.xticks(range(len(class_labels)), class_labels)
    plt.yticks(range(len(class_labels)), class_labels)

    # Add values inside the confusion matrix
    threshold = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black"
            )
    plt.title(f"Confusion Matrix: {name}")
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.tight_layout()

    safe_name = re.sub(r"[^A-Za-z0-9_-]", "_", name)
    plt.savefig(
        os.path.join(OUTPUT_DIR, f"confusion_matrix_{safe_name}.png"),
        dpi=300
    )
    plt.close()

# ------------------------------------------------------------
# 7. SAVE AND DISPLAY COMPARISON RESULTS
# ------------------------------------------------------------

results_df = pd.DataFrame(results).sort_values(
    by="Accuracy", ascending=False
)

print("\n" + "=" * 90)
print("FINAL MODEL COMPARISON")
print("=" * 90)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    os.path.join(OUTPUT_DIR, "model_comparison_results.csv"),
    index=False
)

# ------------------------------------------------------------
# 8. PLOT MODEL ACCURACY COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(13, 7))
plt.bar(results_df["Model"], results_df["Accuracy"])
plt.title("Accuracy Comparison: SVM and Decision Tree")
plt.xlabel("Classification Model")
plt.ylabel("Accuracy")
plt.xticks(rotation=65, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "model_accuracy_comparison.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 9. PLOT EXECUTION TIME COMPARISON
# ------------------------------------------------------------

plt.figure(figsize=(13, 7))
plt.bar(results_df["Model"], results_df["Execution_Time_sec"])
plt.title("Execution Time Comparison")
plt.xlabel("Classification Model")
plt.ylabel("Execution Time (seconds)")
plt.xticks(rotation=65, ha="right")
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, "execution_time_comparison.png"),
    dpi=300
)
plt.show()

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\nAll experiments completed successfully.")
print("Results folder:", os.path.abspath(OUTPUT_DIR))
print("Saved files include:")
print("- tsne_scatter_plot.png")
print("- model_comparison_results.csv")
print("- model_accuracy_comparison.png")
print("- execution_time_comparison.png")
print("- Confusion matrix images for every model")
