# ================================================================
# ASSIGNMENT 3 - CLASSIFICATION
# WATER POTABILITY DATASET
#
# SVM: Linear, Polynomial, RBF
# Decision Tree: max_depth and min_samples_split
# t-SNE: 2D visualization
# Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, Time
# ================================================================

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ================================================================
# 1. LOAD DATASET
# ================================================================

# IMPORTANT:
# Keep the CSV file in the SAME folder as this Python file.
# If your CSV has a different name, change the line below.

FILE_NAME = "water_potability(3).csv"

print("\n================================================")
print("ASSIGNMENT 3 - CLASSIFICATION")
print("================================================")

print("\nCurrent working folder:")
print(os.getcwd())

print("\nLooking for dataset:", FILE_NAME)

if not os.path.exists(FILE_NAME):
    print("\nERROR: CSV file was not found!")
    print("Put the CSV file in the same folder as this Python file.")
    print("\nFiles currently found in this folder:")
    print(os.listdir())
    raise FileNotFoundError(
        "\nPlease put 'water_potability(3).csv' in the same folder."
    )

df = pd.read_csv(FILE_NAME)

print("\nDataset loaded successfully!")
print("Dataset shape:", df.shape)

# ================================================================
# 2. DATASET INFORMATION
# ================================================================

print("\n================ DATASET INFORMATION ================")

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(list(df.columns))

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["Potability"].value_counts())

# ================================================================
# 3. SEPARATE FEATURES AND TARGET
# ================================================================

X = df.drop("Potability", axis=1)
y = df["Potability"]

print("\nNumber of input features:", X.shape[1])
print("Target variable: Potability")

# ================================================================
# 4. HANDLE MISSING VALUES
# ================================================================

print("\n================ DATA PREPROCESSING ================")

imputer = SimpleImputer(strategy="median")

X = pd.DataFrame(
    imputer.fit_transform(X),
    columns=X.columns
)

print("\nMissing values after median imputation:")
print(X.isnull().sum())

# ================================================================
# 5. TRAIN-TEST SPLIT
# ================================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))

# ================================================================
# 6. FEATURE SCALING
# ================================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ================================================================
# 7. RESULTS STORAGE
# ================================================================

results = []


# ================================================================
# 8. MODEL EVALUATION FUNCTION
# ================================================================

def evaluate_model(model, model_name, Xtr, Xte):

    print("\n================================================")
    print(model_name)
    print("================================================")

    start_time = time.perf_counter()

    model.fit(Xtr, y_train)

    y_pred = model.predict(Xte)

    end_time = time.perf_counter()

    execution_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Execution Time (sec)": execution_time
    })

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1-Score :", round(f1, 4))
    print("Time     :", round(execution_time, 6), "seconds")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Potable", "Potable"],
        yticklabels=["Not Potable", "Potable"]
    )

    plt.title("Confusion Matrix - " + model_name)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    return model


# ================================================================
# 9. SVM - LINEAR KERNEL
# ================================================================

svm_linear = SVC(
    kernel="linear",
    C=1.0
)

evaluate_model(
    svm_linear,
    "SVM - Linear Kernel",
    X_train_scaled,
    X_test_scaled
)


# ================================================================
# 10. SVM - POLYNOMIAL KERNEL
# ================================================================

svm_poly = SVC(
    kernel="poly",
    degree=3,
    C=1.0
)

evaluate_model(
    svm_poly,
    "SVM - Polynomial Kernel",
    X_train_scaled,
    X_test_scaled
)


# ================================================================
# 11. SVM - RBF KERNEL
# ================================================================

svm_rbf = SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)

evaluate_model(
    svm_rbf,
    "SVM - RBF Kernel",
    X_train_scaled,
    X_test_scaled
)


# ================================================================
# 12. DECISION TREE - MAX DEPTH
# ================================================================

depth_values = [3, 5, 10, 15]

for depth in depth_values:

    tree = DecisionTreeClassifier(
        max_depth=depth,
        random_state=42
    )

    evaluate_model(
        tree,
        "Decision Tree - max_depth=" + str(depth),
        X_train,
        X_test
    )


# ================================================================
# 13. DECISION TREE - MIN SAMPLES SPLIT
# ================================================================

split_values = [2, 10, 20]

for split in split_values:

    tree = DecisionTreeClassifier(
        min_samples_split=split,
        random_state=42
    )

    evaluate_model(
        tree,
        "Decision Tree - min_samples_split=" + str(split),
        X_train,
        X_test
    )


# ================================================================
# 14. FINAL RESULTS TABLE
# ================================================================

results_df = pd.DataFrame(results)

print("\n\n================================================")
print("FINAL MODEL COMPARISON")
print("================================================")

print(
    results_df.to_string(index=False)
)

# Save results
results_df.to_csv(
    "classification_results.csv",
    index=False
)

print("\nResults saved as: classification_results.csv")


# ================================================================
# 15. ACCURACY GRAPH
# ================================================================

plt.figure(figsize=(13, 6))

plt.bar(
    results_df["Model"],
    results_df["Accuracy"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Accuracy")
plt.title("Accuracy Comparison of Classification Models")
plt.ylim(0, 1)

plt.tight_layout()
plt.show()


# ================================================================
# 16. PRECISION, RECALL AND F1 GRAPH
# ================================================================

x = np.arange(len(results_df))
width = 0.25

plt.figure(figsize=(14, 6))

plt.bar(
    x - width,
    results_df["Precision"],
    width,
    label="Precision"
)

plt.bar(
    x,
    results_df["Recall"],
    width,
    label="Recall"
)

plt.bar(
    x + width,
    results_df["F1-Score"],
    width,
    label="F1-Score"
)

plt.xticks(
    x,
    results_df["Model"],
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.title("Precision, Recall and F1-Score Comparison")
plt.ylim(0, 1)
plt.legend()

plt.tight_layout()
plt.show()


# ================================================================
# 17. EXECUTION TIME GRAPH
# ================================================================

plt.figure(figsize=(13, 6))

plt.bar(
    results_df["Model"],
    results_df["Execution Time (sec)"]
)

plt.xticks(
    rotation=60,
    ha="right"
)

plt.xlabel("Model")
plt.ylabel("Execution Time (seconds)")
plt.title("Execution Time Comparison")

plt.tight_layout()
plt.show()


# ================================================================
# 18. 2D t-SNE VISUALIZATION
# ================================================================

print("\n================================================")
print("2D t-SNE VISUALIZATION")
print("================================================")

# Scale the complete dataset
X_scaled_all = scaler.fit_transform(X)

print("\nRunning t-SNE...")
print("Please wait...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    max_iter=1000,
    random_state=42
)

X_tsne = tsne.fit_transform(X_scaled_all)

print("t-SNE completed!")

plt.figure(figsize=(9, 7))

scatter = plt.scatter(
    X_tsne[:, 0],
    X_tsne[:, 1],
    c=y,
    cmap="viridis",
    alpha=0.7
)

plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.title(
    "2D t-SNE Visualization - Water Potability Dataset"
)

plt.colorbar(
    scatter,
    label="Potability"
)

plt.tight_layout()
plt.show()


# ================================================================
# 19. SAVE t-SNE DATA
# ================================================================

tsne_df = pd.DataFrame({
    "tSNE_1": X_tsne[:, 0],
    "tSNE_2": X_tsne[:, 1],
    "Potability": y.values
})

tsne_df.to_csv(
    "tsne_results.csv",
    index=False
)

print("\nt-SNE results saved as: tsne_results.csv")


# ================================================================
# 20. FINAL MESSAGE
# ================================================================

print("\n================================================")
print("ASSIGNMENT 3 COMPLETED SUCCESSFULLY")
print("================================================")

print("\nDataset:", FILE_NAME)
print("Samples:", len(df))
print("Features:", X.shape[1])
print("Target:", "Potability")

print("\nAlgorithms completed:")
print("1. SVM - Linear Kernel")
print("2. SVM - Polynomial Kernel")
print("3. SVM - RBF Kernel")
print("4. Decision Tree - max_depth")
print("5. Decision Tree - min_samples_split")

print("\nRequired evaluation completed:")
print("- Accuracy")
print("- Precision")
print("- Recall")
print("- F1-Score")
print("- Confusion Matrix")
print("- Execution Time")
print("- 2D t-SNE Visualization")

print("\nGenerated files:")
print("- classification_results.csv")
print("- tsne_results.csv")
