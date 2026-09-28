import os
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
    confusion_matrix,
    classification_report
)

# ============================================================
# 1. PROJECT SETUP
# ============================================================

os.makedirs("../figures", exist_ok=True)
os.makedirs("../figures/tsne", exist_ok=True)
os.makedirs("../figures/confusion_matrix", exist_ok=True)
os.makedirs("../figures/comparison", exist_ok=True)
os.makedirs("../results", exist_ok=True)

print("Project folders verified.")


# ============================================================
# 2. LOAD DATASET
# ============================================================

data_path = "../Data/HTRU_2.csv"

column_names = [
    "Mean_IP",
    "SD_IP",
    "Excess_Kurtosis_IP",
    "Skewness_IP",
    "Mean_DM_SNR",
    "SD_DM_SNR",
    "Excess_Kurtosis_DM_SNR",
    "Skewness_DM_SNR",
    "Class"
]

df = pd.read_csv(
    data_path,
    header=None,
    names=column_names
)

print("\nDATASET INFORMATION")
print("=" * 60)
print("Dataset shape:", df.shape)

display(df.head())


# ============================================================
# 3. DATASET EXPLORATION
# ============================================================

print("\nCOLUMN INFORMATION")
print("=" * 60)
print(df.dtypes)

print("\nMISSING VALUES")
print("=" * 60)
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print("=" * 60)
print(df.duplicated().sum())

print("\nDESCRIPTIVE STATISTICS")
print("=" * 60)
display(df.describe())


# ============================================================
# 4. FEATURES AND TARGET
# ============================================================

X = df.drop("Class", axis=1)
y = df["Class"]

X = X.apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.mean())

print("\nMODEL DATA")
print("=" * 60)
print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", sorted(y.unique()))

print("\nCLASS DISTRIBUTION")
print("=" * 60)
display(y.value_counts().sort_index())


# ============================================================
# 5. CLASS DISTRIBUTION PLOT
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(
    x=y
)

plt.title("Pulsar vs Non-Pulsar Class Distribution")
plt.xlabel("Class")
plt.ylabel("Number of Samples")
plt.xticks(
    [0, 1],
    ["Non-Pulsar", "Pulsar"]
)

plt.tight_layout()

plt.savefig(
    "../figures/class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 6. FEATURE DISTRIBUTIONS
# ============================================================

X.hist(
    figsize=(14, 10),
    bins=30
)

plt.suptitle(
    "Feature Distributions",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "../figures/feature_distributions.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 7. CORRELATION HEATMAP
# ============================================================

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    "../figures/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 8. TRAIN-TEST SPLIT
# ============================================================

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTRAIN-TEST SPLIT")
print("=" * 60)
print("Training samples:", X_train_raw.shape[0])
print("Testing samples:", X_test_raw.shape[0])


# ============================================================
# 9. FEATURE STANDARDIZATION
# ============================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

X_scaled = scaler.transform(X)

print("\nFEATURE STANDARDIZATION COMPLETED")
print("Training feature matrix:", X_train.shape)
print("Testing feature matrix:", X_test.shape)


# ============================================================
# 10. 2D t-SNE VISUALIZATION
# ============================================================

print("\nGenerating 2D t-SNE visualization...")

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    max_iter=1000
)

X_tsne = tsne.fit_transform(X_scaled)

tsne_df = pd.DataFrame({
    "TSNE1": X_tsne[:, 0],
    "TSNE2": X_tsne[:, 1],
    "Class": y.values
})

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=tsne_df,
    x="TSNE1",
    y="TSNE2",
    hue="Class",
    palette="Set1",
    alpha=0.65,
    s=30
)

plt.title("2D t-SNE Visualization of Pulsar Feature Space")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")

plt.legend(
    title="Class",
    labels=["Non-Pulsar", "Pulsar"]
)

plt.tight_layout()

plt.savefig(
    "../figures/tsne/tsne_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()

print("t-SNE plot saved.")


# ============================================================
# 11. MODEL EVALUATION FUNCTION
# ============================================================

results = []


def evaluate_model(model, model_name):

    process = psutil.Process(os.getpid())

    memory_before = (
        process.memory_info().rss / (1024 ** 2)
    )

    start_train = time.perf_counter()

    model.fit(
        X_train,
        y_train
    )

    train_time = (
        time.perf_counter() - start_train
    )

    start_prediction = time.perf_counter()

    y_pred = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter() - start_prediction
    )

    memory_after = (
        process.memory_info().rss / (1024 ** 2)
    )

    memory_used = max(
        memory_after - memory_before,
        0
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        pos_label=1,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    safe_name = (
        model_name
        .replace(" ", "_")
        .replace("=", "")
        .replace(",", "_")
    )

    # Confusion Matrix
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Non-Pulsar", "Pulsar"],
        yticklabels=["Non-Pulsar", "Pulsar"]
    )

    plt.title(
        f"Confusion Matrix - {model_name}"
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.tight_layout()

    plt.savefig(
        f"../figures/confusion_matrix/cm_{safe_name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close()

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Train Time (s)": train_time,
        "Prediction Time (s)": prediction_time,
        "Total Time (s)": train_time + prediction_time,
        "Memory Used (MB)": memory_used,
        "Memory After (MB)": memory_after,
        "Confusion Matrix": cm,
        "Classification Report": classification_report(
            y_test,
            y_pred,
            target_names=[
                "Non-Pulsar",
                "Pulsar"
            ],
            zero_division=0
        )
    }


# ============================================================
# 12. SVM MODELS
# ============================================================

print("\nSVM MODELS")
print("=" * 60)

for kernel in ["linear", "poly", "rbf"]:

    print(
        f"\nTraining SVM with {kernel.upper()} kernel..."
    )

    svm_model = SVC(
        kernel=kernel,
        random_state=42
    )

    result = evaluate_model(
        svm_model,
        f"SVM - {kernel}"
    )

    results.append(result)

print("\nAll SVM models completed.")


# ============================================================
# 13. DECISION TREE MODELS
# ============================================================

print("\nDECISION TREE MODELS")
print("=" * 60)

for depth in [3, 5, 10, None]:

    for min_split in [2, 5, 10]:

        model_name = (
            f"Decision Tree - "
            f"depth={depth}, "
            f"min_split={min_split}"
        )

        print(
            f"\nTraining {model_name}..."
        )

        dt_model = DecisionTreeClassifier(
            max_depth=depth,
            min_samples_split=min_split,
            random_state=42
        )

        result = evaluate_model(
            dt_model,
            model_name
        )

        results.append(result)

print("\nAll Decision Tree models completed.")


# ============================================================
# 14. COMPARISON TABLE
# ============================================================

comparison_df = pd.DataFrame([
    {
        "Model": result["Model"],
        "Accuracy": result["Accuracy"],
        "Precision": result["Precision"],
        "Recall": result["Recall"],
        "F1-Score": result["F1-Score"],
        "Train Time (s)": result["Train Time (s)"],
        "Prediction Time (s)": result["Prediction Time (s)"],
        "Total Time (s)": result["Total Time (s)"],
        "Memory Used (MB)": result["Memory Used (MB)"]
    }
    for result in results
])

comparison_df = comparison_df.sort_values(
    by="F1-Score",
    ascending=False
)

print("\nFINAL MODEL COMPARISON")
print("=" * 100)

display(comparison_df)

comparison_df.to_csv(
    "../results/comparison_results.csv",
    index=False
)


# ============================================================
# 15. SAVE CLASSIFICATION REPORTS
# ============================================================

with open(
    "../results/classification_reports.txt",
    "w"
) as file:

    for result in results:

        file.write("\n")
        file.write("=" * 70)
        file.write("\n")
        file.write(
            f"Model: {result['Model']}\n"
        )
        file.write("=" * 70)
        file.write("\n")

        file.write(
            result["Classification Report"]
        )

        file.write("\n")

print(
    "Classification reports saved."
)


# ============================================================
# 16. PERFORMANCE COMPARISON
# ============================================================

metrics = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1-Score"
]

plt.figure(figsize=(15, 7))

comparison_df.set_index(
    "Model"
)[metrics].plot(
    kind="bar",
    figsize=(15, 7)
)

plt.title(
    "Performance Comparison of SVM and Decision Tree Models"
)

plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1.05)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.legend(
    title="Metrics"
)

plt.tight_layout()

plt.savefig(
    "../figures/comparison/performance_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 17. EXECUTION TIME COMPARISON
# ============================================================

plt.figure(figsize=(15, 7))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="Total Time (s)"
)

plt.title(
    "Execution Time Comparison"
)

plt.xlabel("Model")
plt.ylabel("Total Time (seconds)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "../figures/comparison/time_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 18. MEMORY UTILIZATION COMPARISON
# ============================================================

plt.figure(figsize=(15, 7))

sns.barplot(
    data=comparison_df,
    x="Model",
    y="Memory Used (MB)"
)

plt.title(
    "Memory Utilization Comparison"
)

plt.xlabel("Model")
plt.ylabel("Memory Used (MB)")

plt.xticks(
    rotation=45,
    ha="right"
)

plt.tight_layout()

plt.savefig(
    "../figures/comparison/resource_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 19. SAVE RESOURCE UTILIZATION
# ============================================================

resource_df = comparison_df[
    [
        "Model",
        "Train Time (s)",
        "Prediction Time (s)",
        "Total Time (s)",
        "Memory Used (MB)",
        "Memory After (MB)"
    ]
]

resource_df.to_csv(
    "../results/resource_utilization.csv",
    index=False
)


# ============================================================
# 20. SAVE FINAL MODEL SUMMARY
# ============================================================

best_model = comparison_df.iloc[0]

summary = pd.DataFrame({
    "Parameter": [
        "Dataset",
        "Number of Samples",
        "Number of Features",
        "Number of Classes",
        "Training Samples",
        "Testing Samples",
        "Best Model by F1-Score",
        "Best Model F1-Score"
    ],
    "Value": [
        "HTRU_2 Pulsar Dataset",
        len(df),
        X.shape[1],
        y.nunique(),
        len(X_train),
        len(X_test),
        best_model["Model"],
        best_model["F1-Score"]
    ]
})

summary.to_csv(
    "../results/final_model_summary.csv",
    index=False
)


# ============================================================
# 21. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 100)
print("ASSIGNMENT 3 ANALYSIS COMPLETED")
print("=" * 100)

print("\nDataset:")
print("HTRU_2 Pulsar Star Classification")

print("\nSamples:", len(df))
print("Features:", X.shape[1])
print("Classes:", y.nunique())

print("\nBest model based on F1-Score:")
print(best_model["Model"])

print(
    f"F1-Score: {best_model['F1-Score']:.4f}"
)

print("\nResults saved in:")
print("../results/")

print("\nFigures saved in:")
print("../figures/")

print("\nGenerated model configurations:")
print(len(results))