import pandas as pd
import matplotlib.pyplot as plt
import time, psutil, os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 1. Load
df = pd.read_csv("gender_classification_v7.csv")
print(df.shape)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

le = LabelEncoder()
y = le.fit_transform(y) # Female=0, Male=1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_full_s = scaler.fit_transform(X)

# 2. t-SNE Plot
print("Generating t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_full_s)

plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap='coolwarm', alpha=0.6, s=12)
plt.colorbar(label='Gender')
plt.title('2D t-SNE Visualization - Color coded by Gender')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.savefig('tsne_plot.png', dpi=300)

# 3. Evaluation function
def evaluate(model, name):
    proc = psutil.Process(os.getpid())
    mem_before = proc.memory_info().rss / 1024**2
    start = time.time()
    
    model.fit(X_train_s, y_train)
    
    elapsed = time.time() - start
    mem_after = proc.memory_info().rss / 1024**2
    y_pred = model.predict(X_test_s)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")
    print(f"Time: {elapsed:.4f}s | Memory: {mem_after-mem_before:.2f}MB")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return [name, acc, prec, rec, f1, elapsed, mem_after-mem_before]

results = []

# SVM - 3 kernels
for k in ['linear', 'poly', 'rbf']:
    results.append(evaluate(SVC(kernel=k, random_state=42), f"SVM_{k}"))

# Decision Tree - varying max_depth and min_samples_split
for depth in [3, 5, 10, None]:
    for min_split in [2, 5, 10]:
        results.append(evaluate(DecisionTreeClassifier(max_depth=depth, min_samples_split=min_split, random_state=42),
                                f"DT_depth={depth}_minSplit={min_split}"))

# Save results
results_df = pd.DataFrame(results, columns=["Model","Accuracy","Precision","Recall","F1","Time","Memory_MB"])
results_df.to_csv("comparison_results.csv", index=False)
print("\nSaved: comparison_results.csv and tsne_plot.png")