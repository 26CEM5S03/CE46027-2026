
import pandas as pd, numpy as np, time, csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("riceClassification.csv")
if 'id' in df.columns:
    df = df.drop(columns=['id'])
X = df.drop(columns=['Class'])
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
X_full = scaler.fit_transform(X)

# t-SNE
print("t-SNE...")
idx = np.random.choice(len(X_full), 3000, replace=False)
tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_full[idx])
plt.figure(figsize=(8,6))
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y.iloc[idx], cmap='coolwarm', alpha=0.6, s=15)
plt.title('2D t-SNE of Rice Feature Space'); plt.savefig('tsne_plot.png', dpi=300); plt.close()
print("Saved tsne_plot.png")

# --- SAVE COMPARISON FILE ---
with open('comparison_results.csv','w',newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Model','Kernel/Params','Accuracy','F1-Score','Train_Time(s)','Infer_Time(s)','Precision_0','Recall_0','Precision_1','Recall_1'])

    print("\nSVM Comparison:")
    for kernel in ['linear','poly','rbf']:
        clf = SVC(kernel=kernel, random_state=42)
        t0=time.time(); clf.fit(X_train_s, y_train); train_t=time.time()-t0
        t0=time.time(); y_pred=clf.predict(X_test_s); infer_t=time.time()-t0
        acc=accuracy_score(y_test,y_pred)
        f1=f1_score(y_test,y_pred,average='weighted')
        report=classification_report(y_test,y_pred,output_dict=True)
        print(f"SVM {kernel} Acc={acc:.4f}")
        writer.writerow([f'SVM', kernel, f"{acc:.4f}", f"{f1:.4f}", f"{train_t:.3f}", f"{infer_t:.5f}",
                         f"{report['0']['precision']:.3f}", f"{report['0']['recall']:.3f}", f"{report['1']['precision']:.3f}", f"{report['1']['recall']:.3f}"])
        print(confusion_matrix(y_test,y_pred))

    print("\nDT Comparison:")
    for md in [3,5,10,None]:
        for ms in [2,5,10]:
            clf = DecisionTreeClassifier(max_depth=md, min_samples_split=ms, random_state=42)
            t0=time.time(); clf.fit(X_train_s, y_train); train_t=time.time()-t0
            t0=time.time(); y_pred=clf.predict(X_test_s); infer_t=time.time()-t0
            acc=accuracy_score(y_test,y_pred)
            f1=f1_score(y_test,y_pred,average='weighted')
            print(f"DT depth={md} split={ms} Acc={acc:.4f}")
            writer.writerow([f'DT', f'depth={md}_split={ms}', f"{acc:.4f}", f"{f1:.4f}", f"{train_t:.3f}", f"{infer_t:.5f}", '-','-','-','-'])

print("\nDONE - Created comparison_results.csv and tsne_plot.png")