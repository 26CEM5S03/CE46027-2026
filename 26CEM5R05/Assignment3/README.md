
# Rice Classification - 26CEM5R05

## 1. Dataset
File: riceClassification.csv, 3810 rows, 7 features + Class (Osmancik, Cammeo)

## 2. Preprocessing
Train-Test 80-20 stratified random_state=7, StandardScaler for SVM.

## 3. t-SNE
File: tsne_rice.png, perplexity=30. Shows 2 clusters with slight overlap.

## 4. Results - From My Run (Exact)

Model       Params          Accuracy    TrainTime   InferTime
SVM-linear  linear          0.964912    0.003427    0.000328
SVM-poly    poly            0.894737    0.003373    0.0006
SVM-rbf     rbf             0.964912    0.004313    0.000998
DT          depth=3_s=2     0.912281    0           0
DT          depth=3_s=5     0.912281    0           0
DT          depth=5_s=2     0.921053    0           0
DT          depth=5_s=5     0.929825    0           0
DT          depth=10_s=2    0.921053    0           0
DT          depth=10_s=5    0.929825    0           0
DT          depth=None_s=2  0.921053    0           0
DT          depth=None_s=5  0.929825    0           0

## 5. Analysis With My Results
- Best Model: SVM-linear and SVM-rbf both 96.49% (tie). Linear is preferred as it is simpler and inference faster (0.000328s vs 0.000998s).
- Worst: SVM-poly 89.47% - poly kernel overfits for this dataset.
- DT Best: 92.98% with depth=5_split=5, depth=10_split=5, and depth=None_split=5 (all same 0.929825).
- Computational: SVM train time ~0.003-0.004 sec, DT negligible. RBF train heaviest (0.004313s). Inference: Linear fastest (0.000328s).

## 6. Conclusion
For riceClassification.csv, linear separability is high, so linear SVM achieves 96.49% with fastest inference, making it ideal. RBF also 96.49% but costlier. DT reaches 92.98% max, good for interpretability but lower accuracy.

## 7. How to Run
pip install pandas scikit-learn matplotlib seaborn
python data.py

## 8. Author
26CEM5R05 - Assignment 3