# Assignment 3 — Classification

**Dataset:** [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification) (Kaggle)

## 1. Dataset

The dataset contains specifications of mobile phones (battery power, RAM,
camera quality, screen size, connectivity flags, etc.) and a target column
`price_range` with four classes (0 = low cost, 1 = medium cost, 2 = high
cost, 3 = very high cost). It has 2000 rows and 20 features, evenly split
across the four price classes (500 rows each).

## 2. What the code does (`classification_assignment3.py`)

1. **Load & preprocess:** Reads `train.csv`, splits it into features `X`
   and target `y` (`price_range`), and standardizes the features with
   `StandardScaler` (important for SVM, and it makes the t-SNE embedding
   more meaningful since features are on very different scales, e.g. `ram`
   in the thousands vs `blue` as 0/1).

2. **t-SNE visualization:** Projects the full, scaled feature space down to
   2 dimensions with `sklearn.manifold.TSNE` and saves a scatter plot
   (`tsne_plot.png`), color-coded by `price_range`, to visualize how
   separable the four price classes are before any model is trained.

3. **Train/test split:** 80/20 split, stratified by class so all four price
   ranges are proportionally represented in both sets.

4. **SVM comparison (three kernels):** Trains an `SVC` with `linear`,
   `poly`, and `rbf` kernels (default hyperparameters otherwise) on the
   same train/test split.

5. **Decision Tree comparison:**
   - **Varying `max_depth`:** `2, 4, 8, None` (fully grown) — shows the
     classic underfitting → good fit → overfitting progression.
   - **Varying `min_samples_split`:** `2, 10, 50, 100` — shows how forcing
     larger splits regularizes the tree.

6. **Metrics, for every one of the 11 models trained:**
   - Accuracy
   - Precision (macro-averaged, since this is a 4-class problem)
   - Recall (macro-averaged)
   - F1-score (macro-averaged)
   - Confusion matrix (saved as its own PNG per model)
   - Training time
   - Peak memory used during training

7. **Outputs saved:**
   - `tsne_plot.png` — the 2D t-SNE class-separation plot
   - `confusion_matrix_<model>.png` — one per model (11 total)
   - `compare_accuracy.png`, `compare_f1.png`, `compare_time.png`,
     `compare_memory.png` — bar charts comparing all 11 models
   - `results.csv` — full numeric results table (one row per model)

## 3. How to run

```bash
pip install pandas numpy scikit-learn matplotlib
python classification_assignment3.py
```

Or open it in Google Colab, upload `train.csv` when prompted, and run all
cells in order.

## 4. Results

| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) |
|---|---|---|---|---|
| SVM (linear kernel) | 0.9625 | 0.9627 | 0.9625 | 0.9625 |
| SVM (RBF kernel) | 0.8950 | 0.8969 | 0.8950 | 0.8956 |
| SVM (poly kernel) | 0.8050 | 0.8155 | 0.8050 | 0.8073 |
| DT max_depth=8 | 0.8500 | 0.8523 | 0.8500 | 0.8501 |
| DT min_samples_split=10 | 0.8500 | 0.8546 | 0.8500 | 0.8511 |
| DT max_depth=None (full) | 0.8300 | 0.8319 | 0.8300 | 0.8302 |
| DT min_samples_split=50 | 0.8375 | 0.8461 | 0.8375 | 0.8399 |
| DT min_samples_split=100 | 0.8000 | 0.8040 | 0.8000 | 0.8012 |
| DT max_depth=4 | 0.7900 | 0.8015 | 0.7900 | 0.7888 |
| DT max_depth=2 | 0.7425 | 0.7572 | 0.7425 | 0.7459 |

Full per-model timing and memory figures are in `results.csv`.

## 5. Interpretation

- **Linear SVM performed best by a clear margin (96.25% accuracy)**, ahead
  of RBF and polynomial kernels. The t-SNE plot shows the four price
  classes forming a smooth, continuous gradient rather than tight, well-
  separated clusters — this kind of structure suits a simple linear
  decision boundary better than more flexible kernels, which can overfit
  to local noise instead of capturing the overall trend.
- **Decision Tree — `max_depth`:** Accuracy rises from a shallow tree
  (`max_depth=2`, 74.25%) up to `max_depth=8` (85.00%), then drops slightly
  for a fully grown tree (`max_depth=None`, 83.00%) — a textbook
  underfitting-then-overfitting curve.
- **Decision Tree — `min_samples_split`:** A moderate value
  (`min_samples_split=10`, 85.00%) performs best; both very small
  (`=2`, unrestricted) and very large (`=100`, overly restrictive) values
  give worse results, showing the regularizing effect of this parameter.
- Decision trees trained roughly 3-6x faster than the SVM models on this
  dataset (see `compare_time.png`), though all model/dataset combinations
  here train in well under a second.

## 6. Files in this submission

```
/26CEM5R10/Assignment3/
├── classification_assignment3.py
├── train.csv
├── README.md
├── results.csv
├── tsne_plot.png
├── confusion_matrix_SVM_(linear_kernel).png
├── confusion_matrix_SVM_(poly_kernel).png
├── confusion_matrix_SVM_(rbf_kernel).png
├── confusion_matrix_DT_max_depth=2.png
├── confusion_matrix_DT_max_depth=4.png
├── confusion_matrix_DT_max_depth=8.png
├── confusion_matrix_DT_max_depth=None_(full).png
├── confusion_matrix_DT_min_samples_split=2.png
├── confusion_matrix_DT_min_samples_split=10.png
├── confusion_matrix_DT_min_samples_split=50.png
├── confusion_matrix_DT_min_samples_split=100.png
├── compare_accuracy.png
├── compare_f1.png
├── compare_time.png
└── compare_memory.png
```