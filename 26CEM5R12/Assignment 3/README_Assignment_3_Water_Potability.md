# Assignment 3 - Classification: Water Potability

## 1. Title
**Comparison of SVM and Decision Tree Classifiers for Water Potability Classification**

## 2. Objective

The objective of this assignment is to classify water samples based on their potability and compare the performance of Support Vector Machine (SVM) classifiers with different kernels and Decision Tree classifiers with different structural parameters.

The assignment also includes a 2D t-SNE visualization of the feature space before/alongside model evaluation.

## 3. Dataset

**Dataset:** Water Potability Dataset  
**File:** `water_potability(3).csv`  
**Target variable:** `Potability`

Target classes:

- `0` - Not Potable
- `1` - Potable

The dataset contains water-quality measurements used to predict whether a water sample is potable.

### Input Features

1. `ph`
2. `Hardness`
3. `Solids`
4. `Chloramines`
5. `Sulfate`
6. `Conductivity`
7. `Organic_carbon`
8. `Trihalomethanes`
9. `Turbidity`

## 4. Software and Libraries

The Python program is designed to run in Spyder/Anaconda.

Required Python libraries:

- Python
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install missing packages, if required, using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 5. Project Files

Keep the Python script and CSV dataset in the same folder:

```text
Assignment_3/
│
├── Assignment_3_Water_Potability_FINAL.py
├── water_potability(3).csv
├── classification_results.csv
└── tsne_results.csv
```

The last two files are generated automatically after successful execution.

## 6. Data Preprocessing

The Python program performs the following preprocessing steps:

1. Loads the CSV dataset using pandas.
2. Checks the dataset shape and column names.
3. Checks missing values.
4. Separates the input features (`X`) and target (`y`).
5. Handles missing feature values using **median imputation**.
6. Splits the data into training and testing sets.
7. Uses an **80:20 train-test split**.
8. Uses `random_state=42`.
9. Uses stratification based on the target class.
10. Applies `StandardScaler` to the training and testing data for SVM models.

## 7. Parameters Used

### Train-Test Split

```text
test_size = 0.20
random_state = 42
stratify = y
```

### Missing Value Treatment

```text
SimpleImputer(strategy="median")
```

### Feature Scaling

```text
StandardScaler()
```

## 8. SVM Classification

Three SVM kernels are implemented as required by the assignment.

### 8.1 Linear Kernel

```python
SVC(
    kernel="linear",
    C=1.0
)
```

### 8.2 Polynomial Kernel

```python
SVC(
    kernel="poly",
    degree=3,
    C=1.0
)
```

### 8.3 RBF Kernel

```python
SVC(
    kernel="rbf",
    C=1.0,
    gamma="scale"
)
```

## 9. Decision Tree Classification

The program evaluates Decision Tree performance by changing two structural parameters.

### 9.1 Varying `max_depth`

The following values are tested:

```text
3
5
10
15
```

Example:

```python
DecisionTreeClassifier(
    max_depth=depth,
    random_state=42
)
```

### 9.2 Varying `min_samples_split`

The following values are tested:

```text
2
10
20
```

Example:

```python
DecisionTreeClassifier(
    min_samples_split=split,
    random_state=42
)
```

## 10. Evaluation Metrics

Every model is evaluated using the metrics required by the assignment:

### Accuracy

Measures the proportion of correctly classified samples.

### Precision

Measures how many samples predicted as a particular positive class are actually positive.

### Recall

Measures how many actual positive samples are correctly identified.

### F1-Score

Provides a combined measure of precision and recall.

### Confusion Matrix

Shows:

- True Negative
- False Positive
- False Negative
- True Positive

### Execution Time

The training and prediction execution time is measured using Python's `time.perf_counter()`.

## 11. 2D t-SNE Visualization

The program performs dimensionality reduction using **t-SNE (t-Distributed Stochastic Neighbor Embedding)**.

Parameters used:

```text
n_components = 2
perplexity = 30
max_iter = 1000
random_state = 42
```

The 9-dimensional water-quality feature space is transformed into two dimensions and plotted as a scatter plot.

The points are color-coded according to the `Potability` class.

## 12. Output Graphs

The program generates:

1. Confusion matrix for every tested model.
2. Accuracy comparison graph.
3. Precision, Recall and F1-Score comparison graph.
4. Execution-time comparison graph.
5. 2D t-SNE scatter plot.

## 13. Output Files

After successful execution, the program creates:

### `classification_results.csv`

Contains:

- Model
- Accuracy
- Precision
- Recall
- F1-Score
- Execution Time

### `tsne_results.csv`

Contains:

- `tSNE_1`
- `tSNE_2`
- `Potability`

## 14. How to Run the Program in Spyder

1. Open Spyder.
2. Open `Assignment_3_Water_Potability_FINAL.py`.
3. Make sure `water_potability(3).csv` is in the same folder as the Python file.
4. Set Spyder's current working directory to this folder.
5. Press **F5** or click the Run button.
6. Wait while the SVM, Decision Tree and t-SNE calculations are performed.
7. Observe the metrics, confusion matrices and graphs in the Spyder console and plots pane.
8. Check the generated CSV result files.

## 15. Expected Execution Flow

```text
Load Water Potability Dataset
          ↓
Check Dataset Information
          ↓
Handle Missing Values
          ↓
Separate Features and Target
          ↓
Train-Test Split
          ↓
Feature Scaling
          ↓
      ┌───────────────┐
      │      SVM      │
      ├───────────────┤
      │ Linear        │
      │ Polynomial    │
      │ RBF           │
      └───────────────┘
          ↓
   Decision Tree
          ↓
  max_depth = 3,5,10,15
          ↓
  min_samples_split = 2,10,20
          ↓
Accuracy / Precision / Recall / F1
          ↓
Confusion Matrix
          ↓
Execution Time
          ↓
2D t-SNE Visualization
          ↓
Final Model Comparison
```

## 16. Conclusion

This assignment compares SVM classifiers using Linear, Polynomial and RBF kernels with Decision Tree classifiers using different `max_depth` and `min_samples_split` settings for water potability classification.

The final comparison is based on Accuracy, Precision, Recall, F1-Score, Confusion Matrix and Execution Time. The 2D t-SNE visualization is also used to visualize the feature space and the distribution of the target classes.

The actual performance values should be taken from `classification_results.csv` after the program is executed on the dataset.

## 17. Assignment Requirements Covered

- [x] Unique classification dataset
- [x] 2D t-SNE visualization
- [x] SVM classifier
- [x] Linear kernel
- [x] Polynomial kernel
- [x] RBF kernel
- [x] Decision Tree classifier
- [x] Varying `max_depth`
- [x] Varying `min_samples_split`
- [x] Accuracy
- [x] Precision
- [x] Recall
- [x] F1-Score
- [x] Confusion Matrix
- [x] Execution time comparison
- [x] README documentation
