# Assignment 3 – Classification

## 1. Project Title

**Classification Using SVM and Decision Tree with 2D t-SNE Visualization**

## 2. Dataset

**Dataset Name:** `AILM DATASET 1.csv`

The program uses the column `cp` as the target variable for classification. The remaining numerical columns are used as input features.

## 3. Objective

The objectives of this assignment are:

- To load and preprocess the dataset.
- To identify numerical input features and the target class.
- To divide the dataset into training and testing sets.
- To visualize the dataset using 2D t-SNE.
- To implement Support Vector Machine (SVM) classifiers.
- To implement Decision Tree classifiers.
- To evaluate classification performance using different metrics.
- To compare model execution time and memory usage.
- To generate confusion matrices and comparison graphs.

## 4. Algorithms Used

### Support Vector Machine (SVM)

Three SVM models are implemented:

1. SVM Linear – Linear kernel
2. SVM Polynomial – Polynomial kernel, degree 3
3. SVM RBF – RBF kernel

StandardScaler is applied before SVM classification.

### Decision Tree

Decision Tree models are tested using different combinations of:

- `max_depth = 3, 5, None`
- `min_samples_split = 2, 10, 20`

## 5. Data Preprocessing

The program performs the following steps:

1. Reads the CSV dataset using Pandas.
2. Checks the dataset shape and columns.
3. Checks for missing values.
4. Removes rows containing missing values.
5. Separates the target variable `cp` from the input features.
6. Selects numerical features.
7. Checks that the target contains at least two classes.
8. Splits the data into 80% training and 20% testing data.
9. Uses `random_state=42` and stratification for the train-test split.

## 6. t-SNE Visualization

The program generates a 2D t-SNE scatter plot to visualize the distribution of the dataset.

Parameters include:

- Components: `2`
- Random state: `42`
- Initialization: `PCA`
- Learning rate: `auto`

Output:

`tsne_scatter_plot.png`

## 7. Model Evaluation Metrics

Each classification model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Execution Time
- CPU Time
- Peak Memory Usage

## 8. Software and Libraries Required

### Software

- Python 3.x
- Spyder / Anaconda
- Windows, Linux, or macOS

### Python Libraries

Install the required libraries using:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## 9. Project Folder Structure

```text
Assignment3_Classification/
│
├── Assignment3_Classification_SPYDER_READY.py
├── AILM DATASET 1.csv
│
└── Assignment3_Results/
    ├── tsne_scatter_plot.png
    ├── model_comparison_results.csv
    ├── model_accuracy_comparison.png
    ├── execution_time_comparison.png
    └── confusion_matrix_*.png
```

## 10. How to Run the Program in Spyder

### Step 1
Open Spyder.

### Step 2
Open:

`Assignment3_Classification_SPYDER_READY.py`

### Step 3
Make sure:

`AILM DATASET 1.csv`

is available in the same folder as the Python program.

### Step 4
Click the **Run ▶** button in Spyder.

### Step 5
If the dataset is not automatically found, the program opens a file-selection window. Select:

`AILM DATASET 1.csv`

### Step 6
The program creates an:

`Assignment3_Results`

folder containing the generated outputs.

## 11. Output Files

### t-SNE Plot

`tsne_scatter_plot.png`

Shows the dataset in two-dimensional t-SNE space.

### Model Comparison Results

`model_comparison_results.csv`

Contains the performance results of the classification models.

### Accuracy Comparison

`model_accuracy_comparison.png`

Shows the accuracy of the different models.

### Execution Time Comparison

`execution_time_comparison.png`

Shows the execution time of each classification model.

### Confusion Matrices

Separate confusion-matrix images are generated for every model.

## 12. Expected Console Output

The program displays:

- Dataset shape
- Dataset columns
- First five rows
- Missing values
- Feature columns
- Target class distribution
- Training samples
- Testing samples

For every model, it displays:

- Accuracy
- Precision
- Recall
- F1-Score
- Execution time
- CPU time
- Peak memory
- Classification report

Finally, it displays the **Final Model Comparison** table.

## 13. Result

The classification models are compared based on:

- Accuracy
- Precision
- Recall
- F1-score
- Execution time
- CPU time
- Peak memory usage

The final model comparison is sorted according to Accuracy in descending order.

**Note:** The actual numerical results depend on the contents of `AILM DATASET 1.csv` and the execution environment.

## 14. Conclusion

This assignment demonstrates the application of supervised machine-learning classification techniques using Support Vector Machine and Decision Tree algorithms. The models are evaluated using standard classification metrics, while t-SNE provides a two-dimensional visualization of the dataset.

The generated graphs, confusion matrices, and CSV comparison file can be used to analyze the performance of the implemented classification models.

## 15. Author

**Name:** __________________________

**Roll Number:** ___________________

**Course:** _________________________

**Institution:** NIT Warangal

**Assignment:** Assignment 3 – Classification
