
# Assignment-3: Classification

## 1. Objective

The objective of this assignment is to perform a classification experiment using a Kaggle classification dataset and compare Support Vector Machine (SVM) and Decision Tree classifiers.

The assignment includes:

- 2D t-SNE visualization
- SVM with Linear kernel
- SVM with Polynomial kernel
- SVM with RBF kernel
- Decision Tree with different max_depth values
- Decision Tree with different min_samples_split values
- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Computational resource utilization
- Execution time comparison

---

## 2. Dataset

### Dataset Name

**Breast Cancer Wisconsin Diagnostic Dataset**

The selected dataset is a binary classification dataset containing numerical measurements of cell nuclei obtained from breast cancer diagnostic images.

### Dataset Characteristics

- Number of samples: **569**
- Number of input features: **30**
- Target variable: **diagnosis**
- Number of classes: **2**

### Target Classes

- B = Benign
- M = Malignant

The `id` column was treated as an identifier and was not used as a machine-learning feature.

---

## 3. Software and Libraries Used

The following Python libraries were used:

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- psutil
- openpyxl

Spyder IDE was used for developing and executing the Python program.

---

## 4. Methodology

### 4.1 Data Loading

The dataset was loaded using Pandas. The program supports Excel and CSV input files.

### 4.2 Data Inspection

The dataset was inspected for:

- Number of rows and columns
- Data types
- Missing values
- Duplicate records
- Class distribution

### 4.3 Data Preprocessing

The following preprocessing steps were performed:

1. Duplicate records were checked and removed if present.
2. The `id` column was removed because it is only an identifier.
3. The `diagnosis` column was selected as the target variable.
4. Target labels were encoded as B = 0 and M = 1.
5. Missing numerical values were handled using median values.
6. The data was divided into training and testing subsets using an 80:20 split.
7. Feature standardization was performed using StandardScaler.

The scaler was fitted only on the training data and then applied to the training and testing data to avoid data leakage.

---

## 5. 2D t-SNE Visualization

t-SNE (t-Distributed Stochastic Neighbor Embedding) was used to reduce the 30-dimensional feature space into two dimensions.

The purpose of t-SNE in this assignment is to visualize the distribution and separation of the two target classes before model training.

The generated plot is saved as:

`tsne_2D_scatter_plot.png`

---

## 6. Support Vector Machine Classification

Three SVM kernels were implemented and compared.

### 6.1 Linear Kernel

```python
SVC(kernel="linear", C=1.0)
```

### 6.2 Polynomial Kernel

```python
SVC(kernel="poly", degree=3, C=1.0, gamma="scale")
```

### 6.3 RBF Kernel

```python
SVC(kernel="rbf", C=1.0, gamma="scale")
```

---

## 7. Decision Tree Classification

Decision Tree classification was performed by varying structural parameters.

### 7.1 Varying max_depth

The following values were tested:

- 3
- 5
- 10
- 15
- None

### 7.2 Varying min_samples_split

The following values were tested:

- 2
- 5
- 10
- 20

---

## 8. Performance Metrics

The following metrics were calculated for every model:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Execution time and computational resource measurements were also recorded.

---

## 9. Model Comparison Results

The following table contains the actual results obtained from the experiment:

| Model                             |   Accuracy |   Precision |   Recall |   F1_Score |   Training_Time_sec |   Prediction_Time_sec |   Memory_Change_MB |   CPU_Utilization_percent |
|:----------------------------------|-----------:|------------:|---------:|-----------:|--------------------:|----------------------:|-------------------:|--------------------------:|
| SVM_Linear                        |     0.9649 |      1.0000 |   0.9048 |     0.9500 |              0.0059 |                0.0010 |             0.1445 |                    2.8000 |
| SVM_Polynomial_Degree3            |     0.8860 |      1.0000 |   0.6905 |     0.8169 |              0.0038 |                0.0005 |             0.0000 |                    2.8000 |
| SVM_RBF                           |     0.9737 |      1.0000 |   0.9286 |     0.9630 |              0.0033 |                0.0009 |             0.0000 |                    6.3000 |
| DecisionTree_max_depth_3          |     0.9035 |      0.9429 |   0.7857 |     0.8571 |              0.0139 |                0.0003 |             0.1523 |                    1.2000 |
| DecisionTree_max_depth_5          |     0.9211 |      0.9459 |   0.8333 |     0.8861 |              0.0061 |                0.0003 |             0.0000 |                    1.3000 |
| DecisionTree_max_depth_10         |     0.9298 |      0.9048 |   0.9048 |     0.9048 |              0.0086 |                0.0003 |             0.0000 |                    1.4000 |
| DecisionTree_max_depth_15         |     0.9298 |      0.9048 |   0.9048 |     0.9048 |              0.0086 |                0.0003 |             0.0000 |                    5.6000 |
| DecisionTree_max_depth_None       |     0.9298 |      0.9048 |   0.9048 |     0.9048 |              0.0073 |                0.0003 |             0.0000 |                    2.6000 |
| DecisionTree_min_samples_split_2  |     0.9298 |      0.9048 |   0.9048 |     0.9048 |              0.0090 |                0.0002 |             0.0000 |                    7.6000 |
| DecisionTree_min_samples_split_5  |     0.9386 |      0.9268 |   0.9048 |     0.9157 |              0.0073 |                0.0003 |             0.0039 |                   13.0000 |
| DecisionTree_min_samples_split_10 |     0.9035 |      0.8780 |   0.8571 |     0.8675 |              0.0074 |                0.0003 |             0.0000 |                    9.0000 |
| DecisionTree_min_samples_split_20 |     0.9211 |      0.9024 |   0.8810 |     0.8916 |              0.0087 |                0.0003 |             0.0000 |                    1.3000 |

---

## 10. Results Interpretation

The highest observed accuracy among the tested models was obtained by:

**SVM_RBF**

with an accuracy of:

**97.37%**

The highest observed F1-score was obtained by:

**SVM_RBF**

with an F1-score of:

**96.30%**

These observations are specific to the dataset, parameter settings, train-test split, software environment and experimental conditions used in this assignment.

---

## 11. Computational Resource Utilization

The program measured:

- Training execution time
- Prediction execution time
- Memory change
- CPU utilization

Training and prediction times were measured using Python's high-resolution performance timer.

Memory and CPU utilization were measured using the psutil library.

The measured values are included in the model comparison results file.

---

## 12. Generated Output Files

The program generates:

```text
Assignment3_Results/
|
|-- tsne_2D_scatter_plot.png
|-- classification_performance_comparison.png
|-- training_time_comparison.png
|-- model_comparison_results.csv
|-- model_comparison_results.xlsx
|-- confusion matrix images
```

---

## 13. How to Run the Program

1. Open Spyder.
2. Open `Assignment3_Classification.py`.
3. Run the program.
4. Select the dataset when the file-selection window appears.
5. The program performs preprocessing.
6. The 2D t-SNE plot is generated and saved.
7. SVM Linear, Polynomial and RBF models are trained.
8. Decision Tree models are evaluated with different parameters.
9. Accuracy, Precision, Recall and F1-score are calculated.
10. Confusion matrices are generated.
11. Execution time and computational measurements are recorded.
12. Results are saved in the `Assignment3_Results` folder.

---

## 14. Project Structure

```text
Assignment3/
|
|-- Assignment3_Classification.py
|-- data.xlsx
|-- README.md
|
|-- Assignment3_Results/
|   |-- tsne_2D_scatter_plot.png
|   |-- classification_performance_comparison.png
|   |-- training_time_comparison.png
|   |-- model_comparison_results.csv
|   |-- model_comparison_results.xlsx
|   `-- confusion matrix images
```

---

## 15. Conclusion

This assignment implemented and compared SVM and Decision Tree classification approaches on the selected classification dataset.

Three SVM kernels (Linear, Polynomial and RBF) were evaluated. Decision Tree performance was evaluated by varying max_depth and min_samples_split.

The models were compared using Accuracy, Precision, Recall, F1-score and Confusion Matrix. Computational aspects including training time, prediction time, memory change and CPU utilization were also measured.

The experimental results demonstrate how classifier type and model parameters can influence classification performance and computational requirements.
