# Assignment 3 – Classification: SVM and Decision Tree

## 1. Objective

This assignment compares Support Vector Machine (SVM) and Decision Tree classifiers on a Kaggle Pulsar Star Classification dataset. The work includes 2D t-SNE visualization, three SVM kernels, Decision Tree parameter variation, classification metrics, confusion matrices, execution time, and memory utilization.

## 2. Dataset

**Dataset:** Pulsar Star Classification / HTRU2  
**Source:** Kaggle  
**Kaggle URL:** https://www.kaggle.com/charitarth/pulsar-dataset-htru2

The dataset contains 17,898 observations, 8 numerical input features, and a binary target variable named `Class`.

- Class 0: Non-Pulsar
- Class 1: Pulsar

The dataset is imbalanced, so Accuracy was evaluated together with Precision, Recall, and F1-Score.

## 3. Features

The eight input features are:

1. Mean of the integrated profile
2. Standard deviation of the integrated profile
3. Excess kurtosis of the integrated profile
4. Skewness of the integrated profile
5. Mean of the DM-SNR curve
6. Standard deviation of the DM-SNR curve
7. Excess kurtosis of the DM-SNR curve
8. Skewness of the DM-SNR curve

## 4. Software and Libraries

The analysis was performed in Python using Jupyter Notebook.

Main libraries:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- psutil

## 5. Data Preprocessing

The dataset was loaded from the `Data` folder. Column names were assigned because the downloaded CSV was used without a header row.

The workflow included:

1. Dataset inspection
2. Missing-value checking
3. Duplicate checking
4. Descriptive statistics
5. Feature-target separation
6. 80:20 train-test split
7. Stratified sampling
8. Feature standardization using `StandardScaler`

The scaler was fitted on the training data and then applied to the test data to avoid data leakage.

## 6. Exploratory Data Analysis

The following figures were generated:

- Class distribution
- Feature distributions
- Feature correlation heatmap

All figures are stored in the `figures` folder.

## 7. 2D t-SNE Visualization

A 2D t-SNE visualization was generated using:

- Components: 2
- Perplexity: 30
- Random state: 42
- Maximum iterations: 1000

The resulting plot is saved as:

`figures/tsne/tsne_plot.png`

The plot provides a visual representation of the distribution of the two target classes in the transformed feature space.

## 8. SVM Classification

Three SVM kernels were evaluated:

- Linear
- Polynomial
- RBF

Each model was evaluated using Accuracy, Precision, Recall, F1-Score, training time, prediction time, total time, memory utilization, and a confusion matrix.

## 9. Decision Tree Classification

Decision Tree performance was evaluated by varying:

### max_depth

- 3
- 5
- 10
- None

### min_samples_split

- 2
- 5
- 10

This produced 12 Decision Tree configurations.

## 10. Evaluation Metrics

The following metrics were used:

- **Accuracy:** proportion of correctly classified observations.
- **Precision:** proportion of predicted positive observations that were actually positive.
- **Recall:** proportion of actual positive observations correctly identified.
- **F1-Score:** harmonic mean of Precision and Recall.
- **Confusion Matrix:** summarizes correct and incorrect predictions for both classes.
- **Execution Time:** training, prediction, and total measured time.
- **Memory Utilization:** process-level memory change measured during model evaluation.

## 11. Model Results

| Model | Accuracy | Precision | Recall | F1-Score | Train Time (s) | Prediction Time (s) | Total Time (s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| SVM - RBF | 0.980726 | 0.948097 | 0.835366 | 0.888169 | 0.775506 | 0.282498 | 1.058004 |
| SVM - Linear | 0.979888 | 0.957143 | 0.817073 | 0.881579 | 0.520374 | 0.081293 | 0.601666 |
| SVM - Polynomial | 0.977933 | 0.939929 | 0.810976 | 0.870704 | 1.160183 | 0.108240 | 1.268423 |
| Decision Tree - depth=5, split=2 | 0.978771 | 0.934483 | 0.826220 | 0.877023 | 0.059854 | 0.000742 | 0.060596 |
| Decision Tree - depth=5, split=5 | 0.978771 | 0.934483 | 0.826220 | 0.877023 | 0.060046 | 0.000705 | 0.060750 |
| Decision Tree - depth=5, split=10 | 0.978771 | 0.934483 | 0.826220 | 0.877023 | 0.045791 | 0.000641 | 0.046432 |
| Decision Tree - depth=3, split=5 | 0.977654 | 0.897436 | 0.853659 | 0.875000 | 0.038086 | 0.000577 | 0.038663 |
| Decision Tree - depth=3, split=2 | 0.977654 | 0.897436 | 0.853659 | 0.875000 | 0.040410 | 0.000602 | 0.041011 |
| Decision Tree - depth=3, split=10 | 0.977654 | 0.897436 | 0.853659 | 0.875000 | 0.029445 | 0.000320 | 0.029765 |
| Decision Tree - depth=10, split=10 | 0.974302 | 0.888158 | 0.823171 | 0.854430 | 0.095042 | 0.000637 | 0.095678 |
| Decision Tree - depth=10, split=2 | 0.972905 | 0.878689 | 0.817073 | 0.846761 | 0.101905 | 0.000665 | 0.102571 |
| Decision Tree - depth=10, split=5 | 0.972626 | 0.878289 | 0.814024 | 0.844937 | 0.099140 | 0.000604 | 0.099744 |
| Decision Tree - depth=None, split=10 | 0.969553 | 0.824926 | 0.847561 | 0.836090 | 0.122210 | 0.000928 | 0.123138 |
| Decision Tree - depth=None, split=2 | 0.968436 | 0.824773 | 0.832317 | 0.828528 | 0.128405 | 0.000824 | 0.129229 |
| Decision Tree - depth=None, split=5 | 0.965084 | 0.804805 | 0.817073 | 0.810893 | 0.192764 | 0.000831 | 0.193595 |

The complete table is also saved in `results/comparison_results.csv`.

## 12. Observations

The SVM models achieved high classification performance on the HTRU2 dataset. Among the tested configurations, the RBF SVM produced an F1-Score of 0.888169, while the Linear SVM produced 0.881579 and the Polynomial SVM produced 0.870704.

Decision Tree models had substantially shorter measured training and prediction times. For example, the depth-5 tree with `min_samples_split=10` had a total measured time of approximately 0.046 seconds.

The Decision Tree results also show that increasing tree depth did not consistently improve F1-Score in this experiment.

Because the dataset is imbalanced, Accuracy alone does not fully describe classification performance. Precision, Recall, and F1-Score were therefore included.

## 13. Computational Resource Utilization

Process-level memory usage was measured before and after model evaluation using `psutil`.

Some models show very small or zero measured memory changes because Python and scikit-learn can reuse memory already allocated by the running process. Therefore, these values represent measured process-level changes rather than exact standalone memory requirements.

The resource results are stored in:

`results/resource_utilization.csv`

## 14. Generated Outputs

### Figures

- `figures/class_distribution.png`
- `figures/feature_distributions.png`
- `figures/correlation_heatmap.png`
- `figures/tsne/tsne_plot.png`
- Confusion matrix plots in `figures/confusion_matrix/`
- `figures/comparison/performance_comparison.png`
- `figures/comparison/time_comparison.png`
- `figures/comparison/resource_comparison.png`

### Results

- `results/comparison_results.csv`
- `results/classification_reports.txt`
- `results/resource_utilization.csv`
- `results/final_model_summary.csv`

## 15. Project Structure

```text
Assignment3/
├── Data/
│   └── HTRU_2.csv
├── figures/
│   ├── class_distribution.png
│   ├── feature_distributions.png
│   ├── correlation_heatmap.png
│   ├── tsne/
│   ├── confusion_matrix/
│   └── comparison/
├── notebook/
│   └── Assignment3_SVM_DecisionTree.ipynb
├── results/
│   ├── comparison_results.csv
│   ├── classification_reports.txt
│   ├── resource_utilization.csv
│   └── final_model_summary.csv
├── .gitignore
└── README.md
