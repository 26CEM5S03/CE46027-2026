# EuroSAT Remote Sensing Image Classification

## 1. Project Overview

This project focuses on land-cover classification using the EuroSAT RGB remote sensing image dataset.

The EuroSAT dataset contains satellite images representing different land-cover classes. Classical machine learning classification algorithms were applied to the RGB images after converting the images into numerical feature vectors.

The following classification algorithms were evaluated:

- Linear Support Vector Machine (SVM)
- Polynomial Support Vector Machine (SVM)
- Radial Basis Function (RBF) Support Vector Machine (SVM)
- Decision Tree with different parameter configurations

The models were evaluated using classification performance metrics and computational performance measures.

---

## 2. Dataset

**Dataset:** EuroSAT RGB Dataset

The dataset contains satellite images belonging to 10 land-cover classes.

### Land-cover Classes

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

### Image Processing

The RGB images were resized to:

**32 × 32 pixels**

Each image contains three colour channels:

- Red
- Green
- Blue

The image pixels were converted into numerical feature vectors for classical machine learning classification.

Each image therefore contains:

**32 × 32 × 3 = 3072 features**

---

## 3. Data Exploration

The dataset was explored to understand:

- Number of land-cover classes
- Number of images in each class
- Class distribution
- Visual characteristics of different land-cover classes

Sample images from all 10 classes were visualized.

A class distribution graph was also generated.

---

## 4. Data Preprocessing

The following preprocessing steps were performed:

1. EuroSAT RGB images were loaded.
2. Images were converted to RGB format.
3. Images were resized to 32 × 32 pixels.
4. RGB pixel values were converted into numerical arrays.
5. Each image was flattened into a 3072-dimensional feature vector.
6. Class labels were assigned to the images.
7. Missing-value and data-quality checks were performed.
8. The dataset was divided into training and testing datasets using an 80:20 split.
9. Stratified sampling was applied to maintain class proportions.
10. StandardScaler was applied to the feature data.

---

## 5. Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

A fixed random state was used to make the experiment reproducible.

Stratification was applied so that the class distribution was maintained in both training and testing datasets.

---

## 6. Feature Scaling

Standardization was performed using `StandardScaler`.

The scaler was fitted only on the training data and then applied to both the training and testing data.

This prevents information from the testing dataset from influencing the scaling process.

---

## 7. t-SNE Visualization

t-SNE was applied to a subset of the training data to visualize the high-dimensional EuroSAT feature space in two dimensions.

The original image feature vectors contain 3072 dimensions. t-SNE reduces these features to two dimensions for visualization.

The resulting 2D visualization helps examine the distribution and separation of different land-cover classes.

**Important:** t-SNE was used only for visualization and not as a classification algorithm.

---

# 8. Classification Models

## 8.1 Linear SVM

A Support Vector Machine with a linear kernel was trained on the scaled training features.

The model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Training Time
- Prediction Time

---

## 8.2 Polynomial SVM

A Support Vector Machine with a polynomial kernel was evaluated.

The polynomial degree was set to:

**Degree = 3**

The model was evaluated using the same performance metrics as the Linear SVM.

---

## 8.3 RBF SVM

A Support Vector Machine with a Radial Basis Function kernel was evaluated.

Parameters:

- Kernel = RBF
- C = 1.0
- Gamma = scale

Due to the computational cost of training the RBF SVM on the complete training dataset, a subset of 5,000 training samples was used for RBF model training.

The testing dataset remained unchanged.

Therefore, the RBF results should be interpreted with the difference in training sample size in mind.

---

# 9. Decision Tree Classification

Four Decision Tree configurations were evaluated to study the effect of:

- `max_depth`
- `min_samples_split`

### Decision Tree 1

- `max_depth = 10`
- `min_samples_split = 2`

### Decision Tree 2

- `max_depth = 20`
- `min_samples_split = 2`

### Decision Tree 3

- `max_depth = 10`
- `min_samples_split = 10`

### Decision Tree 4

- `max_depth = 20`
- `min_samples_split = 10`

These configurations allow comparison of tree depth and minimum sample requirements during tree splitting.

---

# 10. Evaluation Metrics

The classification models were evaluated using the following metrics.

### Accuracy

Accuracy represents the proportion of correctly classified test samples among all test samples.

### Precision

Precision measures the proportion of correctly predicted samples among all samples predicted as a particular class.

### Recall

Recall measures the proportion of correctly identified samples among all actual samples belonging to a class.

### F1-score

F1-score is the harmonic mean of precision and recall.

### Confusion Matrix

Confusion matrices were generated to examine the classification performance for each EuroSAT land-cover class and identify class-level misclassifications.

---

# 11. Computational Performance

Computational performance was also recorded.

The following measures were considered:

- Training time
- Prediction time
- Total classification experiment time
- CPU utilization
- Memory utilization

These measures provide information about the computational requirements of the classification models.

---

# 12. Results and Visualizations

The project generated the following outputs.

### Dataset Visualizations

- Class distribution graph
- Sample images from all land-cover classes
- 2D t-SNE visualization

### Model Evaluation Outputs

- Confusion matrix for Linear SVM
- Confusion matrix for Polynomial SVM
- Confusion matrix for RBF SVM
- Confusion matrix for Decision Tree 1
- Confusion matrix for Decision Tree 2
- Confusion matrix for Decision Tree 3
- Confusion matrix for Decision Tree 4

### Comparison Graphs

- Model performance comparison
- Accuracy comparison
- Training time comparison
- Prediction time comparison
- Training and prediction time comparison

### Result Files

- Model comparison CSV
- Final model results CSV
- Individual classification reports
- System resource utilization CSV
- Total execution time
- Complete experiment summary

---

# 13. Project Structure

```text
EuroSAT_Classification/
│
├── figures/
│   ├── class_distribution.png
│   ├── sample_images.png
│   ├── tsne_2d.png
│   ├── confusion_matrix_linear_svm.png
│   ├── confusion_matrix_polynomial_svm.png
│   ├── confusion_matrix_rbf_svm.png
│   ├── confusion_matrix_decision_tree_1.png
│   ├── confusion_matrix_decision_tree_2.png
│   ├── confusion_matrix_decision_tree_3.png
│   ├── confusion_matrix_decision_tree_4.png
│   ├── model_performance_comparison.png
│   ├── accuracy_comparison.png
│   ├── training_time_comparison.png
│   ├── prediction_time_comparison.png
│   └── training_prediction_time_comparison.png
│
├── results/
│   ├── model_comparison_results.csv
│   ├── final_model_results.csv
│   ├── system_resource_utilization.csv
│   ├── total_execution_time.txt
│   ├── experiment_summary.txt
│   ├── linear_svm_classification_report.txt
│   ├── polynomial_svm_classification_report.txt
│   ├── rbf_svm_classification_report.txt
│   ├── decision_tree_1_classification_report.txt
│   ├── decision_tree_2_classification_report.txt
│   ├── decision_tree_3_classification_report.txt
│   └── decision_tree_4_classification_report.txt
│
└── README.md