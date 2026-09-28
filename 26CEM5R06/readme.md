# Assignment 2: Performance Comparison of MLR and KNNR

## 1. Objective

This assignment compares Multiple Linear Regression (MLR) and K-Nearest Neighbors Regression (KNNR).

The models are evaluated using R², MAE, RMSE, execution time, and runtime memory utilization.

## 2. Dataset

Dataset: Hospital Synthetic Dataset
Number of observations: 5000
Number of original variables: 18
Target variable: Admission_Deposit

The dataset is stored in data/HospitalSynthetic1_cleaned.csv.

## 3. Methodology

The dataset was inspected for missing values, duplicate records, data types, and descriptive statistics.

The case_id identifier was excluded because it is an identification field rather than a meaningful predictor.

The data was divided into 80% training data and 20% testing data using random_state=42.

StandardScaler was used to standardize the predictor variables.

Multiple Linear Regression was trained as the linear regression model.

KNN Regression was evaluated for K values from 1 to 30.

The selected K value was 27, based on the lowest RMSE.

## 4. Model Performance

| Model | R² | MAE | RMSE |
|---|---:|---:|---:|
| Multiple Linear Regression | 0.0677 | 738.03 | 949.32 |
| KNN Regression | 0.0593 | 733.63 | 953.60 |

## 5. Execution Time

| Model | Training Time (s) | Prediction Time (s) |
|---|---:|---:|
| Multiple Linear Regression | 0.003904 | 0.000507 |
| KNN Regression | 0.001789 | 0.015340 |

## 6. KNN Hyperparameter Tuning

K values from 1 to 30 were tested.

Selected K: 27

The tuning results are stored in results/knn_tuning_results.csv.

The K vs RMSE plot is stored in figures/knn_k_vs_rmse.png.

## 7. Evaluation Metrics

R² measures the proportion of variation in the target variable explained by the model.

MAE represents the average absolute prediction error.

RMSE represents the square root of the average squared prediction error and gives greater weight to larger errors.

## 8. Visualizations

The assignment includes target distribution, correlation heatmap, actual-versus-predicted plots, residual plots, KNN tuning plots, metric comparison plots, and execution-time comparison.

All figures are stored in the figures folder.

## 9. Resource Utilization

Runtime memory usage was recorded using psutil.

The result is stored in results/resource_utilization.csv.

This represents runtime process memory and should not be interpreted as theoretical algorithmic memory complexity.

## 10. Results

MLR R² = 0.0677
KNNR R² = 0.0593

The relatively low R² values indicate that the selected predictor variables explain only a limited proportion of the variation in Admission_Deposit.

The MAE and RMSE values provide additional information about prediction error magnitude.

## 11. Project Structure

Assignment2/
├── data/
│   └── HospitalSynthetic1_cleaned.csv
├── figures/
├── notebook/
│   └── Assignment2_MLR_KNNR.ipynb
├── results/
├── .gitignore
└── README.md

## 12. Technologies Used

Python, Jupyter Notebook, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and psutil.

## 13. Reproducibility

A fixed random state of 42 was used for the train-test split.

The complete analysis is contained in the Jupyter Notebook.
