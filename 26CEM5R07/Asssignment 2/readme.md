# California Housing: MLR vs KNNR

## Project Overview

This project analyzes California housing prices using two regression approaches:

- Multiple Linear Regression (MLR)
- K-Nearest Neighbors Regression (KNNR)

The models are developed using Python and evaluated using R² Score, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

## Objectives

- Build a Multiple Linear Regression model for California housing price prediction.
- Build a K-Nearest Neighbors Regression model.
- Test different K values for KNNR.
- Evaluate both models using R², MAE, and RMSE.
- Compare the model performance.

## Dataset

The project uses the California Housing dataset.

The dataset contains housing-related features used to predict the target variable:

`median_house_value`

## Methodology

### 1. Data Preparation

The dataset is loaded and prepared for regression analysis. The relevant features are separated from the target variable, and the data is divided into training and testing sets.

### 2. Multiple Linear Regression

A Multiple Linear Regression model is trained using the housing features.

### 3. K-Nearest Neighbors Regression

KNN Regression is tested using the following K values:

- 3
- 5
- 7
- 10
- 15
- 20

The results are compared using R² Score, MAE, and RMSE.

## Results

### Multiple Linear Regression

| Metric | Result |
|---|---:|
| R² Score | 0.613866 |
| MAE | 51,810.48 |
| RMSE | 71,133.17 |

### KNN Regression

The KNNR analysis tested multiple K values. The K=7 model produced the following results:

| Metric | Result |
|---|---:|
| R² Score | 0.709497 |
| MAE | 41,661.51 |
| RMSE | 61,699.10 |

### KNNR K-Value Analysis

| K | R² Score | MAE | RMSE |
|---:|---:|---:|---:|
| 3 | 0.680595 | 43,383.52 | 64,695.50 |
| 5 | 0.704574 | 41,991.73 | 62,219.70 |
| 7 | 0.709497 | 41,661.51 | 61,699.10 |
| 10 | 0.707553 | 41,756.47 | 61,905.17 |
| 15 | 0.706820 | 42,146.29 | 61,982.75 |
| 20 | 0.701002 | 42,776.59 | 62,594.69 |

## MLR vs KNNR

| Metric | MLR | KNNR (K=7) |
|---|---:|---:|
| R² Score | 0.613866 | 0.709497 |
| MAE | 51,810.48 | 41,661.51 |
| RMSE | 71,133.17 | 61,699.10 |

These results summarize the performance obtained on the test data used in the notebook.

## Project Structure

```text
california-housing-mlr-knnr/
│
├── data/
│   └── housing.csv
│
├── notebook/
│   └── california_mlr.ipynb
│
├── results/
│   ├── MLR_Coefficients.csv
│   ├── MLR_Predictions.csv
│   ├── MLR_Model_Performance.csv
│   ├── MLR_vs_KNNR_Comparison.csv
│   ├── KNNR_K_Value_Comparison.csv
│   ├── Final_MLR_vs_KNNR_Comparison.csv
│   └── KNNR_Predictions.csv
│
└── .gitignore
## Tools and Technologies

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- GitHub
- QGIS

## Evaluation Metrics

**R² Score:** Measures the proportion of variation in the target variable explained by the model.

**MAE:** Measures the average absolute difference between actual and predicted values.

**RMSE:** Measures the square root of the average squared prediction error and gives greater weight to larger errors.

## Author

Vyshnavi Perumal
