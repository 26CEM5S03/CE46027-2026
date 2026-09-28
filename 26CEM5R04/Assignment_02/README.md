# Assignment 2 - Building Energy Efficiency

## Objective

This project compares two machine learning regression models:

1. Multiple Linear Regression (MLR)
2. K-Nearest Neighbors Regression (KNNR)

The models are used to predict:

- Heating Load
- Cooling Load

## Dataset

The dataset used in this project is the Building Energy Efficiency
dataset obtained from Kaggle.

The dataset contains 768 observations and 8 input features.

### Input Features

- Relative Compactness
- Surface Area
- Wall Area
- Roof Area
- Overall Height
- Orientation
- Glazing Area
- Glazing Area Distribution

### Target Variables

- Heating Load
- Cooling Load

## Methodology

The dataset was divided into 80% training data and 20% testing data.

Random state used: 42

Two models were implemented:

- Multiple Linear Regression
- K-Nearest Neighbors Regression

For KNNR:

- K = 5
- Weight = distance
- StandardScaler was used for feature scaling

## Evaluation Metrics

The models were evaluated using:

- R² Score
- RMSE
- MAE
- Execution Time
- Peak Memory Usage

## Results

The model comparison results are saved in:

`comparison_results.csv`

The actual and predicted values are saved in:

`actual_vs_predicted_values.csv`

## Graphs

The following graphs were created:

- `actual_vs_predicted_comparison.png`
- `error_comparison.png`
- `r2_comparison.png`
- `time_comparison.png`
- `memory_comparison.png`

## Project Files

- `Assignment2.ipynb`
- `Building Energy Efficiency.csv`
- `comparison_results.csv`
- `actual_vs_predicted_values.csv`
- `actual_vs_predicted_comparison.png`
- `error_comparison.png`
- `r2_comparison.png`
- `time_comparison.png`
- `memory_comparison.png`
- `README.md`

## Dataset Source

Kaggle URL:

(https://www.kaggle.com/datasets/winternguyen/energy-efficiency-on-buildings)

## GitHub

GitHub ID:Jaswanthi-kommuru

