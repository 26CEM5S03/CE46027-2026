# Smart City Traffic Stress Prediction

## 📌 Project Overview

This project focuses on predicting **traffic-related stress levels** in a smart city environment using machine learning regression techniques.

The dataset contains different traffic, road, weather, and driver-related parameters that may influence the **Stress Index**. Two machine learning regression models were developed and compared:

1. **Multiple Linear Regression (MLR)**
2. **K-Nearest Neighbors Regression (KNNR)**

The models were evaluated not only on prediction accuracy but also on **execution time** and **peak memory usage**.

---

## 🎯 Objectives

The main objectives of this project are:

* To analyze factors associated with traffic stress.
* To prepare numerical and categorical data for machine learning.
* To develop a Multiple Linear Regression model.
* To develop a K-Nearest Neighbors Regression model.
* To compare the predictive performance of both models.
* To evaluate models using R², RMSE, and MAE.
* To compare computational performance using execution time and peak memory.
* To visualize actual and predicted stress index values.
* To identify the better-performing model for this dataset.

---

## 📂 Dataset

The project uses the following dataset:

```text
smart_city_traffic_stress_dataset.csv
```

The dataset contains **50,000 observations and 8 columns**.

### Dataset Features

| Feature                   | Description                      | Type        |
| ------------------------- | -------------------------------- | ----------- |
| `traffic_density`         | Traffic density                  | Numerical   |
| `horn_events_per_min`     | Number of horn events per minute | Numerical   |
| `avg_speed`               | Average vehicle speed            | Numerical   |
| `signal_wait_time`        | Waiting time at traffic signals  | Numerical   |
| `weather_condition`       | Weather condition                | Categorical |
| `road_quality_score`      | Road quality score               | Numerical   |
| `driver_experience_level` | Driver experience category       | Categorical |
| `stress_index`            | Traffic-related stress index     | Target      |

The notebook confirms the dataset shape as **(50,000, 8)**.

---

## 🛠️ Technologies and Libraries Used

The project was developed using Python and Jupyter Notebook.

### Main Libraries

```text
Python
Jupyter Notebook
Pandas
NumPy
Matplotlib
Scikit-learn
```

### Scikit-learn Components

The following components were used:

* `train_test_split`
* `ColumnTransformer`
* `StandardScaler`
* `OneHotEncoder`
* `Pipeline`
* `SimpleImputer`
* `LinearRegression`
* `KNeighborsRegressor`
* `r2_score`
* `mean_squared_error`
* `mean_absolute_error`

Additional Python modules were used for computational performance measurement:

* `time`
* `tracemalloc`
* `gc`

These imports and components are present in the notebook.

---

# 🔄 Project Workflow

The complete workflow followed in this project is:

```text
Dataset
   ↓
Load CSV File
   ↓
Inspect Dataset
   ↓
Check Missing Values
   ↓
Separate Input and Target
   ↓
Identify Numerical & Categorical Features
   ↓
Train-Test Split
   ↓
Data Preprocessing
   ↓
Multiple Linear Regression
   ↓
K-Nearest Neighbors Regression
   ↓
Model Evaluation
   ↓
Performance Comparison
   ↓
Save Results
   ↓
Visualization
   ↓
Conclusion
```

---

# 1. Import Required Libraries

The first step is importing all required Python libraries.

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tracemalloc
import gc

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
```

These libraries provide the functionality required for data handling, preprocessing, machine learning, evaluation, visualization, and performance measurement.

---

# 2. Load the Dataset

The CSV dataset was loaded using Pandas.

```python
file_path = "smart_city_traffic_stress_dataset.csv"

data = pd.read_csv(file_path)

print("Dataset loaded successfully!")
```

The notebook successfully loaded the dataset.

---

# 3. Explore the Dataset

The dataset shape, column names, and first five records were examined.

```python
print(data.shape)
print(data.columns.tolist())
display(data.head())
```

### Dataset Shape

```text
50,000 rows
8 columns
```

### Columns

```text
traffic_density
horn_events_per_min
avg_speed
signal_wait_time
weather_condition
road_quality_score
driver_experience_level
stress_index
```

The first five observations were also displayed to understand the structure and values in the dataset.

---

# 4. Check Missing Values

Missing values were checked for every column.

```python
print(data.isnull().sum())
```

### Result

All eight columns contained:

```text
0 missing values
```

Therefore, the dataset did not contain missing values in the loaded data.

Even though no missing values were present, imputation steps were included in the preprocessing pipeline to make the machine learning workflow more robust.

---

# 5. Define Input and Target Variables

The target variable selected for prediction was:

```text
stress_index
```

All remaining columns were used as input variables.

```python
target = "stress_index"

X = data.drop(columns=[target])
y = data[target]
```

### Input Variables

```text
traffic_density
horn_events_per_min
avg_speed
signal_wait_time
weather_condition
road_quality_score
driver_experience_level
```

### Target Variable

```text
stress_index
```

This separation is shown in the notebook output.

---

# 6. Identify Numerical and Categorical Features

The input variables were divided into numerical and categorical features.

### Numerical Features

```text
traffic_density
horn_events_per_min
avg_speed
signal_wait_time
road_quality_score
```

### Categorical Features

```text
weather_condition
driver_experience_level
```

The notebook automatically identified these feature types using Pandas data types.

---

# 7. Train-Test Split

The dataset was divided into training and testing sets.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
```

The split resulted in:

| Dataset  | Samples |
| -------- | ------: |
| Training |  40,000 |
| Testing  |  10,000 |

An 80:20 train-test split was used, with `random_state=42` to make the split reproducible.

---

# 8. Data Preprocessing

A preprocessing pipeline was created using `ColumnTransformer`.

## Numerical Data

For numerical features:

1. Missing values are handled using median imputation.
2. Features are standardized using `StandardScaler`.

```text
Median Imputation
       ↓
Standard Scaling
```

## Categorical Data

For categorical features:

1. Missing values are handled using the most frequent category.
2. Categories are converted into numerical form using One-Hot Encoding.

```text
Most Frequent Imputation
       ↓
One-Hot Encoding
```

The preprocessing pipeline was implemented as:

```python
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]),
            numeric_features
        ),

        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        )
    ]
)
```

This preprocessing setup is directly implemented in the notebook.

---

# 9. Multiple Linear Regression

The first machine learning model was **Multiple Linear Regression (MLR)**.

The preprocessing pipeline and linear regression model were combined into one pipeline.

```python
mlr_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", LinearRegression())
])
```

The model was trained using the training dataset and then used to predict stress index values for the test dataset.

---

# 10. K-Nearest Neighbors Regression

The second model was **K-Nearest Neighbors Regression (KNNR)**.

The model used:

```text
Number of neighbors (K) = 5
Weights = uniform
```

The model was implemented as:

```python
knnr_model = Pipeline([
    ("preprocessing", preprocessor),
    ("model", KNeighborsRegressor(
        n_neighbors=5,
        weights="uniform",
        n_jobs=-1
    ))
])
```

The notebook therefore uses K = 5 for the KNN regression model.

---

# 11. Model Evaluation

Three prediction-performance metrics were used:

### R² Score

R² measures how well the model explains the variation in the target variable.

Higher values indicate better performance.

### RMSE

Root Mean Squared Error measures the typical magnitude of prediction errors.

Lower values indicate better performance.

### MAE

Mean Absolute Error measures the average absolute difference between actual and predicted values.

Lower values indicate better performance.

The project also measured:

* Execution Time
* Peak Memory Usage

This allows both predictive and computational performance to be compared.

---

# 12. Multiple Linear Regression Results

The MLR model produced the following results:

| Metric         |             Result |
| -------------- | -----------------: |
| R² Score       |         **0.8968** |
| RMSE           |         **5.2180** |
| MAE            |         **4.1698** |
| Execution Time | **2.2809 seconds** |
| Peak Memory    |     **12.2415 MB** |

The notebook records an R² of 0.8968, RMSE of 5.218, and MAE of 4.1698.

---

# 13. K-Nearest Neighbors Regression Results

The KNNR model produced:

| Metric         |             Result |
| -------------- | -----------------: |
| R² Score       |         **0.8914** |
| RMSE           |         **5.3539** |
| MAE            |         **4.2773** |
| Execution Time | **3.2073 seconds** |
| Peak Memory    |     **11.2280 MB** |

These values were obtained directly from the notebook's KNNR evaluation.

---

# 14. Model Comparison

The final comparison was created using a Pandas DataFrame.

| Model                          |         R² |       RMSE |        MAE | Execution Time (s) | Peak Memory (MB) |
| ------------------------------ | ---------: | ---------: | ---------: | -----------------: | ---------------: |
| Multiple Linear Regression     | **0.8968** | **5.2180** | **4.1698** |         **2.2809** |          12.2415 |
| K-Nearest Neighbors Regression |     0.8914 |     5.3539 |     4.2773 |             3.2073 |      **11.2280** |

The notebook also saves this comparison as:

```text
model_comparison_results.csv
```

The result-saving step is included in the notebook.

---

# 📊 15. Results Analysis

The comparison shows that **Multiple Linear Regression performed slightly better in predictive accuracy**.

### R²

MLR achieved:

```text
0.8968
```

while KNNR achieved:

```text
0.8914
```

Therefore, MLR had the higher R² score.

### RMSE

MLR:

```text
5.2180
```

KNNR:

```text
5.3539
```

Since lower RMSE is better, MLR performed better.

### MAE

MLR:

```text
4.1698
```

KNNR:

```text
4.2773
```

Again, MLR produced the lower error.

### Execution Time

MLR:

```text
2.2809 seconds
```

KNNR:

```text
3.2073 seconds
```

MLR required less execution time.

### Memory

KNNR used slightly less peak memory:

```text
11.2280 MB
```

compared with MLR:

```text
12.2415 MB
```

Therefore, KNNR had a small advantage in peak memory usage.

---

# 📈 16. Visualization

The notebook includes a scatter plot comparing the actual and predicted stress index values for the MLR model.

The plot uses:

* X-axis: **Actual Stress Index**
* Y-axis: **Predicted Stress Index**
* Title: **MLR: Actual vs Predicted Stress Index**

```python
plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    mlr_predictions,
    alpha=0.5
)

plt.xlabel("Actual Stress Index")
plt.ylabel("Predicted Stress Index")
plt.title("MLR: Actual vs Predicted Stress Index")

plt.grid(True)
plt.tight_layout()

plt.show()
```

This visualization helps assess how closely the predicted stress values follow the actual stress values.

---

# 🏆 17. Best Performing Model

Based on the obtained results, **Multiple Linear Regression is the better overall model for this dataset**.

It achieved:

* Higher R²
* Lower RMSE
* Lower MAE
* Lower execution time

Although KNNR consumed slightly less peak memory, its prediction performance and execution time were less favorable compared with MLR.

Therefore:

```text
Recommended Model:
Multiple Linear Regression (MLR)
```

---

# 📝 18. Conclusion

This project successfully developed and compared two regression-based machine learning models for predicting traffic-related stress levels.

The dataset contained 50,000 observations with traffic, weather, road quality, and driver experience features. The data was divided into numerical and categorical variables and processed using a Scikit-learn preprocessing pipeline.

Both Multiple Linear Regression and K-Nearest Neighbors Regression achieved good predictive performance. However, Multiple Linear Regression performed slightly better across the main prediction metrics.

The final results show that MLR achieved an R² score of **0.8968**, compared with **0.8914** for KNNR. MLR also obtained lower RMSE and MAE values and required less execution time.

Based on these results, **Multiple Linear Regression was selected as the better-performing model for this particular dataset and experimental setup**.

---

# 🔍 19. Key Findings

* The dataset contains **50,000 observations**.
* There are **8 variables** in total.
* `stress_index` is the target variable.
* There are **5 numerical input features**.
* There are **2 categorical input features**.
* No missing values were found in the dataset.
* An **80:20 train-test split** was used.
* KNNR was implemented with **K = 5**.
* MLR achieved an R² of **0.8968**.
* KNNR achieved an R² of **0.8914**.
* MLR had lower RMSE and MAE.
* MLR was faster in execution.
* KNNR used slightly less peak memory.
* Overall, **MLR was the better-performing model**.

---

# 📁 20. Project Structure

A recommended GitHub repository structure is:

```text
smart-city-traffic-stress/
│
├── smart_city_traffic_stress_analysis.ipynb
│
├── smart_city_traffic_stress_dataset.csv
│
├── model_comparison_results.csv
│
├── README.md
│
└── images/
    └── actual_vs_predicted.png
```

> The notebook itself currently refers to the dataset as `smart_city_traffic_stress_dataset.csv` and generates `model_comparison_results.csv`. Make sure these files are present in the GitHub repository if you want the notebook to run directly after cloning.

---

# ▶️ 21. How to Run the Project

## Step 1: Clone the Repository

```bash
git clone <your-github-repository-url>
```

## Step 2: Open the Project Folder

```bash
cd smart-city-traffic-stress
```

## Step 3: Install Required Libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

## Step 4: Start Jupyter Notebook

```bash
jupyter notebook
```

## Step 5: Open the Notebook

Open:

```text
smart_city_traffic_stress_analysis.ipynb
```

## Step 6: Keep the Dataset in the Same Folder

Make sure the following file is present:

```text
smart_city_traffic_stress_dataset.csv
```

## Step 7: Run All Cells

Run the notebook from beginning to end.

The notebook will:

```text
Load Dataset
→ Explore Dataset
→ Check Missing Values
→ Prepare Features
→ Preprocess Data
→ Train MLR
→ Train KNNR
→ Evaluate Models
→ Compare Models
→ Save Results
→ Generate Visualization
```

---

# 📊 22. Performance Summary

| Category       | Best Model |
| -------------- | ---------- |
| R² Score       | **MLR**    |
| RMSE           | **MLR**    |
| MAE            | **MLR**    |
| Execution Time | **MLR**    |
| Peak Memory    | **KNNR**   |
| Overall        | **MLR**    |

---

# 🚀 23. Future Improvements

The current project can be extended in several ways:

* Test additional regression algorithms.
* Tune the K value of KNNR.
* Perform cross-validation.
* Use hyperparameter optimization.
* Analyze feature importance.
* Add more visualization techniques.
* Compare additional performance metrics.
* Test the models on new traffic datasets.
* Investigate relationships between individual traffic factors and stress.
* Deploy the best model as a simple prediction application.

---

# ⚠️ 24. Limitations

The conclusions in this README are based on the results available in the current Jupyter Notebook.

The notebook compares only:

* Multiple Linear Regression
* K-Nearest Neighbors Regression

The current workflow does not include extensive hyperparameter tuning or cross-validation. Therefore, the conclusion that MLR is the better model applies to **this dataset and this experimental setup**.

---

# 📌 25. Final Conclusion

The project demonstrates that machine learning can be used to predict traffic-related stress from traffic, road, weather, and driver-related variables.

Among the two tested models, **Multiple Linear Regression produced the best overall results**, achieving an R² score of **0.8968**, RMSE of **5.2180**, and MAE of **4.1698**.

KNNR also produced strong results but had slightly lower predictive performance and a longer execution time.

Therefore, based on the experimental results:

> **Multiple Linear Regression is the recommended model for predicting the Stress Index in this project.**

---

## 👨‍💻 Project Information

**Project:** Smart City Traffic Stress Prediction
**Platform:** Jupyter Notebook
**Language:** Python
**Machine Learning Type:** Supervised Learning – Regression
**Models:** Multiple Linear Regression and K-Nearest Neighbors Regression
**Dataset Size:** 50,000 × 8
**Best Model:** Multiple Linear Regression
**Target Variable:** `stress_index`
