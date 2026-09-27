# Assignment 3 -- Regression Model Comparison

## 1. Introduction

This assignment compares the performance of two regression techniques
using the **Medical Cost Personal Dataset (`insurance.csv`)**:

1.  Multiple Linear Regression (MLR)
2.  K-Nearest Neighbours (KNN) Regression

The Python program is designed to run in **Spyder** and does not require
`scikit-learn`. Multiple Linear Regression and KNN Regression are
implemented using NumPy, while pandas is used for data handling and
Matplotlib is used for visualization.

The program tests different K values for KNN Regression and evaluates
the models using standard regression performance metrics and execution
time.

------------------------------------------------------------------------

## 2. Dataset

### Dataset Name

**Medical Cost Personal Dataset**

### Dataset File

``` text
insurance.csv
```

### Dataset Location

The program is configured to use:

``` text
C:\a\insurance.csv
```

The Python program should be placed in the same folder:

``` text
C:\a\
│
├── assignment_regression_C_a.py
└── insurance.csv
```

The program automatically creates the output directory:

``` text
C:\a\outputs\
```

------------------------------------------------------------------------

## 3. Dataset Variables

The dataset contains the following variables:

  Variable     Description                     Type
  ------------ ------------------------------- ------------------
  `age`        Age of the individual           Numerical
  `sex`        Sex of the individual           Categorical
  `bmi`        Body Mass Index                 Numerical
  `children`   Number of children/dependents   Numerical
  `smoker`     Smoking status                  Categorical
  `region`     Residential region              Categorical
  `charges`    Medical insurance charges       Numerical target

The target variable used for regression is:

``` text
charges
```

------------------------------------------------------------------------

## 4. Software Requirements

The program is intended for execution in **Spyder**.

The code requires:

-   Python
-   NumPy
-   Pandas
-   Matplotlib

`psutil` is optional and is used for CPU and memory information when it
is available.

### Important

This version does **not** require:

``` text
scikit-learn
```

This was done so that the program can run in a Spyder environment where
`sklearn` is unavailable.

------------------------------------------------------------------------

## 5. Folder Structure

Create the following folder:

``` text
C:\a\
```

Place the files as follows:

``` text
C:\a\
│
├── assignment_regression_C_a.py
├── insurance.csv
│
└── outputs\
```

The `outputs` folder is created automatically by the program if it does
not already exist.

------------------------------------------------------------------------

# 6. Procedure

## Step 1 -- Download the Dataset

Download the `insurance.csv` dataset from Kaggle.

Save the CSV file as:

``` text
insurance.csv
```

Place it inside:

``` text
C:\a\
```

------------------------------------------------------------------------

## Step 2 -- Place the Python Program

Place:

``` text
assignment_regression_C_a.py
```

inside the same folder:

``` text
C:\a\
```

The program is explicitly configured with:

``` python
BASE_FOLDER = r"C:\a"
```

and loads:

``` python
DATA_FILE = os.path.join(
    BASE_FOLDER,
    "insurance.csv"
)
```

Therefore, the dataset must be available at:

``` text
C:\a\insurance.csv
```

------------------------------------------------------------------------

## Step 3 -- Open the Program in Spyder

Open Spyder.

Select:

``` text
File → Open
```

Open:

``` text
C:\a\assignment_regression_C_a.py
```

Run the program using:

``` text
F5
```

or click the **Run** button.

------------------------------------------------------------------------

# 7. Loading the Dataset

The program loads the dataset using pandas:

``` python
df = pd.read_csv(DATA_FILE)
```

It then displays:

-   Dataset shape
-   First five records
-   Column names
-   Missing values
-   Number of duplicate rows

This provides an initial inspection of the dataset.

------------------------------------------------------------------------

# 8. Target Variable

The target variable is:

``` text
charges
```

The program separates the target from the predictor variables:

``` python
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(float).values
```

Here:

-   `X` contains the predictor variables.
-   `y` contains the insurance charges to be predicted.

------------------------------------------------------------------------

# 9. Data Preprocessing

The dataset contains both numerical and categorical variables.

### Numerical variables

``` text
age
bmi
children
```

### Categorical variables

``` text
sex
smoker
region
```

Categorical variables are converted into numerical dummy variables using
pandas:

``` python
X_processed = pd.get_dummies(
    X,
    columns=categorical_features,
    drop_first=True,
    dtype=float
)
```

This converts categorical information into numerical values that can be
used by the regression algorithms.

------------------------------------------------------------------------

# 10. Train-Test Split

The dataset is divided into:

-   **80% training data**
-   **20% testing data**

The program creates a reproducible random split using:

``` python
np.random.seed(42)
```

The training data is used to build the models, while the testing data is
used to evaluate their predictions.

------------------------------------------------------------------------

# 11. Feature Standardization

Feature scaling is performed before the regression calculations.

The training mean and standard deviation are calculated:

``` python
train_mean = X_train.mean(axis=0)
train_std = X_train.std(axis=0)
```

The data is then standardized using:

``` python
X_train_scaled = (
    X_train - train_mean
) / train_std
```

and:

``` python
X_test_scaled = (
    X_test - train_mean
) / train_std
```

The training statistics are also used for the test data to avoid using
information from the test set during preprocessing.

Scaling is particularly important for KNN because KNN uses distances
between observations.

------------------------------------------------------------------------

# 12. Multiple Linear Regression

Multiple Linear Regression models the relationship between multiple
predictor variables and the continuous target variable.

The general form is:

``` text
Y = β0 + β1X1 + β2X2 + ... + βnXn
```

where:

-   `Y` = predicted insurance charges
-   `β0` = intercept
-   `β1 ... βn` = regression coefficients
-   `X1 ... Xn` = predictor variables

The program adds an intercept column and estimates the regression
coefficients using the NumPy pseudoinverse:

``` python
beta = np.linalg.pinv(
    X_train_mlr
).dot(y_train)
```

Predictions are then calculated using:

``` python
linear_predictions = (
    X_test_mlr.dot(beta)
)
```

------------------------------------------------------------------------

# 13. K-Nearest Neighbours Regression

KNN Regression predicts the target value of a new observation by
identifying the nearest training observations.

The program uses Euclidean distance:

``` text
Distance = √Σ(Xi - Yi)²
```

For each test observation:

1.  The distance to every training observation is calculated.
2.  The distances are sorted.
3.  The nearest K observations are selected.
4.  Their target values are averaged.
5.  The average becomes the predicted value.

The program implements KNN directly using NumPy.

------------------------------------------------------------------------

# 14. KNN Parameter Variation

The following K values are tested:

``` text
K = 3
K = 5
K = 7
K = 9
K = 11
```

This allows the effect of changing the number of neighbours to be
examined.

The relevant code is:

``` python
k_values = [3, 5, 7, 9, 11]
```

The results for each K value are stored for later comparison.

------------------------------------------------------------------------

# 15. Performance Metrics

The program evaluates every model using the following metrics.

## 15.1 Mean Absolute Error -- MAE

MAE measures the average absolute difference between actual and
predicted values.

``` text
MAE = (1/n) Σ |yi - ŷi|
```

A smaller MAE indicates smaller average absolute prediction errors.

------------------------------------------------------------------------

## 15.2 Mean Squared Error -- MSE

MSE calculates the average squared prediction error.

``` text
MSE = (1/n) Σ(yi - ŷi)²
```

Because the errors are squared, larger errors have greater influence on
MSE.

------------------------------------------------------------------------

## 15.3 Root Mean Squared Error -- RMSE

RMSE is the square root of MSE.

``` text
RMSE = √MSE
```

RMSE is expressed in the same units as the target variable.

------------------------------------------------------------------------

## 15.4 R² Score

R² measures the proportion of variation in the target that is explained
by the regression model.

The program calculates:

``` text
R² = 1 - SSres/SStot
```

A higher R² indicates that more variation in the target is explained by
the model.

------------------------------------------------------------------------

# 16. Computational Resource Measurement

The program also records:

-   Execution time
-   CPU utilization, when `psutil` is available
-   Memory utilization, when `psutil` is available

Execution time is measured using:

``` python
time.perf_counter()
```

This allows computational performance to be compared along with
prediction performance.

------------------------------------------------------------------------

# 17. Visualizations Generated

The program generates several plots.

## 17.1 Target Distribution

Shows the distribution of insurance charges.

File:

``` text
target_distribution.png
```

------------------------------------------------------------------------

## 17.2 Actual vs Predicted Plot

Shows actual insurance charges against predictions from Multiple Linear
Regression.

File:

``` text
linear_actual_vs_predicted.png
```

A reference diagonal line is included to help visually compare actual
and predicted values.

------------------------------------------------------------------------

## 17.3 Residual Plot

The residual plot shows prediction errors against predicted values.

File:

``` text
linear_residual_plot.png
```

Residuals are calculated as:

``` text
Residual = Actual - Predicted
```

------------------------------------------------------------------------

## 17.4 KNN RMSE vs K

Shows how RMSE changes for:

``` text
K = 3, 5, 7, 9, 11
```

File:

``` text
knn_rmse_vs_k.png
```

------------------------------------------------------------------------

## 17.5 KNN R² vs K

Shows how the R² score changes as K is varied.

File:

``` text
knn_r2_vs_k.png
```

------------------------------------------------------------------------

## 17.6 RMSE Comparison

Compares RMSE for Multiple Linear Regression and each KNN configuration.

File:

``` text
rmse_comparison.png
```

------------------------------------------------------------------------

## 17.7 R² Comparison

Compares R² scores for the tested models.

File:

``` text
r2_comparison.png
```

------------------------------------------------------------------------

## 17.8 MAE Comparison

Compares MAE values for the tested models.

File:

``` text
mae_comparison.png
```

------------------------------------------------------------------------

## 17.9 Execution Time Comparison

Compares the execution time of the models.

File:

``` text
execution_time_comparison.png
```

------------------------------------------------------------------------

# 18. Output Files

After successful execution, the following files are saved inside:

``` text
C:\a\outputs\
```

The output directory contains:

``` text
outputs/
│
├── target_distribution.png
├── linear_actual_vs_predicted.png
├── linear_residual_plot.png
├── knn_rmse_vs_k.png
├── knn_r2_vs_k.png
├── rmse_comparison.png
├── r2_comparison.png
├── mae_comparison.png
├── execution_time_comparison.png
├── regression_results.csv
└── summary.txt
```

------------------------------------------------------------------------

# 19. Results CSV

The file:

``` text
regression_results.csv
```

contains the performance results for all tested models.

The table includes:

-   Model
-   Parameter
-   MAE
-   MSE
-   RMSE
-   R² Score
-   Execution Time
-   CPU
-   Memory

The actual values should be taken from the results produced by running
the program on the dataset.

------------------------------------------------------------------------

# 20. Summary File

The program also creates:

``` text
summary.txt
```

This file contains:

-   Dataset information
-   Training and testing sample counts
-   Complete model comparison
-   Highest R² result
-   Lowest RMSE result
-   Lowest MAE result

The numerical results should be reported directly from the execution
output rather than entered manually.

------------------------------------------------------------------------

# 21. Expected Console Execution Sequence

When the program runs successfully, the console follows approximately
this sequence:

``` text
Python script folder: C:\a
Dataset found: C:\a\insurance.csv
Output folder: C:\a\outputs

MULTIPLE LINEAR REGRESSION vs KNN REGRESSION

Loading dataset...

Dataset loaded successfully.
Dataset shape: ...

First 5 rows:
...

Missing values:
...

Training samples: ...
Testing samples: ...

MULTIPLE LINEAR REGRESSION

MAE:
MSE:
RMSE:
R2:
Execution time:

K-NEAREST NEIGHBOURS REGRESSION

K = 3
MAE:
MSE:
RMSE:
R2:

K = 5
...

K = 7
...

K = 9
...

K = 11
...

FINAL MODEL COMPARISON

...
```

The exact numerical values depend on the dataset used during execution.

------------------------------------------------------------------------

# 22. Conclusion

This program provides a complete comparison between Multiple Linear
Regression and KNN Regression for predicting medical insurance charges.

Multiple Linear Regression is implemented using the NumPy pseudoinverse,
while KNN Regression is implemented using Euclidean distance and
neighbour averaging. KNN is evaluated using five different values of K:
3, 5, 7, 9, and 11.

The models are evaluated using MAE, MSE, RMSE, R² score, and execution
time. Additional plots are generated to visualize the target
distribution, prediction performance, residuals, KNN parameter effects,
and model comparisons.

The final numerical results should be taken from
`regression_results.csv` and `summary.txt` after executing the program.
