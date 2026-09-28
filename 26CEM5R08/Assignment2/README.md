# Assignment 2: Performance Comparison of MLR and KNNR
This project evaluates and compares the performance of two supervised machine learning algorithms—*Multiple Linear Regression (MLR)* and *K-Nearest Neighbors Regression (KNNR)*—for predicting agricultural crop yield (Crop_Yield_ton_per_hectare).
---
## 📌 Project Overview
- *Dataset*: dataset.csv (10,000 rows × 20 features)
- *Target Variable*: Crop_Yield_ton_per_hectare
- *Goal*: Measure and compare predictive accuracy ($R^2$, RMSE, MAE) and computational efficiency (execution time) between a parametric model (MLR) and a non-parametric model (KNNR).
---
## 🛠️ Methodology & Implementation Workflow
1. *Data Preprocessing*:
   - Column names were stripped of trailing/leading whitespace.
   - Categorical features were converted to numeric representations using *One-Hot Encoding* (pd.get_dummies with drop_first=True to prevent dummy variable trap).
2. *Data Splitting*:
   - Partitioned dataset into *80% Training Set* and *20% Testing Set* (random_state=42 for reproducibility).
3. *Feature Standardization*:
   - Applied StandardScaler to ensure zero mean and unit variance across features, particularly critical for distance-based KNN algorithms.
4. *Model Execution & Timing*:
   - *Multiple Linear Regression (MLR)* fitted on scaled training data.
   - *K-Nearest Neighbors Regression (KNNR)* initialized with $k=5$.
   - Measured execution time for both model fitting and prediction phases in milliseconds (ms).
---
## 📊 Comparative Performance Results

| Metric | Multiple Linear Regression (MLR) | K-Nearest Neighbors (KNNR, k=5) | Superior Model |
| :--- | :--- | :--- | :--- |
| *$R^2$ Score* (Higher is better) | *0.9980* | 0.9445 | *MLR* |
| *RMSE* (Lower is better) | *1.0810* | 5.6896 | *MLR* |
| *MAE* (Lower is better) | *0.8638* | 2.9864 | *MLR* |
| *Execution Time* (Lower is better) | *531.79 ms* | 9662.42 ms | *MLR* |

---
## 💡 Key Insights & Conclusion
- *Accuracy: Multiple Linear Regression achieved an $R^2$ of **0.9980, capturing 99.8% of target variance compared to KNNR's **0.9445*.
- *Error Margin: MLR significantly reduced prediction errors, achieving an RMSE of **1.0810* vs *5.6896* for KNNR.
- *Computational Efficiency: MLR executed roughly **18 times faster* than KNNR (531.79 ms vs 9662.42 ms), highlighting the computational overhead of instance-based learning on larger datasets during testing.
- *Final Verdict*: Multiple Linear Regression is optimal for this dataset due to higher predictive performance and vastly faster inference speed.
---
## 📂 File Structure
```text
Assignment2/
│
├── dataset.csv          # Input dataset for Crop Yield Prediction
├── Assignment2.py       # Main Python script for data processing & model training
└── README.md            # Project documentation and performance analysis