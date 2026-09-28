Assignment 2 - Comparing Multiple Linear Regression vs K-Nearest Neighbourhood Regression


Objective:

To compare Multiple Linear Regression and K-Nearest Neighbourhood Regression


Dataset:

This dataset contain Water Quality metrics obtained from Kaggle. It has 2371 data points. After eliminating the null values, we get 1320 data points


Independent Variables:

1. Salinity (ppt)
2. pH
3. SecchiDepth (m)
4. WaterDepth (m)
5. WaterTemp (C)
6. AirTemp (C) 


Dependent Variable: Dissolved Oxygen 


Methodology: 

The dataset was divided into two sets - 80% training and 20% testing with random state as 40. 
For KNNR, k was chosen as 13


Evaluation Metrics:
1. R2 score
2. RMSE
3. MAE
4. Fit Time
5. Predict Time
6. Fit Memory
7. Predict Memory

It was evaluated on the basis of three metrics - 
Accuracy metrics (R2, RMSE, MAE)
Time Metrics (Fit Time and Predict Time) 
Peak Memory Metrics (Fit Memory and Predict Memory)


Result: 

KNNR achieved better predictive accuracy than MLR, with a higher R2 and lower RMSE and MAE. However, MLR had lower prediction time and lower prediction memory.


File Structure:

Assignment-2
--waterquality.csv
--Water_quality_mlr_knnr.ipynb
--results_comparision.csv
--mlr_predicted_vs_actual.png
--knnr_predicted_vs_actual.png
--comparision_metrics.png




