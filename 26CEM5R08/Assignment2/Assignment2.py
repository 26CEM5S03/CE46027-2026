import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def main():
    print("Loading dataset...")
    df = pd.read_csv("dataset.csv") 
    df.columns = df.columns.str.strip()

    y = df['Crop_Yield_ton_per_hectare']
    X = df.drop(columns=['Crop_Yield_ton_per_hectare'])
    
    X_encoded = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Multiple Linear Regression (MLR)
    t0 = time.time()
    mlr = LinearRegression()
    mlr.fit(X_train_scaled, y_train)
    mlr_preds = mlr.predict(X_test_scaled)
    mlr_time = (time.time() - t0) * 1000

    mlr_r2 = r2_score(y_test, mlr_preds)
    mlr_rmse = np.sqrt(mean_squared_error(y_test, mlr_preds))
    mlr_mae = mean_absolute_error(y_test, mlr_preds)

    # 2. K-Nearest Neighbors Regression (KNNR)
    t0 = time.time()
    knnr = KNeighborsRegressor(n_neighbors=5)
    knnr.fit(X_train_scaled, y_train)
    knnr_preds = knnr.predict(X_test_scaled)
    knnr_time = (time.time() - t0) * 1000

    knnr_r2 = r2_score(y_test, knnr_preds)
    knnr_rmse = np.sqrt(mean_squared_error(y_test, knnr_preds))
    knnr_mae = mean_absolute_error(y_test, knnr_preds)

    print("\n================ PERFORMANCE METRICS ================")
    print(f"MLR  -> R2: {mlr_r2:.4f} | RMSE: {mlr_rmse:.4f} | MAE: {mlr_mae:.4f} | Time: {mlr_time:.2f} ms")
    print(f"KNNR -> R2: {knnr_r2:.4f} | RMSE: {knnr_rmse:.4f} | MAE: {knnr_mae:.4f} | Time: {knnr_time:.2f} ms")
    print("=====================================================")

if __name__ == "__main__":
    main()

