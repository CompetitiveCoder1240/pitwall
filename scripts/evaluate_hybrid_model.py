import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.train_hybrid_model import fetch_and_clean_data, XGB_PARAMS
from backend.logger import logger
import fastf1

def evaluate():
    # Ensure cache is hooked up so we don't re-download data
    fastf1.Cache.enable_cache('fastf1_cache')
    
    logger.info("Fetching cached telemetry data for evaluation...")
    df = fetch_and_clean_data()
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    
    logger.info(f"Total clean laps: {len(df)}")
    
    # 80/20 Train-Test Split (random_state for reproducibility)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    logger.info(f"Training on {len(train_df)} laps, Testing on {len(test_df)} unseen laps.")
    
    # --- 1. MVR Anchor (Linear) ---
    # Get dummies but ensure columns align between train and test
    df_linear_train = pd.get_dummies(train_df, columns=['Circuit', 'Compound', 'Driver'], drop_first=True)
    df_linear_test = pd.get_dummies(test_df, columns=['Circuit', 'Compound', 'Driver'], drop_first=True)
    
    # Align test columns to train columns (fill missing with 0)
    df_linear_test = df_linear_test.reindex(columns=df_linear_train.columns, fill_value=0)
    
    X_train_lin = df_linear_train.drop(columns=['LapTime_Sec', 'TyreLife'])
    y_train_lin = df_linear_train['LapTime_Sec']
    
    X_test_lin = df_linear_test.drop(columns=['LapTime_Sec', 'TyreLife'])
    
    mvr = LinearRegression()
    mvr.fit(X_train_lin, y_train_lin)
    
    train_df['Predicted_Linear'] = mvr.predict(X_train_lin)
    test_df['Predicted_Linear'] = mvr.predict(X_test_lin)
    
    train_df['Residual'] = train_df['LapTime_Sec'] - train_df['Predicted_Linear']
    
    # --- 2. XGBoost Booster ---
    X_train_xgb = train_df[['Circuit', 'Compound', 'TyreLife', 'TrackTemp']].copy()
    X_test_xgb = test_df[['Circuit', 'Compound', 'TyreLife', 'TrackTemp']].copy()
    
    # Apply categorical types for XGBoost native support
    for col in ['Circuit', 'Compound']:
        X_train_xgb[col] = X_train_xgb[col].astype('category')
        # Important: test set categories must strictly match train set categories
        X_test_xgb[col] = pd.Categorical(X_test_xgb[col], categories=X_train_xgb[col].cat.categories)
        
    xgb = XGBRegressor(**XGB_PARAMS, enable_categorical=True)
    xgb.fit(X_train_xgb, train_df['Residual'])
    
    test_df['Pred_Residual'] = xgb.predict(X_test_xgb)
    test_df['Final_Pred'] = test_df['Predicted_Linear'] + test_df['Pred_Residual']
    
    # --- 3. Calculate Metrics ---
    # Baseline: How accurate is it if we ONLY use the linear physics math?
    mae_lin = mean_absolute_error(test_df['LapTime_Sec'], test_df['Predicted_Linear'])
    
    # Hybrid: How accurate is it when we add the XGBoost tyre cliff residuals?
    mae_hyb = mean_absolute_error(test_df['LapTime_Sec'], test_df['Final_Pred'])
    rmse_hyb = np.sqrt(mean_squared_error(test_df['LapTime_Sec'], test_df['Final_Pred']))
    
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS (80/20 SPLIT)")
    print("="*50)
    print(f"Linear-Only Baseline MAE : {mae_lin:.3f} seconds per lap")
    print(f"Hybrid ML Model MAE      : {mae_hyb:.3f} seconds per lap")
    print(f"Hybrid ML Model RMSE     : {rmse_hyb:.3f} seconds")
    
    improvement = ((mae_lin - mae_hyb) / mae_lin) * 100
    print(f"\nXGBoost improved accuracy by {improvement:.1f}% by capturing the tyre cliff!")
    print("="*50 + "\n")

if __name__ == "__main__":
    evaluate()
