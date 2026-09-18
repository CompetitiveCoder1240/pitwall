import os
import pandas as pd
import numpy as np
import fastf1
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import joblib
from backend.logger import logger

# Cache path for FastF1
os.makedirs('fastf1_cache', exist_ok=True)
fastf1.Cache.enable_cache('fastf1_cache')

# ---------------------------------------------------------
# Hyperparameters & Model Configuration
# ---------------------------------------------------------
MODEL_DIR = "models"
SEASONS = [2022, 2023, 2024, 2025]
MIN_LAPS_FOR_STINT = 5

# XGBoost robust parameters to prevent overfitting
XGB_PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,             # Shallow trees to prevent memorizing specific driver anomalies
    "learning_rate": 0.05,      # Smooth learning curve
    "min_child_weight": 10,     # Needs evidence from 10+ laps to split (ignores isolated lockups)
    "subsample": 0.8,           # Use 80% of data per tree (reduces variance)
    "colsample_bytree": 0.8,    # Use 80% of features per tree
    "random_state": 42
}


def _get_track_temp_for_lap(weather_data, lap_time_obj):
    """Get the closest TrackTemp reading for a given lap's timestamp.
    
    FastF1 weather_data has a 'Time' column (timedelta from session start).
    Each lap has a 'Time' field (timedelta when the lap ended).
    We find the weather reading closest to the lap's end time.
    """
    if weather_data is None or weather_data.empty or lap_time_obj is None:
        return None
    
    try:
        # Find the weather reading with the smallest time difference
        time_diffs = (weather_data['Time'] - lap_time_obj).abs()
        closest_idx = time_diffs.idxmin()
        return float(weather_data.loc[closest_idx, 'TrackTemp'])
    except Exception:
        return None


def fetch_and_clean_data():
    """Fetch FastF1 data and build the raw feature dataset.
    
    Uses circuit_name (e.g. 'Monza') instead of EventName (e.g. 'Italian Grand Prix')
    to align with the SQLite DB schema used during inference.
    Extracts real TrackTemp from FastF1 weather data per lap.
    """
    all_laps = []
    
    for season in SEASONS:
        schedule = fastf1.get_event_schedule(season)
        for _, event in schedule.iterrows():
            if event['EventFormat'] == 'testing':
                continue
            
            try:
                session = fastf1.get_session(season, event['RoundNumber'], 'R')
                # Load with weather data to get Track Temperature
                session.load(telemetry=False, weather=True)
                
                laps = session.laps
                if laps is None or laps.empty:
                    continue
                
                # 1. Base Filters (Remove anomalies: Safety cars, in/out laps)
                # pick_quicklaps() automatically drops VSC/SC and in/out laps
                laps = laps.pick_quicklaps() 
                
                # Get weather data for TrackTemp alignment
                weather_data = session.weather_data
                
                # Use the circuit short name (e.g. "Monza") to match DB inference
                circuit_name = event.get('Location', event['EventName'])
                
                total_laps = session.total_laps if hasattr(session, 'total_laps') and session.total_laps else laps['LapNumber'].max()
                
                # Compute session-average TrackTemp as fallback
                session_avg_temp = 35.0
                if weather_data is not None and not weather_data.empty and 'TrackTemp' in weather_data.columns:
                    session_avg_temp = float(weather_data['TrackTemp'].mean())
                
                for driver in laps['Driver'].unique():
                    driver_laps = laps.pick_drivers(driver).copy()
                    driver_laps.dropna(subset=['LapTime', 'TyreLife', 'Compound'], inplace=True)
                    
                    if driver_laps.empty:
                        continue
                        
                    # Calculate Fuel Load Proxy (Remaining Laps)
                    driver_laps['FuelLoad_Laps'] = total_laps - driver_laps['LapNumber']
                    driver_laps['LapTime_Sec'] = driver_laps['LapTime'].dt.total_seconds()
                    
                    # 2. Anomaly Filtering (Lock-ups, mistakes, severe traffic/dirty air)
                    # We drop laps that are > 103% of the median lap time for that specific stint.
                    # This cleanly slices out dirty air traffic jams and driver lockups.
                    for stint, stint_data in driver_laps.groupby('Stint'):
                        if len(stint_data) < MIN_LAPS_FOR_STINT:
                            continue
                            
                        stint_median = stint_data['LapTime_Sec'].median()
                        clean_stint = stint_data[stint_data['LapTime_Sec'] <= (stint_median * 1.03)]
                        
                        for _, row in clean_stint.iterrows():
                            # Extract real TrackTemp for this specific lap
                            lap_temp = _get_track_temp_for_lap(weather_data, row.get('Time'))
                            track_temp = lap_temp if lap_temp is not None else session_avg_temp
                            
                            all_laps.append({
                                "Circuit": circuit_name,
                                "Compound": row['Compound'],
                                "Driver": row['Driver'],
                                "TyreLife": row['TyreLife'],
                                "FuelLoad": max(0, row['FuelLoad_Laps']),
                                "TrackTemp": track_temp,
                                "LapTime_Sec": row['LapTime_Sec']
                            })
                            
            except Exception as e:
                logger.warning(f"Failed to process {season} {event['EventName']}: {e}")
                
    return pd.DataFrame(all_laps)


def train_hybrid_model():
    """Train the Physics-Anchored Residual Boosting model."""
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        
    logger.info("Fetching and cleaning historical telemetry...")
    df = fetch_and_clean_data()
    
    if df.empty:
        logger.error("No data fetched. Aborting training.")
        return

    # Filter invalid compounds
    df = df[df['Compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]
    
    logger.info(f"Dataset compiled: {len(df)} clean racing laps.")
    logger.info(f"TrackTemp range: {df['TrackTemp'].min():.1f} - {df['TrackTemp'].max():.1f} deg C (mean: {df['TrackTemp'].mean():.1f})")
    logger.info(f"Circuits: {df['Circuit'].nunique()} unique circuit names")

    # ---------------------------------------------------------
    # STEP 1: The Physics Anchor (Multivariate Regression)
    # ---------------------------------------------------------
    logger.info("Training Linear Physics Anchor...")
    # One-hot encode Circuit, Compound, and Driver to set exact baseline pace
    df_linear = pd.get_dummies(df, columns=['Circuit', 'Compound', 'Driver'], drop_first=True)
    
    # Linear features: Baseline pace + Linear Fuel Burn + TrackTemp
    X_linear = df_linear.drop(columns=['LapTime_Sec', 'TyreLife'])
    y = df_linear['LapTime_Sec']
    
    mvr = LinearRegression()
    mvr.fit(X_linear, y)
    
    # Calculate Residuals (Actual - Predicted)
    df['Predicted_Linear_Pace'] = mvr.predict(X_linear)
    df['Residual_Time'] = df['LapTime_Sec'] - df['Predicted_Linear_Pace']
    
    # ---------------------------------------------------------
    # STEP 2: The ML Booster (XGBoost)
    # ---------------------------------------------------------
    logger.info("Training XGBoost Residual Predictor...")
    
    # Convert categoricals for XGBoost natively (using Pandas categorical types)
    df['Circuit'] = df['Circuit'].astype('category')
    df['Compound'] = df['Compound'].astype('category')
    
    X_xgb = df[['Circuit', 'Compound', 'TyreLife', 'TrackTemp']]
    y_xgb = df['Residual_Time']
    
    xgb = XGBRegressor(**XGB_PARAMS, enable_categorical=True)
    xgb.fit(X_xgb, y_xgb)
    
    # ---------------------------------------------------------
    # STEP 3: Save the Hybrid Pipeline
    # ---------------------------------------------------------
    logger.info(f"Training complete. Saving models to {MODEL_DIR}...")
    
    # Save categorical mapping for exact inference alignment
    circuit_cats = list(df['Circuit'].cat.categories)
    compound_cats = list(df['Compound'].cat.categories)
    
    joblib.dump({
        "mvr_model": mvr,
        "mvr_features": list(X_linear.columns),
        "circuit_categories": circuit_cats,
        "compound_categories": compound_cats
    }, os.path.join(MODEL_DIR, "mvr_anchor.pkl"))
    
    xgb.save_model(os.path.join(MODEL_DIR, "xgb_booster.json"))
    
    logger.info("Success! Hybrid model is ready for real-time inference.")


if __name__ == "__main__":
    train_hybrid_model()
