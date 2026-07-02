import sys
import gzip
import pickle
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.preprocessing import MinMaxScaler
import statsmodels
import sklearn

def main():
    print("=" * 65)
    print("STARTING BULLETPROOF SARIMAX TRAINING PIPELINE")
    print("=" * 65)

    # --- ANTI-CRASH MEASURE 1: The Pickle Version Trap ---
    # Log versions so you know exactly what your Flask app needs to run this
    print(f"\n[Environment Info]")
    print(f"Python version:      {sys.version.split(' ')[0]}")
    print(f"Pandas version:      {pd.__version__}")
    print(f"Statsmodels version: {statsmodels.__version__}")
    print(f"Scikit-Learn:        {sklearn.__version__}")
    print("-> Note: Ensure your Flask app runs in an environment with these versions.\n")

    # 1. Load the Historical Dataset
    print("[1/6] Loading energy consumption data...")
    try:
        df = pd.read_csv("energy.csv")
        # Coerce errors to NaT (Not a Time) so bad date formats don't crash the script
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])  # Drop rows with totally unreadable dates
        df = df.set_index('timestamp')
        
        # --- ANTI-CRASH MEASURE 2: Timeline Gaps ---
        # Resample explicitly to hourly ('h') and interpolate to fill any missing timeline hours
        df = df.resample('h').interpolate(method='linear')
    except FileNotFoundError:
        print("ERROR: 'energy.csv' not found. Please place it in this directory.")
        return
    except Exception as e:
        print(f"ERROR loading dataset: {e}")
        return

    # 2. Filter Training Window
    print("[2/6] Extracting historical training timeline window...")
    train_data = df[(df.index >= '2012-01-01 00:00:00') & (df.index <= '2014-12-31 23:00:00')].copy()
    
    if train_data.empty:
        print("ERROR: Training window selection returned an empty dataframe. Check your CSV date bounds.")
        return

    # --- ANTI-CRASH MEASURE 3: Missing Data (NaNs) ---
    if train_data.isnull().values.any():
        print("-> Warning: Missing data (NaNs) detected in CSV. Applying automated linear interpolation...")
        train_data = train_data.interpolate(method='linear').bfill().ffill()

    # 3. Create and Fit a Fresh Scaler
    print("[3/6] Generating new MinMaxScaler from scratch...")
    scaler = MinMaxScaler()
    scaled_load = scaler.fit_transform(train_data[['load']]).flatten()
    print("-> Data successfully scaled to the 0-1 distribution range.")

    # 4. Extract Exogenous Temperature Feature
    print("[4/6] Extracting Exogenous Features...")
    if 'temp' in train_data.columns:
        exog_train = train_data[['temp']]
        print("-> Found 'temp' column.")
    elif 'temperature' in train_data.columns:
        exog_train = train_data[['temperature']]
        print("-> Found 'temperature' column.")
    else:
        exog_train = pd.DataFrame(np.zeros(len(train_data)), index=train_data.index, columns=['temp'])
        print("-> Warning: No explicit temperature column found. Using structural baseline array.")

    # 5. Initialize & Train the Model
    print("\n[5/6] Initializing SARIMAX(4, 1, 0)x(1, 1, 0, 24) architecture...")
    model = SARIMAX(
        endog=scaled_load,  
        exog=exog_train,
        order=(4, 1, 0),
        seasonal_order=(1, 1, 0, 24),
        initialization='approximate_diffuse'
    )

    print("      Executing mathematical optimization loop (Max 50 iterations)...")
    print("      (Please wait. The script is safely calculating mathematical weights.)")
    model_results = model.fit(disp=True, maxiter=50, cov_type='none', low_memory=True)

    # 6. Serialization & Exporting
    print("\n[6/6] Exporting freshly trained assets to disk...")
    try:
        with gzip.open("model.pkl", "wb") as f:
            pickle.dump(model_results, f)
        print("-> SUCCESS: 'model.pkl' created with converged weights.")

        with gzip.open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)
        print("-> SUCCESS: 'scaler.pkl' created with custom distribution bounds.")

    except Exception as e:
        print(f"ERROR saving pickle files: {e}")

    print("\n" + "=" * 65)
    print("TRAINING COMPLETE. YOU ARE READY TO RUN FLASK.")
    print("=" * 65)

if __name__ == "__main__":
    main()