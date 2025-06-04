import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
import os

# Load model
model = tf.keras.models.load_model("models/model.h5", compile=False)

# Load scalers
scalers = {}
scaler_dir = "models/scalers"
for fname in os.listdir(scaler_dir):
    if fname.endswith(".pkl"):
        symbol = fname.replace("_scaler.pkl", "")
        scalers[symbol] = joblib.load(os.path.join(scaler_dir, fname))

# Load stock data
df_all = pd.read_csv("stocks.csv", index_col=0, parse_dates=True)

def make_prediction(symbol: str, window_size=30):
    if symbol not in df_all.columns:
        raise ValueError(f"{symbol} not found in dataset.")
    
    if symbol not in scalers:
        raise ValueError(f"No scaler found for {symbol}")

    series = df_all[symbol].dropna().values
    if len(series) < window_size:
        raise ValueError("Not enough data for prediction.")

    scaler = scalers[symbol]
    scaled_series = scaler.transform(series.reshape(-1, 1)).squeeze()
    input_seq = scaled_series[-window_size:]

    forecast_days = {
        "1_day": 1,
        "1_week": 7,
        "1_month": 30,
        "3_months": 90,
        "6_months": 180
    }

    predictions = {}
    current_input = input_seq.copy()

    for horizon_name, steps_ahead in forecast_days.items():
        input_window = current_input.copy()
        for _ in range(steps_ahead):
            X = np.array(input_window[-window_size:]).reshape(1, window_size, 1)
            next_scaled = model.predict(X, verbose=0)[0][0]
            input_window = np.append(input_window, next_scaled)

        final_scaled = input_window[-1]
        final_pred = scaler.inverse_transform([[final_scaled]])[0][0]
        predictions[horizon_name] = round(float(final_pred), 2)

    return predictions