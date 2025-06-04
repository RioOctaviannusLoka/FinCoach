import yfinance as yf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

tickers = ['BBCA.JK', 'BYAN.JK', 'TPIA.JK', 'BBRI.JK', 'BMRI.JK', 'DSSA.JK', 'TLKM.JK', 'ASII.JK', 'BBNI.JK', 'ICBP.JK']
window_size = 30

# Download data
df_all = pd.DataFrame()
scalers = {}
X_all, y_all = [], []

for ticker in tickers:
    df = yf.download(ticker, start="2018-01-01", end=None)
    close = df["Close"].dropna()
    df_all[ticker] = close

    series = close.values.reshape(-1, 1)
    scaler = StandardScaler()
    series_scaled = scaler.fit_transform(series).squeeze()
    scalers[ticker] = scaler

    for i in range(len(series_scaled) - window_size):
        X_all.append(series_scaled[i:i+window_size])
        y_all.append(series_scaled[i+window_size])

X_all = np.array(X_all).reshape(-1, window_size, 1)
y_all = np.array(y_all)


# Load and fine-tune model (contoh sederhana)
model = load_model("models/model.h5", compile=False)

model.compile(loss='mae', optimizer='adam', metrics=['mape'])

# Buat data untuk training ulang (X, y) — contoh dummy
model.fit(X_all, y_all, epochs=10, batch_size=32, verbose=1)

# Save the updated model
model.save("models/model.h5")

df_all.to_csv("stocks.csv")