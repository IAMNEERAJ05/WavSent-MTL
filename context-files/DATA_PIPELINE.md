# WavSent-MTL — Data Pipeline Specification

## CRITICAL INSTRUCTIONS
- Follow steps in EXACT ORDER
- NEVER compute features on raw prices
- NEVER fit scaler on val or test data
- NEVER shuffle data at any point
- NEVER touch test set until notebook 09

---

## Overview

Step 1:  Download price data (Yahoo Finance)
Step 2:  Load sentiment outputs (reused from prior work)
Step 3:  Merge price + sentiment by trading date
Step 4:  Coif3 wavelet denoising on OHLCV
Step 5:  Compute all 15 candidate features on denoised prices
Step 6:  MI-based feature ranking → keep top 10
Step 7:  SHAP-based filter → keep top 6-7
Step 8:  Save selected_features.json
Step 9:  Build final feature dataframe
Step 10: Verify no missing values
Step 11: 70/15/15 temporal split
Step 12: Fit MinMaxScaler on train only
Step 13: Build sliding windows [5 × n_features]
Step 14: Generate targets (clf + reg)
Step 15: Check class imbalance
Step 16: Save all arrays

---

## Step 1 — Download Price Data

import yfinance as yf
import pandas as pd

nifty = yf.download('^NSEI',
                    start='2017-01-01',
                    end='2024-05-31')
nifty = nifty[['Open','High','Low','Close','Volume']]
nifty.index = pd.to_datetime(nifty.index)
nifty = nifty.sort_index()
nifty.to_csv('data/raw/nifty50_ohlcv.csv')
# Expected shape: ~(1,800, 5)

---

## Step 2 — Load Sentiment Outputs

### Kotekar (Study 1)
kotekar_sent = pd.read_csv(
    'data/finbert_outputs/kotekar_sentiment.csv')
# Columns: date, company, polarity_mean
# Aggregate to daily mean per date
kotekar_daily = kotekar_sent.groupby('date').agg(
    polarity_mean=('polarity_mean','mean')
).reset_index()

### Kaggle (Study 2)
k1 = pd.read_csv(
    'data/finbert_outputs/kaggle1_polarity.csv')
k2 = pd.read_csv(
    'data/finbert_outputs/kaggle2_polarity.csv')
# Columns: date, polarity_mean, polarity_max
kaggle_sent = pd.concat([k1, k2]).sort_values('date')
kaggle_daily = kaggle_sent.groupby('date').agg(
    polarity_mean=('polarity_mean','mean'),
    polarity_max=('polarity_max','max')
).reset_index()

---

## Step 3 — Merge Price + Sentiment

### Kotekar merge
price_df = pd.read_csv('data/raw/nifty50_ohlcv.csv')
price_df['date'] = pd.to_datetime(
    price_df['Date']).dt.date

# Filter to kotekar period
kotekar_price = price_df[
    (price_df['date'] >= pd.to_datetime(
        '2020-01-01').date()) &
    (price_df['date'] <= pd.to_datetime(
        '2024-05-31').date())
].copy()

df_kotekar = kotekar_price.merge(
    kotekar_daily, on='date', how='left')
df_kotekar['polarity_mean'] = \
    df_kotekar['polarity_mean'].fillna(0.0)
df_kotekar.to_csv(
    'data/processed/kotekar/merged_data.csv',
    index=False)

### Kaggle merge
kaggle_price = price_df[
    (price_df['date'] >= pd.to_datetime(
        '2017-01-01').date()) &
    (price_df['date'] <= pd.to_datetime(
        '2024-05-31').date())
].copy()

df_kaggle = kaggle_price.merge(
    kaggle_daily, on='date', how='left')

# Gap period May–Dec 2021: fill with 0
df_kaggle['polarity_mean'] = \
    df_kaggle['polarity_mean'].fillna(0.0)
df_kaggle['polarity_max'] = \
    df_kaggle['polarity_max'].fillna(0.0)
df_kaggle.to_csv(
    'data/processed/kaggle/merged_data.csv',
    index=False)

# Verify gap fill
gap = df_kaggle[
    (df_kaggle['date'] >= '2021-05-01') &
    (df_kaggle['date'] <= '2021-12-31')]
assert (gap['polarity_mean'] == 0).all()
assert (gap['polarity_max'] == 0).all()

---

## Step 4 — Coif3 Wavelet Denoising
## MUST happen before Step 5

import pywt
import numpy as np

def coif3_denoise(series):
    """
    Coif3, level=1, soft threshold.
    Universal threshold via median absolute deviation.
    """
    coeffs = pywt.wavedec(series, 'coif3', level=1)
    sigma = np.median(
        np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(
        2 * np.log(len(series)))
    coeffs[1:] = [
        pywt.threshold(c, threshold, mode='soft')
        for c in coeffs[1:]]
    return pywt.waverec(
        coeffs, 'coif3')[:len(series)]

for col in ['Open','High','Low','Close','Volume']:
    df[f'{col}_d'] = coif3_denoise(df[col].values)

# Apply to both df_kotekar and df_kaggle

---

## Step 5 — Compute All 15 Candidate Features
## ALL on DENOISED prices only

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta>0,0).rolling(period).mean()
    loss = -delta.where(delta<0,0).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def compute_macd(series, fast=12, slow=26):
    ema_fast = series.ewm(span=fast,adjust=False).mean()
    ema_slow = series.ewm(span=slow,adjust=False).mean()
    return ema_fast - ema_slow

def compute_bb_width(series, period=20, std=2):
    sma = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    return ((sma + std*sigma) - (sma - std*sigma)) / sma

def compute_roc(series, period=5):
    return series.pct_change(periods=period) * 100

def compute_ema(series, period=9):
    return series.ewm(span=period,adjust=False).mean()

def compute_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def compute_obv(close, volume):
    direction = np.sign(close.diff())
    return (direction * volume).cumsum()

def compute_stoch_k(high, low, close, period=14):
    low_min = low.rolling(period).min()
    high_max = high.rolling(period).max()
    return 100 * (close - low_min) / (
        high_max - low_min)

def compute_williams_r(high, low, close, period=14):
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    return -100 * (high_max - close) / (
        high_max - low_min)

def compute_cci(high, low, close, period=20):
    typical = (high + low + close) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())))
    return (typical - sma) / (0.015 * mad)

# Apply all to df['Close_d'], df['High_d'] etc.
df['RSI_14']     = compute_rsi(df['Close_d'])
df['MACD']       = compute_macd(df['Close_d'])
df['BB_width']   = compute_bb_width(df['Close_d'])
df['ROC_5']      = compute_roc(df['Close_d'])
df['EMA_9']      = compute_ema(df['Close_d'])
df['ATR_14']     = compute_atr(
    df['High_d'], df['Low_d'], df['Close_d'])
df['OBV']        = compute_obv(
    df['Close_d'], df['Volume_d'])
df['STOCH_K']    = compute_stoch_k(
    df['High_d'], df['Low_d'], df['Close_d'])
df['WILLIAMS_R'] = compute_williams_r(
    df['High_d'], df['Low_d'], df['Close_d'])
df['CCI_20']     = compute_cci(
    df['High_d'], df['Low_d'], df['Close_d'])

---

## Step 6 — MI-Based Feature Ranking (Kotekar only)

from sklearn.feature_selection import (
    mutual_info_classif)

CANDIDATE_FEATURES = [
    'Close_d','Open_d','High_d','Low_d','Volume_d',
    'RSI_14','MACD','BB_width','ROC_5','EMA_9',
    'ATR_14','OBV','STOCH_K','WILLIAMS_R','CCI_20'
]

# Use training set only — after split in Step 11
# (run this after Step 11 on train split)
mi_scores = mutual_info_classif(
    train_df[CANDIDATE_FEATURES],
    y_clf_train,
    random_state=42)

mi_ranking = pd.Series(
    mi_scores,
    index=CANDIDATE_FEATURES
).sort_values(ascending=False)

top_10 = mi_ranking.head(10).index.tolist()
print("Top 10 by MI:", top_10)

---

## Step 7 — SHAP Filter (Kotekar only)

# Train lightweight LSTM (32 units, 10 seeds)
# on top_10 features, compute SHAP
# Keep top 6-7 by mean absolute SHAP value
# Implementation in src/data/feature_selection.py

from src.data.feature_selection import (
    run_feature_selection)

selected = run_feature_selection(
    X_train_top10,
    y_clf_train,
    top_n_final=7)  # or 6 — check SHAP gap

---

## Step 8 — Save Selected Features

import json

selected_features_kotekar = selected + ['polarity_mean']
selected_features_kaggle = selected + [
    'polarity_mean', 'polarity_max']

with open(
    'data/processed/kotekar/selected_features.json',
    'w') as f:
    json.dump(selected_features_kotekar, f)

# Kaggle uses same technical features
with open(
    'data/processed/kaggle/selected_features.json',
    'w') as f:
    json.dump(selected_features_kaggle, f)

---

## Step 9 — Build Final Feature Dataframe

df = df[['date'] + SELECTED_FEATURES + ['Close']]
# Drop warmup rows (26 days for MACD)
df = df.dropna(
    subset=SELECTED_FEATURES).reset_index(drop=True)
print(f"Final shape: {df.shape}")
df.to_csv(
    'data/processed/{dataset}/featured_data.csv',
    index=False)

---

## Step 10 — Verify No Missing Values

assert df[SELECTED_FEATURES].isnull().sum().sum() == 0,\
    "Missing values found — fix pipeline"
print(f"Verified. Shape: {df.shape}")
print(f"Date range: {df['date'].min()} "
      f"to {df['date'].max()}")

---

## Step 11 — Temporal Split

n = len(df)
train_end = int(n * 0.70)
val_end   = int(n * 0.85)

train_df = df.iloc[:train_end].reset_index(drop=True)
val_df   = df.iloc[train_end:val_end].reset_index(
    drop=True)
test_df  = df.iloc[val_end:].reset_index(drop=True)

assert train_df['date'].max() < val_df['date'].min()
assert val_df['date'].max() < test_df['date'].min()
print(f"Train: {len(train_df)} | "
      f"Val: {len(val_df)} | "
      f"Test: {len(test_df)}")

---

## Step 12 — Scaling (fit on train ONLY)

from sklearn.preprocessing import MinMaxScaler
import joblib

scaler = MinMaxScaler()  # per-feature by default
train_scaled = scaler.fit_transform(
    train_df[SELECTED_FEATURES])
val_scaled   = scaler.transform(
    val_df[SELECTED_FEATURES])
test_scaled  = scaler.transform(
    test_df[SELECTED_FEATURES])

joblib.dump(scaler,
    'data/processed/{dataset}/scaler.pkl')

---

## Step 13 — Sliding Windows

WINDOW_SIZE = 5

def create_windows(scaled_data, window_size=5):
    X = []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i])
    return np.array(X)

X_train = create_windows(train_scaled)
X_val   = create_windows(val_scaled)
X_test  = create_windows(test_scaled)

---

## Step 14 — Generate Targets

def generate_targets(df, window_size=5):
    y_clf, y_reg = [], []
    close = df['Close'].values
    for i in range(window_size, len(df)):
        # Classification: up or down
        y_clf.append(
            1 if close[i] > close[i-1] else 0)
        # Regression: return magnitude
        y_reg.append(
            (close[i] - close[i-1]) / close[i-1])
    return np.array(y_clf), np.array(y_reg)

y_clf_train, y_reg_train = generate_targets(train_df)
y_clf_val,   y_reg_val   = generate_targets(val_df)
y_clf_test,  y_reg_test  = generate_targets(test_df)

# Normalize regression targets using scaler
# (scale to match model output range)
from sklearn.preprocessing import StandardScaler
reg_scaler = StandardScaler()
y_reg_train = reg_scaler.fit_transform(
    y_reg_train.reshape(-1,1)).ravel()
y_reg_val   = reg_scaler.transform(
    y_reg_val.reshape(-1,1)).ravel()
y_reg_test  = reg_scaler.transform(
    y_reg_test.reshape(-1,1)).ravel()

---

## Step 15 — Class Imbalance Check

from collections import Counter
from sklearn.utils.class_weight import (
    compute_class_weight)
import json

counts = Counter(y_clf_train)
ratio = max(counts.values()) / min(counts.values())
print(f"Up: {counts[1]/len(y_clf_train)*100:.1f}% | "
      f"Down: {counts[0]/len(y_clf_train)*100:.1f}% | "
      f"Ratio: {ratio:.2f}")

if ratio > 1.5:
    weights = compute_class_weight(
        'balanced',
        classes=np.array([0,1]),
        y=y_clf_train)
    cw = {0: float(weights[0]),
          1: float(weights[1])}
else:
    cw = None

with open(
    'data/processed/{dataset}/class_weights.json',
    'w') as f:
    json.dump(cw, f)

---

## Step 16 — Save All Arrays

arrays = {
    'X_train': X_train, 'X_val': X_val,
    'X_test': X_test,
    'y_clf_train': y_clf_train,
    'y_clf_val': y_clf_val,
    'y_clf_test': y_clf_test,
    'y_reg_train': y_reg_train,
    'y_reg_val': y_reg_val,
    'y_reg_test': y_reg_test
}

for name, arr in arrays.items():
    np.save(
        f'data/processed/{{dataset}}/{name}.npy',
        arr)

print("All arrays saved.")
print(f"X_train: {X_train.shape}")
print(f"X_val:   {X_val.shape}")
print(f"X_test:  {X_test.shape}")

---

## Expected Final Shapes

### Kotekar (~1,090 total days)
X_train:      (~730, 5, n_feat)  float32
X_val:        (~155, 5, n_feat)  float32
X_test:       (~155, 5, n_feat)  float32
y_clf_*:      (~n,)              int32 {0,1}
y_reg_*:      (~n,)              float32

### Kaggle (~1,800 total days)
X_train:      (~1,230, 5, n_feat) float32
X_val:        (~260, 5, n_feat)   float32
X_test:       (~260, 5, n_feat)   float32
y_clf_*:      (~n,)               int32 {0,1}
y_reg_*:      (~n,)               float32

---

## Pipeline Run Order Per Dataset
notebooks/01_data_prep_kotekar.ipynb → Steps 1,2,3
notebooks/02_data_prep_kaggle.ipynb  → Steps 1,2,3
notebooks/03_feature_engineering.ipynb → Steps 4,5
notebooks/04_feature_selection.ipynb → Steps 6,7,8,9,
                                        10,11,12,13,
                                        14,15,16