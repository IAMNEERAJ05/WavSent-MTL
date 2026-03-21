"""
src/data/preprocessor.py
=========================
Wavelet denoising and MinMaxScaler application.

Responsibilities:
- coif3_denoise()  : apply Coif3 wavelet soft-threshold to a 1-D series
- denoise_ohlcv()  : apply coif3_denoise to all OHLCV columns in a DataFrame
- apply_scaler()   : fit MinMaxScaler on train, transform train/val/test
- handle_missing() : forward-fill then drop remaining NaNs
"""

import numpy as np
import pandas as pd
import pywt
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List
from config.config import CONFIG


def coif3_denoise(series: np.ndarray) -> np.ndarray:
    """Apply Coif3 wavelet soft-thresholding to a 1-D signal.

    Uses universal threshold (sigma * sqrt(2 * log(N))) with
    sigma estimated via median absolute deviation of detail coefficients.

    Args:
        series: 1-D numpy array of raw values (e.g., Close prices).

    Returns:
        Denoised array of the same length as input.

    Example:
        >>> import numpy as np
        >>> x = np.sin(np.linspace(0, 10, 200)) + np.random.randn(200) * 0.3
        >>> d = coif3_denoise(x)
        >>> len(d) == len(x)
        True
    """
    wavelet = CONFIG['wavelet']       # 'coif3'
    level = CONFIG['wavelet_level']   # 1
    mode = CONFIG['wavelet_mode']     # 'soft'

    coeffs = pywt.wavedec(series, wavelet, level=level)
    # Estimate noise sigma from detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(series)))
    # Apply soft threshold to all detail coefficient arrays
    coeffs[1:] = [
        pywt.threshold(c, threshold, mode=mode)
        for c in coeffs[1:]
    ]
    denoised = pywt.waverec(coeffs, wavelet)
    return denoised[:len(series)]


def denoise_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Apply Coif3 denoising to each OHLCV column independently.

    Creates new columns: Open_d, High_d, Low_d, Close_d, Volume_d.
    Original columns are preserved unchanged.

    Args:
        df: DataFrame containing columns Open, High, Low, Close, Volume.

    Returns:
        DataFrame with 5 new *_d columns appended.
        Original columns untouched.

    Example:
        >>> df_out = denoise_ohlcv(df)
        >>> 'Close_d' in df_out.columns
        True
    """
    result = df.copy()
    for col in CONFIG['ohlcv_cols']:   # ['Open','High','Low','Close','Volume']
        result[f'{col}_d'] = coif3_denoise(df[col].values.astype(float))
    return result


def handle_missing(df: pd.DataFrame,
                   feature_cols: List[str]) -> pd.DataFrame:
    """Forward-fill then drop rows with remaining NaN in feature columns.

    Args:
        df: Input DataFrame.
        feature_cols: List of column names to check for NaN.

    Returns:
        Cleaned DataFrame with no NaN in feature_cols, index reset.

    Example:
        >>> df_clean = handle_missing(df, ['RSI_14', 'MACD'])
        >>> df_clean['RSI_14'].isnull().sum()
        0
    """
    result = df.copy()
    result[feature_cols] = result[feature_cols].ffill()
    result = result.dropna(subset=feature_cols).reset_index(drop=True)
    return result


def apply_scaler(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    save_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Fit MinMaxScaler on train only; transform train, val, test.

    CRITICAL: scaler is NEVER fit on val or test data.

    Args:
        train: 2-D array shape [n_train, n_features].
        val:   2-D array shape [n_val, n_features].
        test:  2-D array shape [n_test, n_features].
        save_path: Full path to save scaler.pkl via joblib.

    Returns:
        Tuple (train_scaled, val_scaled, test_scaled, scaler).
        All scaled arrays are float32.

    Example:
        >>> tr_s, v_s, te_s, sc = apply_scaler(tr, v, te, 'scaler.pkl')
        >>> tr_s.max() <= 1.0 and tr_s.min() >= 0.0
        True
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train).astype(np.float32)
    val_scaled = scaler.transform(val).astype(np.float32)
    test_scaled = scaler.transform(test).astype(np.float32)
    joblib.dump(scaler, save_path)
    return train_scaled, val_scaled, test_scaled, scaler


def apply_reg_scaler(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    save_path: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on regression targets using train only.

    Normalizes return magnitude targets so model output range is consistent.

    Args:
        y_train: 1-D array of train regression targets.
        y_val:   1-D array of val regression targets.
        y_test:  1-D array of test regression targets.
        save_path: Full path to save reg_scaler.pkl via joblib.

    Returns:
        Tuple (y_train_s, y_val_s, y_test_s, reg_scaler).

    Example:
        >>> yt_s, yv_s, yte_s, rsc = apply_reg_scaler(yt, yv, yte, 'r.pkl')
        >>> abs(yt_s.mean()) < 0.5
        True
    """
    reg_scaler = StandardScaler()
    y_train_s = reg_scaler.fit_transform(
        y_train.reshape(-1, 1)).ravel().astype(np.float32)
    y_val_s = reg_scaler.transform(
        y_val.reshape(-1, 1)).ravel().astype(np.float32)
    y_test_s = reg_scaler.transform(
        y_test.reshape(-1, 1)).ravel().astype(np.float32)
    joblib.dump(reg_scaler, save_path)
    return y_train_s, y_val_s, y_test_s, reg_scaler
