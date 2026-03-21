"""
src/data/feature_engineering.py
================================
Compute all 15 candidate technical features on DENOISED prices.

CRITICAL: ALL functions operate on denoised columns (*_d).
          Never pass raw OHLCV columns to these functions.

Responsibilities:
- compute_rsi()           : RSI-14
- compute_macd()          : MACD (EMA12 - EMA26)
- compute_bb_width()      : Bollinger Band width
- compute_roc()           : Rate of Change (5-day)
- compute_ema()           : EMA-9
- compute_atr()           : Average True Range (14)
- compute_obv()           : On-Balance Volume
- compute_stoch_k()       : Stochastic %K
- compute_williams_r()    : Williams %R
- compute_cci()           : Commodity Channel Index (20)
- compute_all_features()  : apply all above to a DataFrame
"""

import numpy as np
import pandas as pd
from config.config import CONFIG


def compute_rsi(series: pd.Series, period: int = None) -> pd.Series:
    """Compute Relative Strength Index.

    Args:
        series: Denoised Close price series (pd.Series).
        period: Lookback window. Defaults to CONFIG['rsi_period'] (14).

    Returns:
        pd.Series of RSI values in [0, 100], NaN for warmup rows.

    Example:
        >>> rsi = compute_rsi(df['Close_d'])
        >>> rsi.dropna().between(0, 100).all()
        True
    """
    if period is None:
        period = CONFIG['rsi_period']
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series,
                 fast: int = None,
                 slow: int = None) -> pd.Series:
    """Compute MACD line (EMA_fast - EMA_slow).

    Args:
        series: Denoised Close price series.
        fast:   Fast EMA span. Defaults to CONFIG['macd_fast'] (12).
        slow:   Slow EMA span. Defaults to CONFIG['macd_slow'] (26).

    Returns:
        pd.Series of MACD values. First ~26 rows are NaN-like (small).

    Example:
        >>> macd = compute_macd(df['Close_d'])
        >>> macd.isna().sum() == 0  # EWM has no hard NaN cutoff
        True
    """
    if fast is None:
        fast = CONFIG['macd_fast']
    if slow is None:
        slow = CONFIG['macd_slow']
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    return ema_fast - ema_slow


def compute_bb_width(series: pd.Series,
                     period: int = None,
                     n_std: int = None) -> pd.Series:
    """Compute Bollinger Band width: (Upper - Lower) / Middle.

    Args:
        series: Denoised Close price series.
        period: Rolling window. Defaults to CONFIG['bb_period'] (20).
        n_std:  Number of standard deviations. Defaults to CONFIG['bb_std'] (2).

    Returns:
        pd.Series of BB width values. First (period-1) rows are NaN.

    Example:
        >>> bb = compute_bb_width(df['Close_d'])
        >>> (bb.dropna() >= 0).all()
        True
    """
    if period is None:
        period = CONFIG['bb_period']
    if n_std is None:
        n_std = CONFIG['bb_std']
    sma = series.rolling(period).mean()
    sigma = series.rolling(period).std()
    upper = sma + n_std * sigma
    lower = sma - n_std * sigma
    return (upper - lower) / sma.replace(0, np.nan)


def compute_roc(series: pd.Series, period: int = None) -> pd.Series:
    """Compute Rate of Change over n periods.

    Args:
        series: Denoised Close price series.
        period: Lookback. Defaults to CONFIG['roc_period'] (5).

    Returns:
        pd.Series of ROC values in percent. First (period) rows are NaN.

    Example:
        >>> roc = compute_roc(df['Close_d'])
        >>> roc.dropna().shape[0] > 0
        True
    """
    if period is None:
        period = CONFIG['roc_period']
    return series.pct_change(periods=period) * 100


def compute_ema(series: pd.Series, period: int = None) -> pd.Series:
    """Compute Exponential Moving Average.

    Args:
        series: Denoised Close price series.
        period: EMA span. Defaults to CONFIG['ema_period'] (9).

    Returns:
        pd.Series of EMA values. No hard NaN rows (EWM).

    Example:
        >>> ema = compute_ema(df['Close_d'])
        >>> ema.isna().sum() == 0
        True
    """
    if period is None:
        period = CONFIG['ema_period']
    return series.ewm(span=period, adjust=False).mean()


def compute_atr(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = None) -> pd.Series:
    """Compute Average True Range.

    Args:
        high:   Denoised High series.
        low:    Denoised Low series.
        close:  Denoised Close series.
        period: Lookback. Defaults to CONFIG['atr_period'] (14).

    Returns:
        pd.Series of ATR values (always >= 0). First period rows are NaN.

    Example:
        >>> atr = compute_atr(df['High_d'], df['Low_d'], df['Close_d'])
        >>> (atr.dropna() >= 0).all()
        True
    """
    if period is None:
        period = CONFIG['atr_period']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume.

    Args:
        close:  Denoised Close series.
        volume: Denoised Volume series.

    Returns:
        pd.Series of cumulative OBV values.

    Example:
        >>> obv = compute_obv(df['Close_d'], df['Volume_d'])
        >>> obv.isna().sum() == 0
        True
    """
    direction = np.sign(close.diff().fillna(0))
    return (direction * volume).cumsum()


def compute_stoch_k(high: pd.Series,
                    low: pd.Series,
                    close: pd.Series,
                    period: int = None) -> pd.Series:
    """Compute Stochastic Oscillator %K.

    Args:
        high:   Denoised High series.
        low:    Denoised Low series.
        close:  Denoised Close series.
        period: Lookback. Defaults to CONFIG['stoch_period'] (14).

    Returns:
        pd.Series of %K values in [0, 100]. First period rows are NaN.

    Example:
        >>> sk = compute_stoch_k(df['High_d'], df['Low_d'], df['Close_d'])
        >>> sk.dropna().between(0, 100).all()
        True
    """
    if period is None:
        period = CONFIG['stoch_period']
    low_min = low.rolling(period).min()
    high_max = high.rolling(period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    return 100 * (close - low_min) / denom


def compute_williams_r(high: pd.Series,
                       low: pd.Series,
                       close: pd.Series,
                       period: int = None) -> pd.Series:
    """Compute Williams %R.

    Args:
        high:   Denoised High series.
        low:    Denoised Low series.
        close:  Denoised Close series.
        period: Lookback. Defaults to CONFIG['williams_period'] (14).

    Returns:
        pd.Series of %R values in [-100, 0]. First period rows are NaN.

    Example:
        >>> wr = compute_williams_r(df['High_d'], df['Low_d'], df['Close_d'])
        >>> wr.dropna().between(-100, 0).all()
        True
    """
    if period is None:
        period = CONFIG['williams_period']
    high_max = high.rolling(period).max()
    low_min = low.rolling(period).min()
    denom = (high_max - low_min).replace(0, np.nan)
    return -100 * (high_max - close) / denom


def compute_cci(high: pd.Series,
                low: pd.Series,
                close: pd.Series,
                period: int = None) -> pd.Series:
    """Compute Commodity Channel Index.

    Args:
        high:   Denoised High series.
        low:    Denoised Low series.
        close:  Denoised Close series.
        period: Lookback. Defaults to CONFIG['cci_period'] (20).

    Returns:
        pd.Series of CCI values. First period rows are NaN.

    Example:
        >>> cci = compute_cci(df['High_d'], df['Low_d'], df['Close_d'])
        >>> cci.dropna().shape[0] > 0
        True
    """
    if period is None:
        period = CONFIG['cci_period']
    typical = (high + low + close) / 3
    sma = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(
        lambda x: np.mean(np.abs(x - x.mean())), raw=True
    )
    return (typical - sma) / (0.015 * mad.replace(0, np.nan))


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 15 candidate technical features on denoised OHLCV columns.

    Assumes df already has columns: Open_d, High_d, Low_d, Close_d, Volume_d
    (output of preprocessor.denoise_ohlcv).

    Args:
        df: DataFrame with denoised OHLCV columns (*_d).

    Returns:
        DataFrame with all 15 candidate feature columns added.
        Original columns preserved.

    Example:
        >>> df_feat = compute_all_features(df_denoised)
        >>> 'RSI_14' in df_feat.columns
        True
    """
    result = df.copy()
    c = result['Close_d']
    h = result['High_d']
    l = result['Low_d']
    v = result['Volume_d']

    result['RSI_14'] = compute_rsi(c)
    result['MACD'] = compute_macd(c)
    result['BB_width'] = compute_bb_width(c)
    result['ROC_5'] = compute_roc(c)
    result['EMA_9'] = compute_ema(c)
    result['ATR_14'] = compute_atr(h, l, c)
    result['OBV'] = compute_obv(c, v)
    result['STOCH_K'] = compute_stoch_k(h, l, c)
    result['WILLIAMS_R'] = compute_williams_r(h, l, c)
    result['CCI_20'] = compute_cci(h, l, c)

    return result
