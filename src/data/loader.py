"""
src/data/loader.py
==================
Load raw price data and FinBERT sentiment outputs,
then merge them into per-dataset DataFrames.

Responsibilities:
- load_price_data()        : read nifty50_ohlcv.csv
- load_kotekar_sentiment() : read kotekar_sentiment.csv, aggregate by date
- load_kaggle_sentiment()  : read kaggle1/2, fill gap, concat
- merge_kotekar()          : merge price + kotekar sentiment
- merge_kaggle()           : merge price + kaggle sentiment
"""

import pandas as pd
from config.config import CONFIG


def load_price_data() -> pd.DataFrame:
    """Load Nifty50 OHLCV CSV and parse dates.

    Args:
        None — path taken from CONFIG['raw_data_dir'].

    Returns:
        pd.DataFrame with columns [Date, Open, High, Low, Close, Volume]
        sorted ascending by Date, index reset.

    Example:
        >>> df = load_price_data()
        >>> df.shape[1]
        6
    """
    path = CONFIG['raw_data_dir'] + 'nifty50_ohlcv.csv'
    df = pd.read_csv(path, header=0)
    # Handle multi-level header that yfinance sometimes produces
    if df.columns[0].startswith('Price') or df.iloc[0, 0] == 'Ticker':
        df = pd.read_csv(path, header=[0, 1])
        df.columns = [col[0] if col[1] == '' else col[0]
                      for col in df.columns]
    # Normalize Date column
    date_candidates = [c for c in df.columns if c.lower() in ('date', 'date_')]
    if date_candidates:
        df = df.rename(columns={date_candidates[0]: 'Date'})
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    available = [c for c in keep if c in df.columns]
    for col in available[1:]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[available]


def load_kotekar_sentiment() -> pd.DataFrame:
    """Load kotekar FinBERT output and aggregate to daily polarity_mean.

    Company-level sentiment is averaged across all companies per date.

    Args:
        None — path taken from CONFIG['finbert_output_dir'].

    Returns:
        pd.DataFrame with columns [date, polarity_mean] sorted by date.

    Example:
        >>> df = load_kotekar_sentiment()
        >>> 'polarity_mean' in df.columns
        True
    """
    path = CONFIG['finbert_output_dir'] + 'kotekar_sentiment.csv'
    df = pd.read_csv(path)
    date_col = CONFIG['kotekar_date_col']
    pol_col = CONFIG['kotekar_polarity_col']
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col])
    # Handle actual column name 'polarity' vs config name 'polarity_mean'
    if pol_col not in df.columns and 'polarity' in df.columns:
        df = df.rename(columns={'polarity': pol_col})
    daily = (
        df.groupby(date_col)
        .agg(polarity_mean=(pol_col, 'mean'))
        .reset_index()
    )
    return daily.sort_values(date_col).reset_index(drop=True)


def load_kaggle_sentiment() -> pd.DataFrame:
    """Load kaggle1 + kaggle2 FinBERT outputs and combine.

    DS1: Jan 2017 – Apr 2021 (polarity_mean + polarity_max)
    DS2: Jan 2022 – May 2024 (polarity_mean + polarity_max)
    Gap May 2021 – Dec 2021 will be zero-filled at merge step.

    Args:
        None — paths taken from CONFIG['finbert_output_dir'].

    Returns:
        pd.DataFrame with columns [date, polarity_mean, polarity_max]
        sorted by date.

    Example:
        >>> df = load_kaggle_sentiment()
        >>> set(df.columns) >= {'date', 'polarity_mean', 'polarity_max'}
        True
    """
    base = CONFIG['finbert_output_dir']
    k1 = pd.read_csv(base + 'kaggle1_polarity.csv')
    k2 = pd.read_csv(base + 'kaggle2_polarity.csv')

    date_col = CONFIG['kaggle_date_col']
    mean_col = CONFIG['kaggle_mean_col']
    max_col = CONFIG['kaggle_max_col']

    for df in [k1, k2]:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    combined = pd.concat([k1, k2], ignore_index=True)
    combined = combined.dropna(subset=[date_col])
    daily = (
        combined.groupby(date_col)
        .agg(
            polarity_mean=(mean_col, 'mean'),
            polarity_max=(max_col, 'max'),
        )
        .reset_index()
    )
    return daily.sort_values(date_col).reset_index(drop=True)


def merge_kotekar(price_df: pd.DataFrame,
                  sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Nifty50 price data with Kotekar daily sentiment.

    Filters price data to Kotekar period (Jan 2020 – May 2024).
    Missing sentiment dates filled with polarity_mean=0.0.

    Args:
        price_df: DataFrame from load_price_data().
        sentiment_df: DataFrame from load_kotekar_sentiment().

    Returns:
        pd.DataFrame with OHLCV + polarity_mean, filtered to kotekar period.
        polarity_mean has zero NaNs.

    Example:
        >>> merged = merge_kotekar(load_price_data(), load_kotekar_sentiment())
        >>> merged['polarity_mean'].isnull().sum()
        0
    """
    start = pd.to_datetime(CONFIG['kotekar_start'])
    end = pd.to_datetime(CONFIG['kotekar_end'])

    price = price_df.copy()
    price['Date'] = pd.to_datetime(price['Date'])
    price = price[(price['Date'] >= start) & (price['Date'] <= end)].copy()

    sent = sentiment_df.rename(
        columns={CONFIG['kotekar_date_col']: 'Date'}
    ).copy()
    sent['Date'] = pd.to_datetime(sent['Date'])

    merged = price.merge(sent, on='Date', how='left')
    merged['polarity_mean'] = merged['polarity_mean'].fillna(
        CONFIG['missing_polarity_mean']
    )
    return merged.reset_index(drop=True)


def merge_kaggle(price_df: pd.DataFrame,
                 sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Merge Nifty50 price data with Kaggle daily sentiment.

    Filters price data to full Kaggle period (Jan 2017 – May 2024).
    Gap May–Dec 2021 filled with polarity_mean=0, polarity_max=0
    per DECISIONS.md.

    Args:
        price_df: DataFrame from load_price_data().
        sentiment_df: DataFrame from load_kaggle_sentiment().

    Returns:
        pd.DataFrame with OHLCV + polarity_mean + polarity_max.
        Both sentiment columns have zero NaNs.

    Example:
        >>> merged = merge_kaggle(load_price_data(), load_kaggle_sentiment())
        >>> merged['polarity_max'].isnull().sum()
        0
    """
    start = pd.to_datetime(CONFIG['price_start'])
    end = pd.to_datetime(CONFIG['price_end'])

    price = price_df.copy()
    price['Date'] = pd.to_datetime(price['Date'])
    price = price[(price['Date'] >= start) & (price['Date'] <= end)].copy()

    sent = sentiment_df.rename(
        columns={CONFIG['kaggle_date_col']: 'Date'}
    ).copy()
    sent['Date'] = pd.to_datetime(sent['Date'])

    merged = price.merge(sent, on='Date', how='left')
    merged['polarity_mean'] = merged['polarity_mean'].fillna(
        CONFIG['missing_polarity_mean']
    )
    merged['polarity_max'] = merged['polarity_max'].fillna(
        CONFIG['missing_polarity_max']
    )

    # Verify gap is zero-filled
    gap_start = pd.to_datetime(CONFIG['gap_start'])
    gap_end = pd.to_datetime(CONFIG['gap_end'])
    gap = merged[
        (merged['Date'] >= gap_start) & (merged['Date'] <= gap_end)
    ]
    assert (gap['polarity_mean'] == 0).all(), \
        "Gap polarity_mean not zero-filled"
    assert (gap['polarity_max'] == 0).all(), \
        "Gap polarity_max not zero-filled"

    return merged.reset_index(drop=True)
