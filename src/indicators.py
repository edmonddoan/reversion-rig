"""
Technical indicators for mean reversion strategy.

This module provides functions to calculate various technical indicators
used in mean reversion trading strategies.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def sma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average (SMA).
    
    Args:
        series: Price series (e.g., close prices)
        period: Window size for moving average
        
    Returns:
        Series containing SMA values
        
    Example:
        >>> prices = pd.Series([1, 2, 3, 4, 5])
        >>> sma(prices, 3)
        0    NaN
        1    NaN
        2    2.0
        3    3.0
        4    4.0
        dtype: float64
    """
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average (EMA).
    
    Args:
        series: Price series (e.g., close prices)
        period: Window size for EMA
        
    Returns:
        Series containing EMA values
        
    Example:
        >>> prices = pd.Series([1, 2, 3, 4, 5])
        >>> ema(prices, 3)
    """
    return series.ewm(span=period, adjust=False).mean()


def bollinger_bands(
    series: pd.Series, 
    period: int = 20, 
    std_multiplier: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.
    
    Bollinger Bands consist of:
    - Middle Band: SMA of the price series
    - Upper Band: SMA + (std_multiplier * standard deviation)
    - Lower Band: SMA - (std_multiplier * standard deviation)
    
    Args:
        series: Price series (e.g., close prices)
        period: Window size for SMA and STD calculation
        std_multiplier: Number of standard deviations for bands
        
    Returns:
        Tuple of (middle_band, upper_band, lower_band) Series
        
    Example:
        >>> close = pd.Series([100, 102, 101, 103, 104])
        >>> middle, upper, lower = bollinger_bands(close, period=5)
    """
    middle_band = sma(series, period)
    std = series.rolling(window=period, min_periods=period).std()
    
    upper_band = middle_band + (std_multiplier * std)
    lower_band = middle_band - (std_multiplier * std)
    
    return middle_band, upper_band, lower_band


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index (RSI).
    
    RSI is a momentum oscillator that measures the speed and 
    change of price movements. It ranges from 0 to 100.
    
    Traditional interpretations:
    - RSI >= 70: Overbought
    - RSI <= 30: Oversold
    
    Args:
        series: Price series (e.g., close prices)
        period: Lookback period for RSI calculation
        
    Returns:
        Series containing RSI values (0-100)
        
    Example:
        >>> close = pd.Series([100, 102, 101, 103, 104])
        >>> rsi_values = rsi(close, period=14)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    
    # Use EMA for smoothing
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss
    rsi_series = 100 - (100 / (1 + rs))
    
    # Handle edge case where avg_loss is 0
    rsi_series = rsi_series.fillna(100)
    
    return rsi_series


def zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculate Z-Score of a price series.
    
    Z-score measures how many standard deviations a value is 
    from the mean. Useful for identifying mean reversion opportunities.
    
    Args:
        series: Price series
        window: Rolling window for mean and std calculation
        
    Returns:
        Series containing Z-scores
        
    Example:
        >>> close = pd.Series([100, 102, 101, 103, 104])
        >>> z = zscore(close, window=5)
    """
    rolling_mean = series.rolling(window=window, min_periods=window).mean()
    rolling_std = series.rolling(window=window, min_periods=window).std()
    
    zscore_series = (series - rolling_mean) / rolling_std
    zscore_series = zscore_series.fillna(0)
    
    return zscore_series


def calculate_all_indicators(
    df: pd.DataFrame,
    price_col: str = "Close",
    sma_period: int = 20,
    ema_period: int = 20,
    rsi_period: int = 14,
    bb_period: int = 20,
    bb_std: float = 2.0,
    zscore_window: int = 20
) -> pd.DataFrame:
    """
    Calculate all technical indicators and add to DataFrame.
    
    Args:
        df: DataFrame with price data (must have price_col)
        price_col: Name of the price column
        sma_period: Period for SMA
        ema_period: Period for EMA
        rsi_period: Period for RSI
        bb_period: Period for Bollinger Bands
        bb_std: Standard deviation multiplier for Bollinger Bands
        zscore_window: Window for Z-score calculation
        
    Returns:
        DataFrame with all indicators added as columns
        
    Example:
        >>> df = pd.DataFrame({'Close': [100, 102, 101, ...]})
        >>> df_with_indicators = calculate_all_indicators(df)
    """
    result = df.copy()
    price = result[price_col]
    
    # Moving Averages
    result[f"SMA_{sma_period}"] = sma(price, sma_period)
    result[f"EMA_{ema_period}"] = ema(price, ema_period)
    
    # Bollinger Bands
    bb_middle, bb_upper, bb_lower = bollinger_bands(
        price, period=bb_period, std_multiplier=bb_std
    )
    result["BB_Middle"] = bb_middle
    result["BB_Upper"] = bb_upper
    result["BB_Lower"] = bb_lower
    result["BB_Width"] = (bb_upper - bb_lower) / bb_middle
    result["BB_Position"] = (price - bb_lower) / (bb_upper - bb_lower)
    
    # RSI
    result[f"RSI_{rsi_period}"] = rsi(price, period=rsi_period)
    
    # Z-Score
    result[f"ZScore_{zscore_window}"] = zscore(price, window=zscore_window)
    
    return result
