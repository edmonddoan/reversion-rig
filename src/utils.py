"""
Utility functions for the reversion-rig project.

This module provides helper functions for data processing, validation,
formatting, and general utilities used across the project.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, List, Dict, Any
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logger(name: str = "reversion-rig") -> logging.Logger:
    """
    Set up a logger for the project.
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(logging.INFO)
    
    # Add console handler if not already present
    if not logger_instance.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger_instance.addHandler(handler)
    
    return logger_instance


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 10
) -> bool:
    """
    Validate a DataFrame has required structure.
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        min_rows: Minimum number of rows required
        
    Returns:
        True if valid, raises ValueError otherwise
    """
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    if len(df) < min_rows:
        raise ValueError(f"DataFrame has fewer than {min_rows} rows")
    
    if required_columns:
        missing = set(required_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    return True


def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess price data.
    
    Args:
        df: DataFrame with price data
        
    Returns:
        Cleaned DataFrame
    """
    result = df.copy()
    
    # Convert all column names to lowercase
    result.columns = [col.lower() for col in result.columns]
    
    # Remove any unnamed columns (from CSV import)
    result = result.loc[:, ~result.columns.str.contains('^unnamed')]
    
    # Ensure index is datetime
    if not isinstance(result.index, pd.DatetimeIndex):
        try:
            result.index = pd.to_datetime(result.index)
        except Exception as e:
            logger.warning(f"Could not convert index to datetime: {e}")
    
    # Remove duplicate rows
    result = result[~result.index.duplicated(keep='first')]
    
    # Forward fill missing values (for price data)
    result = result.ffill()
    
    # Drop remaining NaN rows
    result = result.dropna()
    
    return result


def calculate_returns(
    prices: Union[pd.Series, pd.DataFrame],
    method: str = "simple",
    period: int = 1
) -> Union[pd.Series, pd.DataFrame]:
    """
    Calculate returns from price series.
    
    Args:
        prices: Price series or DataFrame
        method: 'simple' for simple returns, 'log' for log returns
        period: Period for returns calculation
        
    Returns:
        Series or DataFrame of returns
    """
    if method == "simple":
        returns = prices.pct_change(periods=period)
    elif method == "log":
        returns = np.log(prices / prices.shift(periods=period))
    else:
        raise ValueError(f"Unknown return method: {method}")
    
    return returns


def annualize_returns(
    returns: pd.Series,
    periods_per_year: int = 252
) -> float:
    """
    Annualize returns from periodic returns.
    
    Args:
        returns: Series of periodic returns
        periods_per_year: Number of periods per year
        
    Returns:
        Annualized return
    """
    if returns.empty:
        return 0.0
    
    total_return = (1 + returns).prod()
    n_periods = len(returns)
    years = n_periods / periods_per_year
    
    if years > 0:
        annualized = (1 + total_return) ** (1 / years) - 1
    else:
        annualized = total_return - 1
    
    return annualized


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        years: Number of years
        
    Returns:
        CAGR as decimal
    """
    if initial_value <= 0 or years <= 0:
        return 0.0
    
    cagr = (final_value / initial_value) ** (1 / years) - 1
    return cagr


def format_currency(value: float, currency: str = "$") -> str:
    """
    Format a value as currency.
    
    Args:
        value: Value to format
        currency: Currency symbol
        
    Returns:
        Formatted string
    """
    if abs(value) >= 1e9:
        return f"{currency}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{currency}{value/1e6:.2f}M"
    elif abs(value) >= 1e3:
        return f"{currency}{value/1e3:.2f}K"
    else:
        return f"{currency}{value:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """
    Format a value as percentage.
    
    Args:
        value: Value to format (decimal form, e.g., 0.05 for 5%)
        decimals: Number of decimal places
        
    Returns:
        Formatted percentage string
    """
    return f"{value * 100:.{decimals}f}%"


def format_number(value: float, decimals: int = 2) -> str:
    """
    Format a number with commas and decimals.
    
    Args:
        value: Value to format
        decimals: Number of decimal places
        
    Returns:
        Formatted number string
    """
    return f"{value:,.{decimals}f}"


def save_results(
    results: Dict[str, Any],
    filepath: Union[str, Path],
    format: str = "json"
) -> None:
    """
    Save backtest results to file.
    
    Args:
        results: Results dictionary
        filepath: Output file path
        format: Output format ('json' or 'csv')
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if format == "json":
        # Convert non-serializable items
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, pd.Series):
                serializable_results[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                serializable_results[key] = value.to_dict(orient='records')
            elif hasattr(value, 'item'):  # numpy types
                serializable_results[key] = value.item()
            else:
                serializable_results[key] = value
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
    
    elif format == "csv":
        # Save portfolio values if present
        if "portfolio_values" in results:
            portfolio_df = results["portfolio_values"].to_frame("portfolio_value")
            portfolio_df.to_csv(filepath)
    
    logger.info(f"Results saved to {filepath}")


def load_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load backtest results from file.
    
    Args:
        filepath: Path to results file
        
    Returns:
        Results dictionary
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    return results


def get_date_range(
    start_date: Optional[Union[str, datetime]] = None,
    end_date: Optional[Union[str, datetime]] = None,
    days: int = 365
) -> tuple:
    """
    Get standardized date range.
    
    Args:
        start_date: Start date (optional)
        end_date: End date (optional)
        days: Default number of days if start_date not provided
        
    Returns:
        Tuple of (start_date, end_date) as datetime objects
    """
    if end_date is None:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
    
    if start_date is None:
        start_date = end_date - timedelta(days=days)
    elif isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
    
    return start_date, end_date


def normalize_prices(df: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """
    Normalize prices to a base value for comparison.
    
    Args:
        df: DataFrame with price data
        base: Base value for normalization
        
    Returns:
        Normalized DataFrame
    """
    result = df.copy()
    for col in result.columns:
        if result[col].iloc[0] != 0:
            result[col] = (result[col] / result[col].iloc[0]) * base
    return result


def merge_on_date(
    *dfs: pd.DataFrame,
    how: str = "inner"
) -> pd.DataFrame:
    """
    Merge multiple DataFrames on their date index.
    
    Args:
        *dfs: DataFrames to merge
        how: Merge method ('inner', 'outer', 'left', 'right')
        
    Returns:
        Merged DataFrame
    """
    from functools import reduce
    
    merged = reduce(
        lambda left, right: pd.merge(
            left, right, left_index=True, right_index=True, how=how
        ),
        dfs
    )
    
    return merged


def resample_prices(
    df: pd.DataFrame,
    frequency: str = "1W",
    method: str = "last"
) -> pd.DataFrame:
    """
    Resample price data to a different frequency.
    
    Args:
        df: DataFrame with price data
        frequency: Pandas offset alias ('1D', '1W', '1M', etc.)
        method: Aggregation method ('last', 'mean', 'max', 'min')
        
    Returns:
        Resampled DataFrame
    """
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    resampled = df.resample(frequency).agg(ohlc_dict)
    
    # Drop rows with no data
    resampled = resampled.dropna(how='all')
    
    return resampled
