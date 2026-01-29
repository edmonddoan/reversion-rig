"""
Data fetcher module for retrieving market data using yfinance.

This module provides functions to download historical price data
from Yahoo Finance for backtesting the mean reversion strategy.
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Union
from ..config import DATA_DIR, DATA_CONFIG


def download_data(
    tickers: Union[str, List[str]],
    period: str = DATA_CONFIG["period"],
    interval: str = DATA_CONFIG["interval"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    prepost: bool = DATA_CONFIG["prepost"],
    auto_adjust: bool = DATA_CONFIG["auto_adjust"],
    progress: bool = True
) -> pd.DataFrame:
    """
    Download historical market data from Yahoo Finance.
    
    Args:
        tickers: Single ticker (str) or list of tickers
        period: Data period (e.g., '1y', '5y', 'max')
        interval: Data interval (e.g., '1d', '1h', '1m')
        start_date: Start date in 'YYYY-MM-DD' format (optional)
        end_date: End date in 'YYYY-MM-DD' format (optional)
        prepost: Include pre-market and post-market data
        auto_adjust: Adjust for splits and dividends
        progress: Show progress bar
        
    Returns:
        DataFrame with multi-index columns (ticker, feature)
        or single-level columns if single ticker
        
    Example:
        >>> data = download_data("AAPL", period="1y")
        >>> data = download_data(["AAPL", "MSFT"], period="6mo")
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    
    # Handle custom date range
    if start_date or end_date:
        yf_tickers = yf.Tickers(" ".join(tickers)) if len(tickers) > 1 else yf.Ticker(tickers[0])
        
        if len(tickers) == 1:
            df = yf_tickers.history(
                period="max",
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=prepost,
                auto_adjust=auto_adjust,
                progress=progress
            )
        else:
            df = yf.download(
                tickers=" ".join(tickers),
                start=start_date,
                end=end_date,
                interval=interval,
                prepost=prepost,
                auto_adjust=auto_adjust,
                progress=progress,
                group_by="ticker"
            )
    else:
        df = yf.download(
            tickers=" ".join(tickers),
            period=period,
            interval=interval,
            prepost=prepost,
            auto_adjust=auto_adjust,
            progress=progress,
            group_by="ticker"
        )
    
    # Flatten multi-index columns if single ticker
    if isinstance(tickers, list) and len(tickers) == 1:
        ticker = tickers[0]
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]
        df = df.rename(columns=lambda x: f"{ticker}_{x}")
    elif isinstance(df.columns, pd.MultiIndex):
        df.columns = [f"{col[0]}_{col[1]}" if col[1] else col[0] for col in df.columns]
    
    return df


def download_single_ticker(
    ticker: str,
    period: str = DATA_CONFIG["period"],
    interval: str = DATA_CONFIG["interval"],
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> pd.DataFrame:
    """
    Download data for a single ticker.
    
    Args:
        ticker: Stock ticker symbol
        period: Data period
        interval: Data interval
        start_date: Start date (optional)
        end_date: End date (optional)
        
    Returns:
        DataFrame with OHLCV data
    """
    if start_date or end_date:
        data = yf.Ticker(ticker).history(
            period="max",
            start=start_date,
            end=end_date,
            interval=interval,
            auto_adjust=True
        )
    else:
        data = yf.Ticker(ticker).history(period=period, interval=interval)
    
    return data


def get_ticker_info(ticker: str) -> dict:
    """
    Get basic information about a ticker.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Dictionary with ticker information
    """
    info = yf.Ticker(ticker).info
    return {
        "symbol": info.get("symbol", ticker),
        "company_name": info.get("longName", info.get("shortName", "N/A")),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("marketCap", "N/A"),
        "pe_ratio": info.get("trailingPE", "N/A"),
        "fifty_two_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
        "fifty_two_week_low": info.get("fiftyTwoWeekLow", "N/A"),
    }


def save_data(df: pd.DataFrame, filename: str, data_dir: Path = DATA_DIR) -> Path:
    """
    Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        filename: Output filename
        data_dir: Directory to save data
        
    Returns:
        Path to saved file
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / filename
    
    # Handle multi-index columns
    if isinstance(df.columns, pd.MultiIndex):
        df.to_csv(filepath, header=True)
    else:
        df.to_csv(filepath)
    
    return filepath


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load DataFrame from CSV file.
    
    Args:
        filepath: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return df


def download_and_save(
    tickers: Union[str, List[str]],
    filename: str = "market_data.csv",
    **kwargs
) -> Tuple[pd.DataFrame, Path]:
    """
    Download data and save to file.
    
    Args:
        tickers: Ticker(s) to download
        filename: Output filename
        **kwargs: Additional arguments for download_data()
        
    Returns:
        Tuple of (DataFrame, file_path)
    """
    df = download_data(tickers, **kwargs)
    filepath = save_data(df, filename)
    return df, filepath


def get_available_tickers() -> List[str]:
    """
    Get list of commonly traded tickers for quick testing.
    
    Returns:
        List of ticker symbols
    """
    return [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META",
        "TSLA", "NVDA", "JPM", "V", "JNJ",
        "WMT", "PG", "MA", "HD", "DIS",
        "NFLX", "PYPL", "INTC", "CSCO", "ADBE"
    ]
