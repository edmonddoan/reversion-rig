"""
reversion-rig: Mean Reversion Trading Strategy Package.

A comprehensive mean reversion trading strategy implementation with
backtesting capabilities, technical indicators, and performance analysis.
"""

__version__ = "1.0.0"
__author__ = "reversion-rig"

from .indicators import (
    sma,
    ema,
    bollinger_bands,
    rsi,
    zscore,
    calculate_all_indicators,
)

from .strategy import (
    MeanReversionStrategy,
    Trade,
    Signal,
    Position,
    create_strategy,
)

from .backtester import (
    Backtester,
    PerformanceMetrics,
    run_backtest,
)

from .data_fetcher import (
    download_data,
    download_single_ticker,
    get_ticker_info,
    save_data,
    load_data,
    download_and_save,
    get_available_tickers,
)

from .utils import (
    setup_logger,
    validate_dataframe,
    clean_price_data,
    calculate_returns,
    format_currency,
    format_percentage,
    save_results,
    load_results,
)

__all__ = [
    # Version
    "__version__",
    
    # Indicators
    "sma",
    "ema",
    "bollinger_bands",
    "rsi",
    "zscore",
    "calculate_all_indicators",
    
    # Strategy
    "MeanReversionStrategy",
    "Trade",
    "Signal",
    "Position",
    "create_strategy",
    
    # Backtester
    "Backtester",
    "PerformanceMetrics",
    "run_backtest",
    
    # Data Fetcher
    "download_data",
    "download_single_ticker",
    "get_ticker_info",
    "save_data",
    "load_data",
    "download_and_save",
    "get_available_tickers",
    
    # Utils
    "setup_logger",
    "validate_dataframe",
    "clean_price_data",
    "calculate_returns",
    "format_currency",
    "format_percentage",
    "save_results",
    "load_results",
]
