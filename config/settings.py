"""
Configuration settings for reversion-rig strategy.
"""

from pathlib import Path
from typing import Dict, Any

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Default trading parameters
DEFAULT_PARAMS: Dict[str, Any] = {
    # Strategy parameters
    "sma_period": 20,
    "std_multiplier": 2.0,  # Bollinger Bands multiplier
    "rsi_period": 14,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "zscore_window": 20,
    
    # Risk management
    "stop_loss_pct": 0.05,  # 5% stop loss
    "take_profit_pct": 0.10,  # 10% take profit
    "position_size_pct": 0.20,  # 20% of capital per trade
    
    # Backtest settings
    "initial_capital": 100000.0,
    "risk_free_rate": 0.02,  # 2% annual risk-free rate
}

# Data fetch settings
DATA_CONFIG = {
    "period": "1y",  # 1 year of data
    "interval": "1d",  # Daily data
    "prepost": False,
    "auto_adjust": True,
}

# Supported tickers
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    "TSLA", "NVDA", "JPM", "V", "JNJ"
]

# Visualization settings
VISUALIZATION_CONFIG = {
    "figsize": (14, 8),
    "style": "plotly_dark",
    "save_plots": True,
    "plot_dir": RESULTS_DIR / "plots",
}
