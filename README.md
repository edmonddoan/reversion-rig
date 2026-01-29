# reversion-rig

> Mean Reversion Trading Strategy with Backtesting

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

## Overview

reversion-rig is a quantitative trading strategy implementation focused on **mean reversion** - the principle that asset prices tend to return to their historical average over time. This project provides a complete framework for:

- Downloading historical market data
- Calculating technical indicators (SMA, EMA, Bollinger Bands, RSI, Z-Score)
- Implementing mean reversion trading logic
- Backtesting strategies on historical data
- Analyzing performance metrics

## Key Features

- **Technical Indicators**: SMA, EMA, Bollinger Bands, RSI, Z-Score
- **Mean Reversion Strategy**: Buy oversold, sell overbought
- **Risk Management**: Stop-loss and take-profit levels
- **Backtesting Engine**: Comprehensive performance analytics
- **Configurable Parameters**: Fine-tune strategy behavior

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/reversion-rig.git
cd reversion-rig

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from src.data_fetcher import download_data
from src.strategy import MeanReversionStrategy
from src.backtester import run_backtest

# Download historical data
data = download_data("AAPL", period="1y")

# Create and configure strategy
strategy = MeanReversionStrategy(
    sma_period=20,
    bb_std=2.0,
    stop_loss_pct=0.05,
    take_profit_pct=0.10,
    position_size_pct=0.20
)

# Run backtest
results = run_backtest(data, strategy, initial_capital=100000)

# View results
print(f"Total Return: {results['metrics']['total_return']:.2%}")
print(f"Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")
print(f"Win Rate: {results['metrics']['win_rate']:.2%}")
```

## Project Structure

```
reversion-rig/
├── src/
│   ├── __init__.py          # Package initialization
│   ├── indicators.py        # Technical indicators
│   ├── data_fetcher.py      # Market data retrieval (yfinance)
│   ├── strategy.py          # Mean reversion strategy
│   ├── backtester.py        # Backtesting framework
│   └── utils.py             # Utility functions
├── tests/
│   ├── __init__.py
│   ├── test_indicators.py   # Indicator unit tests
│   └── test_backtester.py   # Backtester unit tests
├── config/
│   ├── __init__.py
│   └── settings.py          # Configuration parameters
├── data/                     # Market data storage
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── run.py                    # CLI entry point
```

## Strategy Logic

### Mean Reversion Principles

The strategy identifies trading opportunities when prices deviate significantly from their moving average:

```
Entry Signal (BUY):
- Price below lower Bollinger Band, OR
- RSI < 30 (oversold), OR
- Z-Score < -2.0

Exit Signal (SELL):
- Price reaches upper Bollinger Band, OR
- RSI > 70 (overbought), OR
- Stop-loss or take-profit triggered
```

### Bollinger Bands

```
Upper Band = SMA + (K × σ)
Lower Band = SMA - (K × σ)

Where:
- SMA = Simple Moving Average
- σ = Standard deviation
- K = Multiplier (default: 2.0)
```

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `sma_period` | 20 | Period for moving average |
| `bb_std` | 2.0 | Standard deviation multiplier |
| `rsi_period` | 14 | RSI calculation period |
| `rsi_oversold` | 30 | Oversold threshold |
| `rsi_overbought` | 70 | Overbought threshold |
| `stop_loss_pct` | 0.05 | Stop loss percentage |
| `take_profit_pct` | 0.10 | Take profit percentage |
| `position_size_pct` | 0.20 | Position size (% of capital) |

## Usage

### Command Line

```bash
# Run with default settings
python run.py --ticker AAPL

# Custom parameters
python run.py -t MSFT -p 1y -c 50000 --sma-period 30 --bb-std 2.5

# Verbose output
python run.py -t GOOGL --verbose

# Run quick test with synthetic data
python run.py --test

# Batch backtest on multiple tickers
python run.py --batch
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_indicators.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| **Total Return** | Overall percentage return |
| **Annual Return** | Annualized return |
| **Sharpe Ratio** | Risk-adjusted return (risk-free rate: 2%) |
| **Sortino Ratio** | Downside risk-adjusted return |
| **Max Drawdown** | Largest peak-to-trough decline |
| **Win Rate** | Percentage of winning trades |
| **Profit Factor** | Gross profit / Gross loss |
| **Expectancy** | Average expected return per trade |

## Example Output

```
============================================================
REVERSION-RIG: Mean Reversion Trading Strategy
============================================================

Ticker: AAPL
Period: 1y
Initial Capital: $100,000.00

📥 Downloading data for AAPL...
   Downloaded 252 rows of data
   Date range: 2023-01-01 to 2023-12-31

📊 Strategy Configuration:
   SMA Period: 20
   Bollinger Bands Std: 2.0
   Stop Loss: 5.0%
   Take Profit: 10.0%

🔄 Running backtest...

============================================================
BACKTEST RESULTS SUMMARY
============================================================

📊 RETURNS:
   Total Return:       15.23%
   Annual Return:      14.87%

📈 RISK METRICS:
   Volatility:         18.52%
   Sharpe Ratio:       0.72
   Sortino Ratio:      0.95
   Max Drawdown:       -8.32%
   Max DD Duration:    45 days

💰 TRADING STATS:
   Number of Trades:   24
   Win Rate:           58.33%
   Profit Factor:      1.45
   Avg Trade Return:   0.64%
   Avg Trade Duration: 8.5 days
   Expectancy:         0.0032
   Kelly Fraction:     12.5%

✅ Demo completed successfully!
```

## Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **scipy**: Statistical functions
- **yfinance**: Market data retrieval
- **matplotlib**: Plotting
- **plotly**: Interactive charts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -m 'feat: add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details.

---

**Happy Trading! 📈**
