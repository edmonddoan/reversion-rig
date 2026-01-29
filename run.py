#!/usr/bin/env python3
"""
reversion-rig: Mean Reversion Trading Strategy

Example usage and demonstration script.
Run with: python run.py --ticker AAPL --period 1y
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

from src.data_fetcher import download_data, download_and_save, get_available_tickers
from src.strategy import MeanReversionStrategy, create_strategy
from src.backtester import Backtester, run_backtest
from src.utils import (
    setup_logger,
    clean_price_data,
    save_results,
    format_currency,
    format_percentage,
)


def setup_argument_parser():
    """Set up command line arguments."""
    parser = argparse.ArgumentParser(
        description="Mean Reversion Trading Strategy Backtester"
    )
    
    parser.add_argument(
        "--ticker", "-t",
        type=str,
        default="AAPL",
        help="Stock ticker symbol (default: AAPL)"
    )
    
    parser.add_argument(
        "--period", "-p",
        type=str,
        default="1y",
        help="Data period (default: 1y)"
    )
    
    parser.add_argument(
        "--capital", "-c",
        type=float,
        default=100000.0,
        help="Initial capital (default: $100,000)"
    )
    
    parser.add_argument(
        "--sma-period",
        type=int,
        default=20,
        help="SMA period (default: 20)"
    )
    
    parser.add_argument(
        "--bb-std",
        type=float,
        default=2.0,
        help="Bollinger Bands standard deviation multiplier (default: 2.0)"
    )
    
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=0.05,
        help="Stop loss percentage (default: 5%%)"
    )
    
    parser.add_argument(
        "--take-profit",
        type=float,
        default=0.10,
        help="Take profit percentage (default: 10%%)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser


def run_demo():
    """Run a demonstration of the mean reversion strategy."""
    logger = setup_logger()
    logger.info("Starting reversion-rig demo...")
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    print("\n" + "=" * 60)
    print("REVERSION-RIG: Mean Reversion Trading Strategy")
    print("=" * 60)
    print(f"\nTicker: {args.ticker}")
    print(f"Period: {args.period}")
    print(f"Initial Capital: {format_currency(args.capital)}")
    
    # Download data
    print(f"\n📥 Downloading data for {args.ticker}...")
    data = download_data(args.ticker, period=args.period)
    
    if data.empty:
        print(f"❌ No data found for {args.ticker}")
        return
    
    print(f"   Downloaded {len(data)} rows of data")
    print(f"   Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
    
    # Clean data
    data = clean_price_data(data)
    
    # Create strategy
    strategy = create_strategy(
        sma_period=args.sma_period,
        bb_period=args.sma_period,
        bb_std=args.bb_std,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        position_size_pct=0.20,
    )
    
    print(f"\n📊 Strategy Configuration:")
    print(f"   SMA Period: {args.sma_period}")
    print(f"   Bollinger Bands Std: {args.bb_std}")
    print(f"   Stop Loss: {args.stop_loss:.1%}")
    print(f"   Take Profit: {args.take_profit:.1%}")
    
    # Run backtest
    print(f"\n🔄 Running backtest...")
    backtester = Backtester(strategy)
    results = backtester.run(data, initial_capital=args.capital)
    
    # Display results
    metrics = backtester.metrics
    
    print(f"\n💰 FINAL RESULTS:")
    print(f"   Final Capital: {format_currency(results['final_capital'])}")
    print(f"   Total Return: {format_percentage(metrics.total_return)}")
    print(f"   Annual Return: {format_percentage(metrics.annual_return)}")
    
    print(f"\n📈 RISK METRICS:")
    print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
    print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
    print(f"   Max Drawdown: {format_percentage(metrics.max_drawdown)}")
    print(f"   Max DD Duration: {metrics.max_drawdown_duration} days")
    
    print(f"\n💹 TRADING STATS:")
    print(f"   Total Trades: {metrics.num_trades}")
    print(f"   Win Rate: {format_percentage(metrics.win_rate)}")
    print(f"   Profit Factor: {metrics.profit_factor:.2f}")
    print(f"   Avg Trade: {format_percentage(metrics.avg_trade_return)}")
    print(f"   Expectancy: {metrics.expectancy:.4f}")
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    # Save portfolio values
    portfolio_file = output_dir / f"{args.ticker}_portfolio.csv"
    results['portfolio_values'].to_csv(portfolio_file)
    print(f"\n💾 Results saved to {portfolio_file}")
    
    # Generate report
    report_file = output_dir / f"{args.ticker}_report.txt"
    backtester.generate_report(output_dir=output_dir, filename=f"{args.ticker}_report.txt")
    print(f"📄 Report saved to {report_file}")
    
    # Save strategy parameters
    params_file = output_dir / f"{args.ticker}_params.json"
    import json
    with open(params_file, 'w') as f:
        json.dump(strategy.get_parameters(), f, indent=2)
    print(f"⚙️  Parameters saved to {params_file}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    print("=" * 60)


def run_batch_demo():
    """Run demo on multiple tickers."""
    logger = setup_logger()
    
    tickers = ["AAPL", "MSFT", "GOOGL"]
    period = "6mo"
    initial_capital = 50000
    
    print("\n" + "=" * 60)
    print("BATCH BACKTEST: Multiple Tickers")
    print("=" * 60)
    
    all_results = []
    
    for ticker in tickers:
        print(f"\n📊 Processing {ticker}...")
        
        try:
            data = download_data(ticker, period=period)
            if data.empty:
                print(f"   ❌ No data for {ticker}")
                continue
            
            strategy = MeanReversionStrategy()
            results = run_backtest(data, strategy, initial_capital, verbose=False)
            
            result_summary = {
                "ticker": ticker,
                "final_capital": results['final_capital'],
                "total_return": results['metrics']['total_return'],
                "sharpe_ratio": results['metrics']['sharpe_ratio'],
                "max_drawdown": results['metrics']['max_drawdown'],
                "num_trades": results['metrics']['num_trades'],
                "win_rate": results['metrics']['win_rate'],
            }
            all_results.append(result_summary)
            
            print(f"   ✅ Return: {format_percentage(result_summary['total_return'])} | "
                  f"Sharpe: {result_summary['sharpe_ratio']:.2f} | "
                  f"Win Rate: {format_percentage(result_summary['win_rate'])}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    if all_results:
        print("\n" + "-" * 60)
        print("SUMMARY:")
        print("-" * 60)
        
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('total_return', ascending=False)
        
        print(results_df.to_string(index=False))
        
        # Save batch results
        results_df.to_csv("batch_results.csv", index=False)
        print(f"\n💾 Batch results saved to batch_results.csv")


def quick_test():
    """Quick test with synthetic data."""
    print("\n" + "=" * 60)
    print("QUICK TEST: Synthetic Mean-Reverting Data")
    print("=" * 60)
    
    # Create synthetic mean-reverting data
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
    
    # Oscillating data (mean reversion pattern)
    t = np.linspace(0, 4 * np.pi, 252)
    base = 100 + 10 * np.sin(t)
    noise = np.random.randn(252) * 2
    prices = base + noise
    
    data = pd.DataFrame({'Close': prices}, index=dates)
    
    print(f"\n📊 Created {len(data)} days of synthetic data")
    
    # Run backtest
    strategy = MeanReversionStrategy(
        sma_period=20,
        bb_std=2.0,
        stop_loss_pct=0.05,
        take_profit_pct=0.10,
    )
    
    results = run_backtest(data, strategy, initial_capital=100000)
    
    print(f"\n💰 Final Capital: {format_currency(results['final_capital'])}")
    print(f"📈 Total Return: {format_percentage(results['metrics']['total_return'])}")
    print(f"📉 Max Drawdown: {format_percentage(results['metrics']['max_drawdown'])}")
    print(f"🎯 Win Rate: {format_percentage(results['metrics']['win_rate'])}")


if __name__ == "__main__":
    # Check for command line arguments
    import sys
    
    if len(sys.argv) > 1:
        run_demo()
    else:
        print("\nreversion-rig: Mean Reversion Trading Strategy")
        print("\nUsage examples:")
        print("  python run.py --ticker AAPL --period 1y")
        print("  python run.py -t MSFT -p 6mo -c 50000 --verbose")
        print("  python run.py --help")
        print("\nAlso available:")
        print("  - Quick test with synthetic data: python run.py --test")
        print("  - Batch backtest: python run.py --batch")
        print()
        
        # Run quick test by default
        quick_test()
