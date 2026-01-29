"""
Backtesting Framework for Mean Reversion Strategy.

This module provides a comprehensive backtesting framework to evaluate
the performance of the mean reversion trading strategy with various metrics
and visualization capabilities.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .strategy import MeanReversionStrategy, Trade, Signal, Position
from .indicators import calculate_all_indicators
from .config import DEFAULT_PARAMS, VISUALIZATION_CONFIG


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtest results."""
    total_return: float
    annual_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int  # Days
    num_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_trade_duration: float
    expectancy: float
    Kelly_fraction: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "volatility": self.volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "max_drawdown_duration": self.max_drawdown_duration,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "avg_trade_duration": self.avg_trade_duration,
            "expectancy": self.expectancy,
            "Kelly_fraction": self.Kelly_fraction,
        }


class Backtester:
    """
    Backtesting framework for trading strategies.
    
    This class provides methods to:
    - Run backtests with configurable parameters
    - Calculate comprehensive performance metrics
    - Generate performance reports
    - Visualize results
    """
    
    def __init__(
        self,
        strategy: Optional[MeanReversionStrategy] = None,
        risk_free_rate: float = DEFAULT_PARAMS["risk_free_rate"],
    ):
        """
        Initialize the backtester.
        
        Args:
            strategy: MeanReversionStrategy instance (creates default if None)
            risk_free_rate: Annual risk-free rate for Sharpe ratio
        """
        self.strategy = strategy or MeanReversionStrategy()
        self.risk_free_rate = risk_free_rate
        self.results: Optional[Dict[str, Any]] = None
        self.metrics: Optional[PerformanceMetrics] = None
    
    def run(
        self,
        df: pd.DataFrame,
        initial_capital: float = DEFAULT_PARAMS["initial_capital"],
        price_col: str = "Close",
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Run a backtest on historical data.
        
        Args:
            df: DataFrame with price data
            initial_capital: Starting capital
            price_col: Name of the price column
            verbose: Print progress messages
            
        Returns:
            Dictionary with backtest results
        """
        if verbose:
            print(f"Running backtest...")
            print(f"  Initial Capital: ${initial_capital:,.2f}")
            print(f"  Strategy: {self.strategy}")
            print(f"  Data: {len(df)} rows, {df.index[0]} to {df.index[-1]}")
        
        # Run strategy
        self.results = self.strategy.run_strategy(
            df=df,
            initial_capital=initial_capital,
            price_col=price_col
        )
        
        # Add metadata
        self.results["strategy_params"] = self.strategy.get_parameters()
        self.results["backtest_period"] = {
            "start": str(df.index[0]),
            "end": str(df.index[-1]),
            "days": (df.index[-1] - df.index[0]).days,
        }
        
        # Calculate metrics
        self.metrics = self.calculate_metrics(initial_capital)
        
        if verbose:
            self.print_summary()
        
        return self.results
    
    def calculate_metrics(
        self,
        initial_capital: float,
        periods_per_year: int = 252
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            initial_capital: Starting capital
            periods_per_year: Trading periods per year
            
        Returns:
            PerformanceMetrics object
        """
        if not self.results:
            raise ValueError("No results available. Run backtest first.")
        
        portfolio_values = self.results["portfolio_values"]
        returns = self.results["returns"]
        trades = self.results["trades"]
        
        # Basic metrics
        total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
        
        # Annualized metrics
        days = (portfolio_values.index[-1] - portfolio_values.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        daily_rf = self.risk_free_rate / periods_per_year
        excess_returns = returns - daily_rf
        sharpe_ratio = (excess_returns.mean() / returns.std()) * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino_ratio = (excess_returns.mean() / downside_std) if downside_std > 0 else float('inf')
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Max Drawdown Duration
        dd_duration = 0
        max_dd_dur = 0
        in_drawdown = False
        for val in drawdown:
            if val < 0:
                if not in_drawdown:
                    in_drawdown = True
                    dd_duration = 1
                else:
                    dd_duration += 1
                max_dd_dur = max(max_dd_dur, dd_duration)
            else:
                in_drawdown = False
        
        max_drawdown_duration = max_dd_dur
        
        # Trading metrics
        num_trades = len(trades)
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl < 0]
        
        win_rate = len(winning_trades) / num_trades if num_trades > 0 else 0
        
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_trade_return = np.mean([t.pnl_pct for t in trades]) if trades else 0
        avg_trade_duration = np.mean([t.holding_period for t in trades]) if trades else 0
        
        # Expectancy
        avg_win_pct = np.mean([t.pnl_pct for t in winning_trades]) if winning_trades else 0
        avg_loss_pct = np.mean([t.pnl_pct for t in losing_trades]) if losing_trades else 0
        expectancy = (win_rate * avg_win_pct) - ((1 - win_rate) * abs(avg_loss_pct))
        
        # Kelly Criterion
        Kelly_fraction = win_rate - ((1 - win_rate) / (avg_win_pct / abs(avg_loss_pct))) if avg_loss_pct != 0 else 0
        
        return PerformanceMetrics(
            total_return=total_return,
            annual_return=annual_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            max_drawdown_duration=max_drawdown_duration,
            num_trades=num_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            avg_trade_return=avg_trade_return,
            avg_trade_duration=avg_trade_duration,
            expectancy=expectancy,
            Kelly_fraction=Kelly_fraction,
        )
    
    def print_summary(self) -> None:
        """Print a formatted summary of backtest results."""
        if not self.metrics:
            raise ValueError("No metrics available. Run backtest first.")
        
        m = self.metrics
        
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS SUMMARY")
        print("=" * 60)
        
        print("\n📊 RETURNS:")
        print(f"   Total Return:       {m.total_return:>10.2%}")
        print(f"   Annual Return:      {m.annual_return:>10.2%}")
        
        print("\n📈 RISK METRICS:")
        print(f"   Volatility:         {m.volatility:>10.2%}")
        print(f"   Sharpe Ratio:       {m.sharpe_ratio:>10.2f}")
        print(f"   Sortino Ratio:      {m.sortino_ratio:>10.2f}")
        print(f"   Max Drawdown:       {m.max_drawdown:>10.2%}")
        print(f"   Max DD Duration:    {m.max_drawdown_duration:>10} days")
        
        print("\n💰 TRADING STATS:")
        print(f"   Number of Trades:   {m.num_trades:>10}")
        print(f"   Win Rate:           {m.win_rate:>10.2%}")
        print(f"   Profit Factor:      {m.profit_factor:>10.2f}")
        print(f"   Avg Trade Return:   {m.avg_trade_return:>10.2%}")
        print(f"   Avg Trade Duration: {m.avg_trade_duration:>10.1f} days")
        print(f"   Expectancy:         {m.expectancy:>10.4f}")
        print(f"   Kelly Fraction:     {m.Kelly_fraction:>10.2%}")
        
        print("\n" + "=" * 60)
    
    def generate_report(
        self,
        output_dir: Optional[Path] = None,
        filename: str = "backtest_report.txt"
    ) -> Path:
        """
        Generate a text report of backtest results.
        
        Args:
            output_dir: Output directory for report
            filename: Report filename
            
        Returns:
            Path to saved report
        """
        if not self.metrics or not self.results:
            raise ValueError("No results available. Run backtest first.")
        
        output_dir = output_dir or Path(".")
        output_dir.mkdir(parents=True, exist_ok=True)
        filepath = output_dir / filename
        
        m = self.metrics
        period = self.results["backtest_period"]
        
        report = f"""
================================================================================
                    MEAN REVERSION STRATEGY - BACKTEST REPORT
================================================================================

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
                              BACKTEST PERIOD
--------------------------------------------------------------------------------
Start Date:          {period['start']}
End Date:            {period['end']}
Duration:            {period['days']} days

--------------------------------------------------------------------------------
                              STRATEGY PARAMETERS
--------------------------------------------------------------------------------
{self._format_params(self.results.get('strategy_params', {}))}

--------------------------------------------------------------------------------
                              RETURNS
--------------------------------------------------------------------------------
Total Return:        {m.total_return:>12.2%}
Annual Return:       {m.annual_return:>12.2%}

--------------------------------------------------------------------------------
                              RISK METRICS
--------------------------------------------------------------------------------
Volatility:          {m.volatility:>12.2%}
Sharpe Ratio:        {m.sharpe_ratio:>12.2f}
Sortino Ratio:       {m.sortino_ratio:>12.2f}
Max Drawdown:        {m.max_drawdown:>12.2%}
Max DD Duration:     {m.max_drawdown_duration:>12} days

--------------------------------------------------------------------------------
                              TRADING STATISTICS
--------------------------------------------------------------------------------
Number of Trades:    {m.num_trades:>12}
Win Rate:            {m.win_rate:>12.2%}
Profit Factor:       {m.profit_factor:>12.2f}
Avg Trade Return:    {m.avg_trade_return:>12.2%}
Avg Trade Duration:  {m.avg_trade_duration:>12.1f} days
Expectancy:          {m.expectancy:>12.4f}
Kelly Fraction:      {m.Kelly_fraction:>12.2%}

================================================================================
"""
        # Add trades list
        if self.results.get('trades'):
            report += "\n                              TRADES LIST\n"
            report += "-" * 80 + "\n"
            for i, trade in enumerate(self.results['trades'], 1):
                status = "WIN" if trade.pnl > 0 else "LOSS"
                report += f"{i:3}. {trade.entry_date.strftime('%Y-%m-%d')} → {trade.exit_date.strftime('%Y-%m-%d')} | "
                report += f"{status:4} | PnL: ${trade.pnl:>10.2f} ({trade.pnl_pct:>6.2%}) | {trade.holding_period} days\n"
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        return filepath
    
    def _format_params(self, params: Dict[str, Any]) -> str:
        """Format parameters for report."""
        lines = []
        for key, value in params.items():
            if isinstance(value, float):
                lines.append(f"{key.replace('_', ' ').title():<25} {value:>12.2f}")
            elif isinstance(value, bool):
                lines.append(f"{key.replace('_', ' ').title():<25} {'Yes' if value else 'No':>12}")
            else:
                lines.append(f"{key.replace('_', ' ').title():<25} {str(value):>12}")
        return "\n".join(lines)
    
    def optimize_parameters(
        self,
        df: pd.DataFrame,
        param_grid: Dict[str, List[Any]],
        initial_capital: float = DEFAULT_PARAMS["initial_capital"],
        metric: str = "sharpe_ratio"
    ) -> pd.DataFrame:
        """
        Optimize strategy parameters using grid search.
        
        Args:
            df: DataFrame with price data
            param_grid: Dictionary of parameter names to list of values
            initial_capital: Starting capital
            metric: Metric to optimize (sharpe_ratio, total_return, etc.)
            
        Returns:
            DataFrame with optimization results
        """
        from itertools import product
        
        results = []
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        
        total_combinations = 1
        for values in param_values:
            total_combinations *= len(values)
        
        print(f"Running optimization with {total_combinations} parameter combinations...")
        
        for i, combination in enumerate(product(*param_values), 1):
            # Create strategy with current parameters
            params = dict(zip(param_names, combination))
            strategy = MeanReversionStrategy(**params)
            
            # Run backtest
            backtester = Backtester(strategy)
            backtester.run(df, initial_capital, verbose=False)
            
            # Get metric value
            metric_value = getattr(backtester.metrics, metric, 0)
            
            result = {
                "sharpe_ratio": backtester.metrics.sharpe_ratio,
                "total_return": backtester.metrics.total_return,
                "max_drawdown": backtester.metrics.max_drawdown,
                "num_trades": backtester.metrics.num_trades,
                "win_rate": backtester.metrics.win_rate,
                **params
            }
            results.append(result)
            
            if i % 10 == 0 or i == total_combinations:
                print(f"Progress: {i}/{total_combinations} ({100*i/total_combinations:.1f}%)")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by optimization metric
        results_df = results_df.sort_values(by=metric, ascending=False)
        
        print(f"\nTop 5 parameter sets by {metric}:")
        print(results_df.head().to_string(index=False))
        
        return results_df


def run_backtest(
    df: pd.DataFrame,
    strategy: Optional[MeanReversionStrategy] = None,
    initial_capital: float = DEFAULT_PARAMS["initial_capital"],
    risk_free_rate: float = DEFAULT_PARAMS["risk_free_rate"],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to run a backtest.
    
    Args:
        df: DataFrame with price data
        strategy: MeanReversionStrategy instance
        initial_capital: Starting capital
        risk_free_rate: Annual risk-free rate
        verbose: Print progress messages
        
    Returns:
        Dictionary with backtest results and metrics
    """
    backtester = Backtester(strategy, risk_free_rate)
    results = backtester.run(df, initial_capital, verbose=verbose)
    results["metrics"] = backtester.metrics.to_dict()
    return results
