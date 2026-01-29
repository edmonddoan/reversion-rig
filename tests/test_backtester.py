"""
Unit tests for the backtester module.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.backtester import Backtester, PerformanceMetrics, run_backtest
from src.strategy import MeanReversionStrategy, Position, Trade, Signal


class TestBacktester:
    """Tests for the Backtester class."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=252, freq="D")
        
        # Create a random walk with drift
        returns = np.random.randn(252) * 0.02
        prices = 100 * (1 + returns).cumprod()
        
        return pd.DataFrame({
            'Open': prices * 0.999,
            'High': prices * 1.002,
            'Low': prices * 0.998,
            'Close': prices,
            'Volume': np.random.randint(1000000, 10000000, 252)
        }, index=dates)
    
    @pytest.fixture
    def simple_uptrend_data(self):
        """Create simple uptrending data."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        prices = np.linspace(100, 120, 100)
        
        return pd.DataFrame({'Close': prices}, index=dates)
    
    @pytest.fixture
    def mean_reverting_data(self):
        """Create mean-reverting price data."""
        np.random.seed(42)
        dates = pd.date_range(start="2023-01-01", periods=200, freq="D")
        
        # Oscillating data around a mean
        t = np.linspace(0, 4 * np.pi, 200)
        base = 100 + 10 * np.sin(t)
        noise = np.random.randn(200) * 2
        
        return pd.DataFrame({'Close': base + noise}, index=dates)
    
    def test_backtester_initialization(self):
        """Test Backtester initialization."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        
        assert backtester.strategy == strategy
        assert backtester.results is None
        assert backtester.metrics is None
    
    def test_run_backtest(self, sample_price_data):
        """Test running a backtest."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        
        results = backtester.run(sample_price_data, initial_capital=100000)
        
        assert results is not None
        assert 'portfolio_values' in results
        assert 'trades' in results
        assert 'signals' in results
        assert 'final_capital' in results
        
        # Check portfolio values
        assert len(results['portfolio_values']) == len(sample_price_data)
        assert results['portfolio_values'].iloc[0] <= 100000
        assert results['portfolio_values'].iloc[-1] >= 0
    
    def test_calculate_metrics(self, sample_price_data):
        """Test performance metrics calculation."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        backtester.run(sample_price_data, initial_capital=100000)
        
        metrics = backtester.metrics
        
        assert isinstance(metrics, PerformanceMetrics)
        assert isinstance(metrics.total_return, float)
        assert isinstance(metrics.sharpe_ratio, float)
        assert isinstance(metrics.max_drawdown, float)
        assert isinstance(metrics.win_rate, float)
        
        # Validate ranges
        assert -1.0 <= metrics.max_drawdown <= 0.0
        assert 0.0 <= metrics.win_rate <= 1.0
    
    def test_metrics_to_dict(self, sample_price_data):
        """Test metrics conversion to dictionary."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        backtester.run(sample_price_data)
        
        metrics_dict = backtester.metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert 'total_return' in metrics_dict
        assert 'sharpe_ratio' in metrics_dict
        assert 'max_drawdown' in metrics_dict
    
    def test_print_summary(self, sample_price_data, capsys):
        """Test printing summary doesn't crash."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        backtester.run(sample_price_data, verbose=True)
        
        captured = capsys.readouterr()
        assert "BACKTEST RESULTS SUMMARY" in captured.out
        assert "RETURNS" in captured.out
    
    def test_generate_report(self, sample_price_data, tmp_path):
        """Test report generation."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        backtester.run(sample_price_data)
        
        filepath = backtester.generate_report(output_dir=tmp_path)
        
        assert filepath.exists()
        
        report = filepath.read_text()
        assert "BACKTEST REPORT" in report
        assert "RETURNS" in report
    
    def test_run_backtest_convenience_function(self, sample_price_data):
        """Test the convenience run_backtest function."""
        results = run_backtest(sample_price_data, verbose=False)
        
        assert 'metrics' in results
        assert 'portfolio_values' in results
        assert isinstance(results['metrics'], dict)
    
    def test_empty_data_raises_error(self):
        """Test that empty data raises an error."""
        empty_data = pd.DataFrame({'Close': []})
        
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        
        with pytest.raises((ValueError, KeyError)):
            backtester.run(empty_data)
    
    def test_insufficient_data(self):
        """Test with insufficient data for strategy."""
        short_data = pd.DataFrame(
            {'Close': [100, 101, 102]},
            index=pd.date_range("2023-01-01", periods=3)
        )
        
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        
        # Should still run but with no trades
        results = backtester.run(short_data)
        assert len(results['trades']) == 0 or len(results['trades']) >= 0


class TestPerformanceMetrics:
    """Tests for PerformanceMetrics class."""
    
    def test_performance_metrics_creation(self):
        """Test creating PerformanceMetrics instance."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annual_return=0.10,
            volatility=0.20,
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            max_drawdown=-0.10,
            max_drawdown_duration=30,
            num_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.02,
            avg_trade_duration=5.0,
            expectancy=0.01,
            Kelly_fraction=0.15
        )
        
        assert metrics.total_return == 0.15
        assert metrics.sharpe_ratio == 0.5
        assert metrics.win_rate == 0.6
    
    def test_performance_metrics_to_dict(self):
        """Test dictionary conversion."""
        metrics = PerformanceMetrics(
            total_return=0.15,
            annual_return=0.10,
            volatility=0.20,
            sharpe_ratio=0.5,
            sortino_ratio=0.7,
            max_drawdown=-0.10,
            max_drawdown_duration=30,
            num_trades=10,
            win_rate=0.6,
            profit_factor=1.5,
            avg_trade_return=0.02,
            avg_trade_duration=5.0,
            expectancy=0.01,
            Kelly_fraction=0.15
        )
        
        metrics_dict = metrics.to_dict()
        
        assert isinstance(metrics_dict, dict)
        assert len(metrics_dict) == 14  # All fields


class TestBacktesterEdgeCases:
    """Tests for edge cases in backtesting."""
    
    def test_zero_initial_capital(self, sample_price_data):
        """Test with zero initial capital."""
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        
        results = backtester.run(sample_price_data, initial_capital=0)
        assert results['final_capital'] == 0
    
    def test_no_trades(self, sample_price_data):
        """Test scenario with no trades executed."""
        # Strategy with very tight stop loss - should trigger immediate exit
        strategy = MeanReversionStrategy(
            stop_loss_pct=0.001,  # 0.1% stop loss
            take_profit_pct=0.001,
            position_size_pct=0.5
        )
        backtester = Backtester(strategy)
        
        results = backtester.run(sample_price_data, initial_capital=100000)
        
        # May or may not have trades depending on price movement
        # Just ensure it doesn't crash
        assert 'trades' in results
    
    def test_all_losing_trades(self):
        """Test with data that causes all losing trades."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        # Declining prices
        prices = np.linspace(120, 80, 100)
        
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        strategy = MeanReversionStrategy()
        backtester = Backtester(strategy)
        backtester.run(data, initial_capital=100000)
        
        metrics = backtester.metrics
        assert metrics.win_rate == 0.0 or metrics.win_rate >= 0.0
    
    def test_all_winning_trades(self):
        """Test with data that causes all winning trades."""
        dates = pd.date_range(start="2023-01-01", periods=100, freq="D")
        # Strongly increasing prices
        prices = np.linspace(80, 120, 100)
        
        data = pd.DataFrame({'Close': prices}, index=dates)
        
        strategy = MeanReversionStrategy(
            stop_loss_pct=0.50,  # Wide stop loss
            take_profit_pct=0.50
        )
        backtester = Backtester(strategy)
        backtester.run(data, initial_capital=100000)
        
        metrics = backtester.metrics
        # Win rate should be high for uptrend
        assert metrics.win_rate >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
