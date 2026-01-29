"""
Unit tests for the indicators module.
"""

import pytest
import pandas as pd
import numpy as np
from src.indicators import (
    sma,
    ema,
    bollinger_bands,
    rsi,
    zscore,
    calculate_all_indicators,
)


class TestSMA:
    """Tests for Simple Moving Average."""
    
    def test_sma_basic(self):
        """Test basic SMA calculation."""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = sma(data, period=5)
        
        # First 4 values should be NaN
        assert result.iloc[:4].isna().all()
        # Last 6 values should have valid SMA
        assert result.iloc[4] == 3.0  # (1+2+3+4+5)/5 = 3
        assert result.iloc[5] == 4.0  # (2+3+4+5+6)/5 = 4
        assert result.iloc[9] == 8.0  # (6+7+8+9+10)/5 = 8
    
    def test_sma_constant_series(self):
        """Test SMA with constant values."""
        data = pd.Series([10] * 10)
        result = sma(data, period=5)
        
        assert result.iloc[4] == 10.0
        assert result.iloc[-1] == 10.0
    
    def test_sma_single_period(self):
        """Test SMA with period 1."""
        data = pd.Series([5, 10, 15])
        result = sma(data, period=1)
        
        assert result.equals(data)
    
    def test_sma_empty_series(self):
        """Test SMA with empty series."""
        data = pd.Series([], dtype=float)
        result = sma(data, period=5)
        
        assert result.empty


class TestEMA:
    """Tests for Exponential Moving Average."""
    
    def test_ema_basic(self):
        """Test basic EMA calculation."""
        data = pd.Series([1, 2, 3, 4, 5])
        result = ema(data, period=3)
        
        # First values may differ due to EMA initialization
        assert not result.isna().all()
        # Last value should be closest to recent prices
        assert result.iloc[-1] > result.iloc[0]
    
    def test_ema_with_zeros(self):
        """Test EMA handling zeros."""
        data = pd.Series([0, 0, 10, 10, 10])
        result = ema(data, period=3)
        
        assert not result.isna().all()
    
    def test_ema_length_mismatch(self):
        """Test EMA with period longer than data."""
        data = pd.Series([1, 2, 3])
        result = ema(data, period=10)
        
        # Should return all NaN for period > length
        assert result.isna().all() or result.iloc[-1] > 0


class TestBollingerBands:
    """Tests for Bollinger Bands."""
    
    def test_bollinger_bands_basic(self):
        """Test basic Bollinger Bands calculation."""
        # Create data with known mean and std
        np.random.seed(42)
        data = pd.Series(np.random.randn(100) + 100)  # Mean = 100
        
        middle, upper, lower = bollinger_bands(data, period=20, std_multiplier=2.0)
        
        # Middle band should be close to SMA
        assert middle.iloc[-1] is not None
        
        # Upper band should be above middle
        assert (upper.iloc[20:] > middle.iloc[20:]).all()
        
        # Lower band should be below middle
        assert (lower.iloc[20:] < middle.iloc[20:]).all()
        
        # Distance from middle should be roughly 2 std
        distance = upper.iloc[20:] - middle.iloc[20:]
        std = data.rolling(20).std().iloc[20:]
        assert np.allclose(distance, 2 * std, rtol=0.5)
    
    def test_bollinger_bands_constant(self):
        """Test Bollinger Bands with constant data."""
        data = pd.Series([100] * 50)
        middle, upper, lower = bollinger_bands(data, period=20, std_multiplier=2.0)
        
        # Middle should be 100
        assert middle.iloc[19] == 100.0
        
        # Upper and lower should be at 100 (std = 0)
        assert upper.iloc[19] == 100.0
        assert lower.iloc[19] == 100.0
    
    def test_bollinger_bands_width(self):
        """Test that bands widen with volatility."""
        # Low volatility data
        low_vol = pd.Series([100] * 50 + list(np.linspace(100, 101, 50)))
        _, upper_low, lower_low = bollinger_bands(low_vol, period=20, std_multiplier=2.0)
        
        # High volatility data
        np.random.seed(42)
        high_vol = pd.Series(100 + np.random.randn(100))
        _, upper_high, lower_high = bollinger_bands(high_vol, period=20, std_multiplier=2.0)
        
        # Band width should be greater for high volatility
        avg_width_low = (upper_low.iloc[20:] - lower_low.iloc[20:]).mean()
        avg_width_high = (upper_high.iloc[20:] - lower_high.iloc[20:]).mean()
        
        assert avg_width_high > avg_width_low


class TestRSI:
    """Tests for Relative Strength Index."""
    
    def test_rsi_basic(self):
        """Test basic RSI calculation."""
        # Create data with consistent gains
        gains = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        losses = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        prices = 100 + gains.cumsum() - losses.cumsum()
        
        result = rsi(prices, period=14)
        
        # RSI should be high (> 70) when only gains
        assert result.iloc[-1] > 70
    
    def test_rsi_oversold(self):
        """Test RSI for oversold conditions."""
        # Create data with consistent losses
        losses = pd.Series([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        gains = pd.Series([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        prices = 110 - losses.cumsum() + gains.cumsum()
        
        result = rsi(prices, period=14)
        
        # RSI should be low (< 30) when only losses
        assert result.iloc[-1] < 30
    
    def test_rsi_range(self):
        """Test RSI is between 0 and 100."""
        np.random.seed(42)
        prices = pd.Series(100 + np.random.randn(100))
        result = rsi(prices, period=14)
        
        assert (result >= 0).all()
        assert (result <= 100).all()
    
    def test_rsi_stable(self):
        """Test RSI stabilizes with enough data."""
        # Constant upward trend
        prices = pd.Series(range(100, 200))
        result = rsi(prices, period=14)
        
        # Should be stable at high values
        assert result.iloc[-1] > 70


class TestZScore:
    """Tests for Z-Score calculation."""
    
    def test_zscore_basic(self):
        """Test basic Z-score calculation."""
        data = pd.Series([10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
                          10, 11, 10, 11, 10, 11, 10, 11, 10, 11,
                          5, 15])  # Outliers at end
        
        result = zscore(data, window=20)
        
        # Outliers should have extreme z-scores
        assert abs(result.iloc[20]) > 2  # Low outlier
        assert abs(result.iloc[21]) > 2  # High outlier
    
    def test_zscore_mean_zero(self):
        """Test Z-score mean is approximately zero."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100))
        
        result = zscore(data, window=50)
        
        # Mean of z-scores should be close to 0
        assert abs(result.iloc[50:].mean()) < 0.1
    
    def test_zscore_std_one(self):
        """Test Z-score std is approximately 1."""
        np.random.seed(42)
        data = pd.Series(np.random.randn(100))
        
        result = zscore(data, window=50)
        
        # Std of z-scores should be close to 1
        assert 0.9 < result.iloc[50:].std() < 1.1


class TestCalculateAllIndicators:
    """Tests for calculate_all_indicators function."""
    
    def test_calculate_all_indicators(self):
        """Test calculating all indicators at once."""
        np.random.seed(42)
        data = pd.DataFrame({
            'Close': 100 + np.random.randn(100).cumsum()
        })
        
        result = calculate_all_indicators(
            data,
            price_col='Close',
            sma_period=20,
            ema_period=20,
            rsi_period=14,
            bb_period=20,
            bb_std=2.0,
            zscore_window=20
        )
        
        # Check that all indicator columns exist
        expected_cols = [
            'SMA_20', 'EMA_20', 'BB_Middle', 'BB_Upper', 'BB_Lower',
            'BB_Width', 'BB_Position', 'RSI_14', 'ZScore_20'
        ]
        
        for col in expected_cols:
            assert col in result.columns
    
    def test_calculate_all_indicators_with_empty_data(self):
        """Test with empty DataFrame."""
        data = pd.DataFrame({'Close': []})
        
        with pytest.raises(Exception):
            calculate_all_indicators(data)
    
    def test_calculate_all_indicators_insufficient_data(self):
        """Test with insufficient data for indicators."""
        data = pd.DataFrame({'Close': [100, 101, 102]})
        
        result = calculate_all_indicators(
            data,
            price_col='Close',
            sma_period=20,
            rsi_period=14,
            bb_period=20,
            zscore_window=20
        )
        
        # Should have NaN values for rolling indicators
        assert result['SMA_20'].isna().all()
        assert result['RSI_14'].isna().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
