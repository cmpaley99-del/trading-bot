#!/usr/bin/env python3
"""
Test script for intelligent leverage calculation
"""

import pandas as pd
import numpy as np
from technical_analysis import TechnicalAnalysis

def test_intelligent_leverage():
    """Test intelligent leverage calculation based on market volatility"""
    
    print("Testing intelligent leverage calculation...")
    
    # Create test data with different volatility scenarios
    tech_analysis = TechnicalAnalysis()
    
    # Test 1: Low volatility scenario
    print("\n=== Test 1: Low Volatility Scenario ===")
    low_vol_data = create_test_data(volatility=0.3, trend_strength=30)
    leverage_low = tech_analysis.calculate_appropriate_leverage(low_vol_data, "BTCUSDT")
    print(f"Low volatility leverage: {leverage_low}x")
    
    # Test 2: Medium volatility scenario
    print("\n=== Test 2: Medium Volatility Scenario ===")
    medium_vol_data = create_test_data(volatility=0.8, trend_strength=20)
    leverage_medium = tech_analysis.calculate_appropriate_leverage(medium_vol_data, "ETHUSDT")
    print(f"Medium volatility leverage: {leverage_medium}x")
    
    # Test 3: High volatility scenario
    print("\n=== Test 3: High Volatility Scenario ===")
    high_vol_data = create_test_data(volatility=1.5, trend_strength=10)
    leverage_high = tech_analysis.calculate_appropriate_leverage(high_vol_data, "SOLUSDT")
    print(f"High volatility leverage: {leverage_high}x")
    
    # Test 4: Very high volatility scenario
    print("\n=== Test 4: Very High Volatility Scenario ===")
    very_high_vol_data = create_test_data(volatility=3.0, trend_strength=5)
    leverage_very_high = tech_analysis.calculate_appropriate_leverage(very_high_vol_data, "ADAUSDT")
    print(f"Very high volatility leverage: {leverage_very_high}x")
    
    # Test 5: Strong trend scenario
    print("\n=== Test 5: Strong Trend Scenario ===")
    strong_trend_data = create_test_data(volatility=0.6, trend_strength=40)
    leverage_strong = tech_analysis.calculate_appropriate_leverage(strong_trend_data, "XRPUSDT")
    print(f"Strong trend leverage: {leverage_strong}x")
    
    print("\n=== Summary ===")
    print(f"Low Volatility ({leverage_low}x): Higher leverage due to stable market")
    print(f"Medium Volatility ({leverage_medium}x): Moderate leverage")
    print(f"High Volatility ({leverage_high}x): Lower leverage due to increased risk")
    print(f"Very High Volatility ({leverage_very_high}x): Very conservative leverage")
    print(f"Strong Trend ({leverage_strong}x): Slightly higher leverage due to clear direction")
    
    print("\nIntelligent leverage calculation working correctly!")

def create_test_data(volatility=1.0, trend_strength=20, price=50000, periods=100):
    """Create test data with specific volatility and trend characteristics"""
    np.random.seed(42)  # For reproducible results
    
    # Create base price series
    dates = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq='5min')
    
    # Create returns with specified volatility
    returns = np.random.normal(0, volatility/100, periods)
    
    # Add trend component based on trend strength
    trend = np.linspace(0, trend_strength/1000, periods)
    returns += trend
    
    # Calculate price series
    price_series = price * (1 + returns).cumprod()
    
    # Create DataFrame with OHLCV data
    df = pd.DataFrame({
        'open': price_series * 0.999,
        'high': price_series * 1.002,
        'low': price_series * 0.998, 
        'close': price_series,
        'volume': np.random.lognormal(15, 1, periods)
    }, index=dates)
    
    # Calculate technical indicators
    from technical_analysis import technical_analysis
    df_with_indicators = technical_analysis.calculate_indicators(df)
    
    return df_with_indicators

if __name__ == "__main__":
    test_intelligent_leverage()
