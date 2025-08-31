#!/usr/bin/env python3
"""
Test script to simulate different market conditions and verify multiple trade call generation
"""

import pandas as pd
import numpy as np
from loguru import logger
from config import Config
from market_data import MarketData
from technical_analysis import technical_analysis
from trading_signals import TradingSignals

def create_simulated_market_data():
    """Create simulated market data with different signal conditions"""
    
    # Simulate bullish overall + bearish scalping scenario
    bullish_overall_bearish_scalp = {
        'overall_signal': 'BULLISH',
        'scalp_signal': 'BEARISH',
        'rsi_signal': 'BULLISH',
        'macd_signal': 'BULLISH',
        'bb_signal': 'BULLISH',
        'trend_signal': 'BULLISH',
        'volume_signal': 'NEUTRAL',
        'pattern_signal': 'NEUTRAL',
        'ema_scalp_signal': 'BEARISH',
        'momentum_signal': 'BEARISH',
        'vwap_signal': 'NEUTRAL'
    }
    
    # Simulate bearish overall + bullish scalping scenario
    bearish_overall_bullish_scalp = {
        'overall_signal': 'BEARISH',
        'scalp_signal': 'BULLISH',
        'rsi_signal': 'BEARISH',
        'macd_signal': 'BEARISH',
        'bb_signal': 'BEARISH',
        'trend_signal': 'BEARISH',
        'volume_signal': 'NEUTRAL',
        'pattern_signal': 'NEUTRAL',
        'ema_scalp_signal': 'BULLISH',
        'momentum_signal': 'BULLISH',
        'vwap_signal': 'NEUTRAL'
    }
    
    # Simulate both bullish scenario
    both_bullish = {
        'overall_signal': 'BULLISH',
        'scalp_signal': 'BULLISH',
        'rsi_signal': 'BULLISH',
        'macd_signal': 'BULLISH',
        'bb_signal': 'BULLISH',
        'trend_signal': 'BULLISH',
        'volume_signal': 'BULLISH',
        'pattern_signal': 'NEUTRAL',
        'ema_scalp_signal': 'BULLISH',
        'momentum_signal': 'BULLISH',
        'vwap_signal': 'BULLISH'
    }
    
    return [bullish_overall_bearish_scalp, bearish_overall_bullish_scalp, both_bullish]

def test_simulated_signals():
    """Test multiple signal generation with simulated data"""
    
    import pandas as pd
    
    logger.info("Testing multiple trade call generation with simulated signals...")
    
    trading_signals = TradingSignals()
    
    # Test different simulated scenarios
    test_scenarios = create_simulated_market_data()
    scenario_names = [
        "Bullish Overall + Bearish Scalping",
        "Bearish Overall + Bullish Scalping", 
        "Both Bullish (should generate only one call)"
    ]
    
    # Create dummy df_with_indicators with 'atr' column for stop loss calculation
    dummy_df = pd.DataFrame({'atr': [100]})
    
    for i, (signals, scenario_name) in enumerate(zip(test_scenarios, scenario_names)):
        try:
            print(f"\n=== Testing Scenario {i+1}: {scenario_name} ===")
            print(f"Overall Signal: {signals['overall_signal']}")
            print(f"Scalp Signal: {signals['scalp_signal']}")
            
            # Simulate current price and leverage
            current_price = 50000.0 if 'BULLISH' in signals['overall_signal'] else 48000.0
            leverage = 20
            
            # Generate trade calls
            trade_calls = trading_signals._generate_trade_calls_for_signals(
                "BTCUSDT", signals, current_price, leverage, 
                dummy_df, {'quality': 'HIGH'}
            )
            
            print(f"Generated {len(trade_calls)} trade calls")
            
            for j, trade_call in enumerate(trade_calls):
                print(f"Trade Call {j+1}:\n{trade_call}")
                print("-" * 50)
                
        except Exception as e:
            print(f"Error testing scenario {scenario_name}: {e}")
            continue
    
    print("\n=== Simulated Signal Test Complete ===")

if __name__ == "__main__":
    # Configure logging
    logger.add("simulated_signals_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    test_simulated_signals()
