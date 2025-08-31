#!/usr/bin/env python3
"""
Test script to verify multiple trade call generation with operation types
"""

import pandas as pd
import numpy as np
from loguru import logger
from config import Config
from market_data import MarketData
from technical_analysis import technical_analysis
from trading_signals import TradingSignals

def test_multiple_signals():
    """Test multiple signal generation with operation types"""
    
    logger.info("Testing multiple trade call generation with operation types...")
    
    market_data = MarketData()
    trading_signals = TradingSignals()
    
    # Test with a specific pair that might have both signals
    test_pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'AVAXUSDT']
    
    for pair in test_pairs:
        try:
            logger.info(f"\n=== Testing {pair} ===")
            
            # Fetch market data
            df = market_data.get_ohlcv_data(pair, '5m', 100)
            if df is None:
                logger.warning(f"Failed to fetch data for {pair}")
                continue
            
            # Calculate indicators
            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                logger.warning(f"Failed to calculate indicators for {pair}")
                continue
            
            # Generate signals
            signals = technical_analysis.generate_signals(df_with_indicators)
            if signals is None:
                logger.warning(f"Failed to generate signals for {pair}")
                continue
            
            # Display signal information
            logger.info(f"Overall Signal: {signals['overall_signal']}")
            logger.info(f"Scalp Signal: {signals['scalp_signal']}")
            print(f"Overall Signal: {signals['overall_signal']}")
            print(f"Scalp Signal: {signals['scalp_signal']}")
            
            # Get market metrics
            metrics = market_data.get_market_metrics(pair)
            if metrics:
                current_price = metrics['current_price']
                
                # Calculate leverage
                leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, pair)
                logger.info(f"Intelligent Leverage: {leverage}x")
                print(f"Intelligent Leverage: {leverage}x")
                
                # Generate multiple trade calls
                trade_calls = trading_signals._generate_trade_calls_for_signals(
                    pair, signals, current_price, leverage, df_with_indicators, {'quality': 'HIGH'}
                )
                
                logger.info(f"Generated {len(trade_calls)} trade calls for {pair}")
                print(f"Generated {len(trade_calls)} trade calls for {pair}")
                
                for i, trade_call in enumerate(trade_calls):
                    logger.info(f"Trade Call {i+1}:\n{trade_call}")
                    logger.info("-" * 50)
                    print(f"Trade Call {i+1}:\n{trade_call}")
                    print("-" * 50)
                    
        except Exception as e:
            logger.error(f"Error testing {pair}: {e}")
            continue
    
    logger.info("\n=== Multiple Signal Test Complete ===")

if __name__ == "__main__":
    # Configure logging
    logger.add("multiple_signals_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    test_multiple_signals()
