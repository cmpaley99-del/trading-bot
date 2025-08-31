#!/usr/bin/env python3
"""
Test script to verify scalping strategy works across all configured trading pairs
"""

import pandas as pd
import numpy as np
from loguru import logger
from config import Config
from market_data import MarketData
from technical_analysis import technical_analysis
from trading_signals import TradingSignals

def test_all_pairs_scalping():
    """Test scalping strategy across all configured trading pairs"""
    
    logger.info("Testing scalping strategy across all trading pairs...")
    logger.info(f"Configured pairs: {Config.TRADING_PAIRS}")
    
    market_data = MarketData()
    trading_signals = TradingSignals()
    
    for pair in Config.TRADING_PAIRS:
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
            
            # Calculate intelligent leverage
            leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, pair)
            
            # Generate trade call
            trade_call = trading_signals._generate_single_trade_call(pair)
            
            # Display results
            current = df_with_indicators.iloc[-1]
            logger.info(f"Price: {current['close']:.2f}")
            logger.info(f"EMA 5/10/20: {current['ema_5']:.2f}/{current['ema_10']:.2f}/{current['ema_20']:.2f}")
            logger.info(f"RSI: {current['rsi']:.1f}, CCI: {current['cci']:.1f}")
            logger.info(f"ADX: {current['adx']:.1f}, Volatility: {(current['atr']/current['close']*100):.2f}%")
            logger.info(f"Scalp Signal: {signals['scalp_signal']}")
            logger.info(f"Overall Signal: {signals['overall_signal']}")
            logger.info(f"Intelligent Leverage: {leverage}x")
            
            if trade_call:
                logger.info(f"TRADE CALL: Generated for {pair}")
            else:
                logger.info("No trade call generated")
                
        except Exception as e:
            logger.error(f"Error testing {pair}: {e}")
            continue
    
    logger.info("\n=== Scalping Strategy Test Complete ===")

if __name__ == "__main__":
    # Configure logging
    logger.add("scalping_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    test_all_pairs_scalping()
