#!/usr/bin/env python3
"""
Test script to verify parallel processing functionality
"""

import time
from loguru import logger
from market_data import market_data

def test_parallel_ohlcv():
    """Test parallel OHLCV data fetching"""
    logger.info("Testing parallel OHLCV data fetching...")
    
    # Test sequential fetching (current method)
    start_time = time.time()
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        df = market_data.get_ohlcv_data(symbol, '5m', 100)
        if df is not None:
            logger.info(f"Sequential: Fetched {len(df)} candles for {symbol}")
    sequential_time = time.time() - start_time
    
    # Test parallel fetching
    start_time = time.time()
    results = market_data.get_ohlcv_data_parallel(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], '5m', 100)
    parallel_time = time.time() - start_time
    
    logger.info(f"Sequential time: {sequential_time:.2f}s")
    logger.info(f"Parallel time: {parallel_time:.2f}s")
    logger.info(f"Speed improvement: {sequential_time/parallel_time:.1f}x faster")
    
    for symbol, df in results.items():
        logger.info(f"Parallel: Fetched {len(df)} candles for {symbol}")

def test_parallel_prices():
    """Test parallel price fetching"""
    logger.info("Testing parallel price fetching...")
    
    # Test sequential fetching
    start_time = time.time()
    for symbol in ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']:
        price = market_data.get_current_price(symbol)
        if price is not None:
            logger.info(f"Sequential: {symbol} = ${price:,.2f}")
    sequential_time = time.time() - start_time
    
    # Test parallel fetching
    start_time = time.time()
    results = market_data.get_current_prices_parallel(['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    parallel_time = time.time() - start_time
    
    logger.info(f"Sequential time: {sequential_time:.2f}s")
    logger.info(f"Parallel time: {parallel_time:.2f}s")
    logger.info(f"Speed improvement: {sequential_time/parallel_time:.1f}x faster")
    
    for symbol, price in results.items():
        logger.info(f"Parallel: {symbol} = ${price:,.2f}")

if __name__ == "__main__":
    # Configure logging
    logger.add("parallel_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")
    
    logger.info("Starting parallel processing tests...")
    
    test_parallel_ohlcv()
    test_parallel_prices()
    
    logger.info("Parallel processing tests completed!")
