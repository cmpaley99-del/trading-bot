#!/usr/bin/env python3
"""
Test script to verify backtesting framework functionality
"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger
from backtest import backtester
from config import Config

async def test_backtesting():
    """Test backtesting framework with sample data"""
    logger.info("Testing backtesting framework...")

    # Test data
    trading_pairs = Config.TRADING_PAIRS[:3]  # Test with first 3 pairs
    start_date = datetime.now() - timedelta(days=7)
    end_date = datetime.now()

    logger.info(f"Running backtest for pairs: {trading_pairs}")
    logger.info(f"Date range: {start_date.date()} to {end_date.date()}")

    try:
        # Run backtest
        metrics = await backtester.run_backtest(start_date, end_date, trading_pairs)

        if metrics:
            logger.info("✅ Backtest completed successfully")
            backtester.print_summary()

            # Save results
            backtester.save_results("test_backtest_results.json")
            logger.info("✅ Backtest results saved to test_backtest_results.json")

        else:
            logger.error("❌ Backtest failed - no metrics returned")

    except Exception as e:
        logger.error(f"❌ Backtest error: {e}")
        return False

    return True

def test_backtester_initialization():
    """Test backtester initialization"""
    logger.info("Testing backtester initialization...")

    try:
        from backtest import backtester
        assert backtester is not None
        assert hasattr(backtester, 'run_backtest')
        assert hasattr(backtester, 'print_summary')
        assert hasattr(backtester, 'save_results')
        logger.info("✅ Backtester initialization successful")
        return True
    except Exception as e:
        logger.error(f"❌ Backtester initialization failed: {e}")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.add("backtesting_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.info("Starting backtesting framework tests...")

    # Test 1: Initialization
    if not test_backtester_initialization():
        logger.error("Backtester initialization test failed")
        exit(1)

    # Test 2: Functional backtesting
    logger.info("Running functional backtesting test...")
    success = asyncio.run(test_backtesting())

    if success:
        logger.info("✅ All backtesting tests passed!")
    else:
        logger.error("❌ Backtesting tests failed!")

    logger.info("Backtesting framework tests completed!")
