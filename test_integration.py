#!/usr/bin/env python3
"""
Comprehensive integration test for the trading bot system
"""

import asyncio
from datetime import datetime, timedelta
from loguru import logger
from config import Config
from market_data import market_data
from trading_signals import trading_signals
from risk_management import risk_management
from telegram_bot import TelegramBot
from backtest import backtester

async def test_market_data_integration():
    """Test market data integration with trading signals"""
    logger.info("Testing market data and trading signals integration...")

    try:
        # Test data fetching
        pair = Config.TRADING_PAIRS[0]
        df = market_data.get_ohlcv_data(pair, '5m', limit=100)

        if df is None or df.empty:
            logger.error("❌ Failed to fetch market data")
            return False

        logger.info(f"✅ Fetched {len(df)} candles for {pair}")

        # Test technical analysis integration
        from technical_analysis import technical_analysis
        df_with_indicators = technical_analysis.calculate_indicators(df)
        if df_with_indicators is None:
            logger.error("❌ Failed to calculate technical indicators")
            return False

        signals = technical_analysis.generate_signals(df_with_indicators)

        if signals is None:
            logger.error("❌ Failed to generate trading signals")
            return False

        logger.info(f"✅ Generated signals for {pair}: {signals.get('overall_signal', 'N/A')}")
        return True

    except Exception as e:
        logger.error(f"❌ Market data integration test failed: {e}")
        return False

async def test_risk_management_integration():
    """Test risk management integration with position sizing"""
    logger.info("Testing risk management integration...")

    try:
        # Test position size calculation
        current_price = 50000  # BTC price
        position_size = risk_management.calculate_position_size(
            current_price, 'BTCUSDT', leverage=5
        )

        if position_size is None or position_size <= 0:
            logger.error("❌ Failed to calculate position size")
            return False

        logger.info(f"✅ Calculated position size: {position_size:.6f} BTC")
        return True

    except Exception as e:
        logger.error(f"❌ Risk management integration test failed: {e}")
        return False

async def test_telegram_integration():
    """Test Telegram bot integration (without actual sending)"""
    logger.info("Testing Telegram bot integration...")

    try:
        # Test bot initialization
        bot = TelegramBot()
        if not hasattr(bot, 'bot'):
            logger.error("❌ Telegram bot not properly initialized")
            return False

        logger.info("✅ Telegram bot initialized successfully")
        return True

    except Exception as e:
        logger.error(f"❌ Telegram integration test failed: {e}")
        return False

async def test_full_system_integration():
    """Test full system integration with backtesting"""
    logger.info("Testing full system integration...")

    try:
        # Run a quick backtest
        start_date = datetime.now() - timedelta(days=1)
        end_date = datetime.now()

        metrics = await backtester.run_backtest(start_date, end_date, Config.TRADING_PAIRS[:2])

        if metrics is None:
            logger.error("❌ Full system integration test failed - no metrics returned")
            return False

        logger.info("✅ Full system integration test passed")
        backtester.print_summary()
        return True

    except Exception as e:
        logger.error(f"❌ Full system integration test failed: {e}")
        return False

async def run_integration_tests():
    """Run all integration tests"""
    logger.info("Starting comprehensive integration tests...")

    tests = [
        ("Market Data Integration", test_market_data_integration),
        ("Risk Management Integration", test_risk_management_integration),
        ("Telegram Integration", test_telegram_integration),
        ("Full System Integration", test_full_system_integration)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} ---")
        try:
            if await test_func():
                passed += 1
                logger.info(f"✅ {test_name} PASSED")
            else:
                logger.error(f"❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"❌ {test_name} ERROR: {e}")

    logger.info(f"\n{'='*50}")
    logger.info("INTEGRATION TEST RESULTS")
    logger.info(f"{'='*50}")
    logger.info(f"Tests Passed: {passed}/{total}")
    logger.info(".1f")
    logger.info(f"{'='*50}")

    if passed == total:
        logger.info("🎉 ALL INTEGRATION TESTS PASSED!")
        return True
    else:
        logger.error("❌ SOME INTEGRATION TESTS FAILED!")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.add("integration_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.info("Starting integration testing suite...")

    success = asyncio.run(run_integration_tests())

    if success:
        logger.info("✅ Integration testing completed successfully!")
    else:
        logger.error("❌ Integration testing failed!")

    logger.info("Integration testing suite completed!")
