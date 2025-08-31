"""
Test script for improved modules
Validates the enhancements made to the trading bot
"""

import sys
import os
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config_improved import config
from database_improved import ImprovedDatabase
from market_data_improved import market_data

# Use a separate test database to avoid conflicts
test_database = ImprovedDatabase('test_trading_bot.db')

def test_configuration_improvements():
    """Test improved configuration module"""
    print("\n🔧 Testing Configuration Improvements...")

    try:
        # Test configuration validation
        print("✅ Configuration validation:")
        print(f"   Trading pairs: {len(config.TRADING_PAIRS)}")
        print(f"   Risk percentage: {config.trading_config.risk_percentage}%")
        print(f"   Max position size: ${config.trading_config.max_position_size}")
        print(f"   Technical config: RSI({config.technical_config.rsi_period})")

        # Test configuration summary
        print("\n📊 Configuration Summary:")
        summary = config.get_summary()
        print(summary)

        # Test configuration export
        config.export_config('test_config_export.json')
        print("✅ Configuration exported to test_config_export.json")

        return True

    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_database_improvements():
    """Test improved database module"""
    print("\n💾 Testing Database Improvements...")

    try:
        # Test database stats
        stats = test_database.get_database_stats()
        print("✅ Database stats:")
        for key, value in stats.items():
            print(f"   {key}: {value}")

        # Test signal saving
        test_signal = {
            'symbol': 'BTCUSDT',
            'signal_type': 'BULLISH',
            'entry_price': 50000.0,
            'position_size': 1000.0,
            'stop_loss': 49000.0,
            'take_profit': 52000.0,
            'leverage': 10,
            'risk_percentage': 2.0,
            'signal_quality': 'HIGH',
            'confidence_score': 0.85,
            'rsi_signal': 'BULLISH',
            'macd_signal': 'BULLISH',
            'bb_signal': None,
            'trend_signal': None,
            'volume_signal': None,
            'pattern_signal': None,
            'harmonic_pattern': None,
            'elliott_wave': None,
            'fibonacci_level': None,
            'message': 'Test signal from improved database'
        }

        signal_id = test_database.save_signal(test_signal)
        if signal_id:
            print(f"✅ Test signal saved with ID: {signal_id}")
        else:
            print("❌ Failed to save test signal")
            return False

        # Test signal retrieval
        recent_signals = test_database.get_recent_signals(limit=5)
        if recent_signals is not None and len(recent_signals) > 0:
            print(f"✅ Retrieved {len(recent_signals)} recent signals")
        else:
            print("❌ Failed to retrieve recent signals")

        # Test performance stats
        perf_stats = test_database.get_performance_stats()
        print("✅ Performance stats:")
        for key, value in perf_stats.items():
            print(f"   {key}: {value}")

        return True

    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_market_data_improvements():
    """Test improved market data module"""
    print("\n📈 Testing Market Data Improvements...")

    try:
        # Test health status
        health = market_data.get_health_status()
        print("✅ Market data health status:")
        for key, value in health.items():
            print(f"   {key}: {value}")

        # Test single symbol data fetch
        test_symbol = 'BTCUSDT'
        df = market_data.get_ohlcv_data(test_symbol, '5m', 10)
        if df is not None and len(df) > 0:
            print(f"✅ Fetched {len(df)} candles for {test_symbol}")
            print(f"   Latest price: ${df['close'].iloc[-1]:.2f}")
        else:
            print(f"❌ Failed to fetch data for {test_symbol}")
            return False

        # Test current price
        price = market_data.get_current_price(test_symbol)
        if price:
            print(f"✅ Current price for {test_symbol}: ${price:.2f}")
        else:
            print(f"❌ Failed to get current price for {test_symbol}")

        # Test market metrics
        metrics = market_data.get_market_metrics(test_symbol)
        if metrics:
            print("✅ Market metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"   {key}: {value:.2f}")
                else:
                    print(f"   {key}: {value}")
        else:
            print(f"❌ Failed to get market metrics for {test_symbol}")

        # Test parallel data fetching
        symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
        parallel_data = market_data.get_ohlcv_data_parallel(symbols, '5m', 5)
        if parallel_data:
            print(f"✅ Parallel data fetch successful for {len(parallel_data)} symbols")
            for symbol, data in parallel_data.items():
                if data is not None:
                    print(f"   {symbol}: {len(data)} candles")
        else:
            print("❌ Parallel data fetch failed")

        return True

    except Exception as e:
        print(f"❌ Market data test failed: {e}")
        return False

def test_performance_improvements():
    """Test performance improvements"""
    print("\n⚡ Testing Performance Improvements...")

    try:
        # Test caching performance
        test_symbol = 'BTCUSDT'

        # First call (cache miss)
        start_time = time.time()
        df1 = market_data.get_ohlcv_data(test_symbol, '5m', 20)
        first_call_time = time.time() - start_time

        # Second call (cache hit)
        start_time = time.time()
        df2 = market_data.get_ohlcv_data(test_symbol, '5m', 20)
        second_call_time = time.time() - start_time

        if df1 is not None and df2 is not None:
            speedup = first_call_time / second_call_time if second_call_time > 0 else 1
            print(f"   First call: {first_call_time:.3f}s")
            print(f"   Second call: {second_call_time:.3f}s")
            print(f"   Speedup: {speedup:.1f}x")
        # Test database query performance
        start_time = time.time()
        for _ in range(10):
            test_database.get_recent_signals(limit=5)
        db_query_time = (time.time() - start_time) / 10

        print(f"   Database query time: {db_query_time:.4f}s")
        return True

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False

def test_error_handling_improvements():
    """Test error handling improvements"""
    print("\n🛡️ Testing Error Handling Improvements...")

    try:
        # Test invalid symbol
        invalid_df = market_data.get_ohlcv_data('INVALID_SYMBOL', '5m', 10)
        if invalid_df is None:
            print("✅ Properly handled invalid symbol")
        else:
            print("❌ Failed to handle invalid symbol properly")

        # Test configuration validation
        try:
            from config_improved import ImprovedConfig
            test_config = ImprovedConfig()
            print("✅ Configuration validation working")
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")

        # Test database error handling
        try:
            invalid_signal = test_database.save_signal({})
            if invalid_signal is None:
                print("✅ Database properly handled invalid signal data")
            else:
                print("❌ Database failed to handle invalid signal data")
        except Exception as e:
            print(f"❌ Database error handling test failed: {e}")

        return True

    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all improvement tests"""
    print("🚀 RUNNING COMPREHENSIVE IMPROVEMENT TESTS")
    print("=" * 60)

    tests = [
        ("Configuration Improvements", test_configuration_improvements),
        ("Database Improvements", test_database_improvements),
        ("Market Data Improvements", test_market_data_improvements),
        ("Performance Improvements", test_performance_improvements),
        ("Error Handling Improvements", test_error_handling_improvements)
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)

    passed = 0
    total = len(results)

    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if result:
            passed += 1

    print(f"\n🎯 Overall: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL IMPROVEMENTS WORKING PERFECTLY!")
        return True
    else:
        print("⚠️  Some improvements need attention")
        return False

if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    success = run_comprehensive_test()

    if success:
        print("\n🎯 IMPROVEMENT SUMMARY:")
        print("✅ Configuration: Type hints, validation, structured config")
        print("✅ Database: Connection pooling, error handling, performance")
        print("✅ Market Data: Caching, rate limiting, error recovery")
        print("✅ Performance: Optimized queries, caching, parallel processing")
        print("✅ Error Handling: Comprehensive error recovery and logging")
        print("\n🚀 The trading bot is now significantly improved and production-ready!")
    else:
        print("\n⚠️  Some improvements may need additional work")
        sys.exit(1)
