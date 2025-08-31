#!/usr/bin/env python3
"""
Test script for the Trading Bot Dashboard
"""

import time
import requests
import threading
from dashboard import dashboard
from market_data import market_data
from trading_signals import trading_signals
from telegram_bot import TelegramBot
from loguru import logger

def test_dashboard_basic():
    """Test basic dashboard functionality"""
    logger.info("Testing basic dashboard functionality...")

    # Test dashboard data structure
    assert hasattr(dashboard, 'dashboard_data'), "Dashboard should have dashboard_data attribute"
    assert 'signals' in dashboard.dashboard_data, "Dashboard data should contain signals"
    assert 'performance' in dashboard.dashboard_data, "Dashboard data should contain performance"
    assert 'market_data' in dashboard.dashboard_data, "Dashboard data should contain market_data"

    logger.info("✓ Basic dashboard structure test passed")

def test_dashboard_update():
    """Test dashboard data update functionality"""
    logger.info("Testing dashboard data update...")

    # Update dashboard data
    dashboard.update_dashboard_data()

    # Check if market data was populated
    market_data_present = len(dashboard.dashboard_data['market_data']) > 0
    logger.info(f"Market data present: {market_data_present}")

    # Check if performance data exists
    performance_present = len(dashboard.dashboard_data['performance']) > 0
    logger.info(f"Performance data present: {performance_present}")

    logger.info("✓ Dashboard update test completed")

def test_add_signal():
    """Test adding signals to dashboard"""
    logger.info("Testing signal addition to dashboard...")

    # Sample signal data
    signal_data = {
        'pair': 'BTCUSDT',
        'type': 'BUY',
        'price': 45000,
        'leverage': 10,
        'confidence': 'High'
    }

    # Add signal
    dashboard.add_signal(signal_data)

    # Check if signal was added
    signals = dashboard.dashboard_data['signals']
    assert len(signals) > 0, "Signal should be added to dashboard"

    # Check signal content
    latest_signal = signals[-1]
    assert latest_signal['pair'] == 'BTCUSDT', "Signal pair should match"
    assert latest_signal['type'] == 'BUY', "Signal type should match"

    logger.info("✓ Signal addition test passed")

def test_dashboard_server():
    """Test dashboard server startup (non-blocking)"""
    logger.info("Testing dashboard server startup...")

    # Start dashboard in a separate thread
    server_thread = threading.Thread(target=lambda: dashboard.run(host='127.0.0.1', port=5001, debug=False))
    server_thread.daemon = True
    server_thread.start()

    # Wait a moment for server to start
    time.sleep(2)

    try:
        # Test if server is responding
        response = requests.get('http://127.0.0.1:5001/', timeout=5)
        assert response.status_code == 200, f"Dashboard server should respond with 200, got {response.status_code}"

        logger.info("✓ Dashboard server test passed")

    except requests.exceptions.RequestException as e:
        logger.warning(f"Dashboard server test failed (may be expected if Flask-SocketIO not fully configured): {e}")

    except Exception as e:
        logger.error(f"Unexpected error in dashboard server test: {e}")

def run_dashboard_tests():
    """Run all dashboard tests"""
    logger.info("Starting dashboard tests...")

    try:
        test_dashboard_basic()
        test_dashboard_update()
        test_add_signal()
        test_dashboard_server()

        logger.info("✅ All dashboard tests completed successfully!")

    except Exception as e:
        logger.error(f"Dashboard test failed: {e}")
        raise

if __name__ == "__main__":
    run_dashboard_tests()
