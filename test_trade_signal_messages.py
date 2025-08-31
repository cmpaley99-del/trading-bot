#!/usr/bin/env python3
"""
Test script to simulate trade signal messages with clear long/short, prices, and leverage
"""

from trading_signals import TradingSignals
from datetime import datetime

def simulate_trade_messages():
    ts = TradingSignals()

    # Simulate bullish signal for BTCUSDT
    message_bull = ts._prepare_trade_message(
        trading_pair="BTCUSDT",
        signal_type="BULLISH",
        entry_price=50000.0,
        position_size=0.01,
        stop_loss=49000.0,
        take_profit=52000.0,
        signal_quality={"quality": "HIGH"},
        leverage=10
    )
    print("Bullish Signal Message:\n", message_bull)

    # Simulate bearish signal for ETHUSDT
    message_bear = ts._prepare_trade_message(
        trading_pair="ETHUSDT",
        signal_type="BEARISH",
        entry_price=3000.0,
        position_size=0.1,
        stop_loss=3100.0,
        take_profit=2800.0,
        signal_quality={"quality": "MEDIUM"},
        leverage=15
    )
    print("\nBearish Signal Message:\n", message_bear)

if __name__ == "__main__":
    simulate_trade_messages()
