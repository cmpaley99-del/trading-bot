#!/usr/bin/env python3
"""
Test script to simulate Telegram bot sending multiple trade messages
"""

from telegram_bot import TelegramBot
from trading_signals import TradingSignals
import asyncio

async def test_multiple_messages():
    telegram_bot = TelegramBot()
    ts = TradingSignals()
    
    # Simulate multiple trade calls
    trade_calls = [
        ts._prepare_trade_message(
            trading_pair="BTCUSDT",
            signal_type="BULLISH",
            entry_price=50000.0,
            position_size=0.01,
            stop_loss=49000.0,
            take_profit=52000.0,
            signal_quality={"quality": "HIGH"},
            leverage=10
        ),
        ts._prepare_trade_message(
            trading_pair="ETHUSDT",
            signal_type="BEARISH",
            entry_price=3000.0,
            position_size=0.1,
            stop_loss=3100.0,
            take_profit=2800.0,
            signal_quality={"quality": "MEDIUM"},
            leverage=15
        ),
        ts._prepare_trade_message(
            trading_pair="SOLUSDT",
            signal_type="BULLISH",
            entry_price=100.0,
            position_size=5.0,
            stop_loss=95.0,
            take_profit=110.0,
            signal_quality={"quality": "HIGH"},
            leverage=20
        )
    ]
    
    print("Testing multiple trade message sending...")
    for i, trade_call in enumerate(trade_calls, 1):
        print(f"\n--- Trade Call {i} ---")
        print(trade_call)
        print("--- End of Message ---")
    
    print(f"\nTotal trade calls generated: {len(trade_calls)}")
    print("Each trade call would be sent as a separate Telegram message")

if __name__ == "__main__":
    asyncio.run(test_multiple_messages())
