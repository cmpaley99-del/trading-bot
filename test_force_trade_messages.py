#!/usr/bin/env python3
"""
Test script to force trade messages to be sent to Telegram for demonstration
"""

from telegram_bot import TelegramBot
from trading_signals import TradingSignals
import asyncio

async def force_trade_messages():
    telegram_bot = TelegramBot()
    ts = TradingSignals()
    
    # Create test trade calls that would be sent to Telegram
    test_trade_calls = [
        """🚀 *LONG BTC*

💰 *Entry:* $50,250.00
📈 *Size:* 0.008500 BTC
⚡ *Leverage:* 18x

🛑 *Stop Loss:* $48,750.00
🎯 *Take Profit:* $53,500.00

🔍 *Quality:* HIGH
⏰ *Time:* 00:10:22

💡 *Action:* LONG at market""",
        
        """📉 *SHORT ETH*

💰 *Entry:* $3,150.00
📈 *Size:* 0.085000 ETH
⚡ *Leverage:* 12x

🛑 *Stop Loss:* $3,280.00
🎯 *Take Profit:* $2,950.00

🔍 *Quality:* MEDIUM
⏰ *Time:* 00:10:22

💡 *Action:* SHORT at market""",
        
        """🚀 *LONG SOL*

💰 *Entry:* $102.50
📈 *Size:* 4.250000 SOL
⚡ *Leverage:* 15x

🛑 *Stop Loss:* $97.50
🎯 *Take Profit:* $112.00

🔍 *Quality:* HIGH
⏰ *Time:* 00:10:22

💡 *Action:* LONG at market"""
    ]
    
    print("Sending test trade messages to Telegram...")
    
    for i, trade_call in enumerate(test_trade_calls, 1):
        print(f"\n--- Sending Trade Call {i} ---")
        print(trade_call)
        try:
            await telegram_bot.send_message(trade_call)
            print("✅ Message sent successfully!")
        except Exception as e:
            print(f"❌ Error sending message: {e}")
    
    print(f"\nTotal trade messages sent: {len(test_trade_calls)}")

if __name__ == "__main__":
    asyncio.run(force_trade_messages())
