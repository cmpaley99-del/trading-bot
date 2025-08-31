#!/usr/bin/env python3
"""
Test script to verify Telegram bot configuration
"""

import asyncio
from telegram import Bot
from telegram.error import TelegramError
from config import Config
from loguru import logger

async def test_telegram():
    """Test Telegram bot connection and chat ID"""
    try:
        bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        
        # Test if bot token is valid
        me = await bot.get_me()
        print(f"✅ Bot connected successfully: @{me.username}")
        
        # Test if chat ID is valid
        try:
            await bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text="🤖 Telegram bot test message - connection successful!",
                disable_web_page_preview=True
            )
            print(f"✅ Message sent successfully to chat ID: {Config.TELEGRAM_CHAT_ID}")
            
        except TelegramError as e:
            if "chat not found" in str(e).lower():
                print(f"❌ Chat not found error. Please ensure:")
                print(f"   1. You have messaged the bot @{me.username}")
                print(f"   2. The chat ID in .env is correct")
                print(f"   3. To get your chat ID, message the bot and check:")
                print(f"      https://api.telegram.org/bot{Config.TELEGRAM_BOT_TOKEN}/getUpdates")
            else:
                print(f"❌ Telegram error: {e}")
                
    except Exception as e:
        print(f"❌ Failed to connect to Telegram: {e}")
        print("Please check your TELEGRAM_BOT_TOKEN in .env file")

if __name__ == "__main__":
    print("Testing Telegram configuration...")
    asyncio.run(test_telegram())
