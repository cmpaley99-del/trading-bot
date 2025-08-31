#!/usr/bin/env python3
"""
Railway startup script for the Trading Bot
"""

import os
import sys
from loguru import logger

# Set up logging
logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")

def main():
    """Main startup function"""
    try:
        logger.info("🚀 Starting Trading Bot on Railway...")

        # Check for required environment variables
        telegram_token = os.environ.get('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.environ.get('TELEGRAM_CHAT_ID')

        if not telegram_token:
            logger.error("❌ TELEGRAM_BOT_TOKEN environment variable is required")
            sys.exit(1)

        if not telegram_chat_id:
            logger.error("❌ TELEGRAM_CHAT_ID environment variable is required")
            sys.exit(1)

        logger.info("✅ Environment variables validated")

        # Import and run the main bot
        from main import TradingBot

        logger.info("✅ Bot modules loaded successfully")

        bot = TradingBot()
        logger.info("✅ Trading Bot initialized")

        bot.run()

    except Exception as e:
        logger.error(f"❌ Failed to start Trading Bot: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
