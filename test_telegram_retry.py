#!/usr/bin/env python3
"""
Test script to verify Telegram retry mechanism
"""

import asyncio
from unittest.mock import patch, MagicMock
from telegram.error import TelegramError
from loguru import logger
from telegram_bot import TelegramBot

def test_retry_mechanism():
    """Test Telegram retry mechanism with simulated errors"""
    logger.info("Testing Telegram retry mechanism...")

    # Create bot instance
    bot = TelegramBot()

    # Test 1: Successful message on first attempt
    logger.info("Test 1: Successful message on first attempt")
    with patch('telegram.Bot.send_message') as mock_send:
        mock_send.return_value = MagicMock()
        result = bot.send_message("Test message")
        assert result == True
        assert mock_send.call_count == 1
        logger.info("✅ Test 1 passed: Message sent on first attempt")

    # Test 2: Success after retry on pool timeout
    logger.info("Test 2: Success after retry on pool timeout")
    with patch('telegram.Bot.send_message') as mock_send:
        # First call raises pool timeout, second succeeds
        mock_send.side_effect = [
            TelegramError("Pool timeout: All connections in the connection pool are occupied"),
            MagicMock()  # Success on retry
        ]
        result = bot.send_message("Test message")
        assert result == True
        assert mock_send.call_count == 2
        logger.info("✅ Test 2 passed: Message sent after retry")

    # Test 3: Failure after all retries
    logger.info("Test 3: Failure after all retries")
    with patch('telegram.Bot.send_message') as mock_send:
        # Always raise pool timeout
        mock_send.side_effect = TelegramError("Pool timeout: All connections in the connection pool are occupied")
        result = bot.send_message("Test message")
        assert result == False
        assert mock_send.call_count == 3  # Max retries
        logger.info("✅ Test 3 passed: Failed after max retries")

    # Test 4: Success on different error type
    logger.info("Test 4: Success on different error type")
    with patch('telegram.Bot.send_message') as mock_send:
        # First call raises different error, second succeeds
        mock_send.side_effect = [
            TelegramError("Some other error"),
            MagicMock()  # Success on retry
        ]
        result = bot.send_message("Test message")
        assert result == True
        assert mock_send.call_count == 2
        logger.info("✅ Test 4 passed: Message sent after retry on different error")

    logger.info("All Telegram retry tests passed! ✅")

if __name__ == "__main__":
    # Configure logging
    logger.add("telegram_retry_test.log", level="INFO", format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}")

    logger.info("Starting Telegram retry mechanism tests...")
    test_retry_mechanism()
    logger.info("Telegram retry mechanism tests completed!")
