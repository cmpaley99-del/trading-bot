import asyncio
from telegram import Bot
from telegram.constants import ParseMode
from telegram.error import TelegramError
from loguru import logger
from config import Config
from trading_signals import trading_signals
from anomaly_detection import anomaly_detection
import schedule
import time
import threading
import queue
import nest_asyncio

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class TelegramBot:
    def __init__(self):
        self.bot = Bot(token=Config.TELEGRAM_BOT_TOKEN)
        self.chat_id = Config.TELEGRAM_CHAT_ID
        self.message_queue = queue.Queue()
        self.running = True
        logger.info("Telegram Bot initialized")

    async def send_message_async(self, message, max_retries=3, initial_delay=1):
        """Send message using async context with enhanced retry mechanism"""
        for attempt in range(max_retries):
            try:
                await self.bot.send_message(
                    chat_id=self.chat_id,
                    text=message,
                    parse_mode=ParseMode.MARKDOWN,
                    disable_web_page_preview=True
                )
                logger.info(f"Message sent to Telegram successfully on attempt {attempt + 1}")
                return True
            except TelegramError as e:
                error_msg = str(e).lower()

                # Handle specific Telegram errors
                if "pool timeout" in error_msg or "connection pool" in error_msg:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"Telegram connection pool timeout (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Telegram connection pool timeout failed after {max_retries} attempts: {e}")

                elif "rate limit" in error_msg or "too many requests" in error_msg:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt) + 5  # Extra delay for rate limits
                        logger.warning(f"Telegram rate limit hit (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Telegram rate limit error failed after {max_retries} attempts: {e}")

                elif "network" in error_msg or "timeout" in error_msg:
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"Telegram network/timeout error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Telegram network error failed after {max_retries} attempts: {e}")

                else:
                    # Other Telegram errors
                    if attempt < max_retries - 1:
                        delay = initial_delay * (2 ** attempt)
                        logger.warning(f"Telegram error (attempt {attempt + 1}/{max_retries}): {e}, retrying in {delay}s")
                        await asyncio.sleep(delay)
                        continue
                    else:
                        logger.error(f"Telegram error failed after {max_retries} attempts: {e}")

                return False

            except asyncio.TimeoutError:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Async timeout error (attempt {attempt + 1}/{max_retries}), retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Async timeout error failed after {max_retries} attempts")
                    return False

            except Exception as e:
                if attempt < max_retries - 1:
                    delay = initial_delay * (2 ** attempt)
                    logger.warning(f"Unexpected error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}: {e}, retrying in {delay}s")
                    await asyncio.sleep(delay)
                    continue
                else:
                    logger.error(f"Unexpected error failed after {max_retries} attempts: {type(e).__name__}: {e}")
                    return False

        logger.error(f"Failed to send message after {max_retries} attempts")
        return False

    def send_message(self, message):
        """Send message synchronously using a dedicated event loop"""
        try:
            # Run the async function - nest_asyncio is already applied globally
            return asyncio.run(self.send_message_async(message))
            
        except Exception as e:
            logger.error(f"Error in send_message: {e}")
            return False

    def job(self):
        """Scheduled job to generate and send trade calls and anomaly alerts"""
        try:
            if trading_signals.should_generate_signal():
                trade_calls = trading_signals.generate_trade_calls()
                if trade_calls:
                    # Send all trade calls
                    for trade_call in trade_calls:
                        self.send_message(trade_call)
                    trading_signals.update_signal_time()
                    logger.info(f"Sent {len(trade_calls)} trade calls")
                else:
                    logger.info("No trade calls generated at this time")
            else:
                logger.info("Waiting for next analysis interval")

            # Additionally, send anomaly alerts if any
            try:
                anomalies = anomaly_detection.get_anomaly_alerts(min_severity='MEDIUM', min_confidence=0.6)
                if anomalies:
                    for anomaly in anomalies:
                        message = self.format_anomaly_message(anomaly)
                        self.send_message(message)
                    logger.info(f"Sent {len(anomalies)} anomaly alerts")
            except Exception as e:
                logger.error(f"Error sending anomaly alerts: {e}")

        except Exception as e:
            logger.error(f"Error in scheduled job: {e}")

    def send_immediate_signal(self):
        """Send trade calls immediately on startup"""
        try:
            # Check if we should generate signals immediately
            if trading_signals.should_generate_signal():
                trade_calls = trading_signals.generate_trade_calls()
                if trade_calls:
                    # Send all trade calls
                    for trade_call in trade_calls:
                        self.send_message(trade_call)
                    trading_signals.update_signal_time()
                    logger.info(f"Immediate {len(trade_calls)} trade calls sent on startup")
                else:
                    logger.info("No immediate trade calls generated on startup")
            else:
                logger.info("Skipping immediate signals - within analysis interval")

            # Additionally, send anomaly alerts immediately on startup
            try:
                anomalies = anomaly_detection.get_anomaly_alerts(min_severity='MEDIUM', min_confidence=0.6)
                if anomalies:
                    for anomaly in anomalies:
                        message = self.format_anomaly_message(anomaly)
                        self.send_message(message)
                    logger.info(f"Sent {len(anomalies)} immediate anomaly alerts on startup")
            except Exception as e:
                logger.error(f"Error sending immediate anomaly alerts: {e}")

        except Exception as e:
            logger.error(f"Error sending immediate signals: {e}")

    def start_scheduler(self):
        """Run scheduler in a dedicated thread"""
        try:
            # Run immediate signal on startup
            self.send_immediate_signal()
            
            # Schedule regular analysis
            schedule.every(Config.ANALYSIS_INTERVAL).minutes.do(self.job)
            logger.info(f"Scheduler started, running every {Config.ANALYSIS_INTERVAL} minutes")

            # Keep the scheduler running
            while self.running:
                schedule.run_pending()
                time.sleep(1)
                
        except Exception as e:
            logger.error(f"Error in scheduler: {e}")

    def run(self):
        """Start the Telegram bot scheduler"""
        # Run scheduler in a separate thread to avoid blocking
        thread = threading.Thread(target=self.start_scheduler, daemon=True)
        thread.start()
        logger.info("Telegram Bot scheduler running in background")

    def stop(self):
        """Stop the Telegram bot"""
        self.running = False
        logger.info("Telegram Bot stopped")

    def format_anomaly_message(self, anomaly):
        """Format anomaly alert message for Telegram"""
        try:
            anomaly_type = anomaly.get('type', 'ANOMALY')
            severity = anomaly.get('severity', 'MEDIUM')
            description = anomaly.get('description', 'No description')
            timestamp = anomaly.get('timestamp')
            if hasattr(timestamp, 'strftime'):
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                timestamp_str = str(timestamp)
            trading_pair = anomaly.get('trading_pair', 'Multiple Pairs')
            if 'trading_pairs' in anomaly:
                trading_pair = ', '.join(anomaly['trading_pairs'])

            emoji_map = {
                'PRICE_SPIKE': '📈',
                'VOLUME_SPIKE': '🔊',
                'VOLATILITY_SPIKE': '⚡',
                'CORRELATION_BREAK': '🔗',
                'FUNDING_RATE_ANOMALY': '💰',
                'LIQUIDITY_DRYUP': '💧',
                'FLASH_CRASH': '💥',
                'PUMP_DUMP': '🚨',
                'ANOMALY': '❗'
            }
            emoji = emoji_map.get(anomaly_type, '❗')

            message = f"""
{emoji} *Anomaly Alert: {anomaly_type}* ({severity})

⏰ *Time:* {timestamp_str}
📊 *Pair(s):* {trading_pair}

💡 *Details:* {description}
            """
            return message.strip()
        except Exception as e:
            logger.error(f"Error formatting anomaly message: {e}")
            return "⚠️ Anomaly detected, details unavailable."

if __name__ == "__main__":
    bot = TelegramBot()
    bot.run()
