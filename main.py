#!/usr/bin/env python3
"""
Main entry point for the Trading Bot
Runs the complete trading system with dashboard
"""

import asyncio
import threading
import time
from loguru import logger
import os

from config import Config
from market_data import market_data
from trading_signals import trading_signals
from risk_management import risk_management
from technical_analysis import technical_analysis
from telegram_bot import TelegramBot
from dashboard import dashboard
from anomaly_detection import anomaly_detection

class TradingBot:
    def __init__(self):
        self.telegram_bot = TelegramBot()
        self.running = True
        logger.info("Trading Bot initialized")

    def start_signal_generation(self):
        """Start the signal generation loop"""
        def signal_loop():
            while self.running:
                try:
                    if trading_signals.should_generate_signal():
                        trade_calls = trading_signals.generate_trade_calls()
                        if trade_calls:
                            sent_count = 0
                            for trade_call in trade_calls:
                                if self.telegram_bot.send_message(trade_call):
                                    sent_count += 1
                                    # Add signal to dashboard
                                    dashboard.add_signal({
                                        'pair': trade_call.get('pair', 'Unknown'),
                                        'type': trade_call.get('type', 'Unknown'),
                                        'price': trade_call.get('price', 0),
                                        'leverage': trade_call.get('leverage', 1),
                                        'confidence': 'High'
                                    })
                            trading_signals.update_signal_time()
                            logger.info(f"Generated and sent {sent_count} trade signals")
                        else:
                            logger.info("No trade signals generated")
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in signal generation loop: {e}")
                    time.sleep(300)  # Wait 5 minutes on error

        thread = threading.Thread(target=signal_loop, daemon=True)
        thread.start()
        logger.info("Signal generation thread started")

    def start_anomaly_monitoring(self):
        """Start anomaly monitoring"""
        def anomaly_loop():
            while self.running:
                try:
                    anomalies = anomaly_detection.scan_for_anomalies()
                    if anomalies:
                        logger.info(f"Detected {len(anomalies)} anomalies")
                        # Send anomaly alerts via telegram
                        for anomaly in anomalies[-5:]:  # Send last 5 anomalies
                            alert_msg = f"🚨 ANOMALY ALERT: {anomaly['type']} on {anomaly.get('trading_pair', 'Multiple')}\n"
                            alert_msg += f"Severity: {anomaly['severity']}\n"
                            alert_msg += f"Description: {anomaly['description']}"
                            self.telegram_bot.send_message(alert_msg)
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Error in anomaly monitoring: {e}")
                    time.sleep(600)  # Wait 10 minutes on error

        thread = threading.Thread(target=anomaly_loop, daemon=True)
        thread.start()
        logger.info("Anomaly monitoring thread started")

    def run(self):
        """Run the complete trading system"""
        logger.info("Starting Trading Bot System...")

        # Start signal generation
        self.start_signal_generation()

        # Start anomaly monitoring
        self.start_anomaly_monitoring()

        # Start the dashboard
        try:
            port = int(os.environ.get('PORT', 5000))
            dashboard.run(host='0.0.0.0', port=port, debug=False)
        except KeyboardInterrupt:
            logger.info("Trading Bot stopped by user")
            self.running = False
        except Exception as e:
            logger.error(f"Error running trading bot: {e}")
            self.running = False

if __name__ == "__main__":
    bot = TradingBot()
    bot.run()
