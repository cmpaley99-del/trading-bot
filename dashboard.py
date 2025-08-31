#!/usr/bin/env python3
"""
Real-time Trading Bot Dashboard
Provides web interface for monitoring trading signals, performance, and account status
"""

import asyncio
from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import threading
import time
from datetime import datetime, timedelta
from loguru import logger
import json
import os

from config import Config
from market_data import market_data
from trading_signals import trading_signals
from risk_management import risk_management
from technical_analysis import technical_analysis
from telegram_bot import TelegramBot
from backtest import backtester
from anomaly_detection import anomaly_detection

class TradingDashboard:
    def __init__(self):
        self.app = Flask(__name__,
                        template_folder='templates',
                        static_folder='static')
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")

        # Dashboard data
        self.dashboard_data = {
            'signals': [],
            'performance': {},
            'account': {},
            'market_data': {},
            'anomalies': [],
            'anomaly_summary': {},
            'system_status': 'running'
        }

        # Telegram bot for notifications
        self.telegram_bot = TelegramBot()

        # Setup routes
        self.setup_routes()

        # Setup socket events
        self.setup_socket_events()

        logger.info("Trading Dashboard initialized")

    def setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            return render_template('dashboard.html')

        @self.app.route('/api/dashboard-data')
        def get_dashboard_data():
            return jsonify(self.dashboard_data)

        @self.app.route('/api/signals')
        def get_signals():
            return jsonify({
                'signals': self.dashboard_data['signals'][-20:],  # Last 20 signals
                'count': len(self.dashboard_data['signals'])
            })

        @self.app.route('/api/performance')
        def get_performance():
            return jsonify(self.dashboard_data['performance'])

        @self.app.route('/api/market-data')
        def get_market_data():
            return jsonify(self.dashboard_data['market_data'])

        @self.app.route('/api/send-signal', methods=['POST'])
        def send_signal():
            """Manually trigger signal generation and sending"""
            try:
                if trading_signals.should_generate_signal():
                    trade_calls = trading_signals.generate_trade_calls()
                    if trade_calls:
                        sent_count = 0
                        for trade_call in trade_calls:
                            if self.telegram_bot.send_message(trade_call):
                                sent_count += 1
                        trading_signals.update_signal_time()

                        return jsonify({
                            'success': True,
                            'message': f'Sent {sent_count} trade signals',
                            'signals': trade_calls
                        })
                    else:
                        return jsonify({
                            'success': False,
                            'message': 'No trade signals generated'
                        })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Signal generation not due yet'
                    })
            except Exception as e:
                logger.error(f"Error sending manual signal: {e}")
                return jsonify({
                    'success': False,
                    'message': str(e)
                })

        @self.app.route('/api/backtest', methods=['POST'])
        def run_backtest():
            """Run backtest with specified parameters"""
            try:
                data = request.get_json()
                start_date = datetime.fromisoformat(data.get('start_date'))
                end_date = datetime.fromisoformat(data.get('end_date'))
                pairs = data.get('pairs', Config.TRADING_PAIRS)

                # Run backtest asynchronously
                asyncio.run(self._run_backtest_async(start_date, end_date, pairs))

                return jsonify({
                    'success': True,
                    'message': 'Backtest completed',
                    'results': backtester.performance_metrics
                })
            except Exception as e:
                logger.error(f"Error running backtest: {e}")
                return jsonify({
                    'success': False,
                    'message': str(e)
                })

        @self.app.route('/api/anomalies')
        def get_anomalies():
            """Get current anomaly data"""
            return jsonify({
                'anomalies': self.dashboard_data['anomalies'][-20:],  # Last 20 anomalies
                'summary': self.dashboard_data['anomaly_summary'],
                'count': len(self.dashboard_data['anomalies'])
            })

        @self.app.route('/api/scan-anomalies', methods=['POST'])
        def scan_anomalies():
            """Manually trigger anomaly scan"""
            try:
                anomalies = anomaly_detection.scan_for_anomalies()

                return jsonify({
                    'success': True,
                    'message': f'Found {len(anomalies)} anomalies',
                    'anomalies': anomalies[-10:]  # Return last 10 anomalies
                })
            except Exception as e:
                logger.error(f"Error scanning for anomalies: {e}")
                return jsonify({
                    'success': False,
                    'message': str(e)
                })

        @self.app.route('/api/anomaly-alerts')
        def get_anomaly_alerts():
            """Get current anomaly alerts"""
            try:
                alerts = anomaly_detection.get_anomaly_alerts()
                return jsonify({
                    'alerts': alerts,
                    'count': len(alerts)
                })
            except Exception as e:
                logger.error(f"Error getting anomaly alerts: {e}")
                return jsonify({
                    'alerts': [],
                    'count': 0,
                    'error': str(e)
                })

    def setup_socket_events(self):
        """Setup Socket.IO events for real-time updates"""

        @self.socketio.on('connect')
        def handle_connect():
            logger.info("Client connected to dashboard")
            emit('status', {'message': 'Connected to trading dashboard'})

        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info("Client disconnected from dashboard")

        @self.socketio.on('request_update')
        def handle_update_request():
            """Send current dashboard data to client"""
            self.update_dashboard_data()
            emit('dashboard_update', self.dashboard_data)

    async def _run_backtest_async(self, start_date, end_date, pairs):
        """Run backtest asynchronously"""
        metrics = await backtester.run_backtest(start_date, end_date, pairs)
        return metrics

    def update_dashboard_data(self):
        """Update dashboard data with latest information"""
        try:
            # Update market data
            market_info = {}
            for pair in Config.TRADING_PAIRS:
                try:
                    price = market_data.get_current_price(pair)
                    metrics = market_data.get_market_metrics(pair)
                    if price and metrics:
                        market_info[pair] = {
                            'price': price,
                            'change_24h': metrics.get('price_change_24h', 0),
                            'volume_24h': metrics.get('24h_volume', 0),
                            'funding_rate': metrics.get('funding_rate', 0)
                        }
                except Exception as e:
                    logger.error(f"Error updating market data for {pair}: {e}")

            self.dashboard_data['market_data'] = market_info

            # Update performance metrics (from recent backtest or live trading)
            if hasattr(backtester, 'performance_metrics') and backtester.performance_metrics:
                self.dashboard_data['performance'] = backtester.performance_metrics
            else:
                # Default performance data
                self.dashboard_data['performance'] = {
                    'total_return': 0,
                    'total_trades': 0,
                    'win_rate': 0,
                    'current_balance': getattr(Config, 'INITIAL_BALANCE', 10000)
                }

            # Update anomaly data
            try:
                # Get recent anomalies
                recent_anomalies = anomaly_detection.anomaly_history[-20:]  # Last 20 anomalies
                self.dashboard_data['anomalies'] = [
                    {
                        'timestamp': anomaly['timestamp'].isoformat() if hasattr(anomaly['timestamp'], 'isoformat') else str(anomaly['timestamp']),
                        'type': anomaly['type'],
                        'trading_pair': anomaly.get('trading_pair', 'Multiple'),
                        'severity': anomaly['severity'],
                        'confidence': anomaly.get('confidence', 0.5),
                        'description': anomaly['description']
                    }
                    for anomaly in recent_anomalies
                ]

                # Get anomaly summary
                self.dashboard_data['anomaly_summary'] = anomaly_detection.get_anomaly_summary(hours=24)

            except Exception as e:
                logger.error(f"Error updating anomaly data: {e}")
                self.dashboard_data['anomalies'] = []
                self.dashboard_data['anomaly_summary'] = {}

            # Update system status
            self.dashboard_data['system_status'] = 'running'

            # Emit update to connected clients
            self.socketio.emit('dashboard_update', self.dashboard_data)

        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")

    def add_signal(self, signal_data):
        """Add new signal to dashboard"""
        signal_entry = {
            'timestamp': datetime.now().isoformat(),
            'pair': signal_data.get('pair', 'Unknown'),
            'type': signal_data.get('type', 'Unknown'),
            'price': signal_data.get('price', 0),
            'leverage': signal_data.get('leverage', 1),
            'confidence': signal_data.get('confidence', 'Medium')
        }

        self.dashboard_data['signals'].append(signal_entry)

        # Keep only last 100 signals
        if len(self.dashboard_data['signals']) > 100:
            self.dashboard_data['signals'] = self.dashboard_data['signals'][-100:]

        # Emit signal update
        self.socketio.emit('new_signal', signal_entry)
        logger.info(f"Added new signal to dashboard: {signal_entry}")

    def start_background_updates(self):
        """Start background thread for periodic dashboard updates"""
        def update_loop():
            while True:
                try:
                    self.update_dashboard_data()
                    time.sleep(30)  # Update every 30 seconds
                except Exception as e:
                    logger.error(f"Error in dashboard update loop: {e}")
                    time.sleep(60)  # Wait longer on error

        thread = threading.Thread(target=update_loop, daemon=True)
        thread.start()
        logger.info("Dashboard background update thread started")

    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting Trading Dashboard on {host}:{port}")

        # Start background updates
        self.start_background_updates()

        # Run the Flask-SocketIO server
        self.socketio.run(self.app, host=host, port=port, debug=debug)

# Global dashboard instance
dashboard = TradingDashboard()

if __name__ == "__main__":
    dashboard.run()
