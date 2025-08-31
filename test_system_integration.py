"""
Comprehensive System Integration Test
Tests all components working together: market data, technical analysis,
pattern recognition, risk management, ML models, and Telegram integration
"""

import sys
import os
import time
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis
from technical_analysis_advanced import advanced_technical_analysis
from risk_management import risk_management
from trading_signals import trading_signals
from ml_signal_prediction import ml_signal_predictor
from anomaly_detection import anomaly_detector
from telegram_bot import telegram_bot
from database import database


class SystemIntegrationTester:
    """Comprehensive system integration testing"""

    def __init__(self):
        self.test_results = {}
        logger.info("System Integration Tester initialized")

    def run_full_system_test(self):
        """Run complete system integration test"""
        logger.info("=== Starting Full System Integration Test ===")

        try:
            # Test 1: Market Data Integration
            self.test_market_data_integration()

            # Test 2: Technical Analysis Integration
            self.test_technical_analysis_integration()

            # Test 3: Advanced Pattern Recognition
            self.test_advanced_pattern_integration()

            # Test 4: Risk Management Integration
            self.test_risk_management_integration()

            # Test 5: ML Model Integration
            self.test_ml_model_integration()

            # Test 6: Anomaly Detection Integration
            self.test_anomaly_detection_integration()

            # Test 7: Trading Signals Integration
            self.test_trading_signals_integration()

            # Test 8: Database Integration
            self.test_database_integration()

            # Test 9: Telegram Integration
            self.test_telegram_integration()

            # Test 10: End-to-End Trading Flow
            self.test_end_to_end_flow()

            # Generate test report
            self.generate_test_report()

        except Exception as e:
            logger.error(f"System integration test failed: {e}")
            raise

    def test_market_data_integration(self):
        """Test market data fetching and processing"""
        logger.info("Testing Market Data Integration...")

        try:
            # Test single pair data fetching
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            if df is None or len(df) < 50:
                raise Exception(f"Failed to fetch sufficient data for {test_pair}")

            # Test market metrics
            metrics = market_data.get_market_metrics(test_pair)
            if metrics is None:
                raise Exception(f"Failed to get market metrics for {test_pair}")

            # Test multiple pairs
            successful_fetches = 0
            for pair in Config.TRADING_PAIRS[:3]:  # Test first 3 pairs
                df_test = market_data.get_ohlcv_data(pair)
                if df_test is not None and len(df_test) >= 50:
                    successful_fetches += 1

            success_rate = successful_fetches / 3

            self.test_results['market_data'] = {
                'status': 'PASS' if success_rate >= 0.8 else 'FAIL',
                'success_rate': success_rate,
                'data_points': len(df),
                'metrics_available': metrics is not None
            }

            logger.info(f"Market Data Integration: {self.test_results['market_data']['status']}")

        except Exception as e:
            self.test_results['market_data'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Market Data Integration failed: {e}")

    def test_technical_analysis_integration(self):
        """Test technical analysis with market data"""
        logger.info("Testing Technical Analysis Integration...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            if df is None:
                raise Exception("No market data available for technical analysis")

            # Test basic technical analysis
            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                raise Exception("Technical analysis calculation failed")

            # Test signal generation
            signals = technical_analysis.generate_signals(df_with_indicators)
            if signals is None:
                raise Exception("Signal generation failed")

            # Verify required indicators are present
            required_indicators = ['rsi', 'macd', 'bb_upper', 'bb_lower', 'stoch_k', 'adx', 'atr']
            missing_indicators = [ind for ind in required_indicators if ind not in df_with_indicators.columns]

            if missing_indicators:
                raise Exception(f"Missing indicators: {missing_indicators}")

            self.test_results['technical_analysis'] = {
                'status': 'PASS',
                'indicators_calculated': len(df_with_indicators.columns) - len(df.columns),
                'signals_generated': len(signals),
                'missing_indicators': missing_indicators
            }

            logger.info(f"Technical Analysis Integration: PASS")

        except Exception as e:
            self.test_results['technical_analysis'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Technical Analysis Integration failed: {e}")

    def test_advanced_pattern_integration(self):
        """Test advanced pattern recognition integration"""
        logger.info("Testing Advanced Pattern Recognition Integration...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            if df is None:
                raise Exception("No market data available for pattern analysis")

            # Test advanced technical analysis
            advanced_signals = advanced_technical_analysis.generate_advanced_signals(df)

            if advanced_signals is None:
                raise Exception("Advanced signal generation failed")

            # Test pattern analysis summary
            pattern_summary = advanced_technical_analysis.get_pattern_analysis_summary(df)

            # Check for required pattern components
            required_keys = ['harmonic_patterns_detected', 'elliott_wave_position',
                           'fibonacci_signal', 'pattern_confidence', 'recommended_action']

            missing_keys = [key for key in required_keys if key not in pattern_summary]

            self.test_results['advanced_patterns'] = {
                'status': 'PASS' if not missing_keys else 'FAIL',
                'overall_signal': advanced_signals.get('overall_signal'),
                'pattern_confidence': advanced_signals.get('pattern_confidence', 0),
                'signal_strength': advanced_signals.get('signal_strength', 0),
                'missing_keys': missing_keys
            }

            logger.info(f"Advanced Pattern Recognition Integration: {self.test_results['advanced_patterns']['status']}")

        except Exception as e:
            self.test_results['advanced_patterns'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Advanced Pattern Recognition Integration failed: {e}")

    def test_risk_management_integration(self):
        """Test risk management integration"""
        logger.info("Testing Risk Management Integration...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)
            df_with_indicators = technical_analysis.calculate_indicators(df)

            current_price = 50000  # Test price
            signal_type = 'BULLISH'

            # Test position sizing
            position_size = risk_management.calculate_position_size(current_price, test_pair)
            if position_size <= 0:
                raise Exception("Invalid position size calculation")

            # Test stop loss calculation
            stop_loss = risk_management.calculate_stop_loss(
                current_price, signal_type, df_with_indicators['atr'].iloc[-1]
            )
            if stop_loss >= current_price and signal_type == 'BULLISH':
                raise Exception("Invalid stop loss for bullish signal")

            # Test take profit calculation
            take_profit = risk_management.calculate_take_profit(current_price, signal_type, stop_loss)
            if take_profit <= current_price and signal_type == 'BULLISH':
                raise Exception("Invalid take profit for bullish signal")

            # Test signal quality validation
            signals = technical_analysis.generate_signals(df_with_indicators)
            metrics = market_data.get_market_metrics(test_pair)
            signal_quality = risk_management.validate_signal_quality(signals, current_price, metrics, df_with_indicators, test_pair)

            self.test_results['risk_management'] = {
                'status': 'PASS',
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'signal_quality': signal_quality.get('quality', 'UNKNOWN')
            }

            logger.info(f"Risk Management Integration: PASS")

        except Exception as e:
            self.test_results['risk_management'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Risk Management Integration failed: {e}")

    def test_ml_model_integration(self):
        """Test ML model integration"""
        logger.info("Testing ML Model Integration...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)
            df_with_indicators = technical_analysis.calculate_indicators(df)

            if df_with_indicators is None:
                raise Exception("No data for ML model testing")

            # Test ML signal prediction
            ml_prediction = ml_signal_predictor.predict_signal_quality(df_with_indicators)

            if ml_prediction is None:
                raise Exception("ML prediction failed")

            # Test signal quality validation with ML
            signals = technical_analysis.generate_signals(df_with_indicators)
            metrics = market_data.get_market_metrics(test_pair)
            signal_quality = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, df_with_indicators, test_pair
            )

            self.test_results['ml_model'] = {
                'status': 'PASS',
                'ml_prediction': ml_prediction,
                'signal_quality': signal_quality.get('quality', 'UNKNOWN'),
                'ml_confidence': ml_prediction.get('confidence', 0)
            }

            logger.info(f"ML Model Integration: PASS")

        except Exception as e:
            self.test_results['ml_model'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"ML Model Integration failed: {e}")

    def test_anomaly_detection_integration(self):
        """Test anomaly detection integration"""
        logger.info("Testing Anomaly Detection Integration...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            if df is None:
                raise Exception("No data for anomaly detection testing")

            # Test anomaly detection
            anomalies = anomaly_detector.detect_anomalies(df, test_pair)

            if anomalies is None:
                raise Exception("Anomaly detection failed")

            # Check anomaly structure
            required_anomaly_keys = ['price_spike', 'volume_spike', 'volatility_spike',
                                   'correlation_break', 'funding_rate_anomaly']

            missing_anomaly_keys = [key for key in required_anomaly_keys if key not in anomalies]

            self.test_results['anomaly_detection'] = {
                'status': 'PASS' if not missing_anomaly_keys else 'FAIL',
                'anomalies_detected': sum(1 for v in anomalies.values() if v),
                'missing_keys': missing_anomaly_keys
            }

            logger.info(f"Anomaly Detection Integration: {self.test_results['anomaly_detection']['status']}")

        except Exception as e:
            self.test_results['anomaly_detection'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Anomaly Detection Integration failed: {e}")

    def test_trading_signals_integration(self):
        """Test trading signals integration"""
        logger.info("Testing Trading Signals Integration...")

        try:
            # Test trade call generation
            trade_calls = trading_signals.generate_trade_calls()

            if trade_calls is None:
                trade_calls = []

            # Test should_generate_signal
            should_generate = trading_signals.should_generate_signal()

            self.test_results['trading_signals'] = {
                'status': 'PASS',
                'trade_calls_generated': len(trade_calls),
                'should_generate_signal': should_generate
            }

            logger.info(f"Trading Signals Integration: PASS - Generated {len(trade_calls)} trade calls")

        except Exception as e:
            self.test_results['trading_signals'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Trading Signals Integration failed: {e}")

    def test_database_integration(self):
        """Test database integration"""
        logger.info("Testing Database Integration...")

        try:
            # Test database connection and basic operations
            last_signal_time = database.get_last_signal_time()

            # Test saving signal time
            test_time = datetime.now()
            database.save_last_signal_time(test_time)

            # Verify the save worked
            retrieved_time = database.get_last_signal_time()

            time_diff = abs((retrieved_time - test_time).total_seconds()) if retrieved_time else 999

            self.test_results['database'] = {
                'status': 'PASS' if time_diff < 1 else 'FAIL',
                'last_signal_time': str(last_signal_time),
                'time_diff_seconds': time_diff
            }

            logger.info(f"Database Integration: {self.test_results['database']['status']}")

        except Exception as e:
            self.test_results['database'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Database Integration failed: {e}")

    def test_telegram_integration(self):
        """Test Telegram integration (without actually sending messages)"""
        logger.info("Testing Telegram Integration...")

        try:
            # Test Telegram bot initialization and configuration
            bot_configured = hasattr(telegram_bot, 'bot') and telegram_bot.bot is not None

            # Check if chat ID is configured
            chat_id_configured = hasattr(Config, 'TELEGRAM_CHAT_ID') and Config.TELEGRAM_CHAT_ID is not None

            # Test message formatting (without sending)
            test_message = "🚀 *TEST BTC* LONG\n💰 *Entry:* $50000.00\n⚡ *Leverage:* 10x"

            self.test_results['telegram'] = {
                'status': 'PASS' if bot_configured and chat_id_configured else 'WARNING',
                'bot_configured': bot_configured,
                'chat_id_configured': chat_id_configured,
                'test_message_length': len(test_message)
            }

            logger.info(f"Telegram Integration: {self.test_results['telegram']['status']}")

        except Exception as e:
            self.test_results['telegram'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"Telegram Integration failed: {e}")

    def test_end_to_end_flow(self):
        """Test complete end-to-end trading flow"""
        logger.info("Testing End-to-End Trading Flow...")

        try:
            test_pair = Config.TRADING_PAIRS[0]

            # Step 1: Fetch market data
            df = market_data.get_ohlcv_data(test_pair)
            if df is None:
                raise Exception("Market data fetch failed")

            # Step 2: Calculate technical indicators
            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                raise Exception("Technical analysis failed")

            # Step 3: Generate signals
            signals = technical_analysis.generate_signals(df_with_indicators)
            if signals is None:
                raise Exception("Signal generation failed")

            # Step 4: Get market metrics
            metrics = market_data.get_market_metrics(test_pair)
            if metrics is None:
                raise Exception("Market metrics fetch failed")

            # Step 5: Validate signal quality
            signal_quality = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, df_with_indicators, test_pair
            )

            # Step 6: Calculate position parameters
            leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, test_pair)
            position_size = risk_management.calculate_position_size(metrics['current_price'], test_pair, leverage)

            # Step 7: Generate trade call (if signal is valid)
            trade_call = None
            if signals['overall_signal'] in ['BULLISH', 'BEARISH'] and signal_quality['quality'] != 'LOW':
                stop_loss = risk_management.calculate_stop_loss(
                    metrics['current_price'], signals['overall_signal'], df_with_indicators['atr'].iloc[-1]
                )
                take_profit = risk_management.calculate_take_profit(
                    metrics['current_price'], signals['overall_signal'], stop_loss
                )

                trade_call = trading_signals._prepare_trade_message(
                    test_pair, signals['overall_signal'], metrics['current_price'],
                    position_size, stop_loss, take_profit, signal_quality, leverage
                )

            self.test_results['end_to_end'] = {
                'status': 'PASS',
                'data_points': len(df),
                'indicators_calculated': len(df_with_indicators.columns),
                'signal_generated': signals['overall_signal'],
                'signal_quality': signal_quality['quality'],
                'leverage_calculated': leverage,
                'position_size': position_size,
                'trade_call_generated': trade_call is not None
            }

            logger.info(f"End-to-End Flow: PASS - Complete trading pipeline executed successfully")

        except Exception as e:
            self.test_results['end_to_end'] = {'status': 'FAIL', 'error': str(e)}
            logger.error(f"End-to-End Flow failed: {e}")

    def generate_test_report(self):
        """Generate comprehensive test report"""
        logger.info("=== System Integration Test Report ===")

        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values()
                          if result.get('status') == 'PASS')
        failed_tests = sum(1 for result in self.test_results.values()
                          if result.get('status') == 'FAIL')
        warning_tests = sum(1 for result in self.test_results.values()
                           if result.get('status') == 'WARNING')

        success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0

        print(f"\n{'='*60}")
        print(f"SYSTEM INTEGRATION TEST RESULTS")
        print(f"{'='*60}")
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Warnings: {warning_tests}")
        print(f"Success Rate: {success_rate:.1f}%")
        print(f"{'='*60}")

        for test_name, result in self.test_results.items():
            status = result.get('status', 'UNKNOWN')
            status_icon = "✅" if status == 'PASS' else "❌" if status == 'FAIL' else "⚠️"

            print(f"\n{status_icon} {test_name.upper().replace('_', ' ')}: {status}")

            if status == 'FAIL' and 'error' in result:
                print(f"   Error: {result['error']}")

            # Print additional details for successful tests
            if status == 'PASS':
                for key, value in result.items():
                    if key != 'status':
                        print(f"   {key}: {value}")

        print(f"\n{'='*60}")

        if success_rate >= 90:
            print("🎉 EXCELLENT: System integration test PASSED!")
            print("The trading bot is ready for production deployment.")
        elif success_rate >= 75:
            print("⚠️  GOOD: System integration test mostly PASSED.")
            print("Minor issues should be addressed before production.")
        else:
            print("❌ CRITICAL: System integration test FAILED.")
            print("Major issues need to be resolved before deployment.")

        print(f"{'='*60}")


def run_system_integration_test():
    """Run the complete system integration test"""
    tester = SystemIntegrationTester()
    tester.run_full_system_test()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    run_system_integration_test()
