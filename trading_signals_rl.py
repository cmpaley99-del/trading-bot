from datetime import datetime
import pandas as pd
import os
from loguru import logger
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis
from risk_management import risk_management
from database import database
from ml_signal_prediction import ml_signal_predictor
from rl_leverage_complete import rl_leverage_manager


class TradingSignalsRL:
    """Enhanced Trading Signals with RL-based leverage optimization"""

    def __init__(self):
        # Load last signal time from database
        self.last_signal_time = database.get_last_signal_time()
        self.use_rl_leverage = True  # Enable RL leverage by default
        logger.info(f"Trading Signals RL module initialized. Last signal time: {self.last_signal_time}")

    def generate_trade_calls(self):
        """Generate trade calls for all configured trading pairs with RL leverage"""
        trade_calls = []

        for trading_pair in Config.TRADING_PAIRS:
            try:
                # Generate multiple trade calls per pair if multiple opportunities exist
                trade_calls_for_pair = self._generate_multiple_trade_calls(trading_pair)
                if trade_calls_for_pair:
                    trade_calls.extend(trade_calls_for_pair)
            except Exception as e:
                logger.error(f"Error generating trade call for {trading_pair}: {e}")

        return trade_calls

    def _generate_multiple_trade_calls(self, trading_pair):
        """Generate multiple trade calls for a single trading pair with RL leverage"""
        trade_calls = []

        try:
            # Fetch market data for specific pair
            df = market_data.get_ohlcv_data(trading_pair)
            if df is None:
                logger.error(f"Failed to fetch market data for {trading_pair}")
                return []

            # Calculate technical indicators
            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                logger.error(f"Failed to calculate indicators for {trading_pair}")
                return []

            # Generate signals
            signals = technical_analysis.generate_signals(df_with_indicators)
            if signals is None:
                logger.error(f"Failed to generate signals for {trading_pair}")
                return []

            # Get current market metrics
            metrics = market_data.get_market_metrics(trading_pair)
            if metrics is None:
                logger.error(f"Failed to get market metrics for {trading_pair}")
                return []

            # Validate signal quality with ML enhancement
            signal_quality = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, df_with_indicators
            )

            # Only proceed with high or medium quality signals
            if signal_quality['quality'] == 'LOW':
                logger.info(f"Signal quality too low for {trading_pair}, skipping trade calls")
                return []

            current_price = metrics['current_price']

            # Calculate leverage using RL optimization
            leverage = self._calculate_optimal_leverage(
                trading_pair, df_with_indicators, current_price, signals
            )

            # Generate trade calls for different signal types
            trade_calls = self._generate_trade_calls_for_signals(
                trading_pair, signals, current_price, leverage,
                df_with_indicators, signal_quality
            )

            return trade_calls

        except Exception as e:
            logger.error(f"Error generating multiple trade calls for {trading_pair}: {e}")
            return []

    def _calculate_optimal_leverage(self, trading_pair, df_with_indicators, current_price, signals):
        """Calculate optimal leverage using RL or fallback to rule-based"""
        try:
            if self.use_rl_leverage:
                # Prepare market data for RL model
                current_market_data = df_with_indicators.iloc[-1].to_dict()
                current_market_data['close'] = current_price

                # Get RL-optimized leverage
                rl_leverage = rl_leverage_manager.get_optimal_leverage(
                    trading_pair, current_market_data
                )

                # Get rule-based leverage for comparison
                rule_leverage = technical_analysis.calculate_appropriate_leverage(
                    df_with_indicators, trading_pair
                )

                # Use RL leverage but apply safety bounds based on rule-based
                final_leverage = min(rl_leverage, rule_leverage + 5)  # Don't exceed rule-based by more than 5x
                final_leverage = max(final_leverage, rule_leverage - 3)  # Don't go below rule-based by more than 3x
                final_leverage = max(5, min(final_leverage, 30))  # Apply absolute bounds

                logger.info(f"{trading_pair} - RL: {rl_leverage}x, Rule: {rule_leverage}x, Final: {final_leverage}x")

                return final_leverage

            else:
                # Fallback to rule-based leverage
                return technical_analysis.calculate_appropriate_leverage(df_with_indicators, trading_pair)

        except Exception as e:
            logger.error(f"Error calculating RL leverage for {trading_pair}: {e}")
            # Fallback to rule-based leverage
            return technical_analysis.calculate_appropriate_leverage(df_with_indicators, trading_pair)

    def _generate_trade_calls_for_signals(self, trading_pair, signals, current_price, leverage,
                                        df_with_indicators, signal_quality):
        """Generate trade calls for different signal types (regular and scalping)"""
        trade_calls = []

        # Check for regular signal (overall signal)
        if signals['overall_signal'] in ['BULLISH', 'BEARISH']:
            regular_call = self._create_trade_call(
                trading_pair, signals['overall_signal'], current_price, leverage,
                df_with_indicators, signal_quality, operation_type="REGULAR"
            )
            if regular_call:
                trade_calls.append(regular_call)

        # Check for scalping signal (separate from overall signal)
        if (signals['scalp_signal'] in ['BULLISH', 'BEARISH'] and
            signals['scalp_signal'] != signals['overall_signal']):
            scalping_call = self._create_trade_call(
                trading_pair, signals['scalp_signal'], current_price, leverage,
                df_with_indicators, signal_quality, operation_type="SCALPING"
            )
            if scalping_call:
                trade_calls.append(scalping_call)

        return trade_calls

    def _create_trade_call(self, trading_pair, signal_type, current_price, leverage,
                         df_with_indicators, signal_quality, operation_type="REGULAR"):
        """Create a trade call with specified operation type"""
        try:
            # Calculate position details with dynamic leverage
            position_size = risk_management.calculate_position_size(current_price, trading_pair, leverage=leverage)

            # Calculate stop loss and take profit
            stop_loss = risk_management.calculate_stop_loss(
                current_price, signal_type, df_with_indicators['atr'].iloc[-1]
            )
            take_profit = risk_management.calculate_take_profit(
                current_price, signal_type, stop_loss
            )

            # Prepare trade call message
            trade_call = self._prepare_trade_message(
                trading_pair, signal_type, current_price, position_size,
                stop_loss, take_profit, signal_quality, leverage, operation_type
            )

            logger.info(f"Generated {operation_type.lower()} trade call for {trading_pair}: "
                       f"{signal_type} signal with RL-optimized {leverage}x leverage")

            return trade_call

        except Exception as e:
            logger.error(f"Error creating {operation_type.lower()} trade call for {trading_pair}: {e}")
            return None

    def _prepare_trade_message(self, trading_pair, signal_type, entry_price, position_size,
                             stop_loss, take_profit, signal_quality, leverage=None, operation_type="REGULAR"):
        """Prepare trade message for Telegram with operation type and RL leverage indicator"""

        # Determine action based on signal type
        if signal_type == 'BULLISH':
            emoji = "🚀"
            action = "LONG"
        elif signal_type == 'BEARISH':
            emoji = "📉"
            action = "SHORT"
        else:
            return None  # Skip neutral signals

        # Get cryptocurrency name
        crypto_name = trading_pair.replace('USDT', '')

        # Use provided leverage or get from config
        if leverage is None:
            leverage = Config.DEFAULT_LEVERAGE

        # Operation type indicator
        operation_indicator = ""
        if operation_type == "SCALPING":
            operation_indicator = "⚡ *SCALPING*"

        # RL leverage indicator
        rl_indicator = "🤖" if self.use_rl_leverage else ""

        # Create clean message
        message = f"""
{emoji} *{action} {crypto_name}* {operation_indicator} {rl_indicator}

💰 *Entry:* ${entry_price:,.2f}
⚡ *Leverage:* {leverage}x (RL-Optimized)

🛑 *Stop Loss:* ${stop_loss:,.2f}
🎯 *Take Profit:* ${take_profit:,.2f}

🔍 *Quality:* {signal_quality['quality']} ({signal_quality['success_percentage']}% success rate)
⏰ *Time:* {datetime.now().strftime('%H:%M:%S')}

💡 *Action:* {action} at market
        """

        return message.strip()

    def should_generate_signal(self):
        """Check if we should generate a new signal based on time and conditions"""
        if self.last_signal_time is None:
            return True

        # Check time interval
        time_since_last = (datetime.now() - self.last_signal_time).total_seconds() / 60
        if time_since_last < Config.ANALYSIS_INTERVAL:
            return False

        return True

    def update_signal_time(self):
        """Update the last signal time in database"""
        self.last_signal_time = datetime.now()
        database.save_last_signal_time(self.last_signal_time)

    def toggle_rl_leverage(self, enabled=True):
        """Enable or disable RL leverage optimization"""
        self.use_rl_leverage = enabled
        logger.info(f"RL leverage optimization {'enabled' if enabled else 'disabled'}")

    def get_leverage_comparison(self, trading_pair):
        """Get comparison between RL and rule-based leverage for analysis"""
        try:
            # Fetch market data
            df = market_data.get_ohlcv_data(trading_pair)
            if df is None:
                return None

            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                return None

            current_price = market_data.get_current_price(trading_pair)
            if current_price is None:
                return None

            # Get both leverage calculations
            rl_leverage = rl_leverage_manager.get_optimal_leverage(
                trading_pair, df_with_indicators.iloc[-1].to_dict()
            )
            rule_leverage = technical_analysis.calculate_appropriate_leverage(
                df_with_indicators, trading_pair
            )

            return {
                'trading_pair': trading_pair,
                'rl_leverage': rl_leverage,
                'rule_leverage': rule_leverage,
                'difference': rl_leverage - rule_leverage,
                'current_price': current_price,
                'timestamp': datetime.now()
            }

        except Exception as e:
            logger.error(f"Error getting leverage comparison for {trading_pair}: {e}")
            return None


# Singleton instance
trading_signals_rl = TradingSignalsRL()
