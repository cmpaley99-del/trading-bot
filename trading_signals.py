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

class TradingSignals:
    def __init__(self):
        # Load last signal time from database
        self.last_signal_time = database.get_last_signal_time()
        logger.info(f"Trading Signals module initialized. Last signal time: {self.last_signal_time}")

    def generate_trade_calls(self):
        """Generate trade calls for all configured trading pairs"""
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
        """Generate multiple trade calls for a single trading pair"""
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
            
            # Validate signal quality with ML enhancement and anomaly detection
            signal_quality = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, df_with_indicators, trading_pair
            )
            
            # Only proceed with high or medium quality signals
            if signal_quality['quality'] == 'LOW':
                logger.info(f"Signal quality too low for {trading_pair}, skipping trade calls")
                return []
            
            current_price = metrics['current_price']
            
            # Calculate appropriate leverage based on RL model if available, else fallback
            try:
                from trading_signals_rl import rl_leverage_manager
                leverage = rl_leverage_manager.get_optimal_leverage(trading_pair, df_with_indicators.iloc[-1])
            except ImportError:
                leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, trading_pair)
            except Exception as e:
                logger.error(f"Error getting RL leverage for {trading_pair}: {e}")
                leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, trading_pair)
            
            # Generate trade calls for different signal types
            trade_calls = self._generate_trade_calls_for_signals(
                trading_pair, signals, current_price, leverage, 
                df_with_indicators, signal_quality
            )
            
            return trade_calls
            
        except Exception as e:
            logger.error(f"Error generating multiple trade calls for {trading_pair}: {e}")
            return []

    def _generate_single_trade_call(self, trading_pair):
        """Generate a trade call for a single trading pair"""
        try:
            # Fetch market data for specific pair
            df = market_data.get_ohlcv_data(trading_pair)
            if df is None:
                logger.error(f"Failed to fetch market data for {trading_pair}")
                return None
            
            # Calculate technical indicators
            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                logger.error(f"Failed to calculate indicators for {trading_pair}")
                return None
            
            # Generate signals
            signals = technical_analysis.generate_signals(df_with_indicators)
            if signals is None:
                logger.error(f"Failed to generate signals for {trading_pair}")
                return None
            
            # Get current market metrics
            metrics = market_data.get_market_metrics(trading_pair)
            if metrics is None:
                logger.error(f"Failed to get market metrics for {trading_pair}")
                return None
            
            # Validate signal quality with ML enhancement
            signal_quality = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, df_with_indicators
            )
            
            # Only proceed with high or medium quality signals
            if signal_quality['quality'] == 'LOW':
                logger.info(f"Signal quality too low for {trading_pair}, skipping trade call")
                return None
            
            current_price = metrics['current_price']
            signal_type = signals['overall_signal']
            
            # Calculate appropriate leverage based on market analysis
            leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, trading_pair)
            
            # Calculate position details with dynamic leverage
            position_size = risk_management.calculate_position_size(current_price, trading_pair, leverage=leverage)
            
            # Only calculate stop loss and take profit for BULLISH/BEARISH signals
            if signal_type in ['BULLISH', 'BEARISH']:
                stop_loss = risk_management.calculate_stop_loss(
                    current_price, signal_type, df_with_indicators['atr'].iloc[-1]
                )
                take_profit = risk_management.calculate_take_profit(
                    current_price, signal_type, stop_loss
                )
                
                # Prepare trade call message
                trade_call = self._prepare_trade_message(
                    trading_pair, signal_type, current_price, position_size, 
                    stop_loss, take_profit, signal_quality, leverage
                )
                
                logger.info(f"Generated trade call for {trading_pair}: {signal_type} signal with {leverage}x leverage")
                
                return trade_call
            else:
                logger.info(f"No trade call for {trading_pair}: {signal_type} signal (neutral)")
                return None
            
        except Exception as e:
            logger.error(f"Error generating trade call for {trading_pair}: {e}")
            return None

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
                       f"{signal_type} signal with {leverage}x leverage")
            
            return trade_call
            
        except Exception as e:
            logger.error(f"Error creating {operation_type.lower()} trade call for {trading_pair}: {e}")
            return None

    def _prepare_trade_message(self, trading_pair, signal_type, entry_price, position_size, 
                             stop_loss, take_profit, signal_quality, leverage=None, operation_type="REGULAR"):
        """Prepare trade message for Telegram with operation type"""
        
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
        
        # Operation type indicator - only show for scalping
        operation_indicator = ""
        if operation_type == "SCALPING":
            operation_indicator = "⚡ *SCALPING*"
        
        # Create clean message - removed position size as requested
        message = f"""
{emoji} *{action} {crypto_name}* {operation_indicator}

💰 *Entry:* ${entry_price:,.2f}
⚡ *Leverage:* {leverage}x

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

# Singleton instance
trading_signals = TradingSignals()
