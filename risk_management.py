import numpy as np
import os
from loguru import logger
from config import Config
from ml_signal_prediction import ml_signal_predictor
from anomaly_detection import anomaly_detection

class RiskManagement:
    def __init__(self):
        self.account_balance = 10000  # Default starting balance in USDT
        logger.info("Risk Management module initialized")

    def calculate_position_size(self, current_price, trading_pair=None, risk_percentage=None, leverage=None):
        """Calculate position size based on risk management rules"""
        if risk_percentage is None:
            risk_percentage = Config.RISK_PERCENTAGE
        
        try:
            # Calculate risk amount
            risk_amount = self.account_balance * (risk_percentage / 100)
            
            # Get leverage - use provided leverage or get from config
            if leverage is None:
                leverage = Config.DEFAULT_LEVERAGE
            
            # Calculate position size considering leverage
            position_size = (risk_amount * leverage) / current_price
            
            # Apply maximum position size limit
            max_position_usdt = Config.MAX_POSITION_SIZE
            max_position_size = max_position_usdt / current_price
            
            position_size = min(position_size, max_position_size)
            
            crypto_name = trading_pair.replace('USDT', '') if trading_pair else Config.TRADING_PAIR.replace('USDT', '')
            logger.info(f"Calculated position size: {position_size:.6f} {crypto_name} with {leverage}x leverage")
            return round(position_size, 6)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return None

    def calculate_stop_loss(self, entry_price, signal_type, atr):
        """Calculate stop loss price based on ATR volatility"""
        try:
            if atr is None:
                logger.error("ATR is required for stop loss calculation")
                return None
                
            if signal_type == 'BULLISH':
                # For long positions, stop loss below entry (1.5x ATR)
                stop_loss = entry_price - (atr * 1.5)
            elif signal_type == 'BEARISH':
                # For short positions, stop loss above entry (1.5x ATR)
                stop_loss = entry_price + (atr * 1.5)
            else:
                return None
            
            logger.info(f"Calculated stop loss: {stop_loss:.2f}")
            return round(stop_loss, 2)
            
        except Exception as e:
            logger.error(f"Error calculating stop loss: {e}")
            return None

    def calculate_take_profit(self, entry_price, signal_type, stop_loss):
        """Calculate take profit levels based on stop loss"""
        try:
            if stop_loss is None:
                logger.error("Stop loss is required for take profit calculation")
                return None
                
            if signal_type == 'BULLISH':
                # For long positions, 1:2 risk-reward ratio
                risk = entry_price - stop_loss
                take_profit = entry_price + (risk * 2)
            elif signal_type == 'BEARISH':
                # For short positions, 1:2 risk-reward ratio
                risk = stop_loss - entry_price
                take_profit = entry_price - (risk * 2)
            else:
                return None
            
            logger.info(f"Calculated take profit: {take_profit:.2f}")
            return round(take_profit, 2)
            
        except Exception as e:
            logger.error(f"Error calculating take profit: {e}")
            return None

    def calculate_trailing_stop(self, current_price, entry_price, signal_type, highest_price=None, lowest_price=None):
        """Calculate trailing stop loss"""
        try:
            if signal_type == 'BULLISH':
                # For long positions, trail below current price
                if highest_price:
                    trailing_stop = highest_price * (1 - Config.TRAILING_STOP_PERCENTAGE / 100)
                else:
                    trailing_stop = current_price * (1 - Config.TRAILING_STOP_PERCENTAGE / 100)
            elif signal_type == 'BEARISH':
                # For short positions, trail above current price
                if lowest_price:
                    trailing_stop = lowest_price * (1 + Config.TRAILING_STOP_PERCENTAGE / 100)
                else:
                    trailing_stop = current_price * (1 + Config.TRAILING_STOP_PERCENTAGE / 100)
            else:
                return None
            
            return round(trailing_stop, 2)
            
        except Exception as e:
            logger.error(f"Error calculating trailing stop: {e}")
            return None

    def validate_signal_quality(self, signals, current_price, metrics, df_with_indicators=None, trading_pair=None):
        """Validate if the signal meets quality criteria with ML enhancement and anomaly detection"""
        quality_score = 0
        reasons = []
        ml_confidence = 0.5  # Default neutral confidence

        # Check for market anomalies first
        anomaly_alerts = []
        if trading_pair:
            try:
                # Scan for anomalies in the current trading pair
                anomalies = anomaly_detection.scan_for_anomalies([trading_pair])
                if anomalies:
                    # Filter high-confidence anomalies
                    high_confidence_anomalies = [
                        anomaly for anomaly in anomalies
                        if anomaly.get('confidence', 0) > 0.7
                    ]

                    if high_confidence_anomalies:
                        anomaly_alerts = high_confidence_anomalies
                        # Reduce quality score for anomalous market conditions
                        quality_score -= 2
                        reasons.append(f"Market anomalies detected: {len(high_confidence_anomalies)} high-confidence alerts")
                        logger.warning(f"Anomalies detected for {trading_pair}: {len(high_confidence_anomalies)} alerts")
            except Exception as e:
                logger.warning(f"Anomaly detection failed for {trading_pair}: {e}")

        # Get ML prediction confidence if data is available
        if df_with_indicators is not None and signals.get('overall_signal') in ['BULLISH', 'BEARISH']:
            try:
                ml_confidence = ml_signal_predictor.predict_signal_success(
                    df_with_indicators, signals.get('overall_signal')
                )
                logger.info(f"ML confidence for {signals.get('overall_signal')} signal: {ml_confidence:.4f}")
            except Exception as e:
                logger.warning(f"ML prediction failed: {e}")
                ml_confidence = 0.5

        # RSI validation
        if signals.get('rsi_signal') == signals.get('overall_signal'):
            quality_score += 1
        else:
            reasons.append("RSI contradicts overall signal")

        # MACD validation
        if signals.get('macd_signal') == signals.get('overall_signal'):
            quality_score += 1
        else:
            reasons.append("MACD contradicts overall signal")

        # Volume validation
        if signals.get('volume_signal') == signals.get('overall_signal'):
            quality_score += 1
        else:
            reasons.append("Volume contradicts overall signal")

        # Trend validation
        if signals.get('trend_signal') == signals.get('overall_signal'):
            quality_score += 1
        else:
            reasons.append("Trend contradicts overall signal")

        # Funding rate consideration (for futures)
        funding_rate = metrics.get('funding_rate', 0)
        if funding_rate > 0.01 and signals.get('overall_signal') == 'BULLISH':
            quality_score -= 1
            reasons.append("High positive funding rate for long position")
        elif funding_rate < -0.01 and signals.get('overall_signal') == 'BEARISH':
            quality_score -= 1
            reasons.append("High negative funding rate for short position")

        # ML confidence adjustment
        if ml_confidence > 0.7:
            quality_score += 1  # Boost quality for high ML confidence
            reasons.append(f"ML confidence: {ml_confidence:.1%}")
        elif ml_confidence < 0.3:
            quality_score -= 1  # Reduce quality for low ML confidence
            reasons.append(f"ML confidence: {ml_confidence:.1%}")

        # Overall quality assessment
        if quality_score >= 4:
            quality = 'HIGH'
            success_percentage = 80  # 80% success rate for high quality signals
        elif quality_score >= 3:
            quality = 'HIGH'
            success_percentage = 75  # 75% success rate for high quality signals
        elif quality_score >= 2:
            quality = 'MEDIUM'
            success_percentage = 65  # 65% success rate for medium quality signals
        elif quality_score >= 1:
            quality = 'MEDIUM'
            success_percentage = 60  # 60% success rate for medium quality signals
        else:
            quality = 'LOW'
            success_percentage = 45  # 45% success rate for low quality signals

        # Adjust success percentage based on ML confidence
        if ml_confidence > 0.7:
            success_percentage += 10  # Boost success rate for high ML confidence
        elif ml_confidence < 0.3:
            success_percentage -= 10  # Reduce success rate for low ML confidence

        # Adjust success percentage based on additional factors
        if signals.get('overall_signal') == 'BULLISH' and funding_rate > 0.01:
            success_percentage -= 10  # Reduce success rate for long positions with high funding
        elif signals.get('overall_signal') == 'BEARISH' and funding_rate < -0.01:
            success_percentage -= 10  # Reduce success rate for short positions with negative funding

        # Ensure success percentage is within reasonable bounds
        success_percentage = max(25, min(90, success_percentage))

        return {
            'quality': quality,
            'score': quality_score,
            'success_percentage': success_percentage,
            'ml_confidence': ml_confidence,
            'reasons': reasons
        }

    def update_account_balance(self, pnl):
        """Update account balance after a trade"""
        self.account_balance += pnl
        logger.info(f"Account balance updated: {self.account_balance:.2f} USDT")

# Singleton instance
risk_management = RiskManagement()
