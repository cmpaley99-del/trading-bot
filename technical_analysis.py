import pandas as pd
import numpy as np
import talib
from loguru import logger
from config import Config
from pattern_recognition import pattern_recognition

class TechnicalAnalysis:
    def __init__(self):
        self.indicators = {}
        logger.info("Technical Analysis module initialized")

    def calculate_indicators(self, df):
        """Calculate all technical indicators including scalping indicators"""
        if df is None or len(df) < 50:
            logger.warning("Insufficient data for technical analysis")
            return None
        
        try:
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=Config.RSI_PERIOD)
            
            # Calculate MACD
            macd, macd_signal, macd_hist = talib.MACD(
                df['close'], 
                fastperiod=Config.MACD_FAST,
                slowperiod=Config.MACD_SLOW,
                signalperiod=Config.MACD_SIGNAL
            )
            df['macd'] = macd
            df['macd_signal'] = macd_signal
            df['macd_hist'] = macd_hist
            
            # Calculate Bollinger Bands
            upper_bb, middle_bb, lower_bb = talib.BBANDS(
                df['close'], 
                timeperiod=Config.BB_PERIOD,
                nbdevup=Config.BB_STD,
                nbdevdn=Config.BB_STD
            )
            df['bb_upper'] = upper_bb
            df['bb_middle'] = middle_bb
            df['bb_lower'] = lower_bb
            df['bb_percentage'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']) * 100
            
            # Calculate Moving Averages
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_50'] = talib.SMA(df['close'], timeperiod=50)
            df['ema_12'] = talib.EMA(df['close'], timeperiod=12)
            df['ema_26'] = talib.EMA(df['close'], timeperiod=26)
            
            # Scalping-specific EMAs
            df['ema_5'] = talib.EMA(df['close'], timeperiod=5)
            df['ema_10'] = talib.EMA(df['close'], timeperiod=10)
            df['ema_20'] = talib.EMA(df['close'], timeperiod=20)
            
            # Calculate Stochastic Oscillator
            stoch_k, stoch_d = talib.STOCH(
                df['high'], df['low'], df['close'],
                fastk_period=14, slowk_period=3, slowd_period=3
            )
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            
            # Calculate ADX (Average Directional Index)
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Calculate ATR (Average True Range) for volatility
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Calculate OBV (On Balance Volume)
            df['obv'] = talib.OBV(df['close'], df['volume'])
            
            # Calculate Volume indicators
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Advanced indicators for scalping
            # CCI (Commodity Channel Index)
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            
            # Williams %R
            df['williams_r'] = talib.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # VWAP (Volume Weighted Average Price)
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            
            # Price patterns (basic detection)
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['doji'] = talib.CDLDOJI(df['open'], df['high'], df['low'], df['close'])
            df['morning_star'] = talib.CDLMORNINGSTAR(df['open'], df['high'], df['low'], df['close'])
            df['evening_star'] = talib.CDLEVENINGSTAR(df['open'], df['high'], df['low'], df['close'])
            
            logger.info("All technical indicators calculated successfully")
            return df
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {e}")
            return None

    def generate_signals(self, df):
        """Generate trading signals based on technical analysis including scalping signals"""
        if df is None or len(df) < 2:
            return None
        
        current = df.iloc[-1]
        previous = df.iloc[-2]
        
        signals = {
            'rsi_signal': self._get_rsi_signal(current['rsi']),
            'macd_signal': self._get_macd_signal(current['macd'], current['macd_signal']),
            'bb_signal': self._get_bb_signal(current['close'], current['bb_upper'], current['bb_lower']),
            'trend_signal': self._get_trend_signal(current, previous),
            'volume_signal': self._get_volume_signal(current, df['volume'].mean()),
            'pattern_signal': self._get_pattern_signal(current),
            'ema_scalp_signal': self._get_ema_scalp_signal(current, previous),
            'momentum_signal': self._get_momentum_signal(current),
            'vwap_signal': self._get_vwap_signal(current['close'], current['vwap']),
            'overall_signal': 'NEUTRAL',
            'scalp_signal': 'NEUTRAL'
        }
        
        # Determine overall signal
        signals['overall_signal'] = self._determine_overall_signal(signals)
        
        # Generate scalping-specific signal
        signals['scalp_signal'] = self._generate_scalp_signal(signals, current)
        
        return signals

    def _get_rsi_signal(self, rsi):
        if rsi > 70:
            return 'BEARISH'
        elif rsi < 30:
            return 'BULLISH'
        else:
            return 'NEUTRAL'

    def _get_macd_signal(self, macd, macd_signal):
        if macd > macd_signal and macd > 0:
            return 'BULLISH'
        elif macd < macd_signal and macd < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_bb_signal(self, price, bb_upper, bb_lower):
        if price >= bb_upper:
            return 'BEARISH'
        elif price <= bb_lower:
            return 'BULLISH'
        else:
            return 'NEUTRAL'

    def _get_trend_signal(self, current, previous):
        if current['sma_20'] > current['sma_50'] and current['close'] > current['sma_20']:
            return 'BULLISH'
        elif current['sma_20'] < current['sma_50'] and current['close'] < current['sma_20']:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_volume_signal(self, current, avg_volume):
        if current['volume'] > avg_volume * 1.5:
            return 'BULLISH' if current['close'] > current['open'] else 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_pattern_signal(self, current):
        if current['hammer'] > 0 or current['engulfing'] > 0:
            return 'BULLISH'
        elif current['engulfing'] < 0:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_ema_scalp_signal(self, current, previous):
        """Generate EMA crossover scalping signal"""
        # EMA 5 > EMA 10 > EMA 20 = Bullish
        if current['ema_5'] > current['ema_10'] > current['ema_20']:
            return 'BULLISH'
        # EMA 5 < EMA 10 < EMA 20 = Bearish
        elif current['ema_5'] < current['ema_10'] < current['ema_20']:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _get_momentum_signal(self, current):
        """Generate momentum-based scalping signal"""
        # CCI above +100 = overbought, below -100 = oversold
        if current['cci'] > 100:
            return 'BEARISH'
        elif current['cci'] < -100:
            return 'BULLISH'
        
        # Williams %R above -20 = overbought, below -80 = oversold
        if current['williams_r'] > -20:
            return 'BEARISH'
        elif current['williams_r'] < -80:
            return 'BULLISH'
        
        return 'NEUTRAL'

    def _get_vwap_signal(self, price, vwap):
        """Generate VWAP-based signal"""
        # Price above VWAP = Bullish, below VWAP = Bearish
        if price > vwap * 1.01:  # 1% above VWAP
            return 'BULLISH'
        elif price < vwap * 0.99:  # 1% below VWAP
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _generate_scalp_signal(self, signals, current):
        """Generate scalping-specific signal based on multiple factors"""
        scalp_signals = [
            signals['ema_scalp_signal'],
            signals['momentum_signal'],
            signals['vwap_signal'],
            signals['volume_signal']
        ]
        
        bullish_count = sum(1 for signal in scalp_signals if signal == 'BULLISH')
        bearish_count = sum(1 for signal in scalp_signals if signal == 'BEARISH')
        
        # Strong scalping signal requires at least 2 confirming signals
        if bullish_count >= 2:
            return 'BULLISH'
        elif bearish_count >= 2:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def _determine_overall_signal(self, signals):
        bullish_count = sum(1 for signal in signals.values() if signal == 'BULLISH')
        bearish_count = sum(1 for signal in signals.values() if signal == 'BEARISH')
        
        if bullish_count >= 3:
            return 'BULLISH'
        elif bearish_count >= 3:
            return 'BEARISH'
        else:
            return 'NEUTRAL'

    def calculate_appropriate_leverage(self, df, trading_pair):
        """Calculate intelligent leverage based on comprehensive market analysis"""
        if df is None or len(df) < 50:
            return 10  # Default leverage
            
        try:
            current = df.iloc[-1]
            
            # Get volatility metrics
            atr = current['atr']
            current_price = current['close']
            
            # Calculate volatility percentage (ATR as percentage of price)
            volatility_pct = (atr / current_price) * 100
            
            # Calculate historical volatility (standard deviation of returns)
            returns = df['close'].pct_change().dropna()
            historical_volatility = returns.std() * 100 * np.sqrt(252)  # Annualized
            
            # Get trend strength from ADX
            adx_strength = current['adx']
            
            # Calculate momentum indicators
            rsi = current['rsi']
            cci = current['cci']
            williams_r = current['williams_r']
            
            # Calculate trend alignment
            trend_alignment = self._calculate_trend_alignment(current)
            
            # Calculate volume strength
            volume_strength = self._calculate_volume_strength(current, df['volume'].mean())
            
            # Calculate signal confluence
            signal_confluence = self._calculate_signal_confluence(df)
            
            # Determine base leverage based on volatility (more aggressive for scalping)
            if volatility_pct < 0.3:  # Very low volatility
                base_leverage = 25
            elif volatility_pct < 0.6:  # Low volatility
                base_leverage = 20
            elif volatility_pct < 1.0:  # Medium volatility
                base_leverage = 15
            elif volatility_pct < 2.0:  # High volatility
                base_leverage = 12
            else:  # Very high volatility
                base_leverage = 8
            
            # Adjust leverage based on trend strength (more aggressive for strong trends)
            if adx_strength > 30:  # Very strong trend
                trend_adjustment = 1.4
            elif adx_strength > 20:  # Strong trend
                trend_adjustment = 1.2
            elif adx_strength > 15:  # Moderate trend
                trend_adjustment = 1.0
            else:  # Weak trend
                trend_adjustment = 0.7
            
            # Adjust leverage based on momentum (higher for extreme conditions)
            momentum_adjustment = 1.0
            if (rsi < 25 or rsi > 75) or (cci < -150 or cci > 150) or (williams_r < -85 or williams_r > -15):
                momentum_adjustment = 1.3  # Strong momentum
            
            # Adjust leverage based on trend alignment
            leverage_adjustment = trend_adjustment * momentum_adjustment * trend_alignment * volume_strength * signal_confluence
            
            # Calculate final leverage
            final_leverage = int(base_leverage * leverage_adjustment)
            
            # Apply safety limits with higher maximum for scalping
            final_leverage = max(5, min(final_leverage, 30))
            
            logger.info(f"Intelligent leverage for {trading_pair}: {final_leverage}x "
                       f"(Vol: {volatility_pct:.2f}%, ADX: {adx_strength:.1f}, "
                       f"Hist Vol: {historical_volatility:.1f}%, RSI: {rsi:.1f})")
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"Error calculating intelligent leverage for {trading_pair}: {e}")
            return 10  # Fallback to default leverage

    def _calculate_trend_alignment(self, current):
        """Calculate how well different timeframes align"""
        # Check if all EMAs are aligned (5 > 10 > 20 or 5 < 10 < 20)
        ema_aligned = (current['ema_5'] > current['ema_10'] > current['ema_20']) or \
                     (current['ema_5'] < current['ema_10'] < current['ema_20'])
        
        # Check if price is above/below key moving averages
        price_vs_sma = current['close'] > current['sma_20'] > current['sma_50']
        
        if ema_aligned and price_vs_sma:
            return 1.2  # Strong trend alignment
        elif ema_aligned:
            return 1.1  # Good trend alignment
        else:
            return 0.9  # Weak or conflicting trends

    def _calculate_volume_strength(self, current, avg_volume):
        """Calculate volume strength indicator"""
        volume_ratio = current['volume'] / avg_volume
        
        if volume_ratio > 2.0:
            return 1.3  # Very high volume
        elif volume_ratio > 1.5:
            return 1.2  # High volume
        elif volume_ratio > 1.0:
            return 1.1  # Above average volume
        else:
            return 0.9  # Below average volume

    def _calculate_signal_confluence(self, df):
        """Calculate signal confluence across multiple indicators"""
        signals = self.generate_signals(df)
        if not signals:
            return 1.0
            
        bullish_count = sum(1 for signal in signals.values() if signal == 'BULLISH')
        bearish_count = sum(1 for signal in signals.values() if signal == 'BEARISH')
        
        if abs(bullish_count - bearish_count) >= 3:
            return 1.3  # Strong signal confluence
        elif abs(bullish_count - bearish_count) >= 2:
            return 1.2  # Good signal confluence
        elif abs(bullish_count - bearish_count) >= 1:
            return 1.1  # Some signal confluence
        else:
            return 1.0  # Mixed signals

# Singleton instance
technical_analysis = TechnicalAnalysis()
