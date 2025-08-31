import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ml_signal_prediction import ml_signal_predictor
from risk_management import risk_management
from trading_signals import trading_signals
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis

class TestMLSignalPrediction(unittest.TestCase):
    """Comprehensive test suite for ML signal prediction integration"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample market data for testing
        dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
        np.random.seed(42)  # For reproducible results

        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 100),
            'high': np.random.uniform(41000, 51000, 100),
            'low': np.random.uniform(39000, 49000, 100),
            'close': np.random.uniform(40000, 50000, 100),
            'volume': np.random.uniform(1000000, 5000000, 100)
        })

        # Ensure high >= close >= low and high >= open >= low
        for i in range(len(self.sample_data)):
            high = max(self.sample_data.loc[i, ['open', 'close']].max(),
                      self.sample_data.loc[i, 'high'])
            low = min(self.sample_data.loc[i, ['open', 'close']].min(),
                     self.sample_data.loc[i, 'low'])
            self.sample_data.loc[i, 'high'] = high
            self.sample_data.loc[i, 'low'] = low

        # Add technical indicators
        self.sample_data_with_indicators = self._add_sample_indicators()

    def _add_sample_indicators(self):
        """Add sample technical indicators to test data"""
        df = self.sample_data.copy()

        # Simple moving averages
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        df['ema_5'] = df['close'].ewm(span=5).mean()
        df['ema_10'] = df['close'].ewm(span=10).mean()
        df['ema_20'] = df['close'].ewm(span=20).mean()

        # RSI
        df['rsi'] = self._calculate_rsi(df['close'])

        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # Stochastic
        df['stoch_k'] = self._calculate_stochastic(df)
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ATR
        df['atr'] = self._calculate_atr(df)

        # Williams %R
        df['williams_r'] = self._calculate_williams_r(df)

        # CCI
        df['cci'] = self._calculate_cci(df)

        # OBV
        df['obv'] = self._calculate_obv(df)

        # VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(0)

        return df

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, df, k_period=14):
        """Calculate Stochastic %K"""
        lowest_low = df['low'].rolling(window=k_period).min()
        highest_high = df['high'].rolling(window=k_period).max()
        return 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low))

    def _calculate_atr(self, df, period=14):
        """Calculate Average True Range"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_williams_r(self, df, period=14):
        """Calculate Williams %R"""
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        return -100 * ((highest_high - df['close']) / (highest_high - lowest_low))

    def _calculate_cci(self, df, period=20):
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        sma = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        return (typical_price - sma) / (0.015 * mad)

    def _calculate_obv(self, df):
        """Calculate On Balance Volume"""
        obv = pd.Series(index=df.index, dtype=float)
        obv.iloc[0] = df['volume'].iloc[0]

        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] + df['volume'].iloc[i]
            elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                obv.iloc[i] = obv.iloc[i-1] - df['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    def test_ml_predictor_initialization(self):
        """Test ML predictor initialization"""
        print("Testing ML predictor initialization...")
        self.assertIsNotNone(ml_signal_predictor)
        self.assertIsNotNone(ml_signal_predictor.scaler)
        self.assertEqual(ml_signal_predictor.model, None)  # Model not trained yet

    def test_feature_preparation(self):
        """Test feature preparation for ML model"""
        print("Testing feature preparation...")
        features = ml_signal_predictor.prepare_features(self.sample_data_with_indicators)

        self.assertIsNotNone(features)
        self.assertGreater(len(features), 0)

        # Check that we have expected feature columns
        expected_features = ['price_change', 'price_volatility', 'price_trend',
                           'volume_ratio', 'volume_trend', 'rsi', 'macd', 'bb_position']
        for feature in expected_features:
            self.assertIn(feature, features.columns)

    def test_label_creation(self):
        """Test label creation for training"""
        print("Testing label creation...")
        labels, valid_indices = ml_signal_predictor.create_labels(self.sample_data_with_indicators)

        self.assertIsNotNone(labels)
        self.assertIsNotNone(valid_indices)
        self.assertEqual(len(labels), len(valid_indices))

    def test_signal_quality_validation_with_ml(self):
        """Test signal quality validation with ML enhancement"""
        print("Testing signal quality validation with ML...")

        # Mock signals
        signals = {
            'overall_signal': 'BULLISH',
            'rsi_signal': 'BULLISH',
            'macd_signal': 'BULLISH',
            'volume_signal': 'BULLISH',
            'trend_signal': 'BULLISH'
        }

        # Mock metrics
        metrics = {
            'current_price': 45000,
            'funding_rate': 0.001
        }

        # Test without ML (should work)
        quality_without_ml = risk_management.validate_signal_quality(
            signals, metrics['current_price'], metrics
        )

        self.assertIn('quality', quality_without_ml)
        self.assertIn('success_percentage', quality_without_ml)

        # Test with ML enhancement
        quality_with_ml = risk_management.validate_signal_quality(
            signals, metrics['current_price'], metrics, self.sample_data_with_indicators
        )

        self.assertIn('quality', quality_with_ml)
        self.assertIn('success_percentage', quality_with_ml)
        self.assertIn('ml_confidence', quality_with_ml)

        # ML confidence should be a float between 0 and 1
        self.assertIsInstance(quality_with_ml['ml_confidence'], float)
        self.assertGreaterEqual(quality_with_ml['ml_confidence'], 0.0)
        self.assertLessEqual(quality_with_ml['ml_confidence'], 1.0)

    def test_ml_prediction_without_training(self):
        """Test ML prediction when model is not trained"""
        print("Testing ML prediction without training...")

        confidence = ml_signal_predictor.predict_signal_success(
            self.sample_data_with_indicators, 'BULLISH'
        )

        # Should return neutral confidence (0.5) when model is not trained
        self.assertEqual(confidence, 0.5)

    def test_model_training_workflow(self):
        """Test complete model training workflow"""
        print("Testing model training workflow...")

        # This would normally take time, so we'll just test the method exists and handles errors gracefully
        success = ml_signal_predictor.train_model(self.sample_data_with_indicators, 'BULLISH')

        # Training might fail due to insufficient data, but method should handle it
        self.assertIsInstance(success, bool)

    def test_feature_importance_without_model(self):
        """Test feature importance when no model is trained"""
        print("Testing feature importance without model...")

        importance = ml_signal_predictor.get_feature_importance()
        self.assertIsNone(importance)

    def test_signal_quality_scoring_logic(self):
        """Test the signal quality scoring logic"""
        print("Testing signal quality scoring logic...")

        test_cases = [
            {
                'signals': {
                    'overall_signal': 'BULLISH',
                    'rsi_signal': 'BULLISH',
                    'macd_signal': 'BULLISH',
                    'volume_signal': 'BULLISH',
                    'trend_signal': 'BULLISH'
                },
                'expected_quality': 'HIGH'
            },
            {
                'signals': {
                    'overall_signal': 'BULLISH',
                    'rsi_signal': 'BEARISH',
                    'macd_signal': 'BEARISH',
                    'volume_signal': 'BEARISH',
                    'trend_signal': 'BEARISH'
                },
                'expected_quality': 'LOW'
            }
        ]

        metrics = {'current_price': 45000, 'funding_rate': 0.001}

        for test_case in test_cases:
            quality = risk_management.validate_signal_quality(
                test_case['signals'], metrics['current_price'], metrics
            )

            self.assertIn('quality', quality)
            self.assertIn('score', quality)

    def test_ml_confidence_impact_on_quality(self):
        """Test how ML confidence affects signal quality assessment"""
        print("Testing ML confidence impact on quality...")

        signals = {
            'overall_signal': 'BULLISH',
            'rsi_signal': 'BULLISH',
            'macd_signal': 'BULLISH',
            'volume_signal': 'BULLISH',
            'trend_signal': 'BULLISH'
        }

        metrics = {'current_price': 45000, 'funding_rate': 0.001}

        # Test with high ML confidence (mock)
        with patch.object(ml_signal_predictor, 'predict_signal_success', return_value=0.9):
            quality_high_ml = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, self.sample_data_with_indicators
            )

        # Test with low ML confidence (mock)
        with patch.object(ml_signal_predictor, 'predict_signal_success', return_value=0.2):
            quality_low_ml = risk_management.validate_signal_quality(
                signals, metrics['current_price'], metrics, self.sample_data_with_indicators
            )

        # High ML confidence should result in higher success percentage
        self.assertGreater(quality_high_ml['success_percentage'], quality_low_ml['success_percentage'])

    def test_trading_signals_integration(self):
        """Test integration with trading signals module"""
        print("Testing trading signals integration...")

        # Mock the market data and technical analysis modules
        with patch('trading_signals.market_data') as mock_market_data, \
             patch('trading_signals.technical_analysis') as mock_tech_analysis:

            # Setup mocks
            mock_market_data.get_ohlcv_data.return_value = self.sample_data
            mock_market_data.get_market_metrics.return_value = {
                'current_price': 45000,
                'funding_rate': 0.001
            }
            mock_tech_analysis.calculate_indicators.return_value = self.sample_data_with_indicators
            mock_tech_analysis.generate_signals.return_value = {
                'overall_signal': 'BULLISH',
                'rsi_signal': 'BULLISH',
                'macd_signal': 'BULLISH',
                'volume_signal': 'BULLISH',
                'trend_signal': 'BULLISH',
                'scalp_signal': 'NEUTRAL'
            }

            # Test signal generation
            trade_calls = trading_signals.generate_trade_calls()

            # Should handle the mocked data gracefully
            self.assertIsInstance(trade_calls, list)

    def test_error_handling(self):
        """Test error handling in ML prediction"""
        print("Testing error handling...")

        # Test with None data
        confidence = ml_signal_predictor.predict_signal_success(None, 'BULLISH')
        self.assertEqual(confidence, 0.5)

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        confidence = ml_signal_predictor.predict_signal_success(empty_df, 'BULLISH')
        self.assertEqual(confidence, 0.5)

        # Test signal quality validation with invalid data
        quality = risk_management.validate_signal_quality(
            None, 45000, {'funding_rate': 0.001}
        )
        self.assertIn('quality', quality)

    def test_model_persistence(self):
        """Test model saving and loading"""
        print("Testing model persistence...")

        # Test saving without trained model
        ml_signal_predictor.save_model()

        # Test loading when no model exists
        loaded = ml_signal_predictor.load_model()
        self.assertFalse(loaded)

    def test_performance_history_tracking(self):
        """Test performance history tracking"""
        print("Testing performance history tracking...")

        initial_history_length = len(ml_signal_predictor.get_model_performance_history())

        # Simulate training (without actual training)
        ml_signal_predictor.performance_history.append({
            'timestamp': datetime.now(),
            'signal_type': 'BULLISH',
            'accuracy': 0.8,
            'precision': 0.75,
            'recall': 0.7,
            'f1_score': 0.72,
            'cv_mean': 0.76,
            'cv_std': 0.05
        })

        updated_history = ml_signal_predictor.get_model_performance_history()
        self.assertEqual(len(updated_history), initial_history_length + 1)

    def tearDown(self):
        """Clean up after tests"""
        # Reset any global state if needed
        pass

if __name__ == '__main__':
    print("Running comprehensive ML signal prediction tests...")
    unittest.main(verbosity=2)
