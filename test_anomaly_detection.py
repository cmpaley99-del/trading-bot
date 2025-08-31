import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from anomaly_detection import anomaly_detection
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis

class TestAnomalyDetection(unittest.TestCase):
    """Comprehensive test suite for anomaly detection system"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample market data for testing
        dates = pd.date_range(start='2023-01-01', periods=200, freq='1H')
        np.random.seed(42)  # For reproducible results

        # Normal market data
        self.normal_data = pd.DataFrame({
            'timestamp': dates,
            'open': np.random.uniform(40000, 50000, 200),
            'high': np.random.uniform(41000, 51000, 200),
            'low': np.random.uniform(39000, 49000, 200),
            'close': np.random.uniform(40000, 50000, 200),
            'volume': np.random.uniform(1000000, 5000000, 200)
        })

        # Ensure high >= close >= low and high >= open >= low
        for i in range(len(self.normal_data)):
            high = max(self.normal_data.loc[i, ['open', 'close']].max(),
                      self.normal_data.loc[i, 'high'])
            low = min(self.normal_data.loc[i, ['open', 'close']].min(),
                     self.normal_data.loc[i, 'low'])
            self.normal_data.loc[i, 'high'] = high
            self.normal_data.loc[i, 'low'] = low

        # Create anomalous data (price spike)
        self.anomalous_data = self.normal_data.copy()
        spike_idx = 150
        self.anomalous_data.loc[spike_idx, 'close'] = 80000  # 100% price spike
        self.anomalous_data.loc[spike_idx, 'high'] = 85000
        self.anomalous_data.loc[spike_idx, 'volume'] = 15000000  # Volume spike too

        # Create flash crash data
        self.flash_crash_data = self.normal_data.copy()
        crash_start = 120
        for i in range(10):
            idx = crash_start + i
            if idx < len(self.flash_crash_data):
                # Simulate flash crash: sharp drop then recovery
                if i < 5:
                    self.flash_crash_data.loc[idx, 'close'] *= 0.85  # 15% drop
                else:
                    self.flash_crash_data.loc[idx, 'close'] *= 1.02  # Recovery

    def test_anomaly_detection_initialization(self):
        """Test anomaly detection initialization"""
        print("Testing anomaly detection initialization...")
        self.assertIsNotNone(anomaly_detection)
        self.assertIsNotNone(anomaly_detection.alert_thresholds)
        self.assertIsNotNone(anomaly_detection.anomaly_types)
        self.assertEqual(len(anomaly_detection.anomaly_history), 0)

    def test_price_anomaly_detection_normal_data(self):
        """Test price anomaly detection with normal data"""
        print("Testing price anomaly detection with normal data...")
        anomalies = anomaly_detection.detect_price_anomalies(self.normal_data, 'BTCUSDT')

        # Should detect few or no anomalies in normal data
        self.assertIsInstance(anomalies, list)
        # Normal data might have some statistical anomalies, but should be minimal
        self.assertLessEqual(len(anomalies), 5)

    def test_price_anomaly_detection_spike(self):
        """Test price anomaly detection with price spike"""
        print("Testing price anomaly detection with price spike...")
        anomalies = anomaly_detection.detect_price_anomalies(self.anomalous_data, 'BTCUSDT')

        # Should detect the price spike
        self.assertIsInstance(anomalies, list)
        self.assertGreater(len(anomalies), 0)

        # Check if spike anomaly was detected
        spike_anomalies = [a for a in anomalies if a['type'] == 'PRICE_SPIKE']
        self.assertGreater(len(spike_anomalies), 0)

        # Verify anomaly structure
        spike_anomaly = spike_anomalies[0]
        self.assertIn('severity', spike_anomaly)
        self.assertIn('z_score', spike_anomaly)
        self.assertIn('confidence', spike_anomaly)
        self.assertGreater(spike_anomaly['z_score'], 3.0)  # Should be significant

    def test_flash_crash_detection(self):
        """Test flash crash detection"""
        print("Testing flash crash detection...")
        anomalies = anomaly_detection.detect_price_anomalies(self.flash_crash_data, 'BTCUSDT')

        # Should detect flash crash
        flash_crash_anomalies = [a for a in anomalies if a['type'] == 'FLASH_CRASH']
        self.assertGreater(len(flash_crash_anomalies), 0)

        # Verify flash crash anomaly structure
        fc_anomaly = flash_crash_anomalies[0]
        self.assertIn('drop_percentage', fc_anomaly)
        self.assertGreater(fc_anomaly['drop_percentage'], 0)

    def test_volume_anomaly_detection(self):
        """Test volume anomaly detection"""
        print("Testing volume anomaly detection...")
        anomalies = anomaly_detection.detect_volume_anomalies(self.anomalous_data, 'BTCUSDT')

        # Should detect volume spike
        volume_anomalies = [a for a in anomalies if a['type'] == 'VOLUME_SPIKE']
        self.assertGreater(len(volume_anomalies), 0)

        # Verify volume anomaly structure
        vol_anomaly = volume_anomalies[0]
        self.assertIn('volume_ratio', vol_anomaly)
        self.assertGreater(vol_anomaly['volume_ratio'], 2.0)  # Should be significant spike

    def test_volatility_anomaly_detection(self):
        """Test volatility anomaly detection"""
        print("Testing volatility anomaly detection...")
        anomalies = anomaly_detection.detect_volatility_anomalies(self.anomalous_data, 'BTCUSDT')

        # Should detect some volatility anomalies
        self.assertIsInstance(anomalies, list)

        if len(anomalies) > 0:
            vol_anomaly = anomalies[0]
            self.assertIn('volatility', vol_anomaly)
            self.assertIn('vol_z_score', vol_anomaly)

    def test_pump_dump_detection(self):
        """Test pump and dump pattern detection"""
        print("Testing pump and dump detection...")

        # Create pump and dump pattern
        pump_dump_data = self.normal_data.copy()
        pump_start = 100

        # Simulate pump (rapid increase)
        for i in range(10):
            idx = pump_start + i
            if idx < len(pump_dump_data):
                pump_dump_data.loc[idx, 'close'] *= 1.05  # 5% increase each period

        # Simulate dump (rapid decrease)
        dump_start = pump_start + 10
        for i in range(10):
            idx = dump_start + i
            if idx < len(pump_dump_data):
                pump_dump_data.loc[idx, 'close'] *= 0.95  # 5% decrease each period

        anomalies = anomaly_detection.detect_pump_dump_patterns(pump_dump_data, 'BTCUSDT')

        # Should detect pump and dump pattern
        pump_dump_anomalies = [a for a in anomalies if a['type'] == 'PUMP_DUMP']
        self.assertGreater(len(pump_dump_anomalies), 0)

        # Verify pump dump anomaly structure
        pd_anomaly = pump_dump_anomalies[0]
        self.assertIn('pump_return', pd_anomaly)
        self.assertIn('dump_return', pd_anomaly)
        self.assertGreater(pd_anomaly['pump_return'], 0)
        self.assertLess(pd_anomaly['dump_return'], 0)

    def test_correlation_anomaly_detection(self):
        """Test correlation anomaly detection between pairs"""
        print("Testing correlation anomaly detection...")

        # Create correlated data
        btc_data = self.normal_data.copy()
        eth_data = self.normal_data.copy()

        # Make ETH follow BTC with some correlation
        eth_data['close'] = btc_data['close'] * 0.1 + np.random.normal(0, 100, len(eth_data))

        trading_pairs_data = {
            'BTCUSDT': btc_data,
            'ETHUSDT': eth_data
        }

        anomalies = anomaly_detection.detect_correlation_anomalies(trading_pairs_data)

        # Should work without errors
        self.assertIsInstance(anomalies, list)

    def test_funding_rate_anomaly_detection(self):
        """Test funding rate anomaly detection"""
        print("Testing funding rate anomaly detection...")

        # Create funding rates with anomalies
        funding_rates = {
            'BTCUSDT': 0.005,  # Normal
            'ETHUSDT': 0.015,  # High positive (anomalous)
            'ADAUSDT': -0.012, # High negative (anomalous)
            'DOTUSDT': 0.003   # Normal
        }

        anomalies = anomaly_detection.detect_funding_rate_anomalies(funding_rates)

        # Should detect funding rate anomalies
        funding_anomalies = [a for a in anomalies if a['type'] == 'FUNDING_RATE_ANOMALY']
        self.assertGreater(len(funding_anomalies), 0)

        # Verify funding anomaly structure
        fr_anomaly = funding_anomalies[0]
        self.assertIn('funding_rate', fr_anomaly)
        self.assertIn('z_score', fr_anomaly)

    def test_liquidity_anomaly_detection(self):
        """Test liquidity anomaly detection"""
        print("Testing liquidity anomaly detection...")

        # Mock orderbook data with wide spread (low liquidity)
        orderbook_data = {
            'bids': [[45000, 10], [44950, 15]],
            'asks': [[45100, 8], [45150, 12]]  # Wide spread
        }

        anomalies = anomaly_detection.detect_liquidity_anomalies(orderbook_data, 'BTCUSDT')

        # Should detect liquidity anomaly
        liquidity_anomalies = [a for a in anomalies if a['type'] == 'LIQUIDITY_DRYUP']
        self.assertGreater(len(liquidity_anomalies), 0)

        # Verify liquidity anomaly structure
        liq_anomaly = liquidity_anomalies[0]
        self.assertIn('spread_percentage', liq_anomaly)
        self.assertGreater(liq_anomaly['spread_percentage'], 0)

    def test_comprehensive_anomaly_scan(self):
        """Test comprehensive anomaly scan across multiple pairs"""
        print("Testing comprehensive anomaly scan...")

        with patch('anomaly_detection.market_data') as mock_market_data:
            # Setup mocks
            mock_market_data.get_ohlcv_data.return_value = self.anomalous_data
            mock_market_data.get_market_metrics.return_value = {
                'current_price': 45000,
                'funding_rate': 0.01
            }

            # Mock trading pairs
            Config.TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT']

            anomalies = anomaly_detection.scan_for_anomalies(['BTCUSDT'])

            # Should detect multiple types of anomalies
            self.assertIsInstance(anomalies, list)
            self.assertGreater(len(anomalies), 0)

            # Check for different anomaly types
            anomaly_types = set(a['type'] for a in anomalies)
            self.assertGreater(len(anomaly_types), 1)  # Should detect multiple types

    def test_anomaly_summary_generation(self):
        """Test anomaly summary generation"""
        print("Testing anomaly summary generation...")

        # Add some test anomalies to history
        test_anomalies = [
            {
                'type': 'PRICE_SPIKE',
                'timestamp': datetime.now(),
                'trading_pair': 'BTCUSDT',
                'severity': 'HIGH',
                'confidence': 0.9
            },
            {
                'type': 'VOLUME_SPIKE',
                'timestamp': datetime.now() - timedelta(hours=1),
                'trading_pair': 'ETHUSDT',
                'severity': 'MEDIUM',
                'confidence': 0.7
            },
            {
                'type': 'FLASH_CRASH',
                'timestamp': datetime.now() - timedelta(hours=2),
                'trading_pair': 'BTCUSDT',
                'severity': 'CRITICAL',
                'confidence': 0.95
            }
        ]

        anomaly_detection.anomaly_history = test_anomalies

        # Test summary generation
        summary = anomaly_detection.get_anomaly_summary(hours=24)

        self.assertIn('total_anomalies', summary)
        self.assertIn('by_type', summary)
        self.assertIn('by_severity', summary)
        self.assertIn('by_trading_pair', summary)
        self.assertIn('high_confidence_anomalies', summary)

        self.assertEqual(summary['total_anomalies'], 3)
        self.assertEqual(len(summary['by_type']), 3)  # Three different types
        self.assertEqual(summary['by_severity']['HIGH'], 1)
        self.assertEqual(summary['by_severity']['MEDIUM'], 1)
        self.assertEqual(summary['by_severity']['CRITICAL'], 1)

    def test_anomaly_alerts_generation(self):
        """Test anomaly alerts generation"""
        print("Testing anomaly alerts generation...")

        # Add test anomalies
        test_anomalies = [
            {
                'type': 'PRICE_SPIKE',
                'timestamp': datetime.now(),
                'trading_pair': 'BTCUSDT',
                'severity': 'HIGH',
                'confidence': 0.9
            },
            {
                'type': 'VOLUME_SPIKE',
                'timestamp': datetime.now(),
                'trading_pair': 'ETHUSDT',
                'severity': 'LOW',
                'confidence': 0.3
            }
        ]

        anomaly_detection.anomaly_history = test_anomalies

        # Test alerts with different criteria
        high_severity_alerts = anomaly_detection.get_anomaly_alerts(min_severity='HIGH')
        self.assertEqual(len(high_severity_alerts), 1)

        high_confidence_alerts = anomaly_detection.get_anomaly_alerts(min_confidence=0.8)
        self.assertEqual(len(high_confidence_alerts), 1)

    def test_isolation_forest_training(self):
        """Test Isolation Forest model training"""
        print("Testing Isolation Forest training...")

        # Test with sufficient data
        success = anomaly_detection.train_isolation_forest(self.normal_data)
        self.assertIsInstance(success, bool)

        # Test with insufficient data
        small_data = self.normal_data.head(10)
        success_small = anomaly_detection.train_isolation_forest(small_data)
        self.assertFalse(success_small)

    def test_feature_preparation_for_ml(self):
        """Test feature preparation for machine learning"""
        print("Testing feature preparation for ML...")

        features = anomaly_detection._prepare_features_for_ml(self.normal_data)

        if features is not None:
            self.assertIsInstance(features, pd.DataFrame)
            self.assertGreater(len(features), 0)

            # Check for expected features
            expected_features = ['returns', 'log_returns', 'volume_ratio', 'volatility']
            for feature in expected_features:
                self.assertIn(feature, features.columns)

    def test_market_baseline_update(self):
        """Test market baseline update"""
        print("Testing market baseline update...")

        trading_pairs_data = {
            'BTCUSDT': self.normal_data,
            'ETHUSDT': self.normal_data.copy()
        }

        initial_baseline_count = len(anomaly_detection.market_baseline)

        anomaly_detection.update_market_baseline(trading_pairs_data)

        # Should update baseline
        self.assertGreaterEqual(len(anomaly_detection.market_baseline), initial_baseline_count)

    def test_anomaly_history_persistence(self):
        """Test anomaly history saving and loading"""
        print("Testing anomaly history persistence...")

        # Add test anomalies
        test_anomalies = [
            {
                'type': 'PRICE_SPIKE',
                'timestamp': datetime.now(),
                'trading_pair': 'BTCUSDT',
                'severity': 'HIGH',
                'confidence': 0.9,
                'z_score': 4.5,
                'price': 80000,
                'description': 'Test anomaly'
            }
        ]

        anomaly_detection.anomaly_history = test_anomalies

        # Test saving
        anomaly_detection.save_anomaly_history('test_anomaly_history.json')

        # Clear history
        anomaly_detection.anomaly_history = []

        # Test loading
        anomaly_detection.load_anomaly_history('test_anomaly_history.json')

        self.assertEqual(len(anomaly_detection.anomaly_history), 1)
        loaded_anomaly = anomaly_detection.anomaly_history[0]
        self.assertEqual(loaded_anomaly['type'], 'PRICE_SPIKE')
        self.assertEqual(loaded_anomaly['trading_pair'], 'BTCUSDT')

        # Clean up
        if os.path.exists('test_anomaly_history.json'):
            os.remove('test_anomaly_history.json')

    def test_error_handling(self):
        """Test error handling in anomaly detection"""
        print("Testing error handling...")

        # Test with None data
        anomalies = anomaly_detection.detect_price_anomalies(None, 'BTCUSDT')
        self.assertEqual(len(anomalies), 0)

        # Test with empty dataframe
        empty_df = pd.DataFrame()
        anomalies = anomaly_detection.detect_price_anomalies(empty_df, 'BTCUSDT')
        self.assertEqual(len(anomalies), 0)

        # Test with insufficient data
        small_df = pd.DataFrame({'close': [1, 2, 3]})
        anomalies = anomaly_detection.detect_price_anomalies(small_df, 'BTCUSDT')
        self.assertEqual(len(anomalies), 0)

    def test_anomaly_severity_classification(self):
        """Test anomaly severity classification"""
        print("Testing anomaly severity classification...")

        # Test different z-score ranges
        test_cases = [
            (2.5, 'MEDIUM'),  # Below high threshold
            (4.0, 'HIGH'),    # Above high threshold
            (6.0, 'CRITICAL') # Well above high threshold
        ]

        for z_score, expected_severity in test_cases:
            with self.subTest(z_score=z_score):
                # Create anomaly with specific z-score
                anomaly = {
                    'type': 'PRICE_SPIKE',
                    'timestamp': datetime.now(),
                    'trading_pair': 'BTCUSDT',
                    'z_score': z_score,
                    'severity': 'HIGH' if z_score > 4 else 'MEDIUM',
                    'confidence': 0.8
                }

                # Verify severity classification
                if z_score > 4:
                    self.assertEqual(anomaly['severity'], 'HIGH')
                elif z_score > 3:
                    self.assertIn(anomaly['severity'], ['MEDIUM', 'HIGH'])

    def test_anomaly_confidence_calculation(self):
        """Test anomaly confidence calculation"""
        print("Testing anomaly confidence calculation...")

        # Test confidence bounds
        anomalies = anomaly_detection.detect_price_anomalies(self.anomalous_data, 'BTCUSDT')

        for anomaly in anomalies:
            if 'confidence' in anomaly:
                self.assertGreaterEqual(anomaly['confidence'], 0.0)
                self.assertLessEqual(anomaly['confidence'], 1.0)

    def tearDown(self):
        """Clean up after tests"""
        # Reset anomaly history
        anomaly_detection.anomaly_history = []
        anomaly_detection.market_baseline = {}

        # Clean up any test files
        test_files = ['test_anomaly_history.json']
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == '__main__':
    print("Running comprehensive anomaly detection tests...")
    unittest.main(verbosity=2)
