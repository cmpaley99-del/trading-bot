import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import warnings
from datetime import datetime, timedelta
from loguru import logger
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis
import json
import os

warnings.filterwarnings('ignore')

class AnomalyDetection:
    """Advanced anomaly detection system for cryptocurrency trading"""

    def __init__(self):
        self.isolation_forest = None
        self.scaler = StandardScaler()
        self.anomaly_history = []
        self.market_baseline = {}
        self.alert_thresholds = {
            'price_spike': 3.0,  # Standard deviations
            'volume_spike': 2.5,
            'volatility_spike': 2.0,
            'correlation_break': 0.3,
            'funding_rate_anomaly': 2.5
        }
        self.anomaly_types = {
            'PRICE_SPIKE': 'Sudden price movement beyond normal range',
            'VOLUME_SPIKE': 'Unusual trading volume',
            'VOLATILITY_SPIKE': 'Extreme price volatility',
            'CORRELATION_BREAK': 'Breakdown in normal asset correlations',
            'FUNDING_RATE_ANOMALY': 'Abnormal funding rates',
            'LIQUIDITY_DRYUP': 'Sudden reduction in market liquidity',
            'FLASH_CRASH': 'Rapid price decline followed by recovery',
            'PUMP_DUMP': 'Coordinated price manipulation pattern'
        }
        logger.info("Anomaly Detection system initialized")

    def detect_price_anomalies(self, df, trading_pair):
        """Detect price-based anomalies using statistical methods"""
        anomalies = []

        if df is None or len(df) < 20:
            return anomalies

        try:
            # Calculate rolling statistics
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['rolling_mean'] = df['close'].rolling(window=20).mean()
            df['rolling_std'] = df['close'].rolling(window=20).std()
            df['z_score'] = (df['close'] - df['rolling_mean']) / df['rolling_std']

            # Price spike detection
            price_spikes = df[abs(df['z_score']) > self.alert_thresholds['price_spike']]
            for idx, row in price_spikes.iterrows():
                anomaly = {
                    'type': 'PRICE_SPIKE',
                    'timestamp': row.name if hasattr(row, 'name') else idx,
                    'trading_pair': trading_pair,
                    'severity': 'HIGH' if abs(row['z_score']) > 4 else 'MEDIUM',
                    'z_score': row['z_score'],
                    'price': row['close'],
                    'description': f"Price spike detected: {row['z_score']:.2f} standard deviations",
                    'confidence': min(abs(row['z_score']) / 5.0, 1.0)
                }
                anomalies.append(anomaly)

            # Flash crash detection
            recent_prices = df['close'].tail(10)
            if len(recent_prices) >= 10:
                max_price = recent_prices.max()
                min_price = recent_prices.min()
                drop_percentage = (max_price - min_price) / max_price

                if drop_percentage > 0.05:  # 5% drop in recent candles
                    recovery_check = recent_prices.iloc[-1] / recent_prices.iloc[-3]
                    if recovery_check > 0.98:  # Quick recovery
                        anomaly = {
                            'type': 'FLASH_CRASH',
                            'timestamp': df.index[-1],
                            'trading_pair': trading_pair,
                            'severity': 'CRITICAL',
                            'drop_percentage': drop_percentage,
                            'description': f"Flash crash detected: {drop_percentage:.1%} drop with quick recovery",
                            'confidence': 0.9
                        }
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting price anomalies for {trading_pair}: {e}")

        return anomalies

    def detect_volume_anomalies(self, df, trading_pair):
        """Detect volume-based anomalies"""
        anomalies = []

        if df is None or len(df) < 20:
            return anomalies

        try:
            df = df.copy()
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_z'] = (df['volume'] - df['volume_ma']) / df['volume_std']

            # Volume spike detection
            volume_spikes = df[df['volume_z'] > self.alert_thresholds['volume_spike']]
            for idx, row in volume_spikes.iterrows():
                anomaly = {
                    'type': 'VOLUME_SPIKE',
                    'timestamp': row.name if hasattr(row, 'name') else idx,
                    'trading_pair': trading_pair,
                    'severity': 'MEDIUM',
                    'volume_ratio': row['volume_z'],
                    'volume': row['volume'],
                    'description': f"Volume spike: {row['volume_z']:.2f}x normal volume",
                    'confidence': min(row['volume_z'] / 4.0, 1.0)
                }
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting volume anomalies for {trading_pair}: {e}")

        return anomalies

    def detect_volatility_anomalies(self, df, trading_pair):
        """Detect volatility-based anomalies"""
        anomalies = []

        if df is None or len(df) < 20:
            return anomalies

        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()
            df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(24)  # Annualized
            df['vol_ma'] = df['volatility'].rolling(window=50).mean()
            df['vol_std'] = df['volatility'].rolling(window=50).std()
            df['vol_z'] = (df['volatility'] - df['vol_ma']) / df['vol_std']

            # Volatility spike detection
            vol_spikes = df[df['vol_z'] > self.alert_thresholds['volatility_spike']]
            for idx, row in vol_spikes.iterrows():
                anomaly = {
                    'type': 'VOLATILITY_SPIKE',
                    'timestamp': row.name if hasattr(row, 'name') else idx,
                    'trading_pair': trading_pair,
                    'severity': 'HIGH',
                    'volatility': row['volatility'],
                    'vol_z_score': row['vol_z'],
                    'description': f"Volatility spike: {row['vol_z']:.2f} standard deviations above normal",
                    'confidence': min(row['vol_z'] / 3.0, 1.0)
                }
                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting volatility anomalies for {trading_pair}: {e}")

        return anomalies

    def detect_correlation_anomalies(self, trading_pairs_data):
        """Detect correlation breakdown between trading pairs"""
        anomalies = []

        if not trading_pairs_data or len(trading_pairs_data) < 2:
            return anomalies

        try:
            # Calculate returns for each pair
            returns_data = {}
            for pair, df in trading_pairs_data.items():
                if df is not None and len(df) > 20:
                    returns_data[pair] = df['close'].pct_change().dropna()

            if len(returns_data) < 2:
                return anomalies

            # Calculate correlation matrix
            returns_df = pd.DataFrame(returns_data)
            correlation_matrix = returns_df.corr()

            # Check for correlation breakdowns
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    pair1 = correlation_matrix.columns[i]
                    pair2 = correlation_matrix.columns[j]
                    correlation = correlation_matrix.iloc[i, j]

                    # Check against historical baseline
                    baseline_key = f"{pair1}_{pair2}"
                    if baseline_key in self.market_baseline:
                        baseline_corr = self.market_baseline[baseline_key]['mean']
                        baseline_std = self.market_baseline[baseline_key]['std']

                        if baseline_std > 0:
                            corr_z = abs(correlation - baseline_corr) / baseline_std

                            if corr_z > self.alert_thresholds['correlation_break']:
                                anomaly = {
                                    'type': 'CORRELATION_BREAK',
                                    'timestamp': datetime.now(),
                                    'trading_pairs': [pair1, pair2],
                                    'severity': 'MEDIUM',
                                    'correlation': correlation,
                                    'baseline_corr': baseline_corr,
                                    'z_score': corr_z,
                                    'description': f"Correlation breakdown between {pair1} and {pair2}: {correlation:.3f} vs baseline {baseline_corr:.3f}",
                                    'confidence': min(corr_z / 3.0, 1.0)
                                }
                                anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting correlation anomalies: {e}")

        return anomalies

    def detect_funding_rate_anomalies(self, funding_rates):
        """Detect anomalies in funding rates"""
        anomalies = []

        if not funding_rates:
            return anomalies

        try:
            rates = [rate for rate in funding_rates.values() if rate is not None]
            if len(rates) < 5:
                return anomalies

            rates_array = np.array(rates)
            mean_rate = np.mean(rates_array)
            std_rate = np.std(rates_array)

            for pair, rate in funding_rates.items():
                if rate is not None and std_rate > 0:
                    z_score = abs(rate - mean_rate) / std_rate

                    if z_score > self.alert_thresholds['funding_rate_anomaly']:
                        anomaly = {
                            'type': 'FUNDING_RATE_ANOMALY',
                            'timestamp': datetime.now(),
                            'trading_pair': pair,
                            'severity': 'HIGH' if z_score > 4 else 'MEDIUM',
                            'funding_rate': rate,
                            'z_score': z_score,
                            'description': f"Abnormal funding rate for {pair}: {rate:.4f}% ({z_score:.2f} std dev)",
                            'confidence': min(z_score / 4.0, 1.0)
                        }
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting funding rate anomalies: {e}")

        return anomalies

    def detect_liquidity_anomalies(self, orderbook_data, trading_pair):
        """Detect liquidity dry-up anomalies"""
        anomalies = []

        if not orderbook_data:
            return anomalies

        try:
            # Check bid-ask spread
            if 'bids' in orderbook_data and 'asks' in orderbook_data:
                best_bid = orderbook_data['bids'][0][0] if orderbook_data['bids'] else None
                best_ask = orderbook_data['asks'][0][0] if orderbook_data['asks'] else None

                if best_bid and best_ask:
                    spread = (best_ask - best_bid) / best_bid
                    spread_threshold = 0.001  # 0.1% spread threshold

                    if spread > spread_threshold:
                        anomaly = {
                            'type': 'LIQUIDITY_DRYUP',
                            'timestamp': datetime.now(),
                            'trading_pair': trading_pair,
                            'severity': 'MEDIUM',
                            'spread_percentage': spread * 100,
                            'description': f"Wide bid-ask spread: {spread:.4f}% indicates low liquidity",
                            'confidence': min(spread / 0.005, 1.0)
                        }
                        anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting liquidity anomalies for {trading_pair}: {e}")

        return anomalies

    def detect_pump_dump_patterns(self, df, trading_pair):
        """Detect potential pump and dump patterns"""
        anomalies = []

        if df is None or len(df) < 50:
            return anomalies

        try:
            df = df.copy()
            df['returns'] = df['close'].pct_change()

            # Look for rapid price increases followed by declines
            window_size = 20
            for i in range(window_size, len(df) - window_size):
                window = df.iloc[i-window_size:i+window_size]

                if len(window) < window_size * 1.5:
                    continue

                # Check for pump pattern (rapid increase)
                max_return = window['returns'].max()
                pump_threshold = 0.05  # 5% increase

                if max_return > pump_threshold:
                    # Check for dump pattern (subsequent decline)
                    post_pump = df.iloc[i:i+window_size]
                    if len(post_pump) > 5:
                        post_decline = (post_pump['close'].iloc[-1] - post_pump['close'].iloc[0]) / post_pump['close'].iloc[0]

                        if post_decline < -0.03:  # 3% decline after pump
                            anomaly = {
                                'type': 'PUMP_DUMP',
                                'timestamp': df.index[i],
                                'trading_pair': trading_pair,
                                'severity': 'HIGH',
                                'pump_return': max_return,
                                'dump_return': post_decline,
                                'description': f"Pump & dump pattern detected: {max_return:.1%} pump followed by {post_decline:.1%} dump",
                                'confidence': 0.8
                            }
                            anomalies.append(anomaly)

        except Exception as e:
            logger.error(f"Error detecting pump-dump patterns for {trading_pair}: {e}")

        return anomalies

    def train_isolation_forest(self, historical_data):
        """Train Isolation Forest for multivariate anomaly detection"""
        try:
            if historical_data is None or len(historical_data) < 100:
                logger.warning("Insufficient data for Isolation Forest training")
                return False

            # Prepare features for training
            features = self._prepare_features_for_ml(historical_data)

            if features is None or len(features) == 0:
                return False

            # Train the model
            self.isolation_forest = IsolationForest(
                contamination=0.1,  # Expected proportion of anomalies
                random_state=42,
                n_estimators=100
            )

            self.isolation_forest.fit(features)
            logger.info("Isolation Forest trained successfully")
            return True

        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return False

    def _prepare_features_for_ml(self, df):
        """Prepare features for machine learning models"""
        try:
            if df is None or len(df) < 20:
                return None

            features = pd.DataFrame()

            # Price-based features
            features['returns'] = df['close'].pct_change()
            features['log_returns'] = np.log(df['close'] / df['close'].shift(1))

            # Volume features
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()

            # Volatility features
            features['volatility'] = df['close'].pct_change().rolling(20).std()

            # Technical indicators
            features['rsi'] = self._calculate_rsi(df['close'])
            features['macd'] = self._calculate_macd(df['close'])

            # Statistical features
            features['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
            features['volume_zscore'] = (df['volume'] - df['volume'].rolling(20).mean()) / df['volume'].rolling(20).std()

            # Fill NaN values
            features = features.fillna(method='bfill').fillna(0)

            return features

        except Exception as e:
            logger.error(f"Error preparing features for ML: {e}")
            return None

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices):
        """Calculate MACD indicator"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        return ema12 - ema26

    def update_market_baseline(self, trading_pairs_data):
        """Update market baseline statistics for anomaly detection"""
        try:
            if not trading_pairs_data:
                return

            # Calculate correlation baselines
            returns_data = {}
            for pair, df in trading_pairs_data.items():
                if df is not None and len(df) > 50:
                    returns_data[pair] = df['close'].pct_change().dropna()

            if len(returns_data) >= 2:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()

                # Store correlation baselines
                for i in range(len(correlation_matrix.columns)):
                    for j in range(i+1, len(correlation_matrix.columns)):
                        pair1 = correlation_matrix.columns[i]
                        pair2 = correlation_matrix.columns[j]

                        # Calculate rolling correlation statistics
                        rolling_corr = returns_df[pair1].rolling(100).corr(returns_df[pair2])
                        baseline_corr = rolling_corr.mean()
                        baseline_std = rolling_corr.std()

                        if not np.isnan(baseline_corr) and not np.isnan(baseline_std):
                            baseline_key = f"{pair1}_{pair2}"
                            self.market_baseline[baseline_key] = {
                                'mean': baseline_corr,
                                'std': baseline_std,
                                'last_updated': datetime.now()
                            }

            logger.info(f"Market baseline updated with {len(self.market_baseline)} correlation pairs")

        except Exception as e:
            logger.error(f"Error updating market baseline: {e}")

    def scan_for_anomalies(self, trading_pairs=None):
        """Comprehensive anomaly scan across all trading pairs"""
        if trading_pairs is None:
            trading_pairs = Config.TRADING_PAIRS

        all_anomalies = []
        trading_pairs_data = {}

        # Collect data for all pairs
        for pair in trading_pairs:
            try:
                df = market_data.get_ohlcv_data(pair)
                if df is not None:
                    trading_pairs_data[pair] = df

                    # Detect individual pair anomalies
                    price_anomalies = self.detect_price_anomalies(df, pair)
                    volume_anomalies = self.detect_volume_anomalies(df, pair)
                    volatility_anomalies = self.detect_volatility_anomalies(df, pair)
                    pump_dump_anomalies = self.detect_pump_dump_patterns(df, pair)

                    all_anomalies.extend(price_anomalies)
                    all_anomalies.extend(volume_anomalies)
                    all_anomalies.extend(volatility_anomalies)
                    all_anomalies.extend(pump_dump_anomalies)

            except Exception as e:
                logger.error(f"Error scanning anomalies for {pair}: {e}")

        # Cross-pair analysis
        try:
            correlation_anomalies = self.detect_correlation_anomalies(trading_pairs_data)
            all_anomalies.extend(correlation_anomalies)
        except Exception as e:
            logger.error(f"Error in cross-pair anomaly detection: {e}")

        # Funding rate analysis
        try:
            funding_rates = {}
            for pair in trading_pairs:
                metrics = market_data.get_market_metrics(pair)
                if metrics and 'funding_rate' in metrics:
                    funding_rates[pair] = metrics['funding_rate']

            if funding_rates:
                funding_anomalies = self.detect_funding_rate_anomalies(funding_rates)
                all_anomalies.extend(funding_anomalies)
        except Exception as e:
            logger.error(f"Error in funding rate anomaly detection: {e}")

        # Update anomaly history
        self.anomaly_history.extend(all_anomalies)

        # Keep only recent anomalies (last 1000)
        if len(self.anomaly_history) > 1000:
            self.anomaly_history = self.anomaly_history[-1000:]

        logger.info(f"Anomaly scan completed. Found {len(all_anomalies)} anomalies")
        return all_anomalies

    def get_anomaly_summary(self, hours=24):
        """Get summary of anomalies in the specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_anomalies = [
            anomaly for anomaly in self.anomaly_history
            if isinstance(anomaly['timestamp'], datetime) and anomaly['timestamp'] > cutoff_time
        ]

        summary = {
            'total_anomalies': len(recent_anomalies),
            'by_type': {},
            'by_severity': {},
            'by_trading_pair': {},
            'high_confidence_anomalies': []
        }

        for anomaly in recent_anomalies:
            # Count by type
            anomaly_type = anomaly['type']
            summary['by_type'][anomaly_type] = summary['by_type'].get(anomaly_type, 0) + 1

            # Count by severity
            severity = anomaly['severity']
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1

            # Count by trading pair
            if 'trading_pair' in anomaly:
                pair = anomaly['trading_pair']
                summary['by_trading_pair'][pair] = summary['by_trading_pair'].get(pair, 0) + 1
            elif 'trading_pairs' in anomaly:
                for pair in anomaly['trading_pairs']:
                    summary['by_trading_pair'][pair] = summary['by_trading_pair'].get(pair, 0) + 1

            # High confidence anomalies
            if anomaly.get('confidence', 0) > 0.7:
                summary['high_confidence_anomalies'].append(anomaly)

        return summary

    def get_anomaly_alerts(self, min_severity='MEDIUM', min_confidence=0.6):
        """Get current anomaly alerts that meet the criteria"""
        alerts = []

        for anomaly in self.anomaly_history[-100:]:  # Check last 100 anomalies
            if (anomaly['severity'] in ['HIGH', 'CRITICAL'] or
                (anomaly['severity'] == 'MEDIUM' and min_severity == 'MEDIUM')):

                if anomaly.get('confidence', 0) >= min_confidence:
                    alerts.append(anomaly)

        return alerts

    def save_anomaly_history(self, filename='anomaly_history.json'):
        """Save anomaly history to file"""
        try:
            # Convert datetime objects to strings for JSON serialization
            serializable_history = []
            for anomaly in self.anomaly_history:
                anomaly_copy = anomaly.copy()
                if isinstance(anomaly_copy['timestamp'], datetime):
                    anomaly_copy['timestamp'] = anomaly_copy['timestamp'].isoformat()
                serializable_history.append(anomaly_copy)

            with open(filename, 'w') as f:
                json.dump(serializable_history, f, indent=2)

            logger.info(f"Anomaly history saved to {filename}")

        except Exception as e:
            logger.error(f"Error saving anomaly history: {e}")

    def load_anomaly_history(self, filename='anomaly_history.json'):
        """Load anomaly history from file"""
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    history_data = json.load(f)

                # Convert timestamp strings back to datetime objects
                for anomaly in history_data:
                    if 'timestamp' in anomaly and isinstance(anomaly['timestamp'], str):
                        try:
                            anomaly['timestamp'] = datetime.fromisoformat(anomaly['timestamp'])
                        except:
                            anomaly['timestamp'] = datetime.now()

                self.anomaly_history = history_data
                logger.info(f"Anomaly history loaded from {filename}")

        except Exception as e:
            logger.error(f"Error loading anomaly history: {e}")

# Singleton instance
anomaly_detection = AnomalyDetection()
