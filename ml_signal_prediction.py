import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os
from datetime import datetime, timedelta
import logging
from loguru import logger
from config import Config

class MLSignalPredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.model_path = 'models/signal_predictor.pkl'
        self.scaler_path = 'models/scaler.pkl'
        self.performance_history = []
        logger.info("ML Signal Predictor initialized")

        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)

    def prepare_features(self, df, signal_type=None):
        """Prepare features for ML model from technical analysis data"""
        if df is None or len(df) < 50:
            return None

        features = pd.DataFrame(index=df.index)

        try:
            # Price-based features
            features['price_change'] = df['close'].pct_change()
            features['price_volatility'] = df['close'].rolling(window=20).std()
            features['price_trend'] = df['close'] / df['close'].rolling(window=50).mean()

            # Volume features
            features['volume_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
            features['volume_trend'] = df['volume'].rolling(window=10).mean() / df['volume'].rolling(window=50).mean()

            # RSI features
            features['rsi'] = df['rsi']
            features['rsi_trend'] = df['rsi'].diff()
            features['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            features['rsi_oversold'] = (df['rsi'] < 30).astype(int)

            # MACD features
            features['macd'] = df['macd']
            features['macd_signal'] = df['macd_signal']
            features['macd_hist'] = df['macd_hist']
            features['macd_crossover'] = np.where(df['macd'] > df['macd_signal'], 1, -1)

            # Bollinger Bands features
            features['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            features['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            features['bb_squeeze'] = features['bb_width'].rolling(window=20).std()

            # Moving Average features
            features['sma_20_trend'] = df['sma_20'].pct_change(10)
            features['sma_50_trend'] = df['sma_50'].pct_change(10)
            features['ema_12_trend'] = df['ema_12'].pct_change(10)
            features['ema_26_trend'] = df['ema_26'].pct_change(10)

            # Scalping EMAs
            features['ema_5_trend'] = df['ema_5'].pct_change(5)
            features['ema_10_trend'] = df['ema_10'].pct_change(5)
            features['ema_20_trend'] = df['ema_20'].pct_change(5)

            # Stochastic features
            features['stoch_k'] = df['stoch_k']
            features['stoch_d'] = df['stoch_d']
            features['stoch_divergence'] = df['stoch_k'] - df['stoch_d']

            # ADX and momentum
            features['adx'] = df['adx']
            features['cci'] = df['cci']
            features['williams_r'] = df['williams_r']

            # Volatility and ATR
            features['atr'] = df['atr']
            features['atr_ratio'] = df['atr'] / df['close']

            # Volume indicators
            features['obv'] = df['obv']
            features['vwap'] = df['vwap']
            features['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

            # Pattern recognition (simplified)
            features['hammer_pattern'] = df['hammer'].fillna(0)
            features['engulfing_pattern'] = df['engulfing'].fillna(0)
            features['doji_pattern'] = df['doji'].fillna(0)

            # Time-based features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                features['hour'] = df['timestamp'].dt.hour
                features['day_of_week'] = df['timestamp'].dt.dayofweek
                features['month'] = df['timestamp'].dt.month

            # Fill NaN values
            features = features.fillna(method='ffill').fillna(0)

            # Add lag features (previous periods)
            for col in ['rsi', 'macd', 'bb_position', 'volume_ratio']:
                if col in features.columns:
                    features[f'{col}_lag1'] = features[col].shift(1)
                    features[f'{col}_lag2'] = features[col].shift(2)

            # Remove any remaining NaN values
            features = features.dropna()

            return features

        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return None

    def create_labels(self, df, forward_periods=12):
        """Create labels for signal prediction based on future price movement"""
        if df is None or len(df) < forward_periods + 10:
            return None

        try:
            # Calculate future returns
            future_returns = df['close'].shift(-forward_periods) / df['close'] - 1

            # Define success threshold (e.g., 0.5% profit)
            success_threshold = 0.005

            # Create binary labels
            labels = np.where(future_returns > success_threshold, 1,
                            np.where(future_returns < -success_threshold, 0, -1))

            # Remove neutral signals for binary classification
            valid_indices = labels != -1
            labels = labels[valid_indices]

            return labels, valid_indices

        except Exception as e:
            logger.error(f"Error creating labels: {e}")
            return None, None

    def train_model(self, historical_data, signal_type='BULLISH'):
        """Train the ML model using historical data"""
        try:
            logger.info(f"Training ML model for {signal_type} signals...")

            # Prepare features
            features = self.prepare_features(historical_data)
            if features is None:
                logger.error("Failed to prepare features")
                return False

            # Create labels
            labels, valid_indices = self.create_labels(historical_data)
            if labels is None:
                logger.error("Failed to create labels")
                return False

            # Align features and labels
            features = features.iloc[:len(labels)]

            # Store feature columns
            self.feature_columns = features.columns.tolist()

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels, test_size=0.2, random_state=42, stratify=labels
            )

            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)

            # Train model (using Random Forest for better interpretability)
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                class_weight='balanced'
            )

            self.model.fit(X_train_scaled, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')

            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=5)

            logger.info(f"Model Performance for {signal_type}:")
            logger.info(f"Accuracy: {accuracy:.4f}")
            logger.info(f"Precision: {precision:.4f}")
            logger.info(f"Recall: {recall:.4f}")
            logger.info(f"F1-Score: {f1:.4f}")
            logger.info(f"CV Mean Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

            # Store performance metrics
            self.performance_history.append({
                'timestamp': datetime.now(),
                'signal_type': signal_type,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            })

            # Save model
            self.save_model()

            return True

        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False

    def predict_signal_success(self, df, signal_type='BULLISH'):
        """Predict the success probability of a trading signal"""
        if self.model is None:
            logger.warning("Model not trained yet")
            return 0.5  # Return neutral confidence

        try:
            # Prepare features
            features = self.prepare_features(df)
            if features is None or len(features) == 0:
                return 0.5

            # Ensure we have the same features as training
            missing_cols = set(self.feature_columns) - set(features.columns)
            if missing_cols:
                logger.warning(f"Missing feature columns: {missing_cols}")
                return 0.5

            # Select only the features used in training
            features = features[self.feature_columns]

            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get prediction probabilities
            probabilities = self.model.predict_proba(features_scaled)

            # Return confidence for the positive class (successful signal)
            if len(probabilities[0]) > 1:
                confidence = probabilities[-1][1]  # Probability of success (class 1)
            else:
                confidence = probabilities[-1][0]  # Binary case

            # Ensure confidence is between 0 and 1
            confidence = max(0.0, min(1.0, confidence))

            logger.info(f"ML Prediction confidence for {signal_type}: {confidence:.4f}")
            return confidence

        except Exception as e:
            logger.error(f"Error predicting signal success: {e}")
            return 0.5

    def get_feature_importance(self):
        """Get feature importance from the trained model"""
        if self.model is None:
            return None

        try:
            importance = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_columns, importance))

            # Sort by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

            return sorted_features

        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return None

    def save_model(self):
        """Save the trained model and scaler"""
        try:
            if self.model:
                joblib.dump(self.model, self.model_path)
                joblib.dump(self.scaler, self.scaler_path)
                joblib.dump(self.feature_columns, 'models/feature_columns.pkl')
                logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")

    def load_model(self):
        """Load the trained model and scaler"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                self.feature_columns = joblib.load('models/feature_columns.pkl')
                logger.info("Model loaded successfully")
                return True
            else:
                logger.warning("No saved model found")
                return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False

    def get_model_performance_history(self):
        """Get historical performance metrics"""
        return self.performance_history

    def retrain_model(self, new_data, signal_type='BULLISH'):
        """Retrain the model with new data"""
        logger.info("Retraining ML model with new data...")
        return self.train_model(new_data, signal_type)

# Singleton instance
ml_signal_predictor = MLSignalPredictor()
