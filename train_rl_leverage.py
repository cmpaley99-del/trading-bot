"""
Training script for Reinforcement Learning-based leverage optimization
Fetches historical data and trains the RL model for optimal leverage selection
"""

import sys
import os
import pandas as pd
from datetime import datetime
from loguru import logger

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_leverage_complete import RLLeverageManager
from market_data import market_data
from technical_analysis import technical_analysis
from config import Config


def fetch_historical_data_for_training(trading_pair='BTCUSDT', timeframe='5m', limit=2000):
    """Fetch historical OHLCV data for RL training"""
    logger.info(f"Fetching {limit} candles of {timeframe} data for {trading_pair} for RL training")

    try:
        # Fetch raw OHLCV data
        df = market_data.get_ohlcv_data(trading_pair, timeframe, limit)
        if df is None:
            logger.error(f"Failed to fetch historical data for {trading_pair}")
            return None

        logger.info(f"Fetched {len(df)} raw candles for {trading_pair}")

        # Calculate technical indicators required for RL state
        df_with_indicators = technical_analysis.calculate_indicators(df)
        if df_with_indicators is None:
            logger.error(f"Failed to calculate indicators for {trading_pair}")
            return None

        logger.info(f"Calculated technical indicators for {len(df_with_indicators)} candles")

        # Ensure we have enough data after indicator calculation
        if len(df_with_indicators) < 100:
            logger.warning(f"Insufficient data after indicator calculation: {len(df_with_indicators)} candles")
            return None

        return df_with_indicators

    except Exception as e:
        logger.error(f"Error fetching historical data for training: {e}")
        return None


def train_rl_model_for_pair(trading_pair, historical_data, episodes=100):
    """Train RL model for a specific trading pair"""
    logger.info(f"Starting RL training for {trading_pair} with {episodes} episodes")

    try:
        # Get RL manager
        rl_manager = RLLeverageManager()

        # Train the model
        rl_manager.optimizers[trading_pair].train_on_historical_data(historical_data, episodes)

        # Get performance metrics
        metrics = rl_manager.optimizers[trading_pair].get_performance_metrics()

        logger.info(f"Training completed for {trading_pair}")
        logger.info(f"Performance metrics: {metrics}")

        return metrics

    except Exception as e:
        logger.error(f"Error training RL model for {trading_pair}: {e}")
        return None


def train_all_pairs(episodes_per_pair=50):
    """Train RL models for all configured trading pairs"""
    logger.info("Starting RL training for all trading pairs")

    results = {}
    historical_data_dict = {}

    # Fetch historical data for all pairs
    for pair in Config.TRADING_PAIRS:
        logger.info(f"Fetching historical data for {pair}")
        historical_data = fetch_historical_data_for_training(pair, '5m', 1500)

        if historical_data is not None:
            historical_data_dict[pair] = historical_data
            logger.info(f"Successfully fetched {len(historical_data)} data points for {pair}")
        else:
            logger.warning(f"Skipping {pair} due to data fetch failure")

    # Train models for pairs with data
    for pair, data in historical_data_dict.items():
        logger.info(f"Training RL model for {pair}")
        metrics = train_rl_model_for_pair(pair, data, episodes_per_pair)

        if metrics:
            results[pair] = metrics
        else:
            logger.error(f"Failed to train model for {pair}")

    # Save all models
    rl_manager = RLLeverageManager()
    rl_manager.save_all_models()

    logger.info("RL training completed for all available pairs")
    return results


def test_trained_models():
    """Test the trained RL models with current market data"""
    logger.info("Testing trained RL models")

    try:
        rl_manager = RLLeverageManager()

        for pair in Config.TRADING_PAIRS:
            # Get current market data
            df = market_data.get_ohlcv_data(pair, '5m', 50)
            if df is None:
                logger.warning(f"Could not fetch current data for {pair}")
                continue

            df_with_indicators = technical_analysis.calculate_indicators(df)
            if df_with_indicators is None:
                logger.warning(f"Could not calculate indicators for {pair}")
                continue

            current_data = df_with_indicators.iloc[-1].to_dict()

            # Get RL optimal leverage
            rl_leverage = rl_manager.get_optimal_leverage(pair, current_data)

            # Get rule-based leverage for comparison
            rule_leverage = technical_analysis.calculate_appropriate_leverage(df_with_indicators, pair)

            logger.info(f"{pair} - RL Leverage: {rl_leverage}x, Rule-based: {rule_leverage}x")

    except Exception as e:
        logger.error(f"Error testing trained models: {e}")


def main():
    """Main training function"""
    logger.info("=== RL Leverage Training Script Started ===")

    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    try:
        # Train models for all pairs
        logger.info("Starting comprehensive RL training")
        training_results = train_all_pairs(episodes_per_pair=100)

        # Log training results
        logger.info("Training Results Summary:")
        for pair, metrics in training_results.items():
            logger.info(f"{pair}: Episodes={metrics.get('total_episodes', 0)}, "
                       f"Best Reward={metrics.get('best_reward', 0):.2f}")

        # Test trained models
        logger.info("Testing trained models with current market data")
        test_trained_models()

        logger.info("=== RL Training Script Completed Successfully ===")

    except Exception as e:
        logger.error(f"RL training script failed: {e}")
        raise


if __name__ == "__main__":
    main()
