"""
Test script for Reinforcement Learning-based leverage optimization
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_leverage_complete import LeverageOptimizationEnv, LeverageOptimizer, RLLeverageManager
from config import Config
from loguru import logger


def create_test_market_data(num_points=1000):
    """Create synthetic market data for testing"""
    logger.info(f"Creating {num_points} points of synthetic market data")

    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(num_points)]

    # Generate price data with trends and volatility
    base_price = 50000
    prices = []
    current_price = base_price

    for i in range(num_points):
        # Add trend component
        trend = 0.0001 * np.sin(i * 0.01)  # Slow trend

        # Add volatility based on market conditions
        if i < num_points // 3:
            volatility = 0.005  # Low volatility period
        elif i < 2 * num_points // 3:
            volatility = 0.015  # High volatility period
        else:
            volatility = 0.008  # Medium volatility period

        # Generate price movement
        price_change = np.random.normal(trend, volatility)
        current_price *= (1 + price_change)
        prices.append(current_price)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000000, 10000000) for _ in range(num_points)]
    })

    return df


def test_rl_environment():
    """Test the RL environment"""
    logger.info("Testing RL Environment")

    # Create test data
    test_data = create_test_market_data(100)

    # Create environment
    env = LeverageOptimizationEnv(test_data, 'BTCUSDT')

    # Test basic functionality
    state = env.reset()
    logger.info(f"Initial state shape: {state.shape}")

    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()  # Random action
        next_state, reward, done, info = env.step(action)
        total_reward += reward

        logger.info(f"Step {step}: Action={action+5}x, Reward={reward:.2f}, "
                   f"Portfolio={info['portfolio_value']:.2f}")

        if done:
            break

    logger.info(f"Environment test completed. Total reward: {total_reward:.2f}")


def test_leverage_optimizer():
    """Test the leverage optimizer"""
    logger.info("Testing Leverage Optimizer")

    # Create test data
    test_data = create_test_market_data(500)

    # Create optimizer
    optimizer = LeverageOptimizer('BTCUSDT')

    # Test optimal leverage selection
    current_market_data = test_data.iloc[-1].to_dict()
    optimal_leverage = optimizer.get_optimal_leverage(current_market_data)

    logger.info(f"Optimal leverage for current market: {optimal_leverage}x")

    # Test training (short training for demo)
    logger.info("Starting short training session...")
    optimizer.train_on_historical_data(test_data, episodes=5)

    # Get performance metrics
    metrics = optimizer.get_performance_metrics()
    logger.info(f"Training metrics: {metrics}")

    # Test again after training
    optimal_leverage_trained = optimizer.get_optimal_leverage(current_market_data)
    logger.info(f"Optimal leverage after training: {optimal_leverage_trained}x")


def test_rl_manager():
    """Test the RL leverage manager"""
    logger.info("Testing RL Leverage Manager")

    # Create manager
    manager = RLLeverageManager()

    # Create test data for one pair
    test_data = create_test_market_data(200)
    current_market_data = test_data.iloc[-1].to_dict()

    # Test getting optimal leverage
    optimal_leverage = manager.get_optimal_leverage('BTCUSDT', current_market_data)
    logger.info(f"RL Manager optimal leverage for BTCUSDT: {optimal_leverage}x")

    # Test performance metrics
    metrics = manager.get_all_performance_metrics()
    logger.info(f"RL Manager performance metrics: {list(metrics.keys())}")


def test_comparison_with_rule_based():
    """Compare RL-based leverage with rule-based intelligent leverage"""
    logger.info("Comparing RL vs Rule-based leverage")

    from technical_analysis import technical_analysis

    # Create test data
    test_data = create_test_market_data(100)
    test_data_with_indicators = technical_analysis.calculate_indicators(test_data)

    if test_data_with_indicators is not None:
        current_data = test_data_with_indicators.iloc[-1]

        # Get rule-based leverage
        rule_based_leverage = technical_analysis.calculate_appropriate_leverage(
            test_data_with_indicators, 'BTCUSDT'
        )

        # Get RL-based leverage
        manager = RLLeverageManager()
        rl_leverage = manager.get_optimal_leverage('BTCUSDT', current_data.to_dict())

        logger.info(f"Rule-based leverage: {rule_based_leverage}x")
        logger.info(f"RL-based leverage: {rl_leverage}x")
        logger.info(f"Difference: {abs(rl_leverage - rule_based_leverage)}x")

        # Analyze market conditions
        volatility = current_data['atr'] / current_data['close'] * 100
        trend_strength = current_data['adx']
        rsi = current_data['rsi']

        logger.info(f"Market conditions - Volatility: {volatility:.2f}%, "
                   f"Trend Strength: {trend_strength:.1f}, RSI: {rsi:.1f}")


def run_all_tests():
    """Run all RL leverage tests"""
    logger.info("=== Starting RL Leverage Optimization Tests ===")

    try:
        test_rl_environment()
        print("\n" + "="*50 + "\n")

        test_leverage_optimizer()
        print("\n" + "="*50 + "\n")

        test_rl_manager()
        print("\n" + "="*50 + "\n")

        test_comparison_with_rule_based()
        print("\n" + "="*50 + "\n")

        logger.info("=== All RL Leverage Tests Completed Successfully ===")

    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    run_all_tests()
