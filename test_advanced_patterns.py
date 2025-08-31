"""
Test script for Advanced Pattern Recognition
Tests harmonic patterns, Elliott waves, and Fibonacci analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pattern_recognition import pattern_recognition, HarmonicPatterns, ElliottWave, FibonacciAnalysis
from technical_analysis_advanced import advanced_technical_analysis
from loguru import logger


def create_test_market_data(num_points=200):
    """Create synthetic market data for testing patterns"""
    logger.info(f"Creating {num_points} points of synthetic market data for pattern testing")

    # Generate timestamps
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(minutes=i*5) for i in range(num_points)]

    # Generate price data with trend and patterns
    base_price = 50000
    prices = []
    current_price = base_price

    # Create a trending market with some patterns
    for i in range(num_points):
        # Add trend component
        trend = 0.0002 * np.sin(i * 0.02)  # Slow trend

        # Add volatility
        volatility = 0.01

        # Add some pattern-like movements
        if 50 <= i <= 80:  # Create a potential harmonic pattern
            pattern_move = 0.005 * np.sin((i-50) * 0.3)
        elif 120 <= i <= 150:  # Another pattern
            pattern_move = -0.004 * np.sin((i-120) * 0.4)
        else:
            pattern_move = 0

        # Generate price movement
        price_change = np.random.normal(trend + pattern_move, volatility)
        current_price *= (1 + price_change)
        prices.append(current_price)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
        'close': prices,
        'volume': [np.random.uniform(1000000, 10000000) for _ in range(num_points)]
    })

    return df


def test_harmonic_patterns():
    """Test harmonic pattern detection"""
    logger.info("Testing Harmonic Pattern Detection")

    # Create test data
    test_data = create_test_market_data(150)

    # Test harmonic patterns
    harmonic = HarmonicPatterns()
    patterns = harmonic.detect_patterns(test_data)

    logger.info(f"Detected patterns: {list(patterns.keys())}")
    for pattern_type, pattern_list in patterns.items():
        logger.info(f"{pattern_type}: {len(pattern_list)} patterns found")
        if pattern_list:
            best_pattern = max(pattern_list, key=lambda x: x.get('confidence', 0))
            logger.info(f"Best {pattern_type} pattern confidence: {best_pattern.get('confidence', 0):.2f}")


def test_elliott_waves():
    """Test Elliott wave analysis"""
    logger.info("Testing Elliott Wave Analysis")

    # Create test data
    test_data = create_test_market_data(200)

    # Test Elliott wave analysis
    elliott = ElliottWave()
    wave_analysis = elliott.analyze_waves(test_data)

    logger.info(f"Wave analysis: {wave_analysis}")

    if wave_analysis.get('wave_structure'):
        impulse_count = len(wave_analysis['wave_structure'].get('impulse_waves', []))
        corrective_count = len(wave_analysis['wave_structure'].get('corrective_waves', []))
        logger.info(f"Impulse waves: {impulse_count}, Corrective waves: {corrective_count}")


def test_fibonacci_analysis():
    """Test Fibonacci analysis"""
    logger.info("Testing Fibonacci Analysis")

    # Create test data
    test_data = create_test_market_data(150)

    # Test Fibonacci analysis
    fib = FibonacciAnalysis()
    fib_analysis = fib.calculate_fibonacci_levels(test_data)

    logger.info(f"Fibonacci analysis: Trend={fib_analysis.get('trend', 'unknown')}")
    logger.info(f"Swing High: {fib_analysis.get('swing_high', 'N/A')}")
    logger.info(f"Swing Low: {fib_analysis.get('swing_low', 'N/A')}")

    # Test Fibonacci signals
    fib_signals = fib.get_fib_signals(fib_analysis)
    logger.info(f"Fibonacci signals: {fib_signals}")


def test_advanced_technical_analysis():
    """Test advanced technical analysis integration"""
    logger.info("Testing Advanced Technical Analysis Integration")

    # Create test data
    test_data = create_test_market_data(200)

    # Test advanced analysis
    advanced_signals = advanced_technical_analysis.generate_advanced_signals(test_data)

    if advanced_signals:
        logger.info("Advanced signals generated successfully")
        logger.info(f"Overall signal: {advanced_signals.get('overall_signal')}")
        logger.info(f"Signal strength: {advanced_signals.get('signal_strength', 0):.2f}")
        logger.info(f"Pattern confidence: {advanced_signals.get('pattern_confidence', 0):.2f}")

        # Show pattern analysis summary
        pattern_summary = advanced_technical_analysis.get_pattern_analysis_summary(test_data)
        logger.info(f"Pattern summary: {pattern_summary}")
    else:
        logger.warning("Failed to generate advanced signals")


def test_complete_pattern_recognition():
    """Test complete pattern recognition system"""
    logger.info("Testing Complete Pattern Recognition System")

    # Create test data
    test_data = create_test_market_data(200)

    # Test complete analysis
    pattern_signals = pattern_recognition.get_pattern_signals(test_data)

    logger.info("Complete pattern analysis results:")
    logger.info(f"Overall signal: {pattern_signals.get('overall_signal')}")
    logger.info(f"Confidence: {pattern_signals.get('confidence', 0):.2f}")
    logger.info(f"Harmonic patterns detected: {len(pattern_signals.get('harmonic_patterns', {}))}")
    logger.info(f"Elliott position: {pattern_signals.get('elliott_position', 'unknown')}")
    logger.info(f"Fibonacci level: {pattern_signals.get('fibonacci_level', {})}")


def run_all_pattern_tests():
    """Run all pattern recognition tests"""
    logger.info("=== Starting Advanced Pattern Recognition Tests ===")

    try:
        test_harmonic_patterns()
        print("\n" + "="*60 + "\n")

        test_elliott_waves()
        print("\n" + "="*60 + "\n")

        test_fibonacci_analysis()
        print("\n" + "="*60 + "\n")

        test_advanced_technical_analysis()
        print("\n" + "="*60 + "\n")

        test_complete_pattern_recognition()
        print("\n" + "="*60 + "\n")

        logger.info("=== All Advanced Pattern Recognition Tests Completed Successfully ===")

    except Exception as e:
        logger.error(f"Pattern recognition test failed with error: {e}")
        raise


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    run_all_pattern_tests()
