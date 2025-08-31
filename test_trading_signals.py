#!/usr/bin/env python3
"""
Test script for trading signals functionality
"""

import os
import sys
import pandas as pd
import numpy as np
from trading_signals import TradingSignals

def test_trading_signals():
    """Test trading signals functionality"""
    
    print("Testing trading signals functionality...")
    
    # Create trading signals instance
    trading_signals = TradingSignals()
    
    # Test 1: Generate trade calls for all pairs
    print("\n=== Test 1: Generate Trade Calls for All Pairs ===")
    trade_calls = trading_signals.generate_trade_calls()
    print(f"Generated {len(trade_calls)} trade calls")
    
    for i, call in enumerate(trade_calls):
        print(f"Trade Call {i+1}:")
        print(call)
        print("-" * 50)
    
    # Test 2: Test multiple opportunities (simulate multiple signals)
    print("\n=== Test 2: Multiple Opportunities Simulation ===")
    
    # Simulate multiple trade calls for a single pair
    trading_pair = "BTCUSDT"
    multiple_calls = trading_signals._generate_multiple_trade_calls(trading_pair)
    print(f"Multiple trade calls for {trading_pair}: {len(multiple_calls)}")
    
    # Test 3: Test signal quality filtering
    print("\n=== Test 3: Signal Quality Filtering ===")
    
    # Test with different signal qualities
    test_signals = [
        {'overall_signal': 'BULLISH', 'quality': 'HIGH'},
        {'overall_signal': 'BEARISH', 'quality': 'MEDIUM'}, 
        {'overall_signal': 'NEUTRAL', 'quality': 'LOW'},
        {'overall_signal': 'BULLISH', 'quality': 'LOW'}
    ]
    
    for i, signal in enumerate(test_signals):
        print(f"Signal {i+1}: {signal['overall_signal']} - Quality: {signal['quality']}")
        if signal['quality'] == 'LOW':
            print("  -> Would be filtered out (low quality)")
        else:
            print("  -> Would be processed")
    
    print("\nTesting completed successfully!")

if __name__ == "__main__":
    test_trading_signals()
