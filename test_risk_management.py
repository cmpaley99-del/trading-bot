#!/usr/bin/env python3
"""
Test script for risk management with different leverage configurations
"""

import os
import sys
from risk_management import RiskManagement

def test_risk_management():
    """Test risk management with different leverage configurations"""
    
    print("Testing risk management with different leverage configurations...")
    
    # Create risk management instance
    risk_mgmt = RiskManagement()
    
    # Test different leverage scenarios
    test_cases = [
        {"trading_pair": "BTCUSDT", "current_price": 50000, "leverage": 10, "expected_size": "~0.004"},
        {"trading_pair": "ETHUSDT", "current_price": 2500, "leverage": 20, "expected_size": "~0.16"},
        {"trading_pair": "SOLUSDT", "current_price": 100, "leverage": 5, "expected_size": "~2.0"}
    ]
    
    print("\n=== Risk Management Position Size Calculation ===")
    
    for i, test_case in enumerate(test_cases):
        trading_pair = test_case["trading_pair"]
        current_price = test_case["current_price"]
        leverage = test_case["leverage"]
        expected_size = test_case["expected_size"]
        
        # Calculate position size with specific leverage
        position_size = risk_mgmt.calculate_position_size(current_price, trading_pair, leverage=leverage)
        
        print(f"Test {i+1}: {trading_pair}")
        print(f"  Price: ${current_price:,}")
        print(f"  Leverage: {leverage}x")
        print(f"  Position Size: {position_size:.6f} {trading_pair.replace('USDT', '')}")
        print(f"  Expected: {expected_size}")
        print()
    
    # Test stop loss and take profit calculations
    print("\n=== Stop Loss and Take Profit Calculations ===")
    
    test_scenarios = [
        {"entry_price": 50000, "signal_type": "BULLISH", "atr": 500},
        {"entry_price": 2500, "signal_type": "BEARISH", "atr": 25},
        {"entry_price": 100, "signal_type": "BULLISH", "atr": 2}
    ]
    
    for i, scenario in enumerate(test_scenarios):
        entry_price = scenario["entry_price"]
        signal_type = scenario["signal_type"]
        atr = scenario["atr"]
        
        stop_loss = risk_mgmt.calculate_stop_loss(entry_price, signal_type, atr)
        take_profit = risk_mgmt.calculate_take_profit(entry_price, signal_type, stop_loss)
        
        print(f"Scenario {i+1}: {signal_type} signal")
        print(f"  Entry: ${entry_price:,}")
        print(f"  Stop Loss: ${stop_loss:,.2f}")
        print(f"  Take Profit: ${take_profit:,.2f}")
        
        if signal_type == "BULLISH":
            risk = entry_price - stop_loss
            reward = take_profit - entry_price
        else:
            risk = stop_loss - entry_price
            reward = entry_price - take_profit
        
        risk_reward = reward / risk
        print(f"  Risk-Reward Ratio: 1:{risk_reward:.2f}")
        print()
    
    print("Risk management testing completed successfully!")

if __name__ == "__main__":
    test_risk_management()
