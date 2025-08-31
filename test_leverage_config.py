#!/usr/bin/env python3
"""
Test script for different leverage configurations
"""

import os
import sys
from config import Config

def test_leverage_configurations():
    """Test different leverage configurations"""
    
    print("Testing leverage configurations...")
    
    # Test 1: Default configuration (no LEVERAGES env var)
    print("\n=== Test 1: Default Configuration ===")
    if 'LEVERAGES' in os.environ:
        del os.environ['LEVERAGES']
    
    # Re-import config to refresh
    import importlib
    importlib.reload(sys.modules['config'])
    from config import Config
    
    print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
    print(f"LEVERAGE_MAP: {Config.LEVERAGE_MAP}")
    
    # Test 2: Custom leverage configuration
    print("\n=== Test 2: Custom Leverage Configuration ===")
    os.environ['LEVERAGES'] = '15,25,8'
    
    # Re-import config to refresh
    importlib.reload(sys.modules['config'])
    from config import Config
    
    print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
    print(f"LEVERAGE_MAP: {Config.LEVERAGE_MAP}")
    
    # Test 3: Mismatched configuration (should fallback)
    print("\n=== Test 3: Mismatched Configuration ===")
    os.environ['LEVERAGES'] = '10,20'  # Only 2 values for 3 pairs
    
    # Re-import config to refresh
    importlib.reload(sys.modules['config'])
    from config import Config
    
    print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
    print(f"LEVERAGE_MAP: {Config.LEVERAGE_MAP}")
    
    # Test 4: Invalid leverage values
    print("\n=== Test 4: Invalid Leverage Values ===")
    os.environ['LEVERAGES'] = 'invalid,20,15'
    
    # Re-import config to refresh
    importlib.reload(sys.modules['config'])
    from config import Config
    
    print(f"TRADING_PAIRS: {Config.TRADING_PAIRS}")
    print(f"LEVERAGE_MAP: {Config.LEVERAGE_MAP}")

if __name__ == "__main__":
    test_leverage_configurations()
