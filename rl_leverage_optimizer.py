"""
Reinforcement Learning Environment for Leverage Optimization
Uses Q-learning and DQN to optimize leverage based on market conditions and trading performance
"""

import numpy as np
import pandas as pd
import gym
from gym import spaces
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from loguru import logger
from config import Config
import technical_analysis
import risk_management
import json
import os
from datetime import datetime, timedelta


class LeverageOptimizationEnv(gym.Env):
    """
    Custom Gym environment for leverage optimization
    State: Market conditions (volatility, trend strength, momentum, etc.)
    Action: Leverage multiplier (5x to 30x)
    Reward: Profit/Loss based on trading performance
    """

    def __init__(self, historical_data=None, trading_pair='BTCUSDT'):
        super(LeverageOptimizationEnv, self).__init__()

        self.trading_pair = trading_pair
        self.historical_data = historical_data
        self.current_step = 0
        self.max_steps = len(historical_data) - 1 if historical_data is not None else 1000

        # Define action space: leverage from 5x to 30x (26 possible actions)
        self.action_space = spaces.Discrete(26)  # 0-25 corresponding to 5x-30x

        # Define observation space: market state features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(15,), dtype=np.float32
        )

        # Initialize state
        self.state = None
        self.portfolio_value = 10000.0  # Starting portfolio value
        self.position = None
        self.entry_price = None
        self.current_leverage = 10

        # Risk management parameters
        self.max_drawdown = 0.1  # 10% max drawdown
        self.stop_loss_pct = 0.02  # 2% stop loss
        self.take_profit_pct = 0.04  # 4% take profit

        logger.info(f"Leverage Optimization Environment initialized for {trading_pair}")

    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        self.portfolio_value = 10000.0
        self.position = None
        self.entry_price = None
        self.current_leverage = 10

        # Get initial state
        self.state = self._get_state()
        return self.state

    def step(self, action):
        """Execute one step in the environment"""
        # Convert action to leverage (5x to 30x)
        leverage = action + 5

        # Get current market data
        if self.historical_data is not None and self.current_step < len(self.historical_data):
            current_data = self.historical_data.iloc[self.current_step]
            next_data = self.historical_data.iloc[min(self.current_step + 1, len(self.historical_data) - 1)]
        else:
            # Generate synthetic data if no historical data
            current_data = self._generate_synthetic_data()
            next_data = self._generate_synthetic_data()

        # Calculate reward based on leverage performance
        reward = self._calculate_reward(leverage, current_data, next_data)

        # Update portfolio value
        self.portfolio_value += reward

        # Update state
        self.current_step += 1
        self.state = self._get_state()

        # Check if episode is done
        done = self._is_done()

        # Additional info
        info = {
            'leverage': leverage,
            'portfolio_value': self.portfolio_value,
            'reward': reward,
            'step': self.current_step
        }

        return self.state, reward, done, info

    def _get_state(self):
        """Get current market state as observation"""
        if self.historical_data is not None and self.current_step < len(self.historical_data):
            data = self.historical_data.iloc[self.current_step]

            # Extract relevant features for state
            state = np.array([
                data.get('close', 50000),  # Current price
                data.get('rsi', 50),  # RSI
                data.get('adx', 20),  # ADX (trend strength)
                data.get('atr', 1000),  # ATR (volatility)
                data.get('cci', 0),  # CCI (momentum)
                data.get('volume_ratio', 1),  # Volume ratio
                data.get('bb_percentage', 50),  # Bollinger Band position
                data.get('macd_hist', 0),  # MACD histogram
                data.get('stoch_k', 50),  # Stochastic K
                data.get('stoch_d', 50),  # Stochastic D
                data.get('williams_r', -50),  # Williams %R
                data.get('ema_5', data.get('close', 50000)),  # EMA 5
                data.get('ema_10', data.get('close', 50000)),  # EMA 10
                data.get('ema_20', data.get('close', 50000)),  # EMA 20
                self.portfolio_value / 10000.0  # Normalized portfolio value
            ], dtype=np.float32)
        else:
            # Default state for synthetic data
            state = np.array([
                50000, 50, 20, 1000, 0, 1, 50, 0, 50, 50, -50,
                50000, 50000, 50000, 1.0
            ], dtype=np.float32)

        return state

    def _calculate_reward(self, leverage, current_data, next_data):
        """Calculate reward based on leverage performance"""
        try:
            current_price = current_data.get('close', 50000)
            next_price = next_data.get('close', current_price * 1.001)  # Small default change

            # Simulate position sizing with different leverage
            risk_amount = self.portfolio_value *
