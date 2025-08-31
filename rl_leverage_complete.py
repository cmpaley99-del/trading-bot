"""
Reinforcement Learning Environment for Leverage Optimization
Uses Q-learning and DQN to optimize leverage based on market conditions and trading performance
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
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
            risk_amount = self.portfolio_value * Config.RISK_PERCENTAGE / 100
            position_size = (risk_amount * leverage) / current_price

            # Calculate price change
            price_change_pct = (next_price - current_price) / current_price

            # Calculate P&L with leverage
            pnl = position_size * current_price * price_change_pct * leverage

            # Apply transaction costs (0.1% maker fee)
            transaction_cost = abs(pnl) * 0.001
            pnl -= transaction_cost

            # Risk management: penalize excessive drawdown
            if pnl < -self.portfolio_value * self.max_drawdown:
                pnl *= 2  # Double penalty for excessive loss

            # Reward shaping: encourage consistent performance
            reward = pnl

            # Bonus for using optimal leverage in different market conditions
            volatility = current_data.get('atr', 1000) / current_price * 100
            trend_strength = current_data.get('adx', 20)

            if volatility < 0.5 and leverage > 20:  # High leverage in low volatility
                reward += 10
            elif volatility > 2.0 and leverage < 10:  # Conservative in high volatility
                reward += 10
            elif trend_strength > 25 and leverage > 15:  # Higher leverage in strong trends
                reward += 5

            return reward

        except Exception as e:
            logger.error(f"Error calculating reward: {e}")
            return -10  # Penalty for errors

    def _is_done(self):
        """Check if episode should end"""
        # End episode if portfolio is depleted or max steps reached
        if self.portfolio_value <= 1000:  # Less than 10% of initial capital
            return True
        if self.current_step >= self.max_steps:
            return True
        return False

    def _generate_synthetic_data(self):
        """Generate synthetic market data for testing"""
        base_price = 50000
        return {
            'close': base_price + np.random.normal(0, 1000),
            'rsi': np.random.uniform(20, 80),
            'adx': np.random.uniform(15, 35),
            'atr': np.random.uniform(500, 2000),
            'cci': np.random.normal(0, 50),
            'volume_ratio': np.random.uniform(0.5, 2.0),
            'bb_percentage': np.random.uniform(20, 80),
            'macd_hist': np.random.normal(0, 100),
            'stoch_k': np.random.uniform(20, 80),
            'stoch_d': np.random.uniform(20, 80),
            'williams_r': np.random.uniform(-80, -20),
            'ema_5': base_price + np.random.normal(0, 200),
            'ema_10': base_price + np.random.normal(0, 150),
            'ema_20': base_price + np.random.normal(0, 100)
        }


class DQNNetwork(nn.Module):
    """Deep Q-Network for leverage optimization"""

    def __init__(self, input_size=15, output_size=26):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DQN training"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class LeverageOptimizer:
    """Main RL agent for leverage optimization"""

    def __init__(self, trading_pair='BTCUSDT', model_path=None):
        self.trading_pair = trading_pair
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize networks
        self.policy_net = DQNNetwork().to(self.device)
        self.target_net = DQNNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Training parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.memory = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 10

        # Load model if exists
        self.model_path = model_path or f'rl_leverage_model_{trading_pair}.pth'
        if os.path.exists(self.model_path):
            self.load_model()

        # Performance tracking
        self.episode_rewards = []
        self.best_reward = float('-inf')

        logger.info(f"Leverage Optimizer initialized for {trading_pair}")

    def select_action(self, state, training=True):
        """Select action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, 25)  # Random action (5x-30x leverage)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Compute Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        # Compute loss and update
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self):
        """Save model to disk"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'best_reward': self.best_reward
        }, self.model_path)
        logger.info(f"Model saved to {self.model_path}")

    def load_model(self):
        """Load model from disk"""
        try:
            checkpoint = torch.load(self.model_path)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint.get('epsilon', 1.0)
            self.episode_rewards = checkpoint.get('episode_rewards', [])
            self.best_reward = checkpoint.get('best_reward', float('-inf'))
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def get_optimal_leverage(self, market_data):
        """Get optimal leverage for current market conditions"""
        try:
            # Create environment to get state
            env = LeverageOptimizationEnv([market_data], self.trading_pair)
            state = env.reset()

            # Select best action (greedy)
            action = self.select_action(state, training=False)
            leverage = action + 5  # Convert to 5x-30x range

            # Apply bounds
            leverage = max(5, min(30, leverage))

            logger.info(f"RL optimal leverage for {self.trading_pair}: {leverage}x")
            return leverage

        except Exception as e:
            logger.error(f"Error getting optimal leverage: {e}")
            return 10  # Fallback to default

    def train_on_historical_data(self, historical_data, episodes=100):
        """Train the RL agent on historical market data"""
        logger.info(f"Starting RL training for {episodes} episodes on {self.trading_pair}")

        for episode in range(episodes):
            # Create environment with historical data
            env = LeverageOptimizationEnv(historical_data, self.trading_pair)
            state = env.reset()
            episode_reward = 0
            done = False
            steps = 0

            while not done and steps < len(historical_data):
                # Select action
                action = self.select_action(state, training=True)

                # Execute action
                next_state, reward, done, info = env.step(action)

                # Store experience
                self.memory.push(state, action, reward, next_state, done)

                # Train
                loss = self.train_step()

                # Update state and reward
                state = next_state
                episode_reward += reward
                steps += 1

            # Update target network
            if episode % self.target_update == 0:
                self.update_target_network()

            # Track performance
            self.episode_rewards.append(episode_reward)
            if episode_reward > self.best_reward:
                self.best_reward = episode_reward
                self.save_model()

            if episode % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                logger.info(f"Episode {episode}/{episodes}, Avg Reward: {avg_reward:.2f}, "
                          f"Best: {self.best_reward:.2f}, Epsilon: {self.epsilon:.3f}")

        logger.info("RL training completed")
        self.save_model()

    def get_performance_metrics(self):
        """Get training performance metrics"""
        if not self.episode_rewards:
            return {}

        return {
            'total_episodes': len(self.episode_rewards),
            'best_reward': self.best_reward,
            'avg_reward_last_10': np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else 0,
            'avg_reward_all': np.mean(self.episode_rewards),
            'epsilon': self.epsilon,
            'model_path': self.model_path
        }


class RLLeverageManager:
    """Manager class for RL-based leverage optimization across multiple pairs"""

    def __init__(self):
        self.optimizers = {}
        self.performance_data = {}

        # Initialize optimizers for each trading pair
        for pair in Config.TRADING_PAIRS:
            self.optimizers[pair] = LeverageOptimizer(pair)
            self.performance_data[pair] = []

        logger.info("RL Leverage Manager initialized")

    def get_optimal_leverage(self, trading_pair, market_data):
        """Get optimal leverage for a trading pair"""
        if trading_pair not in self.optimizers:
            logger.warning(f"No RL optimizer found for {trading_pair}, using default")
            return Config.DEFAULT_LEVERAGE

        return self.optimizers[trading_pair].get_optimal_leverage(market_data)

    def train_all_pairs(self, historical_data_dict, episodes=50):
        """Train RL models for all trading pairs"""
        logger.info("Starting RL training for all trading pairs")

        for pair in Config.TRADING_PAIRS:
            if pair in historical_data_dict:
                logger.info(f"Training RL model for {pair}")
                self.optimizers[pair].train_on_historical_data(
                    historical_data_dict[pair], episodes
                )
            else:
                logger.warning(f"No historical data found for {pair}")

        logger.info("RL training completed for all pairs")

    def get_all_performance_metrics(self):
        """Get performance metrics for all optimizers"""
        metrics = {}
        for pair, optimizer in self.optimizers.items():
            metrics[pair] = optimizer.get_performance_metrics()
        return metrics

    def save_all_models(self):
        """Save all RL models"""
        for optimizer in self.optimizers.values():
            optimizer.save_model()

    def load_all_models(self):
        """Load all RL models"""
        for optimizer in self.optimizers.values():
            optimizer.load_model()


# Global instance
rl_leverage_manager = RLLeverageManager()
