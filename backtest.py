import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from loguru import logger
from config import Config
from market_data import market_data
from trading_signals import trading_signals
from risk_management import risk_management
import json
import os

class Backtester:
    def __init__(self):
        self.historical_data = {}
        self.performance_metrics = {}
        self.trades = []
        self.portfolio_value = getattr(Config, 'INITIAL_BALANCE', 10000)  # Default to 10000 if not set
        self.initial_balance = getattr(Config, 'INITIAL_BALANCE', 10000)
        logger.info("Backtester initialized")

    async def load_historical_data(self, trading_pairs, start_date, end_date, timeframe='5m'):
        """Load historical OHLCV data for backtesting"""
        logger.info(f"Loading historical data for {len(trading_pairs)} pairs from {start_date} to {end_date}")

        # For backtesting, we'll use the existing market data fetcher
        # In a real implementation, you'd want to use historical data APIs
        for pair in trading_pairs:
            try:
                # Fetch recent data as proxy for historical (limited by API)
                df = market_data.get_ohlcv_data(pair, timeframe, limit=1000)
                if df is not None:
                    # Filter by date range if possible
                    self.historical_data[pair] = df
                    logger.info(f"Loaded {len(df)} candles for {pair}")
                else:
                    logger.warning(f"Failed to load data for {pair}")
            except Exception as e:
                logger.error(f"Error loading historical data for {pair}: {e}")

    async def run_backtest(self, start_date, end_date, trading_pairs):
        """Run backtest simulation"""
        logger.info(f"Starting backtest from {start_date} to {end_date} for pairs: {trading_pairs}")

        # Load historical data
        await self.load_historical_data(trading_pairs, start_date, end_date)

        if not self.historical_data:
            logger.error("No historical data loaded, cannot run backtest")
            return None

        # Reset portfolio
        self.portfolio_value = self.initial_balance
        self.trades = []

        # Simulate trading
        await self._simulate_trading(start_date, end_date, trading_pairs)

        # Calculate performance metrics
        metrics = self._calculate_performance_metrics()

        logger.info(f"Backtest completed. Final portfolio value: ${self.portfolio_value:.2f}")
        return metrics

    async def _simulate_trading(self, start_date, end_date, trading_pairs):
        """Simulate trading based on historical data"""
        # For each trading pair, simulate signals and trades
        for pair in trading_pairs:
            if pair not in self.historical_data:
                continue

            df = self.historical_data[pair]

            # Simulate going through each candle
            for idx, row in df.iterrows():
                current_time = idx
                current_price = row['close']

                # Check if we should generate signals at this time
                # This is a simplified simulation - in reality you'd replay the exact conditions

                # Simulate random signals for demonstration (replace with actual signal logic)
                if np.random.random() < 0.05:  # 5% chance of signal per candle
                    signal_type = 'BUY' if np.random.random() > 0.5 else 'SELL'
                    leverage = getattr(Config, 'DEFAULT_LEVERAGE', 5)  # Default leverage

                    # Calculate position size
                    position_size = risk_management.calculate_position_size(
                        current_price, pair, leverage=leverage
                    )

                    if position_size > 0:
                        # Execute trade
                        self._execute_trade(pair, signal_type, current_price, position_size, current_time)

    def _execute_trade(self, pair, signal_type, price, size, timestamp):
        """Execute a simulated trade"""
        trade = {
            'pair': pair,
            'type': signal_type,
            'price': price,
            'size': size,
            'timestamp': timestamp,
            'portfolio_value': self.portfolio_value
        }

        self.trades.append(trade)

        # Update portfolio value (simplified - in reality would track positions)
        if signal_type == 'BUY':
            cost = price * size
            if cost <= self.portfolio_value:
                self.portfolio_value -= cost
        elif signal_type == 'SELL':
            revenue = price * size
            self.portfolio_value += revenue

        logger.debug(f"Executed {signal_type} trade for {pair} at ${price:.4f}, size: {size}")

    def _calculate_performance_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.trades:
            return {
                'total_return': 0,
                'total_trades': 0,
                'win_rate': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0,
                'profit_factor': 0
            }

        # Calculate returns
        final_value = self.portfolio_value
        total_return = (final_value - self.initial_balance) / self.initial_balance * 100

        # Trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t['type'] == 'SELL'])  # Simplified
        win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0

        # Calculate drawdown
        portfolio_values = [self.initial_balance] + [t['portfolio_value'] for t in self.trades]
        max_drawdown = self._calculate_max_drawdown(portfolio_values)

        # Sharpe ratio (simplified)
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0

        # Profit factor
        gross_profit = sum(t['price'] * t['size'] for t in self.trades if t['type'] == 'SELL')
        gross_loss = sum(t['price'] * t['size'] for t in self.trades if t['type'] == 'BUY')
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        metrics = {
            'total_return': total_return,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'profit_factor': profit_factor,
            'final_portfolio_value': final_value,
            'initial_balance': self.initial_balance
        }

        self.performance_metrics = metrics
        return metrics

    def _calculate_max_drawdown(self, portfolio_values):
        """Calculate maximum drawdown"""
        peak = portfolio_values[0]
        max_drawdown = 0

        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            max_drawdown = max(max_drawdown, drawdown)

        return max_drawdown

    def save_results(self, filename="backtest_results.json"):
        """Save backtest results to file"""
        results = {
            'performance_metrics': self.performance_metrics,
            'trades': self.trades,
            'timestamp': datetime.now().isoformat()
        }

        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"Backtest results saved to {filename}")

    def print_summary(self):
        """Print backtest summary"""
        if not self.performance_metrics:
            logger.info("No backtest results to display")
            return

        metrics = self.performance_metrics
        print("\n" + "="*50)
        print("BACKTEST RESULTS SUMMARY")
        print("="*50)
        print(".2f")
        print(".2f")
        print(f"Total Trades: {metrics['total_trades']}")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print(".2f")
        print("="*50)

# Singleton instance
backtester = Backtester()
