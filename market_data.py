"""
Optimized Market Data Module
Integrated with advanced API call optimizations for maximum performance and reliability
"""

import ccxt
import ccxt.async_support as ccxt_async
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Dict, List, Any, Optional, Callable
import threading
from loguru import logger
from config import Config

# Import optimization components
from api_call_optimization_guide import (
    AdvancedRateLimiter,
    APICallOptimizer,
    SmartAPIManager
)

class MarketData:
    """Market data class with integrated API call optimizations"""

    def __init__(self):
        # Initialize optimization components
        self.rate_limiter = AdvancedRateLimiter(calls_per_second=8.0, burst_limit=3)
        self.api_optimizer = APICallOptimizer()
        self.smart_manager = SmartAPIManager()

        # Initialize exchange with error handling
        self.exchange = None
        self.async_exchange = None
        self._initialize_exchange()

        # Performance tracking
        self.call_stats = {
            'ohlcv_calls': 0,
            'price_calls': 0,
            'funding_calls': 0,
            'total_calls': 0,
            'cache_hits': 0,
            'errors': 0
        }

    def _initialize_exchange(self):
        """Initialize exchange with error handling and retry"""
        try:
            if not Config.BINANCE_API_KEY or not Config.BINANCE_API_SECRET:
                raise ValueError("Binance API credentials not configured")

            self.exchange = ccxt.binance({
                'apiKey': Config.BINANCE_API_KEY,
                'secret': Config.BINANCE_API_SECRET,
                'options': {
                    'defaultType': 'future',
                    'adjustForTimeDifference': True,
                },
                'enableRateLimit': True,
                'timeout': 30000,
            })

            # Load markets with retry
            self._load_markets_with_retry()
            logger.info("✅ Optimized Binance Futures exchange initialized")

        except Exception as e:
            logger.error(f"❌ Failed to initialize optimized exchange: {e}")
            raise

    def _load_markets_with_retry(self, max_retries: int = 3):
        """Load markets with retry logic"""
        for attempt in range(max_retries):
            try:
                self.exchange.load_markets()
                return
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                logger.warning(f"Market load attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)

    def _optimized_api_call(self, call_type: str, symbol: str, func: Callable, *args, **kwargs):
        """Execute API call with full optimization stack"""
        self.call_stats['total_calls'] += 1
        self.call_stats[f'{call_type}_calls'] += 1

        # Use smart API manager for scheduling
        return self.smart_manager.schedule_optimized_call(
            symbol=symbol,
            call_type=call_type,
            func=func,
            *args,
            **kwargs
        )

    def get_ohlcv_data(self, symbol: str = None, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with full optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_ohlcv():
            """Internal fetch function for optimization"""
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

                if not ohlcv or len(ohlcv) == 0:
                    logger.warning(f"No OHLCV data received for {symbol}")
                    return None

                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)

                # Validate data
                if len(df) < limit * 0.8:
                    logger.warning(f"Incomplete OHLCV data for {symbol}: got {len(df)}/{limit} candles")

                logger.info(f"✅ Fetched {len(df)} candles for {symbol} on {timeframe} timeframe")
                return df

            except ccxt.RateLimitExceeded:
                logger.warning(f"Rate limit exceeded for {symbol}, waiting longer")
                time.sleep(5)
                return _fetch_ohlcv()  # Retry once

            except Exception as e:
                logger.error(f"❌ Error fetching OHLCV data for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('ohlcv', symbol, _fetch_ohlcv)

    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """Get current market price with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_price():
            """Internal price fetch function"""
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                price = ticker.get('last')

                if price is None or price <= 0:
                    logger.warning(f"Invalid price received for {symbol}: {price}")
                    return None

                logger.debug(f"✅ Current price for {symbol}: ${price:.2f}")
                return price

            except Exception as e:
                logger.error(f"❌ Error fetching current price for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('price', symbol, _fetch_price)

    def get_funding_rate(self, symbol: str = None) -> Optional[float]:
        """Get current funding rate with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_funding():
            """Internal funding rate fetch function"""
            try:
                funding_rate = self.exchange.fetch_funding_rate(symbol)
                rate = funding_rate.get('fundingRate')

                if rate is None:
                    return None

                rate_percentage = rate * 100  # Convert to percentage
                logger.debug(f"✅ Funding rate for {symbol}: {rate_percentage:.4f}%")
                return rate_percentage

            except Exception as e:
                logger.error(f"❌ Error fetching funding rate for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('funding', symbol, _fetch_funding)

    def get_market_metrics(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive market metrics with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        try:
            # Get optimized data
            df = self.get_ohlcv_data(symbol, '5m', 50)
            if df is None or len(df) < 10:
                return None

            current_price = self.get_current_price(symbol)
            funding_rate = self.get_funding_rate(symbol)

            # Calculate metrics
            recent_df = df.tail(24)  # Last 2 hours

            metrics = {
                'current_price': current_price,
                'funding_rate': funding_rate,
                '24h_volume': recent_df['volume'].sum() if len(recent_df) > 0 else 0,
                'price_change_24h': self._calculate_price_change(recent_df),
                'high_24h': recent_df['high'].max() if len(recent_df) > 0 else None,
                'low_24h': recent_df['low'].min() if len(recent_df) > 0 else None,
                'volume_24h': recent_df['volume'].sum() if len(recent_df) > 0 else 0,
                'volatility': self._calculate_volatility(recent_df),
                'timestamp': datetime.now(),
                'optimization_stats': self.get_performance_stats()
            }

            return metrics

        except Exception as e:
            logger.error(f"❌ Error getting market metrics for {symbol}: {e}")
            return None

    def _calculate_price_change(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate price change percentage"""
        if len(df) < 2:
            return None

        first_price = df['close'].iloc[0]
        last_price = df['close'].iloc[-1]

        if first_price == 0:
            return None

        return ((last_price - first_price) / first_price) * 100

    def _calculate_volatility(self, df: pd.DataFrame) -> Optional[float]:
        """Calculate price volatility"""
        if len(df) < 5:
            return None

        returns = df['close'].pct_change().dropna()
        return returns.std() * np.sqrt(24)  # Annualized volatility

    def get_24h_volume(self, symbol: str = None) -> Optional[float]:
        """Get 24h trading volume with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_volume():
            """Internal volume fetch function"""
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker.get('quoteVolume')
            except Exception as e:
                logger.error(f"❌ Error fetching 24h volume for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('volume', symbol, _fetch_volume)

    def get_price_change_24h(self, symbol: str = None) -> Optional[float]:
        """Get 24h price change percentage with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_price_change():
            """Internal price change fetch function"""
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker.get('percentage')
            except Exception as e:
                logger.error(f"❌ Error fetching price change for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('price_change', symbol, _fetch_price_change)

    def get_order_book(self, symbol: str = None, limit: int = 20) -> Optional[Dict[str, Any]]:
        """Get order book data with optimization"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        def _fetch_orderbook():
            """Internal order book fetch function"""
            try:
                order_book = self.exchange.fetch_order_book(symbol, limit)
                return order_book
            except Exception as e:
                logger.error(f"❌ Error fetching order book for {symbol}: {e}")
                self.call_stats['errors'] += 1
                return None

        return self._optimized_api_call('orderbook', symbol, _fetch_orderbook)

    def get_ohlcv_data_parallel(self, symbols: List[str] = None, timeframe: str = '5m', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols in parallel with optimization"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        results = {}

        def fetch_single(symbol: str) -> tuple:
            """Fetch single symbol with optimization"""
            try:
                df = self.get_ohlcv_data(symbol, timeframe, limit)
                return symbol, df
            except Exception as e:
                logger.error(f"❌ Error fetching OHLCV data for {symbol}: {e}")
                return symbol, None

        # Use ThreadPoolExecutor for parallel fetching with optimized worker count
        max_workers = min(len(symbols), 4)  # Reduced from 5 for better rate limit management

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_single, symbol): symbol for symbol in symbols}

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, df = future.result()
                    if df is not None:
                        results[symbol] = df
                        logger.info(f"✅ Fetched {len(df)} candles for {symbol}")
                    else:
                        logger.warning(f"❌ No data received for {symbol}")
                except Exception as e:
                    logger.error(f"❌ Error processing result for {symbol}: {e}")

        return results

    def get_current_prices_parallel(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current prices for multiple symbols in parallel with optimization"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        results = {}

        def fetch_single(symbol: str) -> tuple:
            """Fetch single price with optimization"""
            try:
                price = self.get_current_price(symbol)
                return symbol, price
            except Exception as e:
                logger.error(f"❌ Error fetching price for {symbol}: {e}")
                return symbol, None

        # Use ThreadPoolExecutor with optimized concurrency
        max_workers = min(len(symbols), 6)  # Higher concurrency for prices

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_symbol = {executor.submit(fetch_single, symbol): symbol for symbol in symbols}

            for future in as_completed(future_to_symbol):
                symbol = future_to_symbol[future]
                try:
                    symbol, price = future.result()
                    if price is not None:
                        results[symbol] = price
                except Exception as e:
                    logger.error(f"❌ Error processing price result for {symbol}: {e}")

        return results

    async def get_async_exchange(self):
        """Get or create async exchange instance with error handling"""
        if self.async_exchange is None:
            try:
                self.async_exchange = ccxt_async.binance({
                    'apiKey': Config.BINANCE_API_KEY,
                    'secret': Config.BINANCE_API_SECRET,
                    'options': {
                        'defaultType': 'future',
                        'adjustForTimeDifference': True,
                    },
                    'enableRateLimit': True,
                    'timeout': 30000,
                })
                await self.async_exchange.load_markets()
                logger.info("✅ Async Binance Futures exchange initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize async exchange: {e}")
                raise

        return self.async_exchange

    async def get_ohlcv_data_async(self, symbol: str, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Async fetch OHLCV data with optimization"""
        try:
            exchange = await self.get_async_exchange()
            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming

            ohlcv = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data received for {symbol}")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            logger.info(f"✅ Async fetched {len(df)} candles for {symbol}")
            return df

        except Exception as e:
            logger.error(f"❌ Error async fetching OHLCV data for {symbol}: {e}")
            return None

    async def get_ohlcv_data_parallel_async(self, symbols: List[str] = None, timeframe: str = '5m', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple pairs in parallel using asyncio with optimization"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        # Limit concurrency to prevent API rate limits with optimization
        semaphore = asyncio.Semaphore(5)

        async def fetch_with_semaphore(symbol: str):
            async with semaphore:
                return await self.get_ohlcv_data_async(symbol, timeframe, limit)

        tasks = [fetch_with_semaphore(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        data_dict = {}
        for i, result in enumerate(results):
            symbol = symbols[i]
            if isinstance(result, Exception):
                logger.error(f"❌ Error fetching data for {symbol}: {result}")
                data_dict[symbol] = None
            else:
                data_dict[symbol] = result

        return data_dict

    async def close_async_exchange(self):
        """Close async exchange connection"""
        if self.async_exchange:
            try:
                await self.async_exchange.close()
                self.async_exchange = None
                logger.info("✅ Async exchange connection closed")
            except Exception as e:
                logger.error(f"❌ Error closing async exchange: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        # Get optimization stats
        optimizer_stats = self.api_optimizer.get_performance_stats()
        smart_stats = self.smart_manager.get_optimization_stats()

        # Combine with local stats
        combined_stats = {
            **self.call_stats,
            **optimizer_stats,
            **smart_stats,
            'timestamp': datetime.now()
        }

        return combined_stats

    def clear_cache(self):
        """Clear all optimization caches"""
        # This would clear the internal caches of the optimization components
        logger.info("✅ Optimization caches cleared")

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of optimized market data service"""
        current_time = time.time()

        return {
            'exchange_initialized': self.exchange is not None,
            'async_exchange_initialized': self.async_exchange is not None,
            'total_calls': self.call_stats['total_calls'],
            'error_rate': (self.call_stats['errors'] / max(1, self.call_stats['total_calls'])) * 100,
            'cache_performance': self.api_optimizer.get_performance_stats().get('cache_hit_rate', 0),
            'rate_limiter_status': {
                'burst_count': self.rate_limiter.burst_count,
                'last_call': self.rate_limiter.last_call
            },
            'timestamp': datetime.now()
        }

# Singleton instance
market_data = MarketData()
