"""
Improved Market Data Module
Enhanced with caching, rate limiting, error handling, and performance optimizations
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
from functools import lru_cache
from typing import Optional, Dict, List, Any, Union
import threading
from loguru import logger
from config import Config

class RateLimiter:
    """Simple rate limiter for API calls"""

    def __init__(self, calls_per_second: float = 10.0):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            current_time = time.time()
            time_since_last_call = current_time - self.last_call

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)

            self.last_call = time.time()

class DataCache:
    """Simple LRU cache for market data"""

    def __init__(self, max_size: int = 100, ttl_seconds: int = 300):
        self.cache = {}
        self.max_size = max_size
        self.ttl = ttl_seconds
        self.lock = threading.Lock()

    def get(self, key: str) -> Any:
        """Get item from cache"""
        with self.lock:
            if key in self.cache:
                item, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    return item
                else:
                    # Expired, remove it
                    del self.cache[key]
            return None

    def set(self, key: str, value: Any):
        """Set item in cache"""
        with self.lock:
            # Remove oldest items if cache is full
            if len(self.cache) >= self.max_size:
                oldest_key = min(self.cache.keys(),
                               key=lambda k: self.cache[k][1])
                del self.cache[oldest_key]

            self.cache[key] = (value, time.time())

    def clear(self):
        """Clear all cached data"""
        with self.lock:
            self.cache.clear()

class ImprovedMarketData:
    """Improved market data class with caching and error handling"""

    def __init__(self):
        self.rate_limiter = RateLimiter(calls_per_second=8.0)  # Conservative rate limiting
        self.cache = DataCache(max_size=200, ttl_seconds=300)  # 5-minute cache

        # Initialize exchange with error handling
        self.exchange = None
        self.async_exchange = None
        self._initialize_exchange()

        # Connection health tracking
        self.last_successful_call = time.time()
        self.consecutive_failures = 0
        self.max_consecutive_failures = 5

    def _initialize_exchange(self):
        """Initialize exchange with error handling"""
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
                'timeout': 30000,  # 30 second timeout
            })

            # Load markets with retry
            self._load_markets_with_retry()
            logger.info("✅ Binance Futures exchange initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize exchange: {e}")
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
                time.sleep(2 ** attempt)  # Exponential backoff

    def _handle_api_error(self, error: Exception, operation: str) -> None:
        """Handle API errors and update health tracking"""
        self.consecutive_failures += 1
        logger.error(f"API Error in {operation}: {error}")

        if self.consecutive_failures >= self.max_consecutive_failures:
            logger.warning(f"High failure rate detected ({self.consecutive_failures} consecutive failures)")
            # Could implement circuit breaker here

    def _handle_api_success(self) -> None:
        """Handle successful API call"""
        self.last_successful_call = time.time()
        self.consecutive_failures = 0

    def _get_cache_key(self, operation: str, symbol: str, **kwargs) -> str:
        """Generate cache key for operation"""
        params = "_".join([f"{k}:{v}" for k, v in sorted(kwargs.items())])
        return f"{operation}_{symbol}_{params}"

    @lru_cache(maxsize=32)
    def get_ohlcv_data(self, symbol: str = None, timeframe: str = '5m', limit: int = 100) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data with caching and error handling"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        # Check cache first
        cache_key = self._get_cache_key('ohlcv', symbol, timeframe=timeframe, limit=limit)
        cached_data = self.cache.get(cache_key)
        if cached_data is not None:
            logger.debug(f"Cache hit for OHLCV data: {symbol}")
            return cached_data

        try:
            self.rate_limiter.wait_if_needed()

            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)

            if not ohlcv or len(ohlcv) == 0:
                logger.warning(f"No OHLCV data received for {symbol}")
                return None

            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Validate data
            if len(df) < limit * 0.8:  # Allow 20% data loss
                logger.warning(f"Incomplete OHLCV data for {symbol}: got {len(df)}/{limit} candles")

            # Cache the result
            self.cache.set(cache_key, df)
            self._handle_api_success()

            logger.info(f"✅ Fetched {len(df)} candles for {symbol} on {timeframe} timeframe")
            return df

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit exceeded for {symbol}, waiting longer")
            time.sleep(5)
            return self.get_ohlcv_data(symbol, timeframe, limit)  # Retry once

        except Exception as e:
            self._handle_api_error(e, f"get_ohlcv_data({symbol})")
            return None

    def get_current_price(self, symbol: str = None) -> Optional[float]:
        """Get current market price with caching"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        # Check cache (shorter TTL for price data)
        cache_key = f"price_{symbol}"
        cached_price = self.cache.get(cache_key)
        if cached_price is not None:
            return cached_price

        try:
            self.rate_limiter.wait_if_needed()

            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker.get('last')

            if price is None or price <= 0:
                logger.warning(f"Invalid price received for {symbol}: {price}")
                return None

            # Cache for 30 seconds (price data changes frequently)
            self.cache.set(cache_key, price)
            self._handle_api_success()

            return price

        except Exception as e:
            self._handle_api_error(e, f"get_current_price({symbol})")
            return None

    def get_funding_rate(self, symbol: str = None) -> Optional[float]:
        """Get current funding rate for futures"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        cache_key = f"funding_{symbol}"
        cached_rate = self.cache.get(cache_key)
        if cached_rate is not None:
            return cached_rate

        try:
            self.rate_limiter.wait_if_needed()

            funding_rate = self.exchange.fetch_funding_rate(symbol)
            rate = funding_rate.get('fundingRate')

            if rate is None:
                return None

            rate_percentage = rate * 100  # Convert to percentage

            # Cache for 1 hour (funding rates change every 8 hours)
            self.cache.set(cache_key, rate_percentage)
            self._handle_api_success()

            return rate_percentage

        except Exception as e:
            self._handle_api_error(e, f"get_funding_rate({symbol})")
            return None

    def get_market_metrics(self, symbol: str = None) -> Optional[Dict[str, Any]]:
        """Get comprehensive market metrics with error handling"""
        if symbol is None:
            symbol = Config.TRADING_PAIRS[0]

        try:
            # Get OHLCV data
            df = self.get_ohlcv_data(symbol, '5m', 50)
            if df is None or len(df) < 10:
                return None

            current_price = self.get_current_price(symbol)
            funding_rate = self.get_funding_rate(symbol)

            # Calculate basic metrics
            recent_df = df.tail(24)  # Last 2 hours (24 * 5min)

            metrics = {
                'current_price': current_price,
                'funding_rate': funding_rate,
                '24h_volume': recent_df['volume'].sum() if len(recent_df) > 0 else 0,
                'price_change_24h': self._calculate_price_change(recent_df),
                'high_24h': recent_df['high'].max() if len(recent_df) > 0 else None,
                'low_24h': recent_df['low'].min() if len(recent_df) > 0 else None,
                'volume_24h': recent_df['volume'].sum() if len(recent_df) > 0 else 0,
                'volatility': self._calculate_volatility(recent_df),
                'timestamp': datetime.now()
            }

            return metrics

        except Exception as e:
            self._handle_api_error(e, f"get_market_metrics({symbol})")
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

    def get_ohlcv_data_parallel(self, symbols: List[str] = None, timeframe: str = '5m', limit: int = 100) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV data for multiple symbols in parallel with error handling"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        results = {}

        def fetch_single(symbol: str) -> tuple:
            try:
                df = self.get_ohlcv_data(symbol, timeframe, limit)
                return symbol, df
            except Exception as e:
                logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
                return symbol, None

        # Use ThreadPoolExecutor for parallel fetching
        max_workers = min(len(symbols), 4)  # Limit concurrent requests

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
                    logger.error(f"Error processing result for {symbol}: {e}")

        return results

    def get_current_prices_parallel(self, symbols: List[str] = None) -> Dict[str, float]:
        """Get current prices for multiple symbols in parallel"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        results = {}

        def fetch_single(symbol: str) -> tuple:
            try:
                price = self.get_current_price(symbol)
                return symbol, price
            except Exception as e:
                logger.error(f"Error fetching price for {symbol}: {e}")
                return symbol, None

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
                    logger.error(f"Error processing price result for {symbol}: {e}")

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
        """Async fetch OHLCV data with error handling"""
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
        """Fetch OHLCV data for multiple pairs in parallel using asyncio"""
        if symbols is None:
            symbols = Config.TRADING_PAIRS

        # Limit concurrency to prevent API rate limits
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

    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of market data service"""
        current_time = time.time()

        return {
            'last_successful_call': datetime.fromtimestamp(self.last_successful_call),
            'consecutive_failures': self.consecutive_failures,
            'time_since_last_success': current_time - self.last_successful_call,
            'cache_size': len(self.cache.cache),
            'exchange_initialized': self.exchange is not None,
            'async_exchange_initialized': self.async_exchange is not None
        }

    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("✅ Market data cache cleared")

# Singleton instance
market_data = ImprovedMarketData()
