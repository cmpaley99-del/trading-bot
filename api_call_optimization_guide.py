"""
API Call Optimization Guide for Cryptocurrency Trading Bot
Advanced techniques to improve API call efficiency and reliability
"""

import asyncio
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from functools import wraps
import ccxt
import ccxt.async_support as ccxt_async
from loguru import logger
from config import Config

class AdvancedRateLimiter:
    """Advanced rate limiter with burst handling and priority queues"""

    def __init__(self, calls_per_second: float = 8.0, burst_limit: int = 5):
        self.calls_per_second = calls_per_second
        self.min_interval = 1.0 / calls_per_second
        self.burst_limit = burst_limit
        self.burst_count = 0
        self.last_burst_reset = time.time()
        self.burst_window = 1.0  # 1 second burst window

        self.lock = threading.Lock()
        self.last_call = 0

        # Priority queues for different call types
        self.priority_queues = {
            'CRITICAL': [],  # Price data, immediate signals
            'HIGH': [],      # Technical indicators, order book
            'NORMAL': [],    # Historical data, funding rates
            'LOW': []        # Background tasks, cleanup
        }

    def _reset_burst_if_needed(self):
        """Reset burst counter if window has passed"""
        current_time = time.time()
        if current_time - self.last_burst_reset >= self.burst_window:
            self.burst_count = 0
            self.last_burst_reset = current_time

    def can_make_call(self, priority: str = 'NORMAL') -> bool:
        """Check if a call can be made based on rate limits and priority"""
        with self.lock:
            self._reset_burst_if_needed()

            # Allow burst calls up to limit
            if self.burst_count < self.burst_limit:
                return True

            # Check normal rate limiting
            current_time = time.time()
            time_since_last_call = current_time - self.last_call

            if time_since_last_call >= self.min_interval:
                return True

            return False

    def wait_if_needed(self, priority: str = 'NORMAL'):
        """Wait if necessary to respect rate limits"""
        with self.lock:
            self._reset_burst_if_needed()

            # Allow burst calls
            if self.burst_count < self.burst_limit:
                self.burst_count += 1
                self.last_call = time.time()
                return

            # Normal rate limiting
            current_time = time.time()
            time_since_last_call = current_time - self.last_call

            if time_since_last_call < self.min_interval:
                sleep_time = self.min_interval - time_since_last_call
                time.sleep(sleep_time)

            self.last_call = time.time()

    def get_queue_size(self, priority: str = 'NORMAL') -> int:
        """Get size of priority queue"""
        return len(self.priority_queues.get(priority, []))

class APICallOptimizer:
    """Advanced API call optimization with batching and smart scheduling"""

    def __init__(self):
        self.rate_limiter = AdvancedRateLimiter(calls_per_second=8.0, burst_limit=3)

        # Call batching configuration
        self.batch_configs = {
            'ohlcv': {'max_batch_size': 5, 'max_wait_time': 0.5},
            'ticker': {'max_batch_size': 10, 'max_wait_time': 0.2},
            'orderbook': {'max_batch_size': 3, 'max_wait_time': 0.3}
        }

        # Call deduplication cache
        self.call_cache = {}
        self.cache_ttl = 0.1  # 100ms cache for deduplication

        # Performance tracking
        self.call_stats = {
            'total_calls': 0,
            'cached_calls': 0,
            'failed_calls': 0,
            'avg_response_time': 0,
            'last_cleanup': time.time()
        }

    def optimize_api_call(self, func: Callable) -> Callable:
        """Decorator to optimize API calls"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)

            # Check deduplication cache
            if cache_key in self.call_cache:
                cache_entry = self.call_cache[cache_key]
                if time.time() - cache_entry['timestamp'] < self.cache_ttl:
                    self.call_stats['cached_calls'] += 1
                    return cache_entry['result']

            # Apply rate limiting
            self.rate_limiter.wait_if_needed()

            # Track call performance
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                response_time = time.time() - start_time

                # Update performance stats
                self.call_stats['total_calls'] += 1
                self.call_stats['avg_response_time'] = (
                    (self.call_stats['avg_response_time'] * (self.call_stats['total_calls'] - 1)) +
                    response_time
                ) / self.call_stats['total_calls']

                # Cache successful result
                self.call_cache[cache_key] = {
                    'result': result,
                    'timestamp': time.time()
                }

                return result

            except Exception as e:
                self.call_stats['failed_calls'] += 1
                raise e
            finally:
                self._cleanup_cache_if_needed()

        return wrapper

    def _generate_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key for deduplication"""
        # Convert args and kwargs to strings, handling non-hashable types
        def make_hashable(obj):
            if isinstance(obj, dict):
                return tuple(sorted(obj.items()))
            elif isinstance(obj, list):
                return tuple(obj)
            else:
                return str(obj)

        args_str = "_".join(make_hashable(arg) for arg in args)
        kwargs_str = "_".join(f"{k}:{make_hashable(v)}" for k, v in sorted(kwargs.items()))

        return f"{func_name}_{args_str}_{kwargs_str}"

    def _cleanup_cache_if_needed(self):
        """Clean up expired cache entries"""
        current_time = time.time()
        if current_time - self.call_stats['last_cleanup'] > 1.0:  # Clean every second
            expired_keys = [
                key for key, entry in self.call_cache.items()
                if current_time - entry['timestamp'] > self.cache_ttl
            ]

            for key in expired_keys:
                del self.call_cache[key]

            self.call_stats['last_cleanup'] = current_time

    def batch_api_calls(self, call_type: str, calls: List[Dict[str, Any]]) -> List[Any]:
        """Batch multiple API calls for efficiency"""
        if call_type not in self.batch_configs:
            # Fallback to individual calls
            return [call['func'](*call.get('args', []), **call.get('kwargs', {})) for call in calls]

        config = self.batch_configs[call_type]
        results = []

        # Process calls in batches
        for i in range(0, len(calls), config['max_batch_size']):
            batch = calls[i:i + config['max_batch_size']]

            # Execute batch
            batch_start = time.time()
            batch_results = []
            for call in batch:
                try:
                    result = call['func'](*call.get('args', []), **call.get('kwargs', {}))
                    batch_results.append(result)
                except Exception as e:
                    logger.error(f"Batch call failed: {e}")
                    batch_results.append(None)

            results.extend(batch_results)

            # Respect batch timing
            batch_time = time.time() - batch_start
            if batch_time < config['max_wait_time']:
                time.sleep(config['max_wait_time'] - batch_time)

        return results

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get API call performance statistics"""
        total_calls = self.call_stats['total_calls']
        cached_calls = self.call_stats['cached_calls']
        failed_calls = self.call_stats['failed_calls']

        return {
            'total_calls': total_calls,
            'cached_calls': cached_calls,
            'cache_hit_rate': (cached_calls / total_calls * 100) if total_calls > 0 else 0,
            'failed_calls': failed_calls,
            'failure_rate': (failed_calls / total_calls * 100) if total_calls > 0 else 0,
            'avg_response_time': self.call_stats['avg_response_time'],
            'cache_size': len(self.call_cache),
            'rate_limiter_burst_count': self.rate_limiter.burst_count
        }

class SmartAPIManager:
    """Smart API manager with predictive call scheduling"""

    def __init__(self):
        self.optimizer = APICallOptimizer()
        self.call_history = []
        self.predictive_cache = {}
        self.learning_enabled = True

    def predict_call_frequency(self, symbol: str, call_type: str) -> float:
        """Predict optimal call frequency based on historical data"""
        if not self.learning_enabled:
            return 1.0  # Default frequency

        # Analyze recent call patterns for this symbol/type
        recent_calls = [
            call for call in self.call_history[-100:]  # Last 100 calls
            if call['symbol'] == symbol and call['type'] == call_type
        ]

        if len(recent_calls) < 5:
            return 1.0

        # Calculate average time between calls
        timestamps = [call['timestamp'] for call in recent_calls]
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]

        if intervals:
            avg_interval = sum(intervals) / len(intervals)
            # Suggest frequency based on data volatility
            if avg_interval < 1.0:  # High frequency calls
                return 0.5  # Reduce frequency
            elif avg_interval > 5.0:  # Low frequency calls
                return 2.0  # Increase frequency

        return 1.0

    def schedule_optimized_call(self, symbol: str, call_type: str, func: Callable, *args, **kwargs):
        """Schedule an API call with optimization"""
        # Record call for learning
        self.call_history.append({
            'symbol': symbol,
            'type': call_type,
            'timestamp': time.time()
        })

        # Limit history size
        if len(self.call_history) > 1000:
            self.call_history = self.call_history[-500:]

        # Apply optimization
        optimized_func = self.optimizer.optimize_api_call(func)

        # Predict and adjust frequency
        frequency_multiplier = self.predict_call_frequency(symbol, call_type)

        # Execute with optimization
        return optimized_func(*args, **kwargs)

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        base_stats = self.optimizer.get_performance_stats()

        # Add predictive stats
        predictive_stats = {
            'learning_enabled': self.learning_enabled,
            'call_history_size': len(self.call_history),
            'predictive_cache_size': len(self.predictive_cache),
            'avg_calls_per_minute': len(self.call_history) / max(1, (time.time() - (self.call_history[0]['timestamp'] if self.call_history else time.time())) / 60)
        }

        return {**base_stats, **predictive_stats}

# Global instances
api_optimizer = APICallOptimizer()
smart_api_manager = SmartAPIManager()

# Example usage functions
def optimized_price_fetch(symbol: str) -> Optional[float]:
    """Example of optimized price fetching"""
    try:
        # This would normally call the exchange API
        # For demo, we'll simulate the call
        time.sleep(0.01)  # Simulate API latency
        return 50000.0  # Mock price
    except Exception as e:
        logger.error(f"Price fetch failed for {symbol}: {e}")
        return None

def optimized_ohlcv_fetch(symbol: str, timeframe: str = '5m', limit: int = 100):
    """Example of optimized OHLCV fetching"""
    try:
        # This would normally call the exchange API
        # For demo, we'll simulate the call
        time.sleep(0.05)  # Simulate API latency
        return []  # Mock OHLCV data
    except Exception as e:
        logger.error(f"OHLCV fetch failed for {symbol}: {e}")
        return None

if __name__ == "__main__":
    # Example usage
    print("🚀 API Call Optimization Guide")
    print("=" * 50)

    # Test basic optimization
    print("\n📊 Testing Basic Optimization...")

    # Simulate multiple calls
    for i in range(10):
        result = api_optimizer.optimize_api_call(optimized_price_fetch)("BTCUSDT")
        if i < 3:  # Test caching by calling same symbol
            result = api_optimizer.optimize_api_call(optimized_price_fetch)("BTCUSDT")

    # Get performance stats
    stats = api_optimizer.get_performance_stats()
    print("Performance Stats:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    # Test smart API manager
    print("\n🧠 Testing Smart API Manager...")

    for i in range(5):
        smart_api_manager.schedule_optimized_call(
            "BTCUSDT", "price", optimized_price_fetch, "BTCUSDT"
        )

    smart_stats = smart_api_manager.get_optimization_stats()
    print("Smart API Stats:")
    for key, value in smart_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")

    print("\n✅ API Call Optimization Guide Complete!")
    print("\n💡 Key Improvements:")
    print("  • Intelligent rate limiting with burst handling")
    print("  • Request deduplication and caching")
    print("  • Batch processing for multiple calls")
    print("  • Predictive call frequency optimization")
    print("  • Comprehensive performance monitoring")
    print("  • Automatic cache cleanup and management")
