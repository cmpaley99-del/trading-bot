"""
Performance Optimization Module
Optimizes the trading bot for production deployment
"""

import sys
import os
import time
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import pandas as pd
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import Config
from market_data import market_data
from technical_analysis import technical_analysis
from technical_analysis_advanced import advanced_technical_analysis
from trading_signals import trading_signals


class PerformanceOptimizer:
    """Performance optimization and monitoring"""

    def __init__(self):
        self.baseline_metrics = {}
        self.optimized_metrics = {}
        logger.info("Performance Optimizer initialized")

    def run_performance_analysis(self):
        """Run comprehensive performance analysis"""
        logger.info("=== Starting Performance Analysis ===")

        try:
            # Test 1: Memory Usage Analysis
            self.analyze_memory_usage()

            # Test 2: CPU Usage Analysis
            self.analyze_cpu_usage()

            # Test 3: API Call Optimization
            self.optimize_api_calls()

            # Test 4: Data Processing Optimization
            self.optimize_data_processing()

            # Test 5: Signal Generation Optimization
            self.optimize_signal_generation()

            # Test 6: Concurrent Processing Analysis
            self.analyze_concurrent_processing()

            # Test 7: Database Query Optimization
            self.optimize_database_queries()

            # Generate optimization report
            self.generate_optimization_report()

        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise

    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        logger.info("Analyzing Memory Usage...")

        try:
            # Get initial memory usage
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            # Test memory usage during market data fetching
            test_pairs = Config.TRADING_PAIRS[:5]  # Test with 5 pairs
            data_fetch_memory = []

            for pair in test_pairs:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                df = market_data.get_ohlcv_data(pair)
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                data_fetch_memory.append(memory_after - memory_before)

            # Test memory usage during technical analysis
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            df_with_indicators = technical_analysis.calculate_indicators(df)
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            ta_memory_usage = memory_after - memory_before

            # Test memory usage during signal generation
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            signals = technical_analysis.generate_signals(df_with_indicators)
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            signal_memory_usage = memory_after - memory_before

            self.baseline_metrics['memory'] = {
                'initial_memory_mb': initial_memory,
                'avg_data_fetch_mb': np.mean(data_fetch_memory),
                'ta_memory_mb': ta_memory_usage,
                'signal_memory_mb': signal_memory_usage,
                'total_memory_mb': psutil.Process().memory_info().rss / 1024 / 1024
            }

            logger.info(f"Memory Analysis: Initial: {initial_memory:.1f}MB, "
                       f"TA: {ta_memory_usage:.1f}MB, Signals: {signal_memory_usage:.1f}MB")

        except Exception as e:
            logger.error(f"Memory analysis failed: {e}")
            self.baseline_metrics['memory'] = {'error': str(e)}

    def analyze_cpu_usage(self):
        """Analyze CPU usage patterns"""
        logger.info("Analyzing CPU Usage...")

        try:
            # Test CPU usage during intensive operations
            cpu_usages = []

            # Test 1: Market data fetching for multiple pairs
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(market_data.get_ohlcv_data, pair)
                          for pair in Config.TRADING_PAIRS[:5]]
                results = [future.result() for future in futures]
            data_fetch_time = time.time() - start_time

            # Test 2: Technical analysis processing
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            start_time = time.time()
            df_with_indicators = technical_analysis.calculate_indicators(df)
            ta_time = time.time() - start_time

            # Test 3: Signal generation
            start_time = time.time()
            signals = technical_analysis.generate_signals(df_with_indicators)
            signal_time = time.time() - start_time

            # Test 4: Advanced pattern analysis
            start_time = time.time()
            advanced_signals = advanced_technical_analysis.generate_advanced_signals(df)
            pattern_time = time.time() - start_time

            self.baseline_metrics['cpu'] = {
                'data_fetch_time': data_fetch_time,
                'ta_time': ta_time,
                'signal_time': signal_time,
                'pattern_time': pattern_time,
                'total_processing_time': data_fetch_time + ta_time + signal_time + pattern_time
            }

            logger.info(f"CPU Analysis: Data Fetch: {data_fetch_time:.2f}s, "
                       f"TA: {ta_time:.2f}s, Signals: {signal_time:.2f}s, "
                       f"Patterns: {pattern_time:.2f}s")

        except Exception as e:
            logger.error(f"CPU analysis failed: {e}")
            self.baseline_metrics['cpu'] = {'error': str(e)}

    def optimize_api_calls(self):
        """Optimize API call patterns"""
        logger.info("Optimizing API Calls...")

        try:
            # Test current API call patterns
            test_pairs = Config.TRADING_PAIRS[:3]

            # Sequential calls
            start_time = time.time()
            sequential_results = []
            for pair in test_pairs:
                result = market_data.get_ohlcv_data(pair)
                sequential_results.append(result)
            sequential_time = time.time() - start_time

            # Concurrent calls
            start_time = time.time()
            with ThreadPoolExecutor(max_workers=3) as executor:
                concurrent_results = list(executor.map(market_data.get_ohlcv_data, test_pairs))
            concurrent_time = time.time() - start_time

            # Calculate optimization metrics
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
            efficiency = speedup / 3  # Ideal speedup would be 3x for 3 workers

            self.optimized_metrics['api_calls'] = {
                'sequential_time': sequential_time,
                'concurrent_time': concurrent_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'recommendation': 'concurrent' if speedup > 1.5 else 'sequential'
            }

            logger.info(f"API Optimization: Sequential: {sequential_time:.2f}s, "
                       f"Concurrent: {concurrent_time:.2f}s, Speedup: {speedup:.2f}x")

        except Exception as e:
            logger.error(f"API optimization failed: {e}")
            self.optimized_metrics['api_calls'] = {'error': str(e)}

    def optimize_data_processing(self):
        """Optimize data processing operations"""
        logger.info("Optimizing Data Processing...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)

            if df is None or len(df) < 100:
                raise Exception("Insufficient data for optimization testing")

            # Test different data processing approaches
            data_sizes = [100, 500, 1000, 2000]

            processing_times = {}
            for size in data_sizes:
                test_df = df.tail(size).copy()

                # Time technical analysis
                start_time = time.time()
                result_df = technical_analysis.calculate_indicators(test_df)
                processing_time = time.time() - start_time

                processing_times[size] = processing_time

                logger.info(f"Data size {size}: {processing_time:.3f}s")

            # Calculate scaling efficiency
            base_time = processing_times[100]
            scaling_factors = {}
            for size in data_sizes[1:]:
                expected_time = base_time * (size / 100)
                actual_time = processing_times[size]
                scaling_factors[size] = actual_time / expected_time

            self.optimized_metrics['data_processing'] = {
                'processing_times': processing_times,
                'scaling_factors': scaling_factors,
                'avg_scaling_efficiency': np.mean(list(scaling_factors.values())),
                'recommendation': 'good' if np.mean(list(scaling_factors.values())) < 1.5 else 'needs_optimization'
            }

            logger.info(f"Data Processing: Average scaling efficiency: {np.mean(list(scaling_factors.values())):.2f}")

        except Exception as e:
            logger.error(f"Data processing optimization failed: {e}")
            self.optimized_metrics['data_processing'] = {'error': str(e)}

    def optimize_signal_generation(self):
        """Optimize signal generation process"""
        logger.info("Optimizing Signal Generation...")

        try:
            test_pair = Config.TRADING_PAIRS[0]
            df = market_data.get_ohlcv_data(test_pair)
            df_with_indicators = technical_analysis.calculate_indicators(df)

            # Test signal generation with different configurations
            signal_configs = [
                {'advanced_patterns': False},
                {'advanced_patterns': True},
            ]

            signal_times = {}

            for config in signal_configs:
                if config['advanced_patterns']:
                    start_time = time.time()
                    signals = advanced_technical_analysis.generate_advanced_signals(df)
                    generation_time = time.time() - start_time
                    config_name = 'with_advanced_patterns'
                else:
                    start_time = time.time()
                    signals = technical_analysis.generate_signals(df_with_indicators)
                    generation_time = time.time() - start_time
                    config_name = 'basic_signals'

                signal_times[config_name] = generation_time

                logger.info(f"Signal generation {config_name}: {generation_time:.3f}s")

            # Calculate overhead of advanced patterns
            basic_time = signal_times.get('basic_signals', 0)
            advanced_time = signal_times.get('with_advanced_patterns', 0)
            overhead = advanced_time - basic_time if advanced_time > basic_time else 0
            overhead_percentage = (overhead / basic_time * 100) if basic_time > 0 else 0

            self.optimized_metrics['signal_generation'] = {
                'basic_time': basic_time,
                'advanced_time': advanced_time,
                'overhead': overhead,
                'overhead_percentage': overhead_percentage,
                'recommendation': 'acceptable' if overhead_percentage < 50 else 'consider_caching'
            }

            logger.info(f"Signal Generation: Advanced patterns overhead: {overhead_percentage:.1f}%")

        except Exception as e:
            logger.error(f"Signal generation optimization failed: {e}")
            self.optimized_metrics['signal_generation'] = {'error': str(e)}

    def analyze_concurrent_processing(self):
        """Analyze concurrent processing capabilities"""
        logger.info("Analyzing Concurrent Processing...")

        try:
            test_pairs = Config.TRADING_PAIRS[:8]  # Test with 8 pairs

            # Test sequential processing
            start_time = time.time()
            sequential_results = []
            for pair in test_pairs:
                df = market_data.get_ohlcv_data(pair)
                if df is not None:
                    df_with_indicators = technical_analysis.calculate_indicators(df)
                    signals = technical_analysis.generate_signals(df_with_indicators)
                    sequential_results.append(signals)
            sequential_time = time.time() - start_time

            # Test concurrent processing
            def process_pair(pair):
                df = market_data.get_ohlcv_data(pair)
                if df is not None:
                    df_with_indicators = technical_analysis.calculate_indicators(df)
                    signals = technical_analysis.generate_signals(df_with_indicators)
                    return signals
                return None

            start_time = time.time()
            with ThreadPoolExecutor(max_workers=4) as executor:
                concurrent_results = list(executor.map(process_pair, test_pairs))
            concurrent_time = time.time() - start_time

            # Calculate metrics
            speedup = sequential_time / concurrent_time if concurrent_time > 0 else 1
            efficiency = speedup / 4  # Ideal speedup would be 4x for 4 workers

            self.optimized_metrics['concurrent_processing'] = {
                'sequential_time': sequential_time,
                'concurrent_time': concurrent_time,
                'speedup': speedup,
                'efficiency': efficiency,
                'workers_used': 4,
                'pairs_processed': len(test_pairs),
                'recommendation': 'concurrent' if speedup > 2 else 'sequential'
            }

            logger.info(f"Concurrent Processing: Sequential: {sequential_time:.2f}s, "
                       f"Concurrent: {concurrent_time:.2f}s, Speedup: {speedup:.2f}x")

        except Exception as e:
            logger.error(f"Concurrent processing analysis failed: {e}")
            self.optimized_metrics['concurrent_processing'] = {'error': str(e)}

    def optimize_database_queries(self):
        """Optimize database query patterns"""
        logger.info("Optimizing Database Queries...")

        try:
            from database import database

            # Test database query performance
            query_times = []

            # Test multiple queries
            for i in range(10):
                start_time = time.time()
                last_signal_time = database.get_last_signal_time()
                query_time = time.time() - start_time
                query_times.append(query_time)

            avg_query_time = np.mean(query_times)
            max_query_time = np.max(query_times)
            min_query_time = np.min(query_times)

            # Test database write performance
            write_times = []
            for i in range(5):
                test_time = datetime.now()
                start_time = time.time()
                database.save_last_signal_time(test_time)
                write_time = time.time() - start_time
                write_times.append(write_time)

            avg_write_time = np.mean(write_times)

            self.optimized_metrics['database'] = {
                'avg_query_time': avg_query_time,
                'max_query_time': max_query_time,
                'min_query_time': min_query_time,
                'avg_write_time': avg_write_time,
                'queries_per_second': 1 / avg_query_time if avg_query_time > 0 else 0,
                'writes_per_second': 1 / avg_write_time if avg_write_time > 0 else 0,
                'recommendation': 'good' if avg_query_time < 0.01 else 'needs_optimization'
            }

            logger.info(f"Database Optimization: Avg query: {avg_query_time:.4f}s, "
                       f"Avg write: {avg_write_time:.4f}s")

        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
            self.optimized_metrics['database'] = {'error': str(e)}

    def generate_optimization_report(self):
        """Generate comprehensive optimization report"""
        logger.info("=== Performance Optimization Report ===")

        print(f"\n{'='*70}")
        print(f"TRADING BOT PERFORMANCE OPTIMIZATION REPORT")
        print(f"{'='*70}")
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*70}")

        # Memory Analysis
        if 'memory' in self.baseline_metrics:
            mem = self.baseline_metrics['memory']
            print(f"\n📊 MEMORY USAGE ANALYSIS:")
            print(f"   Initial Memory: {mem.get('initial_memory_mb', 0):.1f} MB")
            print(f"   Technical Analysis: {mem.get('ta_memory_mb', 0):.1f} MB")
            print(f"   Signal Generation: {mem.get('signal_memory_mb', 0):.1f} MB")
            print(f"   Total Memory: {mem.get('total_memory_mb', 0):.1f} MB")

        # CPU Analysis
        if 'cpu' in self.baseline_metrics:
            cpu = self.baseline_metrics['cpu']
            print(f"\n⚡ CPU PERFORMANCE ANALYSIS:")
            print(f"   Data Fetching: {cpu.get('data_fetch_time', 0):.2f}s")
            print(f"   Technical Analysis: {cpu.get('ta_time', 0):.2f}s")
            print(f"   Signal Generation: {cpu.get('signal_time', 0):.2f}s")
            print(f"   Pattern Analysis: {cpu.get('pattern_time', 0):.2f}s")
            print(f"   Total Processing: {cpu.get('total_processing_time', 0):.2f}s")

        # API Optimization
        if 'api_calls' in self.optimized_metrics:
            api = self.optimized_metrics['api_calls']
            print(f"\n🌐 API CALL OPTIMIZATION:")
            print(f"   Sequential Time: {api.get('sequential_time', 0):.2f}s")
            print(f"   Concurrent Time: {api.get('concurrent_time', 0):.2f}s")
            print(f"   Speedup: {api.get('speedup', 1):.2f}x")
            print(f"   Efficiency: {api.get('efficiency', 0):.1%}")
            print(f"   Recommendation: {api.get('recommendation', 'unknown')}")

        # Data Processing
        if 'data_processing' in self.optimized_metrics:
            dp = self.optimized_metrics['data_processing']
            print(f"\n📈 DATA PROCESSING OPTIMIZATION:")
            print(f"   Average Scaling Efficiency: {dp.get('avg_scaling_efficiency', 1):.2f}")
            print(f"   Recommendation: {dp.get('recommendation', 'unknown')}")

        # Signal Generation
        if 'signal_generation' in self.optimized_metrics:
            sg = self.optimized_metrics['signal_generation']
            print(f"\n🎯 SIGNAL GENERATION OPTIMIZATION:")
            print(f"   Basic Signals Time: {sg.get('basic_time', 0):.3f}s")
            print(f"   Advanced Signals Time: {sg.get('advanced_time', 0):.3f}s")
            print(f"   Overhead: {sg.get('overhead_percentage', 0):.1f}%")
            print(f"   Recommendation: {sg.get('recommendation', 'unknown')}")

        # Concurrent Processing
        if 'concurrent_processing' in self.optimized_metrics:
            cp = self.optimized_metrics['concurrent_processing']
            print(f"\n🔄 CONCURRENT PROCESSING ANALYSIS:")
            print(f"   Sequential Time: {cp.get('sequential_time', 0):.2f}s")
            print(f"   Concurrent Time: {cp.get('concurrent_time', 0):.2f}s")
            print(f"   Speedup: {cp.get('speedup', 1):.2f}x")
            print(f"   Efficiency: {cp.get('efficiency', 0):.1%}")
            print(f"   Workers: {cp.get('workers_used', 0)}")
            print(f"   Recommendation: {cp.get('recommendation', 'unknown')}")

        # Database Optimization
        if 'database' in self.optimized_metrics:
            db = self.optimized_metrics['database']
            print(f"\n💾 DATABASE OPTIMIZATION:")
            print(f"   Avg Query Time: {db.get('avg_query_time', 0):.4f}s")
            print(f"   Avg Write Time: {db.get('avg_write_time', 0):.4f}s")
            print(f"   Queries/sec: {db.get('queries_per_second', 0):.1f}")
            print(f"   Writes/sec: {db.get('writes_per_second', 0):.1f}")
            print(f"   Recommendation: {db.get('recommendation', 'unknown')}")

        # Overall Assessment
        print(f"\n{'='*70}")
        print(f"🎯 OVERALL PERFORMANCE ASSESSMENT:")

        recommendations = []
        if 'api_calls' in self.optimized_metrics:
            if self.optimized_metrics['api_calls'].get('speedup', 1) > 2:
                recommendations.append("✅ API concurrent processing is highly effective")
            else:
                recommendations.append("⚠️  Consider API rate limiting or caching")

        if 'concurrent_processing' in self.optimized_metrics:
            if self.optimized_metrics['concurrent_processing'].get('speedup', 1) > 2:
                recommendations.append("✅ Concurrent processing significantly improves performance")
            else:
                recommendations.append("⚠️  Concurrent processing overhead may not be worth it")

        if 'signal_generation' in self.optimized_metrics:
            overhead = self.optimized_metrics['signal_generation'].get('overhead_percentage', 0)
            if overhead < 30:
                recommendations.append("✅ Advanced pattern analysis overhead is acceptable")
            else:
                recommendations.append("⚠️  Consider caching advanced pattern results")

        if 'database' in self.optimized_metrics:
            if self.optimized_metrics['database'].get('avg_query_time', 1) < 0.01:
                recommendations.append("✅ Database performance is excellent")
            else:
                recommendations.append("⚠️  Database queries may need optimization")

        for rec in recommendations:
            print(f"   {rec}")

        print(f"\n🚀 PRODUCTION READINESS:")
        if len(recommendations) >= 3 and recommendations[0].startswith("✅"):
            print("   🟢 EXCELLENT: System is well-optimized for production")
        elif len(recommendations) >= 2:
            print("   🟡 GOOD: System is ready for production with minor optimizations")
        else:
            print("   🔴 NEEDS WORK: Address performance issues before production")

        print(f"{'='*70}")


def run_performance_optimization():
    """Run the complete performance optimization analysis"""
    optimizer = PerformanceOptimizer()
    optimizer.run_performance_analysis()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    run_performance_optimization()
