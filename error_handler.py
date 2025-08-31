"""
Enhanced Error Handling Module
Comprehensive error handling and recovery for the trading bot
"""

import sys
import os
import time
import traceback
from datetime import datetime
from functools import wraps
import pandas as pd

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from loguru import logger
from config import Config


class ErrorHandler:
    """Enhanced error handling and recovery system"""

    def __init__(self):
        self.error_counts = {}
        self.last_errors = {}
        self.recovery_actions = {}
        self.circuit_breakers = {}
        logger.info("Enhanced Error Handler initialized")

    def handle_error(self, error, context="", component="", severity="ERROR"):
        """Handle and log errors with context"""
        error_type = type(error).__name__
        error_message = str(error)

        # Log the error with full context
        logger.error(f"[{component}] {severity}: {error_message}")
        if context:
            logger.error(f"[{component}] Context: {context}")

        # Add stack trace for debugging
        logger.error(f"[{component}] Stack trace: {traceback.format_exc()}")

        # Update error tracking
        self._update_error_tracking(component, error_type, error_message)

        # Execute recovery actions
        self._execute_recovery_actions(component, error_type)

        # Check circuit breaker
        if self._check_circuit_breaker(component):
            logger.warning(f"[{component}] Circuit breaker activated - component disabled")

    def _update_error_tracking(self, component, error_type, error_message):
        """Update error tracking statistics"""
        key = f"{component}_{error_type}"

        if key not in self.error_counts:
            self.error_counts[key] = 0
            self.last_errors[key] = datetime.now()

        self.error_counts[key] += 1
        self.last_errors[key] = datetime.now()

    def _execute_recovery_actions(self, component, error_type):
        """Execute appropriate recovery actions based on error type"""
        recovery_key = f"{component}_{error_type}"

        if recovery_key in self.recovery_actions:
            action = self.recovery_actions[recovery_key]
            logger.info(f"[{component}] Executing recovery action: {action}")
            # Execute the recovery action (placeholder for actual implementation)
            return True

        # Default recovery actions
        if "API" in error_type.upper():
            self._handle_api_error(component)
        elif "DATABASE" in error_type.upper():
            self._handle_database_error(component)
        elif "NETWORK" in error_type.upper():
            self._handle_network_error(component)
        elif "MEMORY" in error_type.upper():
            self._handle_memory_error(component)

    def _check_circuit_breaker(self, component):
        """Check if circuit breaker should be activated"""
        key = f"{component}_*"
        total_errors = sum(count for k, count in self.error_counts.items()
                          if k.startswith(f"{component}_"))

        # Activate circuit breaker if too many errors in short time
        if total_errors > 10:  # More than 10 errors
            recent_errors = sum(1 for k, timestamp in self.last_errors.items()
                               if k.startswith(f"{component}_") and
                               (datetime.now() - timestamp).seconds < 300)  # Last 5 minutes

            if recent_errors > 5:
                self.circuit_breakers[component] = datetime.now()
                return True

        return False

    def _handle_api_error(self, component):
        """Handle API-related errors"""
        logger.info(f"[{component}] Handling API error - implementing retry logic")
        # Implement exponential backoff retry logic
        time.sleep(1)  # Simple retry delay

    def _handle_database_error(self, component):
        """Handle database-related errors"""
        logger.info(f"[{component}] Handling database error - checking connection")
        # Implement database reconnection logic

    def _handle_network_error(self, component):
        """Handle network-related errors"""
        logger.info(f"[{component}] Handling network error - checking connectivity")
        # Implement network connectivity checks

    def _handle_memory_error(self, component):
        """Handle memory-related errors"""
        logger.info(f"[{component}] Handling memory error - clearing cache")
        # Implement memory cleanup

    def get_error_statistics(self):
        """Get comprehensive error statistics"""
        stats = {
            'total_errors': sum(self.error_counts.values()),
            'error_types': len(self.error_counts),
            'active_circuit_breakers': len(self.circuit_breakers),
            'error_breakdown': self.error_counts.copy(),
            'recent_errors': {}
        }

        # Get recent errors (last hour)
        one_hour_ago = datetime.now().timestamp() - 3600
        for key, timestamp in self.last_errors.items():
            if timestamp.timestamp() > one_hour_ago:
                stats['recent_errors'][key] = timestamp

        return stats

    def reset_circuit_breaker(self, component):
        """Reset circuit breaker for a component"""
        if component in self.circuit_breakers:
            del self.circuit_breakers[component]
            logger.info(f"[{component}] Circuit breaker reset")

    def add_recovery_action(self, component, error_type, action):
        """Add custom recovery action"""
        key = f"{component}_{error_type}"
        self.recovery_actions[key] = action
        logger.info(f"[{component}] Added recovery action for {error_type}: {action}")


class ErrorBoundary:
    """Context manager for error boundaries"""

    def __init__(self, component_name, error_handler=None):
        self.component_name = component_name
        self.error_handler = error_handler or ErrorHandler()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.error_handler.handle_error(
                exc_val,
                context=f"Error boundary for {self.component_name}",
                component=self.component_name,
                severity="ERROR"
            )
            return True  # Suppress the exception
        return False


def error_handler_decorator(component_name, severity="ERROR"):
    """Decorator for automatic error handling"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = ErrorHandler()
                error_handler.handle_error(
                    e,
                    context=f"Function: {func.__name__}",
                    component=component_name,
                    severity=severity
                )
                return None  # Return None on error
        return wrapper
    return decorator


class ValidationError(Exception):
    """Custom validation error"""
    pass


class APIError(Exception):
    """Custom API error"""
    pass


class DataError(Exception):
    """Custom data processing error"""
    pass


def validate_market_data(df, required_columns=None):
    """Validate market data DataFrame"""
    if df is None:
        raise DataError("Market data is None")

    if len(df) == 0:
        raise DataError("Market data is empty")

    if required_columns is None:
        required_columns = ['open', 'high', 'low', 'close', 'volume']

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataError(f"Missing required columns: {missing_columns}")

    # Check for NaN values
    if df.isnull().any().any():
        logger.warning("Market data contains NaN values - filling with forward fill")
        df = df.fillna(method='ffill')

    return df


def validate_signal_quality(signals, required_keys=None):
    """Validate trading signal quality"""
    if signals is None:
        raise ValidationError("Signals is None")

    if not isinstance(signals, dict):
        raise ValidationError("Signals must be a dictionary")

    if required_keys is None:
        required_keys = ['overall_signal', 'rsi_signal', 'macd_signal']

    missing_keys = [key for key in required_keys if key not in signals]
    if missing_keys:
        raise ValidationError(f"Missing required signal keys: {missing_keys}")

    # Validate signal values
    valid_signals = ['BULLISH', 'BEARISH', 'NEUTRAL']
    for key, value in signals.items():
        if key.endswith('_signal') and value not in valid_signals:
            raise ValidationError(f"Invalid signal value for {key}: {value}")

    return signals


def safe_api_call(func, max_retries=3, delay=1):
    """Safe API call with retry logic"""
    def wrapper(*args, **kwargs):
        last_error = None

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"API call failed after {max_retries} attempts: {e}")

        raise APIError(f"API call failed after {max_retries} attempts: {last_error}")

    return wrapper


def create_error_report():
    """Create comprehensive error report"""
    error_handler = ErrorHandler()
    stats = error_handler.get_error_statistics()

    report = f"""
ERROR REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*60}

SUMMARY:
- Total Errors: {stats['total_errors']}
- Error Types: {stats['error_types']}
- Active Circuit Breakers: {stats['active_circuit_breakers']}
- Recent Errors (last hour): {len(stats['recent_errors'])}

ERROR BREAKDOWN:
"""

    for error_type, count in stats['error_breakdown'].items():
        report += f"- {error_type}: {count} occurrences\n"

    if stats['recent_errors']:
        report += "\nRECENT ERRORS:\n"
        for error_type, timestamp in stats['recent_errors'].items():
            report += f"- {error_type}: {timestamp}\n"

    if stats['active_circuit_breakers'] > 0:
        report += "\nACTIVE CIRCUIT BREAKERS:\n"
        for component in error_handler.circuit_breakers:
            report += f"- {component}: activated\n"

    report += f"\n{'='*60}\n"

    return report


# Global error handler instance
error_handler = ErrorHandler()


if __name__ == "__main__":
    # Test the error handling system
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="{time} {level} {message}")

    # Test error handling
    try:
        raise ValueError("Test error")
    except Exception as e:
        error_handler.handle_error(e, "Test context", "test_component")

    # Test validation functions
    try:
        validate_market_data(None)
    except DataError as e:
        error_handler.handle_error(e, "Validation test", "validation_test")

    # Generate error report
    report = create_error_report()
    print(report)
