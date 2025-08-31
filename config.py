"""
Improved Configuration Module
Enhanced with type hints, validation, and better structure
"""

import os
from dotenv import load_dotenv
import logging
from typing import List, Optional
from dataclasses import dataclass, field
from functools import lru_cache

# Load environment variables
load_dotenv()

@dataclass(frozen=True)
class TradingConfig:
    """Trading configuration with validation"""
    risk_percentage: float = field(default=2.0)
    max_position_size: float = field(default=1000.0)
    stop_loss_percentage: float = field(default=2.0)
    take_profit_percentage: float = field(default=4.0)
    trailing_stop_percentage: float = field(default=1.0)

    def __post_init__(self):
        """Validate trading configuration parameters"""
        if not 0 < self.risk_percentage <= 100:
            raise ValueError(f"Risk percentage must be between 0 and 100, got {self.risk_percentage}")
        if not 0 < self.stop_loss_percentage <= 50:
            raise ValueError(f"Stop loss percentage must be between 0 and 50, got {self.stop_loss_percentage}")
        if not 0 < self.take_profit_percentage <= 200:
            raise ValueError(f"Take profit percentage must be between 0 and 200, got {self.take_profit_percentage}")
        if not 0 < self.max_position_size <= 100000:
            raise ValueError(f"Max position size must be between 0 and 100000, got {self.max_position_size}")

@dataclass(frozen=True)
class TechnicalConfig:
    """Technical analysis configuration"""
    rsi_period: int = field(default=14)
    macd_fast: int = field(default=12)
    macd_slow: int = field(default=26)
    macd_signal: int = field(default=9)
    bb_period: int = field(default=20)
    bb_std: float = field(default=2.0)
    atr_period: int = field(default=14)
    stoch_k: int = field(default=14)
    stoch_d: int = field(default=3)

    def __post_init__(self):
        """Validate technical analysis parameters"""
        if not 2 <= self.rsi_period <= 50:
            raise ValueError(f"RSI period must be between 2 and 50, got {self.rsi_period}")
        if not 2 <= self.macd_fast < self.macd_slow:
            raise ValueError(f"MACD fast ({self.macd_fast}) must be less than slow ({self.macd_slow})")
        if not 2 <= self.bb_period <= 50:
            raise ValueError(f"Bollinger Band period must be between 2 and 50, got {self.bb_period}")
        if not 1.0 <= self.bb_std <= 3.0:
            raise ValueError(f"Bollinger Band std must be between 1.0 and 3.0, got {self.bb_std}")

@dataclass(frozen=True)
class APIConfig:
    """API configuration for external services"""
    telegram_bot_token: Optional[str] = field(default_factory=lambda: os.getenv('TELEGRAM_BOT_TOKEN'))
    telegram_chat_id: Optional[str] = field(default_factory=lambda: os.getenv('TELEGRAM_CHAT_ID'))
    binance_api_key: Optional[str] = field(default_factory=lambda: os.getenv('BINANCE_API_KEY'))
    binance_api_secret: Optional[str] = field(default_factory=lambda: os.getenv('BINANCE_API_SECRET'))

    def __post_init__(self):
        """Validate API configuration"""
        required_apis = ['telegram_bot_token', 'telegram_chat_id']
        for api in required_apis:
            value = getattr(self, api)
            if not value or not value.strip():
                raise ValueError(f"Required API configuration missing: {api}")

class Config:
    """Improved configuration class with validation and type safety"""

    # Core Configuration
    TRADING_PAIRS: List[str] = os.getenv('TRADING_PAIRS', 'BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,BNBUSDT,ATOMUSDT').split(',')
    DEFAULT_LEVERAGE: int = int(os.getenv('LEVERAGE', '10'))
    ANALYSIS_INTERVAL: int = int(os.getenv('ANALYSIS_INTERVAL_MINUTES', '5'))

    # System Configuration
    DATABASE_PATH: str = os.getenv('DATABASE_PATH', 'trading_bot.db')
    LOG_LEVEL: int = logging.INFO
    MAX_WORKERS: int = int(os.getenv('MAX_WORKERS', '4'))

    # Cache Configuration
    CACHE_SIZE: int = int(os.getenv('CACHE_SIZE', '100'))
    CACHE_TTL: int = int(os.getenv('CACHE_TTL_SECONDS', '300'))  # 5 minutes

    # Anomaly Detection
    ANOMALY_SCAN_INTERVAL: int = int(os.getenv('ANOMALY_SCAN_INTERVAL_MINUTES', '2'))

    # Backward compatibility attributes
    TRADING_PAIR = TRADING_PAIRS[0] if TRADING_PAIRS else 'BTCUSDT'  # Default to first trading pair
    RISK_PERCENTAGE = float(os.getenv('RISK_PERCENTAGE', '2'))
    RSI_PERIOD = int(os.getenv('RSI_PERIOD', '14'))
    MACD_FAST = int(os.getenv('MACD_FAST', '12'))
    MACD_SLOW = int(os.getenv('MACD_SLOW', '26'))
    MACD_SIGNAL = int(os.getenv('MACD_SIGNAL', '9'))
    BB_PERIOD = int(os.getenv('BB_PERIOD', '20'))
    BB_STD = float(os.getenv('BB_STD', '2'))
    MAX_POSITION_SIZE = float(os.getenv('MAX_POSITION_SIZE_USDT', '1000'))
    STOP_LOSS_PERCENTAGE = float(os.getenv('STOP_LOSS_PERCENTAGE', '2'))
    TAKE_PROFIT_PERCENTAGE = float(os.getenv('TAKE_PROFIT_PERCENTAGE', '4'))
    TRAILING_STOP_PERCENTAGE = float(os.getenv('TRAILING_STOP_PERCENTAGE', '1'))

    # API Configuration (backward compatibility)
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')

    def __init__(self):
        """Initialize and validate configuration"""
        self._api_config: Optional[APIConfig] = None
        self._trading_config: Optional[TradingConfig] = None
        self._technical_config: Optional[TechnicalConfig] = None

        # Validate configuration on initialization
        issues = self.validate_configuration()
        if issues:
            print("Configuration Issues Found:")
            for issue in issues:
                print(f"  ❌ {issue}")
            raise ValueError("Configuration validation failed. Please fix the issues above.")

        print("✅ Configuration validated successfully.")

    @property
    def api_config(self) -> APIConfig:
        """Get API configuration with caching"""
        if self._api_config is None:
            self._api_config = APIConfig()
        return self._api_config

    @property
    def trading_config(self) -> TradingConfig:
        """Get trading configuration with caching"""
        if self._trading_config is None:
            self._trading_config = TradingConfig(
                risk_percentage=float(os.getenv('RISK_PERCENTAGE', '2')),
                max_position_size=float(os.getenv('MAX_POSITION_SIZE_USDT', '1000')),
                stop_loss_percentage=float(os.getenv('STOP_LOSS_PERCENTAGE', '2')),
                take_profit_percentage=float(os.getenv('TAKE_PROFIT_PERCENTAGE', '4')),
                trailing_stop_percentage=float(os.getenv('TRAILING_STOP_PERCENTAGE', '1'))
            )
        return self._trading_config

    @property
    def technical_config(self) -> TechnicalConfig:
        """Get technical analysis configuration with caching"""
        if self._technical_config is None:
            self._technical_config = TechnicalConfig(
                rsi_period=int(os.getenv('RSI_PERIOD', '14')),
                macd_fast=int(os.getenv('MACD_FAST', '12')),
                macd_slow=int(os.getenv('MACD_SLOW', '26')),
                macd_signal=int(os.getenv('MACD_SIGNAL', '9')),
                bb_period=int(os.getenv('BB_PERIOD', '20')),
                bb_std=float(os.getenv('BB_STD', '2')),
                atr_period=int(os.getenv('ATR_PERIOD', '14')),
                stoch_k=int(os.getenv('STOCH_K', '14')),
                stoch_d=int(os.getenv('STOCH_D', '3'))
            )
        return self._technical_config

    def validate_configuration(self) -> List[str]:
        """Comprehensive configuration validation"""
        issues = []

        # Validate trading pairs
        if not self.TRADING_PAIRS:
            issues.append("No trading pairs configured")
        else:
            invalid_pairs = [pair for pair in self.TRADING_PAIRS if not pair.endswith('USDT')]
            if invalid_pairs:
                issues.append(f"Invalid trading pairs (must end with USDT): {invalid_pairs}")

            if len(self.TRADING_PAIRS) > 20:
                issues.append("Too many trading pairs configured (max 20 recommended)")

        # Validate leverage
        if not 1 <= self.DEFAULT_LEVERAGE <= 125:
            issues.append(f"Default leverage must be between 1 and 125, got {self.DEFAULT_LEVERAGE}")

        # Validate analysis interval
        if not 1 <= self.ANALYSIS_INTERVAL <= 60:
            issues.append(f"Analysis interval must be between 1 and 60 minutes, got {self.ANALYSIS_INTERVAL}")

        # Validate system settings
        if self.MAX_WORKERS < 1 or self.MAX_WORKERS > 10:
            issues.append(f"Max workers must be between 1 and 10, got {self.MAX_WORKERS}")

        if self.CACHE_SIZE < 10 or self.CACHE_SIZE > 1000:
            issues.append(f"Cache size must be between 10 and 1000, got {self.CACHE_SIZE}")

        # Validate API configuration
        try:
            api_config = APIConfig()
        except ValueError as e:
            issues.append(f"API configuration error: {e}")

        # Validate trading configuration
        try:
            risk_pct = float(os.getenv('RISK_PERCENTAGE', '2'))
            if not 0 < risk_pct <= 100:
                issues.append(f"Risk percentage must be between 0 and 100, got {risk_pct}")
        except ValueError:
            issues.append("Invalid RISK_PERCENTAGE value")

        # Validate technical configuration
        try:
            macd_fast = int(os.getenv('MACD_FAST', '12'))
            macd_slow = int(os.getenv('MACD_SLOW', '26'))
            if macd_fast >= macd_slow:
                issues.append(f"MACD fast ({macd_fast}) must be less than slow ({macd_slow})")
        except ValueError:
            issues.append("Invalid MACD period values")

        return issues

    def get_summary(self) -> str:
        """Get configuration summary"""
        summary = f"""
🔧 CONFIGURATION SUMMARY
{'='*50}
Trading Pairs: {len(self.TRADING_PAIRS)} configured
Default Leverage: {self.DEFAULT_LEVERAGE}x
Analysis Interval: {self.ANALYSIS_INTERVAL} minutes
Max Workers: {self.MAX_WORKERS}
Cache Size: {self.CACHE_SIZE}
Database: {self.DATABASE_PATH}

📊 TRADING CONFIGURATION
Risk Percentage: {self.trading_config.risk_percentage}%
Max Position Size: ${self.trading_config.max_position_size}
Stop Loss: {self.trading_config.stop_loss_percentage}%
Take Profit: {self.trading_config.take_profit_percentage}%

📈 TECHNICAL ANALYSIS
RSI Period: {self.technical_config.rsi_period}
MACD: {self.technical_config.macd_fast}/{self.technical_config.macd_slow}/{self.technical_config.macd_signal}
Bollinger Bands: {self.technical_config.bb_period} period, {self.technical_config.bb_std} std

🔗 API CONFIGURATION
Telegram: {'✅ Configured' if self.api_config.telegram_bot_token else '❌ Missing'}
Binance: {'✅ Configured' if self.api_config.binance_api_key else '❌ Missing'}
"""
        return summary

    def export_config(self, filepath: str) -> None:
        """Export current configuration to file"""
        import json

        config_dict = {
            'trading_pairs': self.TRADING_PAIRS,
            'default_leverage': self.DEFAULT_LEVERAGE,
            'analysis_interval': self.ANALYSIS_INTERVAL,
            'max_workers': self.MAX_WORKERS,
            'cache_size': self.CACHE_SIZE,
            'database_path': self.DATABASE_PATH,
            'trading_config': {
                'risk_percentage': self.trading_config.risk_percentage,
                'max_position_size': self.trading_config.max_position_size,
                'stop_loss_percentage': self.trading_config.stop_loss_percentage,
                'take_profit_percentage': self.trading_config.take_profit_percentage,
                'trailing_stop_percentage': self.trading_config.trailing_stop_percentage
            },
            'technical_config': {
                'rsi_period': self.technical_config.rsi_period,
                'macd_fast': self.technical_config.macd_fast,
                'macd_slow': self.technical_config.macd_slow,
                'macd_signal': self.technical_config.macd_signal,
                'bb_period': self.technical_config.bb_period,
                'bb_std': self.technical_config.bb_std,
                'atr_period': self.technical_config.atr_period,
                'stoch_k': self.technical_config.stoch_k,
                'stoch_d': self.technical_config.stoch_d
            }
        }

        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)

        print(f"✅ Configuration exported to {filepath}")

# Global configuration instance
config = Config()
