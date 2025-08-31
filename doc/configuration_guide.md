# Configuration Guide

This guide covers all configuration options and settings for the Cryptocurrency Futures Trading Bot.

## Environment Variables

### Required Variables

#### Binance API Configuration
```env
# Your Binance Futures API Key
BINANCE_API_KEY=your_binance_api_key_here

# Your Binance Futures API Secret
BINANCE_API_SECRET=your_binance_api_secret_here
```

#### Telegram Bot Configuration
```env
# Your Telegram Bot Token (get from @BotFather)
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz

# Your Telegram Chat ID (get from @userinfobot or bot)
TELEGRAM_CHAT_ID=123456789
```

### Optional Variables

#### Trading Parameters
```env
# Trading pairs to monitor (comma-separated)
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,BNBUSDT,ATOMUSDT

# Risk percentage per trade (recommended: 1-5)
RISK_PERCENTAGE=2

# Analysis interval in minutes (recommended: 5-15)
ANALYSIS_INTERVAL_MINUTES=5

# Default leverage when intelligent calculation unavailable
LEVERAGE=10
```

#### Technical Analysis Settings
```env
# RSI parameters
RSI_PERIOD=14

# MACD parameters
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9

# Bollinger Bands parameters
BB_PERIOD=20
BB_STD=2.0
```

#### Risk Management
```env
# Maximum position size in USDT
MAX_POSITION_SIZE_USDT=1000

# Stop loss percentage
STOP_LOSS_PERCENTAGE=2

# Take profit percentage
TAKE_PROFIT_PERCENTAGE=4

# Trailing stop percentage
TRAILING_STOP_PERCENTAGE=1
```

#### System Settings
```env
# Database path
DATABASE_PATH=trading_bot.db

# Log level (DEBUG, INFO, WARNING, ERROR)
LOG_LEVEL=INFO
```

## Configuration Files

### .env File Setup

1. **Create .env file:**
   ```bash
   cp .env.example .env
   ```

2. **Edit with your values:**
   ```bash
   nano .env
   ```

3. **Example .env file:**
   ```env
   # Binance API
   BINANCE_API_KEY=abcdefghijklmnopqrstuvwx
   BINANCE_API_SECRET=yzabcdefghijklmnopqrstuvwx

   # Telegram
   TELEGRAM_BOT_TOKEN=123456789:ABCdefGHIjklMNOpqrsTUVwxyz
   TELEGRAM_CHAT_ID=987654321

   # Trading
   TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT
   RISK_PERCENTAGE=2
   ANALYSIS_INTERVAL_MINUTES=5
   ```

### Config Validation

The bot automatically validates required configuration on startup:

```python
# This happens automatically when importing config
from config import Config

# Check if all required variables are set
if not Config.TELEGRAM_BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN is required")
```

## Trading Pairs Configuration

### Supported Pairs
The bot supports all major Binance Futures pairs:

```python
# Major cryptocurrencies
TRADING_PAIRS = [
    "BTCUSDT",   # Bitcoin
    "ETHUSDT",   # Ethereum
    "SOLUSDT",   # Solana
    "ADAUSDT",   # Cardano
    "DOTUSDT",   # Polkadot
    "AVAXUSDT",  # Avalanche
    "MATICUSDT", # Polygon
    "XRPUSDT",   # Ripple
    "LINKUSDT",  # Chainlink
    "DOGEUSDT",  # Dogecoin
    "LTCUSDT",   # Litecoin
    "BNBUSDT",   # Binance Coin
    "ATOMUSDT",  # Cosmos
]
```

### Custom Pair Selection
```env
# Conservative (low risk)
TRADING_PAIRS=BTCUSDT,ETHUSDT

# Balanced (medium risk)
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT

# Aggressive (high risk)
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,BNBUSDT,ATOMUSDT
```

## Risk Management Configuration

### Conservative Settings (Low Risk)
```env
RISK_PERCENTAGE=1
MAX_POSITION_SIZE_USDT=500
STOP_LOSS_PERCENTAGE=1.5
TAKE_PROFIT_PERCENTAGE=3
TRAILING_STOP_PERCENTAGE=0.5
```

### Balanced Settings (Medium Risk)
```env
RISK_PERCENTAGE=2
MAX_POSITION_SIZE_USDT=1000
STOP_LOSS_PERCENTAGE=2
TAKE_PROFIT_PERCENTAGE=4
TRAILING_STOP_PERCENTAGE=1
```

### Aggressive Settings (High Risk)
```env
RISK_PERCENTAGE=5
MAX_POSITION_SIZE_USDT=2000
STOP_LOSS_PERCENTAGE=3
TAKE_PROFIT_PERCENTAGE=6
TRAILING_STOP_PERCENTAGE=1.5
```

## Technical Analysis Configuration

### Trend-Following Strategy
```env
# Longer timeframes for trend following
RSI_PERIOD=21
MACD_FAST=8
MACD_SLOW=21
BB_PERIOD=30
```

### Mean-Reversion Strategy
```env
# Shorter timeframes for mean reversion
RSI_PERIOD=9
MACD_FAST=5
MACD_SLOW=13
BB_PERIOD=15
```

### Scalping Strategy
```env
# Very short timeframes for scalping
RSI_PERIOD=6
MACD_FAST=3
MACD_SLOW=8
BB_PERIOD=10
ANALYSIS_INTERVAL_MINUTES=2
```

## Performance Tuning

### High-Frequency Trading
```env
# For scalping strategies
ANALYSIS_INTERVAL_MINUTES=1
TRADING_PAIRS=BTCUSDT,ETHUSDT  # Limit pairs for performance
LOG_LEVEL=WARNING  # Reduce logging overhead
```

### Low-Latency Setup
```env
# Minimize delays
ANALYSIS_INTERVAL_MINUTES=3
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT  # Focus on major pairs
```

### Resource-Constrained Environment
```env
# For limited CPU/memory
ANALYSIS_INTERVAL_MINUTES=10
TRADING_PAIRS=BTCUSDT,ETHUSDT
LOG_LEVEL=ERROR
```

## Advanced Configuration

### Custom Database Location
```env
# Use different database location
DATABASE_PATH=/var/lib/trading_bot/data.db
```

### Multiple Bot Instances
```env
# For running multiple bots
DATABASE_PATH=trading_bot_instance1.db
TELEGRAM_CHAT_ID=123456789  # Different chat for each instance
```

### Development Settings
```env
# For development and testing
LOG_LEVEL=DEBUG
ANALYSIS_INTERVAL_MINUTES=1
RISK_PERCENTAGE=0.1  # Very low risk for testing
```

## Configuration Validation

### Automatic Validation
The bot validates configuration on startup:

```python
def validate_config():
    """Validate all configuration settings"""
    required_vars = [
        'TELEGRAM_BOT_TOKEN',
        'TELEGRAM_CHAT_ID'
    ]

    for var in required_vars:
        if not getattr(Config, var, None):
            raise ValueError(f"Missing required environment variable: {var}")

    # Validate trading pairs
    if not Config.TRADING_PAIRS:
        raise ValueError("At least one trading pair must be configured")

    # Validate risk parameters
    if not (0.1 <= Config.RISK_PERCENTAGE <= 10):
        raise ValueError("Risk percentage must be between 0.1 and 10")

    # Validate analysis interval
    if not (1 <= Config.ANALYSIS_INTERVAL <= 60):
        raise ValueError("Analysis interval must be between 1 and 60 minutes")
```

### Manual Validation
```bash
# Test configuration loading
python -c "
from config import Config
print('✅ Telegram Token:', bool(Config.TELEGRAM_BOT_TOKEN))
print('✅ Chat ID:', bool(Config.TELEGRAM_CHAT_ID))
print('✅ Trading Pairs:', len(Config.TRADING_PAIRS))
print('✅ Risk %:', Config.RISK_PERCENTAGE)
"
```

## Security Configuration

### API Key Security
```bash
# Set proper file permissions
chmod 600 .env

# Use environment variables instead of .env file
export BINANCE_API_KEY="your_key"
export BINANCE_API_SECRET="your_secret"
```

### Network Security
```python
# Configure proxy if needed
import os
os.environ['HTTP_PROXY'] = 'http://proxy.company.com:8080'
os.environ['HTTPS_PROXY'] = 'http://proxy.company.com:8080'
```

## Monitoring Configuration

### Log Configuration
```python
# Advanced logging setup
LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': 'logs/trading_bot.log',
            'formatter': 'detailed',
            'level': 'DEBUG'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'detailed',
            'level': 'INFO'
        }
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'DEBUG'
    }
}
```

### Performance Monitoring
```python
# Enable performance metrics
ENABLE_PERFORMANCE_MONITORING = True
METRICS_INTERVAL_SECONDS = 60
METRICS_FILE = 'logs/performance_metrics.json'
```

## Troubleshooting Configuration

### Common Issues

1. **Missing Environment Variables:**
   ```bash
   # Check if variables are set
   env | grep -E "(TELEGRAM|BINANCE)"
   ```

2. **Invalid API Keys:**
   ```bash
   # Test API connectivity
   python -c "
   from binance.client import Client
   client = Client(Config.BINANCE_API_KEY, Config.BINANCE_API_SECRET)
   print(client.get_account_status())
   "
   ```

3. **Configuration File Issues:**
   ```bash
   # Validate .env file syntax
   python -c "
   import os
   from dotenv import load_dotenv
   load_dotenv()
   print('Environment loaded successfully')
   "
   ```

## Configuration Templates

### Beginner Template
```env
# Safe, conservative settings for beginners
TRADING_PAIRS=BTCUSDT,ETHUSDT
RISK_PERCENTAGE=1
ANALYSIS_INTERVAL_MINUTES=10
LEVERAGE=5
STOP_LOSS_PERCENTAGE=1.5
TAKE_PROFIT_PERCENTAGE=3
LOG_LEVEL=INFO
```

### Advanced Template
```env
# Advanced settings for experienced traders
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,BNBUSDT,ATOMUSDT
RISK_PERCENTAGE=2
ANALYSIS_INTERVAL_MINUTES=5
LEVERAGE=10
STOP_LOSS_PERCENTAGE=2
TAKE_PROFIT_PERCENTAGE=4
LOG_LEVEL=DEBUG
```

### Scalping Template
```env
# High-frequency scalping configuration
TRADING_PAIRS=BTCUSDT,ETHUSDT
RISK_PERCENTAGE=0.5
ANALYSIS_INTERVAL_MINUTES=2
LEVERAGE=15
STOP_LOSS_PERCENTAGE=0.5
TAKE_PROFIT_PERCENTAGE=1
LOG_LEVEL=WARNING
```

## Configuration Backup and Recovery

### Backup Configuration
```bash
# Backup current configuration
cp .env .env.backup
cp config.py config.py.backup
```

### Restore Configuration
```bash
# Restore from backup
cp .env.backup .env
cp config.py.backup config.py
```

### Version Control
```bash
# Track configuration changes
git add .env.example
git commit -m "Update configuration template"
```

This configuration guide provides comprehensive settings for different trading styles and risk preferences. Always test configurations in a demo environment before deploying to live trading.
