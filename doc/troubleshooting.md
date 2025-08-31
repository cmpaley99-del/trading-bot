# Troubleshooting Guide

## Common Issues and Solutions

### 1. Telegram Bot Not Sending Messages

**Symptoms:**
- Bot starts successfully but no messages are received in Telegram
- Error messages in logs about Telegram API

**Solutions:**

1. **Check Telegram Bot Token:**
   ```bash
   # Verify token format (should start with 'bot')
   echo $TELEGRAM_BOT_TOKEN
   ```

2. **Verify Chat ID:**
   ```bash
   # Send a test message to verify chat ID
   curl -X POST "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage" \
        -d "chat_id=$TELEGRAM_CHAT_ID&text=Test message"
   ```

3. **Check Network Connectivity:**
   ```bash
   # Test internet connection
   ping api.telegram.org
   ```

4. **Review Bot Permissions:**
   - Ensure the bot is added to the Telegram group/channel
   - Verify bot has message sending permissions

### 2. Binance API Connection Issues

**Symptoms:**
- "API key not found" errors
- "Invalid API key" messages
- Market data fetch failures

**Solutions:**

1. **Verify API Keys:**
   ```bash
   # Check if environment variables are set
   echo $BINANCE_API_KEY
   echo $BINANCE_API_SECRET
   ```

2. **Test API Connectivity:**
   ```python
   from binance.client import Client
   client = Client(api_key, api_secret)
   print(client.get_account())
   ```

3. **Check API Permissions:**
   - Ensure "Enable Futures" is checked in API settings
   - Verify "Enable Trading" permission is enabled
   - Check IP whitelist if configured

4. **Rate Limit Issues:**
   - Wait a few minutes if hitting rate limits
   - Consider upgrading to VIP account for higher limits

### 3. Signal Generation Problems

**Symptoms:**
- No trade signals generated
- Signals generated too frequently or infrequently
- Low quality signals

**Solutions:**

1. **Check Market Data:**
   ```python
   # Test market data fetching
   from market_data import market_data
   df = market_data.get_ohlcv_data('BTCUSDT')
   print(f"Data points: {len(df)}")
   ```

2. **Verify Technical Indicators:**
   ```python
   # Check if indicators are calculated correctly
   from technical_analysis import technical_analysis
   indicators = technical_analysis.calculate_indicators(df)
   print(indicators.head())
   ```

3. **Adjust Signal Quality Threshold:**
   ```python
   # Lower quality threshold in config if needed
   RISK_QUALITY_THRESHOLD = 0.3  # Adjust as needed
   ```

4. **Check Analysis Interval:**
   ```python
   # Verify analysis interval setting
   print(f"Analysis interval: {Config.ANALYSIS_INTERVAL} minutes")
   ```

### 4. Database Connection Issues

**Symptoms:**
- "Database locked" errors
- "Table not found" messages
- Data persistence failures

**Solutions:**

1. **Check Database File:**
   ```bash
   # Verify database file exists and permissions
   ls -la trading_bot.db
   ```

2. **Reset Database:**
   ```bash
   # Backup and recreate database
   cp trading_bot.db trading_bot.db.backup
   rm trading_bot.db
   python -c "from database import database; database.initialize()"
   ```

3. **Check Database Schema:**
   ```bash
   # Verify table structure
   sqlite3 trading_bot.db ".schema"
   ```

### 5. Memory and Performance Issues

**Symptoms:**
- High CPU usage
- Memory consumption increasing over time
- Bot becoming unresponsive

**Solutions:**

1. **Monitor Resource Usage:**
   ```bash
   # Check system resources
   top -p $(pgrep -f "python.*main.py")
   ```

2. **Reduce Trading Pairs:**
   ```bash
   # Limit number of pairs in config
   TRADING_PAIRS = "BTCUSDT,ETHUSDT"  # Reduce from 13 to 2
   ```

3. **Increase Analysis Interval:**
   ```bash
   # Reduce frequency of analysis
   ANALYSIS_INTERVAL_MINUTES = 10  # Increase from 5 to 10
   ```

4. **Enable Data Caching:**
   ```python
   # Implement caching in market_data.py
   @lru_cache(maxsize=100)
   def get_cached_data(symbol, timeframe):
       return fetch_fresh_data(symbol, timeframe)
   ```

### 6. Signal Quality Issues

**Symptoms:**
- Too many false signals
- Signals not profitable
- Low success rate

**Solutions:**

1. **Adjust Quality Thresholds:**
   ```python
   # Increase quality requirements
   MIN_SIGNAL_CONFIDENCE = 0.7
   MIN_RSI_CONFIRMATION = True
   MIN_MACD_CONFIRMATION = True
   ```

2. **Fine-tune Technical Parameters:**
   ```python
   # Adjust indicator parameters
   RSI_PERIOD = 21  # Increase from 14
   MACD_FAST = 8    # Decrease from 12
   MACD_SLOW = 21   # Decrease from 26
   ```

3. **Add Market Condition Filters:**
   ```python
   # Filter signals based on market conditions
   MIN_VOLUME_THRESHOLD = 1000000
   MAX_VOLATILITY_THRESHOLD = 0.05
   ```

### 7. Telegram Rate Limiting

**Symptoms:**
- Messages not being sent
- "Too many requests" errors
- Delayed message delivery

**Solutions:**

1. **Implement Message Queuing:**
   ```python
   # Add message queue with rate limiting
   from collections import deque
   message_queue = deque(maxlen=20)  # Limit concurrent messages
   ```

2. **Add Delays Between Messages:**
   ```python
   # Add delays for bulk messages
   import time
   time.sleep(2)  # 2-second delay between messages
   ```

3. **Batch Messages:**
   ```python
   # Combine multiple signals into single message
   combined_message = "\n\n".join(signal_messages)
   send_message(combined_message)
   ```

### 8. Configuration Issues

**Symptoms:**
- Bot not starting
- "Missing required environment variables" error
- Unexpected behavior

**Solutions:**

1. **Verify .env File:**
   ```bash
   # Check .env file exists and has correct format
   cat .env
   ```

2. **Environment Variable Loading:**
   ```bash
   # Test environment variable loading
   python -c "from config import Config; print(Config.TELEGRAM_BOT_TOKEN)"
   ```

3. **Configuration Validation:**
   ```python
   # Add configuration validation
   def validate_config():
       required_vars = ['TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
       for var in required_vars:
           if not getattr(Config, var):
               raise ValueError(f"Missing: {var}")
   ```

## Diagnostic Commands

### System Health Check
```bash
# Run comprehensive health check
python -c "
from config import Config
from market_data import market_data
from telegram_bot import TelegramBot

print('🔍 System Health Check')
print(f'✅ Config loaded: {bool(Config.TELEGRAM_BOT_TOKEN)}')
print(f'✅ Market data: {bool(market_data.get_ohlcv_data(\"BTCUSDT\"))}')
print(f'✅ Telegram: {bool(TelegramBot().send_message(\"Health check\"))}')
"
```

### Log Analysis
```bash
# Analyze recent logs for errors
tail -n 50 logs/trading_bot.log | grep -i error

# Count error types
grep -i "error\|exception" logs/trading_bot.log | \
    sed 's/.*ERROR.*//' | sort | uniq -c | sort -nr
```

### Performance Monitoring
```bash
# Monitor bot performance
watch -n 30 "ps aux | grep 'python.*main.py' | grep -v grep"

# Check memory usage
python -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## Emergency Procedures

### Emergency Stop
```bash
# Kill bot process immediately
pkill -f "python.*main.py"

# Or find and kill specific process
ps aux | grep "python.*main.py"
kill -9 <PID>
```

### Data Recovery
```bash
# Restore from backup
cp trading_bot.db.backup trading_bot.db

# Verify database integrity
python -c "import sqlite3; conn = sqlite3.connect('trading_bot.db'); print('DB OK')"
```

### System Reset
```bash
# Complete system reset
rm -rf logs/*.log
rm trading_bot.db
rm -rf __pycache__/
rm -rf */__pycache__/

# Reinitialize
python main.py
```

## Prevention Best Practices

### Regular Maintenance
1. **Daily Log Review:** Check logs for unusual errors
2. **Weekly Performance Check:** Review signal quality and profitability
3. **Monthly System Update:** Update dependencies and restart bot
4. **Database Backup:** Regular database backups

### Monitoring Setup
1. **Enable System Monitoring:**
   ```bash
   # Install monitoring tools
   pip install psutil
   ```

2. **Set Up Alerts:**
   ```python
   # Add alert system for critical errors
   def send_alert(message):
       # Send critical alerts via email/telegram
       pass
   ```

3. **Performance Tracking:**
   ```python
   # Log performance metrics
   logger.info(f"Signal quality: {signal_quality}")
   logger.info(f"Response time: {response_time}")
   ```

## Support Resources

### Documentation
- [API Reference](api_reference.md)
- [Configuration Guide](configuration_guide.md)
- [Setup Guide](setup_guide.md)

### Community Support
- GitHub Issues: Report bugs and request features
- Telegram Community: Join discussions and get help
- Documentation Wiki: Extended guides and tutorials

### Professional Services
- Code Review: Professional code analysis
- Performance Optimization: System tuning services
- Custom Development: Feature development and integration
