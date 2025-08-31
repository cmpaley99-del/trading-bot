# 🚀 Cryptocurrency Futures Trading Bot - Setup Guide

## 📋 Prerequisites

### 1. Python 3.8+
```bash
python --version
```

### 2. TA-Lib Installation
**Windows:**
1. Download TA-Lib wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
2. Install with: `pip install TA_Lib‑0.4.24‑cp38‑cp38‑win_amd64.whl` (adjust version)

**macOS:**
```bash
brew install ta-lib
```

**Linux:**
```bash
sudo apt-get install libta-lib-dev
```

## 🔧 Installation Steps

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
# Copy the example file
cp .env.example .env

# Edit .env with your credentials
nano .env  # or use any text editor
```

### 3. Get Telegram Bot Token
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow instructions to create bot
4. Copy the token to `.env`

### 4. Get Your Chat ID
1. Message your new bot
2. Check bot logs or use: https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
3. Copy chat_id to `.env`

### 5. Get Binance API Keys
1. Login to Binance
2. Go to API Management
3. Create API Key with Futures trading enabled
4. Copy API Key and Secret to `.env`

## 🏃‍♂️ Running the Bot

### Basic Run
```bash
python main.py
```

### Run in Background (Linux/macOS)
```bash
nohup python main.py > bot.log 2>&1 &
```

### Run as Windows Service
1. Use NSSM: `nssm install TradingBot python main.py`
2. Or use Task Scheduler to run on startup

## ⚙️ Configuration Options

### Trading Parameters (in .env)
```env
TRADING_PAIR=BTCUSDT      # Trading pair
LEVERAGE=10               # Leverage (1-125)
RISK_PERCENTAGE=2         # Risk per trade (%)
ANALYSIS_INTERVAL=5       # Minutes between analysis
```

### Technical Analysis Settings
```env
RSI_PERIOD=14
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
BB_PERIOD=20
BB_STD=2
```

## 📊 Monitoring

### Log Files
- Check `logs/trading_bot.log` for detailed logs
- Logs include all signals, errors, and performance data

### Database
- SQLite database: `trading_bot.db`
- Contains all signals, trades, and performance metrics

### Performance Tracking
The bot automatically tracks:
- Win rate and trade statistics
- Total P&L and average return
- Signal quality metrics
- Drawdown analysis

## 🛠️ Troubleshooting

### Common Issues
1. **TA-Lib not found**: Install TA-Lib system library first
2. **API rate limits**: Bot handles rate limiting automatically
3. **Telegram errors**: Check bot token and chat ID
4. **Binance connection**: Verify API keys and permissions

### Debug Mode
Set `LOG_LEVEL=DEBUG` in `.env` for detailed logging

## 🔒 Security Notes

1. **Never share your .env file**
2. **Use restricted API keys** on Binance
3. **Regularly rotate API keys**
4. **Start with small amounts** for testing

## 📈 Performance Tips

1. **Test first**: Use paper trading or small amounts
2. **Monitor regularly**: Check logs and performance
3. **Adjust parameters**: Fine-tune based on market conditions
4. **Stay updated**: Keep dependencies current

## 🆘 Support

If you encounter issues:
1. Check the logs in `logs/trading_bot.log`
2. Verify all API keys are correct
3. Ensure TA-Lib is properly installed
4. Check internet connection and firewall settings

## 🎯 Success Metrics

The bot is working correctly when:
- ✅ Sends automatic signals every 5 minutes
- ✅ Persists signal timing across restarts
- ✅ Handles market data errors gracefully
- ✅ Provides clear risk management parameters
- ✅ Stores all data in the database

Happy trading! 🚀
