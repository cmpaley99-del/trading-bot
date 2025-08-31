# Cryptocurrency Futures Trading Bot

A sophisticated automated trading system for cryptocurrency futures markets with intelligent leverage management, multi-currency support, and advanced risk management.

## 🚀 Features

### Core Trading Features
- **Multi-Currency Support**: Trade across 13+ cryptocurrency pairs simultaneously
- **Intelligent Leverage**: Dynamic leverage calculation (3x-30x) based on market volatility and trend strength
- **Advanced Risk Management**: Position sizing, stop-loss, take-profit with 1:2 risk-reward ratio
- **Real-time Signal Generation**: Technical analysis with RSI, MACD, Bollinger Bands, and trend indicators
- **Machine Learning Enhancement**: ML-based signal quality prediction and confidence scoring

### Advanced Features
- **Anomaly Detection**: Real-time market anomaly monitoring and alerts
- **Scalping Strategies**: High-frequency trading signals for quick profits
- **Telegram Integration**: Real-time trade notifications and bot control
- **Backtesting Engine**: Historical performance analysis and strategy validation
- **Web Dashboard**: Real-time monitoring and performance visualization

## 📊 Supported Cryptocurrencies

- BTCUSDT (Bitcoin)
- ETHUSDT (Ethereum)
- SOLUSDT (Solana)
- ADAUSDT (Cardano)
- DOTUSDT (Polkadot)
- AVAXUSDT (Avalanche)
- MATICUSDT (Polygon)
- XRPUSDT (Ripple)
- LINKUSDT (Chainlink)
- DOGEUSDT (Dogecoin)
- LTCUSDT (Litecoin)
- BNBUSDT (Binance Coin)
- ATOMUSDT (Cosmos)

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- Binance Futures API keys
- Telegram Bot Token

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd cryptocurrency-trading-bot
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

4. **Run the bot**
   ```bash
   python main.py
   ```

## ⚙️ Configuration

### Environment Variables
```env
# Binance API Configuration
BINANCE_API_KEY=your_binance_api_key
BINANCE_API_SECRET=your_binance_api_secret

# Telegram Configuration
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id

# Trading Parameters
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT
RISK_PERCENTAGE=2
ANALYSIS_INTERVAL_MINUTES=5
```

### Trading Parameters
- **Risk Percentage**: Maximum risk per trade (default: 2%)
- **Analysis Interval**: Signal generation frequency (default: 5 minutes)
- **Leverage Range**: 3x to 30x based on market conditions

## 📈 Trading Strategy

### Signal Generation
The bot uses multiple technical indicators to generate trading signals:

1. **RSI (Relative Strength Index)**: Momentum oscillator
2. **MACD (Moving Average Convergence Divergence)**: Trend-following momentum indicator
3. **Bollinger Bands**: Volatility-based price channels
4. **ATR (Average True Range)**: Volatility measurement
5. **Volume Analysis**: Price-volume relationship confirmation

### Leverage Calculation
Intelligent leverage is calculated based on:
- Market volatility (using ATR and historical volatility)
- Trend strength (using ADX - Average Directional Index)
- Risk tolerance settings

### Risk Management
- **Position Sizing**: Risk-based position calculation
- **Stop Loss**: ATR-based dynamic stop loss placement
- **Take Profit**: 1:2 risk-reward ratio targets
- **Maximum Drawdown**: Portfolio-level risk controls

## 🔧 Architecture

### Core Modules

```
├── main.py                 # Main application entry point
├── config.py              # Configuration management
├── trading_signals.py     # Signal generation engine
├── technical_analysis.py  # Technical indicators and analysis
├── risk_management.py     # Risk management and position sizing
├── market_data.py         # Market data fetching and processing
├── telegram_bot.py        # Telegram integration and notifications
├── database.py            # Data persistence and logging
├── anomaly_detection.py   # Market anomaly monitoring
├── ml_signal_prediction.py # Machine learning signal enhancement
├── backtest.py            # Backtesting engine
└── dashboard.py           # Web dashboard
```

### Data Flow
1. **Market Data** → Fetch real-time OHLCV data
2. **Technical Analysis** → Calculate indicators and generate signals
3. **Risk Assessment** → Validate signal quality and calculate position sizes
4. **Trade Execution** → Execute trades with proper risk management
5. **Monitoring** → Track performance and send notifications

## 📊 Performance Monitoring

### Web Dashboard
Access the web dashboard at `http://localhost:5000` to monitor:
- Real-time account balance and P&L
- Active positions and performance
- Trading signals and execution history
- Risk metrics and drawdown analysis

### Telegram Notifications
Receive real-time notifications for:
- New trading signals with entry/exit points
- Trade execution confirmations
- Risk alerts and position updates
- System status and error notifications

## 🧪 Testing

### Test Coverage
```bash
# Run all tests
python -m pytest

# Run specific test categories
python test_integration.py      # Integration tests
python test_trading_signals.py  # Signal generation tests
python test_risk_management.py  # Risk management tests
python test_anomaly_detection.py # Anomaly detection tests
```

### Backtesting
```bash
# Run backtest on historical data
python backtest.py --symbol BTCUSDT --days 30
```

## 🚨 Risk Disclaimer

**⚠️ IMPORTANT: This software is for educational and research purposes only.**

- **Real Money Trading**: The bot can execute real trades with real money. Use at your own risk.
- **Market Risk**: Cryptocurrency markets are highly volatile. You may lose all your investment.
- **Technical Risk**: Software bugs or connectivity issues may result in unexpected losses.
- **Regulatory Risk**: Ensure compliance with local regulations regarding automated trading.

### Recommended Practices
1. **Start Small**: Begin with small position sizes
2. **Paper Trading**: Test extensively in demo mode first
3. **Risk Management**: Never risk more than you can afford to lose
4. **Monitoring**: Regularly monitor bot performance and intervene if needed
5. **Backup**: Maintain manual override capabilities

## 📚 Documentation

- [Setup Guide](setup_guide.md)
- [Configuration Guide](configuration_guide.md)
- [API Documentation](api_reference.md)
- [Troubleshooting](troubleshooting.md)
- [Changelog](CHANGELOG.md)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Telegram Community**: [Join our Telegram group](https://t.me/your-bot-community)

## 🔄 Version History

### v2.0.0 (Latest)
- ✅ Multi-currency support (13+ pairs)
- ✅ Intelligent leverage calculation
- ✅ Advanced risk management
- ✅ Machine learning signal enhancement
- ✅ Real-time anomaly detection
- ✅ Web dashboard
- ✅ Telegram integration

### v1.0.0
- ✅ Basic signal generation
- ✅ Single currency support
- ✅ Fixed leverage
- ✅ Telegram notifications
