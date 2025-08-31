# API Reference

## Core Classes and Modules

### TradingSignals

The main signal generation engine that coordinates all trading activities.

#### Methods

##### `generate_trade_calls()`
Generates trade calls for all configured trading pairs.

**Returns:** `List[str]` - List of formatted trade call messages

**Example:**
```python
from trading_signals import trading_signals

trade_calls = trading_signals.generate_trade_calls()
for call in trade_calls:
    print(call)
```

##### `should_generate_signal()`
Checks if enough time has passed since the last signal generation.

**Returns:** `bool` - True if signal should be generated

##### `update_signal_time()`
Updates the timestamp of the last signal generation in the database.

### TechnicalAnalysis

Handles all technical indicator calculations and signal generation.

#### Methods

##### `calculate_indicators(df)`
Calculates technical indicators for the given OHLCV dataframe.

**Parameters:**
- `df` (DataFrame): OHLCV data with columns [timestamp, open, high, low, close, volume]

**Returns:** `DataFrame` - DataFrame with added technical indicators

**Indicators Calculated:**
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- ATR (Average True Range)
- EMA (Exponential Moving Averages)
- Stochastic Oscillator
- Williams %R
- CCI (Commodity Channel Index)

##### `generate_signals(df_with_indicators)`
Generates trading signals based on technical indicators.

**Parameters:**
- `df_with_indicators` (DataFrame): DataFrame with technical indicators

**Returns:** `dict` - Dictionary containing signal information

**Signal Types:**
- `overall_signal`: BULLISH, BEARISH, or NEUTRAL
- `rsi_signal`: Signal from RSI analysis
- `macd_signal`: Signal from MACD analysis
- `volume_signal`: Signal from volume analysis
- `trend_signal`: Signal from trend analysis
- `scalp_signal`: Short-term scalping signal

##### `calculate_appropriate_leverage(df_with_indicators, trading_pair)`
Calculates optimal leverage based on market conditions.

**Parameters:**
- `df_with_indicators` (DataFrame): DataFrame with technical indicators
- `trading_pair` (str): Trading pair symbol (e.g., 'BTCUSDT')

**Returns:** `int` - Recommended leverage (3x-30x)

**Factors Considered:**
- Market volatility (using ATR and historical volatility)
- Trend strength (using ADX)
- Current market conditions

### RiskManagement

Handles position sizing, stop-loss, and take-profit calculations.

#### Methods

##### `calculate_position_size(current_price, trading_pair, leverage)`
Calculates position size based on risk management rules.

**Parameters:**
- `current_price` (float): Current asset price
- `trading_pair` (str): Trading pair symbol
- `leverage` (int): Leverage multiplier

**Returns:** `float` - Position size in base currency

**Formula:**
```python
risk_amount = account_balance * (risk_percentage / 100)
position_size = (risk_amount * leverage) / current_price
```

##### `calculate_stop_loss(entry_price, signal_type, atr)`
Calculates stop-loss price using ATR for volatility-based placement.

**Parameters:**
- `entry_price` (float): Entry price
- `signal_type` (str): 'BULLISH' or 'BEARISH'
- `atr` (float): Average True Range value

**Returns:** `float` - Stop-loss price

**Formula:**
```python
# For BULLISH signals
stop_loss = entry_price - (atr * 1.5)

# For BEARISH signals
stop_loss = entry_price + (atr * 1.5)
```

##### `calculate_take_profit(entry_price, signal_type, stop_loss)`
Calculates take-profit price with 1:2 risk-reward ratio.

**Parameters:**
- `entry_price` (float): Entry price
- `signal_type` (str): 'BULLISH' or 'BEARISH'
- `stop_loss` (float): Stop-loss price

**Returns:** `float` - Take-profit price

**Formula:**
```python
# For BULLISH signals
risk = entry_price - stop_loss
take_profit = entry_price + (risk * 2)

# For BEARISH signals
risk = stop_loss - entry_price
take_profit = entry_price - (risk * 2)
```

##### `validate_signal_quality(signals, current_price, metrics, df_with_indicators)`
Validates signal quality using multiple criteria.

**Parameters:**
- `signals` (dict): Signal dictionary from technical analysis
- `current_price` (float): Current asset price
- `metrics` (dict): Market metrics
- `df_with_indicators` (DataFrame): DataFrame with technical indicators

**Returns:** `dict` - Quality assessment with score and confidence

**Quality Criteria:**
- RSI confirmation
- MACD confirmation
- Volume confirmation
- Trend confirmation
- ML confidence score
- Funding rate analysis

### MarketData

Handles market data fetching and processing.

#### Methods

##### `get_ohlcv_data(trading_pair, timeframe='5m', limit=100)`
Fetches OHLCV data from Binance Futures.

**Parameters:**
- `trading_pair` (str): Trading pair symbol
- `timeframe` (str): Timeframe ('1m', '5m', '15m', '1h', '4h', '1d')
- `limit` (int): Number of candles to fetch

**Returns:** `DataFrame` - OHLCV data

##### `get_market_metrics(trading_pair)`
Fetches current market metrics for a trading pair.

**Parameters:**
- `trading_pair` (str): Trading pair symbol

**Returns:** `dict` - Market metrics including:
- `current_price`: Current price
- `volume_24h`: 24-hour volume
- `price_change_24h`: 24-hour price change
- `funding_rate`: Current funding rate

### TelegramBot

Handles Telegram integration for notifications and commands.

#### Methods

##### `send_message(message)`
Sends a message to the configured Telegram chat.

**Parameters:**
- `message` (str): Message to send (supports Markdown formatting)

**Returns:** `bool` - True if message sent successfully

##### `send_trade_notification(trade_call)`
Sends a formatted trade notification.

**Parameters:**
- `trade_call` (dict): Trade call information

### Database

Handles data persistence and retrieval.

#### Methods

##### `save_last_signal_time(timestamp)`
Saves the timestamp of the last signal generation.

**Parameters:**
- `timestamp` (datetime): Signal generation timestamp

##### `get_last_signal_time()`
Retrieves the timestamp of the last signal generation.

**Returns:** `datetime` or `None`

##### `save_trade_record(trade_data)`
Saves trade execution data to database.

**Parameters:**
- `trade_data` (dict): Trade execution information

### AnomalyDetection

Monitors market for anomalous conditions.

#### Methods

##### `scan_for_anomalies()`
Scans all trading pairs for market anomalies.

**Returns:** `list` - List of detected anomalies

**Anomaly Types:**
- Price anomalies (sudden spikes/drops)
- Volume anomalies (unusual volume patterns)
- Volatility anomalies (extreme volatility changes)
- Correlation anomalies (unusual pair correlations)
- Funding rate anomalies

##### `detect_flash_crash(df)`
Detects potential flash crash conditions.

**Parameters:**
- `df` (DataFrame): Price data

**Returns:** `bool` - True if flash crash detected

### MLSignalPredictor

Uses machine learning to enhance signal quality prediction.

#### Methods

##### `predict_signal_success(df, signal_type)`
Predicts the probability of signal success using ML models.

**Parameters:**
- `df` (DataFrame): Historical data with indicators
- `signal_type` (str): 'BULLISH' or 'BEARISH'

**Returns:** `float` - Confidence score (0.0 to 1.0)

##### `train_model(df)`
Trains the ML model on historical data.

**Parameters:**
- `df` (DataFrame): Training data

**Returns:** `bool` - True if training successful

## Configuration

### Config Class

Central configuration management using environment variables.

#### Key Settings

```python
class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    # Trading Pairs
    TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', ...]

    # Risk Parameters
    RISK_PERCENTAGE = 2.0  # 2% risk per trade
    MAX_POSITION_SIZE = 1000  # Max $1000 per position

    # Technical Parameters
    RSI_PERIOD = 14
    MACD_FAST = 12
    MACD_SLOW = 26
    ANALYSIS_INTERVAL = 5  # minutes
```

## Error Handling

### Exception Types

- `ConnectionError`: Network connectivity issues
- `APIError`: Binance API errors
- `ValidationError`: Data validation failures
- `RiskError`: Risk management violations
- `SignalError`: Signal generation failures

### Error Recovery

The system implements automatic retry mechanisms for:
- Network timeouts (up to 3 retries with exponential backoff)
- API rate limits (intelligent delay and retry)
- Temporary service unavailability
- Database connection issues

## Data Structures

### Trade Call Format

```python
{
    'trading_pair': 'BTCUSDT',
    'signal_type': 'BULLISH',
    'entry_price': 50000.0,
    'leverage': 10,
    'position_size': 0.002,
    'stop_loss': 49000.0,
    'take_profit': 52000.0,
    'quality': 'HIGH',
    'confidence': 0.85,
    'timestamp': datetime.now()
}
```

### Signal Dictionary

```python
{
    'overall_signal': 'BULLISH',
    'rsi_signal': 'BULLISH',
    'macd_signal': 'BULLISH',
    'volume_signal': 'BULLISH',
    'trend_signal': 'BULLISH',
    'scalp_signal': 'NEUTRAL',
    'confidence_score': 0.82
}
```

### Market Metrics

```python
{
    'current_price': 50000.0,
    'volume_24h': 1500000.0,
    'price_change_24h': 2.5,
    'funding_rate': 0.0001,
    'open_interest': 100000.0,
    'volatility': 0.05
}
```

## Performance Considerations

### Optimization Techniques

1. **Data Caching**: Market data cached for 1-minute intervals
2. **Batch Processing**: Multiple pairs processed in parallel
3. **Async Operations**: Non-blocking I/O for API calls
4. **Memory Management**: DataFrame optimization for large datasets
5. **Connection Pooling**: Reused HTTP connections for efficiency

### Resource Usage

- **Memory**: ~50MB base usage, scales with number of pairs
- **CPU**: Minimal usage during analysis intervals
- **Network**: ~1MB/minute for 13 pairs at 5-minute intervals
- **Storage**: ~10MB/month for logs and trade history

## Security Considerations

### API Key Management
- Environment variables for sensitive data
- No hardcoded credentials
- Key rotation support

### Risk Controls
- Maximum position size limits
- Daily loss limits
- Emergency stop mechanisms
- Manual override capabilities

### Data Protection
- Encrypted database storage
- Secure logging practices
- No sensitive data in logs
