# 📋 Binance API Setup Guide (Read-Only Access)

## 🔒 Security-First Approach
This guide shows you how to create Binance API keys with **READ-ONLY** access only - no trading permissions. This allows your bot to fetch market data without any risk of making trades.

## 🚀 Step-by-Step Instructions

### 1. Login to Binance
- Go to [Binance.com](https://www.binance.com)
- Login to your account

### 2. Navigate to API Management
- Click on your profile icon (top right)
- Select "API Management"
- Or go directly: https://www.binance.com/en/my/settings/api-management

### 3. Create New API Key
- Click "Create API" button
- Give it a descriptive name like "TradingBot-ReadOnly"
- **IMPORTANT**: DO NOT enable any trading permissions
- Click "Create"

### 4. Set Permissions (CRITICAL STEP)
When creating the API key, **ONLY** enable these permissions:
- ✅ **Enable Reading** (this is all you need)
- ❌ **Disable Spot & Margin Trading**
- ❌ **Disable Futures Trading** 
- ❌ **Disable Withdrawals**
- ❌ **Disable other permissions**

### 5. Copy Your API Keys
- Copy the **API Key** (starts with alphanumeric characters)
- Copy the **Secret Key** (longer string)
- Store them securely

### 6. Update Your .env File
Open your `.env` file and replace the placeholder values:

```env
BINANCE_API_KEY=your_actual_api_key_here
BINANCE_API_SECRET=your_actual_secret_key_here
```

### 7. Test the Configuration
Run this command to verify your API keys work:
```bash
python -c "from config import Config; print('API Key configured successfully')"
```

## 🔐 Security Best Practices

### What READ-ONLY Access Allows:
- ✅ View account balance (read-only)
- ✅ Fetch market data and prices
- ✅ Get order book data
- ✅ Access historical data
- ✅ Check funding rates

### What READ-ONLY Access Prevents:
- ❌ Making any trades
- ❌ Withdrawing funds
- ❌ Modifying orders
- ❌ Changing account settings

### Additional Security Measures:
1. **IP Restrictions**: Consider adding your server IP to the API key restrictions
2. **Regular Rotation**: Rotate API keys every 3-6 months
3. **Monitor Usage**: Regularly check API usage in Binance
4. **Immediate Revocation**: If you suspect any issues, revoke the key immediately

## 🛠️ Troubleshooting

### Common Issues:
1. **"Invalid Api-Key ID"**: Usually means the API key doesn't exist or is incorrect
2. **Permission errors**: Make sure "Enable Reading" is the only permission enabled
3. **IP restrictions**: Check if your IP is whitelisted if you enabled IP restrictions

### Verification Steps:
1. Test API connection: `python main.py` should start without authentication errors
2. Check market data: The bot should be able to fetch prices and indicators
3. Verify Telegram messages: Signals should be sent to your chat ID 1479869528

## 📞 Support
If you encounter issues:
1. Double-check API key permissions in Binance
2. Verify the keys are correctly copied to your .env file
3. Ensure no extra spaces or characters in the .env file

## ⚠️ Important Notes
- Never share your API keys with anyone
- Keep your .env file secure and private
- Regularly monitor your Binance account for any suspicious activity
- Start with small amounts when testing with real funds (if you later enable trading)

Your bot will now be able to:
- Fetch real-time market data from Binance
- Perform technical analysis
- Generate trading signals
- Send Telegram notifications
- **Without** any trading capabilities

Happy trading! 🚀
