# 🚀 Railway Deployment Guide for Trading Bot

This guide will walk you through deploying your trading bot to Railway for 24/7 operation.

## 📋 Prerequisites

- GitHub or GitLab account
- Railway account (free tier available)
- Telegram Bot Token and Chat ID
- (Optional) Binance API credentials

## 📝 Step 1: Prepare Your Code Repository

### 1.1 Create a Git Repository

```bash
# Initialize git in your project directory
git init

# Add all files to git
git add .

# Commit your changes
git commit -m "Initial commit: Trading bot with Railway deployment files"
```

### 1.2 Create Repository on GitHub/GitLab

1. Go to [GitHub](https://github.com) or [GitLab](https://gitlab.com)
2. Click "New Repository"
3. Name it `trading-bot` or similar
4. Don't initialize with README (you already have files)
5. Copy the repository URL

### 1.3 Push Code to Repository

```bash
# Add remote repository
git remote add origin https://github.com/YOUR_USERNAME/trading-bot.git

# Push your code
git push -u origin main
```

## 🚂 Step 2: Deploy to Railway

### 2.1 Connect Repository to Railway

1. Go to [Railway.app](https://railway.app)
2. Sign up/Login with your GitHub/GitLab account
3. Click "New Project"
4. Select "Deploy from GitHub repo" or "Deploy from GitLab repo"
5. Search for your `trading-bot` repository
6. Click "Deploy"

### 2.2 Configure Environment Variables

After deployment starts, go to your project dashboard:

1. Click on your project
2. Go to "Variables" tab
3. Add the following environment variables:

#### Required Variables:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
```

#### Optional Variables (for live trading):
```
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_api_secret_here
```

#### Trading Configuration (optional - defaults provided):
```
TRADING_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT,DOTUSDT,AVAXUSDT,MATICUSDT,XRPUSDT,LINKUSDT,DOGEUSDT,LTCUSDT,BNBUSDT,ATOMUSDT
LEVERAGE=10
ANALYSIS_INTERVAL_MINUTES=5
RISK_PERCENTAGE=2
MAX_POSITION_SIZE_USDT=1000
```

### 2.3 Monitor Deployment

Railway will automatically:
- Install Python dependencies from `requirements.txt`
- Use `railway.json` configuration
- Start your bot using `python main.py`
- Set up health checks and auto-restart

## 🔧 Step 3: Post-Deployment Configuration

### 3.1 Get Your Railway URL

After deployment:
1. Go to your Railway project dashboard
2. Click on "Settings" tab
3. Copy the "Public URL" (something like `https://trading-bot-production.up.railway.app`)

### 3.2 Update Telegram Webhook (Optional)

If you want to receive Telegram updates via webhook instead of polling:

```python
# In your telegram_bot.py, you can add webhook support
# But for Railway, polling works fine
```

### 3.3 Test Your Deployment

1. Check Railway logs to ensure the bot started successfully
2. Send a test message to your Telegram bot
3. Access the dashboard at your Railway URL

## 📊 Step 4: Monitoring and Maintenance

### 4.1 View Logs

```bash
# Railway provides real-time logs in the dashboard
# Go to your project → "Logs" tab
```

### 4.2 Monitor Performance

- Railway provides basic metrics in the dashboard
- Your bot's dashboard is available at: `https://your-app-url.railway.app`
- Telegram notifications will keep you updated on signals

### 4.3 Update Your Bot

```bash
# Make changes to your code
git add .
git commit -m "Update: Improved signal generation"
git push origin main

# Railway will automatically redeploy
```

## 🛠️ Troubleshooting

### Common Issues:

1. **Build Failures**
   - Check `requirements.txt` for correct package versions
   - Ensure all dependencies are listed

2. **Environment Variables Not Set**
   - Double-check variable names (case-sensitive)
   - Restart deployment after adding variables

3. **Telegram Bot Not Responding**
   - Verify `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID`
   - Check Railway logs for authentication errors

4. **Memory/CPU Limits**
   - Railway free tier has limits
   - Monitor usage in Railway dashboard
   - Consider upgrading if needed

### Useful Commands:

```bash
# Check Railway CLI (if installed)
railway logs
railway variables
railway restart
```

## 💰 Railway Pricing

- **Free Tier**: 512MB RAM, 1GB storage, sufficient for most trading bots
- **Hobby Plan**: $5/month for more resources
- **Pro Plan**: $10/month for even more resources

## 🔒 Security Best Practices

1. **Never commit API keys** to your repository
2. **Use Railway environment variables** for sensitive data
3. **Regularly update dependencies** in `requirements.txt`
4. **Monitor logs** for any security issues
5. **Use strong, unique passwords** for your accounts

## 📞 Support

- Railway Documentation: https://docs.railway.app/
- Railway Discord: https://discord.gg/railway
- GitHub Issues: Create issues in your repository for bot-specific problems

## 🎯 Next Steps

1. Monitor your bot's performance
2. Fine-tune trading parameters based on results
3. Add more features as needed
4. Consider setting up alerts for Railway deployment issues

Your trading bot is now running 24/7 on Railway! 🚀
