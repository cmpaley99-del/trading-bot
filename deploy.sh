#!/bin/bash

# 🚀 Trading Bot Railway Deployment Script
# This script helps you deploy your trading bot to Railway

echo "🚀 Trading Bot Railway Deployment Script"
echo "========================================"

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo "📝 Initializing Git repository..."
    git init
    echo "✅ Git repository initialized"
else
    echo "✅ Git repository already exists"
fi

# Check if there are uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    echo "📝 Committing changes..."
    git add .
    git commit -m "Deploy: Trading bot ready for Railway"
    echo "✅ Changes committed"
else
    echo "✅ No uncommitted changes"
fi

# Check if remote origin exists
if git remote get-url origin > /dev/null 2>&1; then
    echo "✅ Remote origin already configured"
else
    echo "❓ Please provide your Git repository URL:"
    echo "   Example: https://github.com/YOUR_USERNAME/trading-bot.git"
    read -p "Repository URL: " repo_url

    if [ -n "$repo_url" ]; then
        git remote add origin "$repo_url"
        echo "✅ Remote origin added"
    else
        echo "❌ No repository URL provided. Please add it manually:"
        echo "   git remote add origin YOUR_REPO_URL"
        exit 1
    fi
fi

# Push to repository
echo "📤 Pushing code to repository..."
if git push -u origin main 2>/dev/null; then
    echo "✅ Code pushed successfully"
elif git push -u origin master 2>/dev/null; then
    echo "✅ Code pushed successfully (to master branch)"
else
    echo "❌ Failed to push code. Please check your repository URL and try again."
    exit 1
fi

echo ""
echo "🎉 Code successfully pushed to repository!"
echo ""
echo "📋 Next Steps:"
echo "1. Go to https://railway.app"
echo "2. Sign up/Login with your GitHub/GitLab account"
echo "3. Click 'New Project'"
echo "4. Select 'Deploy from GitHub repo' or 'Deploy from GitLab repo'"
echo "5. Search for your repository and click 'Deploy'"
echo "6. Add environment variables in Railway dashboard:"
echo "   - TELEGRAM_BOT_TOKEN"
echo "   - TELEGRAM_CHAT_ID"
echo "   - BINANCE_API_KEY (optional)"
echo "   - BINANCE_API_SECRET (optional)"
echo ""
echo "📖 For detailed instructions, see RAILWAY_DEPLOYMENT_GUIDE.md"
echo ""
echo "🔗 Your Railway URL will be available after deployment"
echo "📊 Dashboard will be accessible at: https://your-app-url.railway.app"
