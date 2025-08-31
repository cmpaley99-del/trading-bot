# Cloud Server Deployment Guide

## Options for 24/7 Trading Bot Operation

### 1. DigitalOcean Droplet (Recommended)
**Cost**: ~$5-10/month
**Setup Time**: 15-30 minutes

```bash
# Step 1: Create DigitalOcean account
# Step 2: Create Ubuntu 22.04 Droplet
# Step 3: SSH into your droplet

# Install dependencies
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv git -y

# Clone your trading bot
git clone <your-repository-url>
cd trading-bot

# Setup virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
nano .env  # Edit with your API keys

# Run as a service
sudo nano /etc/systemd/system/trading-bot.service
```

### 2. AWS EC2 Instance
**Cost**: Free tier eligible or ~$5-15/month
**Setup**: Similar to DigitalOcean but with AWS console

### 3. Raspberry Pi (Local 24/7)
**Cost**: One-time $35-75
**Setup**: Runs locally, no monthly fees

## Service File for Systemd

Create `/etc/systemd/system/trading-bot.service`:
```ini
[Unit]
Description=Cryptocurrency Trading Bot
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/trading-bot
ExecStart=/home/ubuntu/trading-bot/venv/bin/python main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Enable and Start Service

```bash
sudo systemctl daemon-reload
sudo systemctl enable trading-bot
sudo systemctl start trading-bot
sudo systemctl status trading-bot
```

## Monitoring

```bash
# View logs
journalctl -u trading-bot -f

# Check status
sudo systemctl status trading-bot

# Restart service
sudo systemctl restart trading-bot
```

## Security Considerations

1. **Use strong passwords** for your server
2. **Enable firewall** (ufw)
3. **Use SSH keys** instead of passwords
4. **Keep system updated** regularly
5. **Secure your API keys** in environment variables

## Backup Strategy

1. **Regular database backups**
2. **Version control** for your code
3. **Backup configuration files**
4. **Monitor server resources**

## Recommended Cloud Providers

1. **DigitalOcean** - Easiest for beginners
2. **AWS EC2** - More features, slightly more complex
3. **Vultr** - Good alternative to DigitalOcean
4. **Linode** - Reliable and affordable

## Next Steps Tomorrow

1. Choose a cloud provider and create account
2. Set up your server instance
3. Deploy the trading bot code
4. Configure as a system service
5. Test and monitor the deployment

The bot will then run 24/7 and send you trade calls whenever profitable opportunities are detected!
