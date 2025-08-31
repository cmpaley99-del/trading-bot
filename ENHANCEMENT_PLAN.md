# Trading Bot Enhancement Plan

## 1. Multiple Trade Calls per Pair
- Modify `_generate_multiple_trade_calls()` to generate both regular and scalping signals
- Create separate trade calls for different signal types when both are present
- Include operation type (scalping/regular) in trade messages

## 2. Enhanced Signal Detection
- Identify when both overall and scalping signals are present
- Generate separate trade calls for different timeframes/strategies
- Add signal strength scoring to prioritize opportunities

## 3. Telegram Message Format
- Include operation type in trade messages
- Add signal strength indicators
- Include multiple opportunities in single messages when appropriate

## 4. Configuration Updates
- Add thresholds for multiple signal generation
- Configure minimum signal strength requirements
- Set maximum trade calls per pair

## 5. Risk Management
- Ensure proper position sizing for multiple signals
- Implement correlation checks between signals
- Add maximum exposure limits per asset
