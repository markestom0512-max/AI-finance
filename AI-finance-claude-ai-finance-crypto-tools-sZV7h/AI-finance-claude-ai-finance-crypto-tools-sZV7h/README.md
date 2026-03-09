# AI Finance — Market Analyzer

An AI-powered tool that analyzes crypto market movements in real-time and generates precise insights & predictions using Claude Opus 4.6.

## Architecture

```
Market Data (CCXT)
    ↓
Technical Analysis Engine
  • RSI, MACD, Bollinger Bands
  • Stochastic, ATR, EMA (9/21/50/200)
  • VWAP, OBV, Volume Profile
  • Candlestick Pattern Detection
  • Support/Resistance Levels
    ↓
Order Book Analysis
  • Bid/Ask Imbalance
  • Spread monitoring
    ↓
Claude Opus 4.6 (Adaptive Thinking)
  • Market context & trend
  • Signal confluence
  • Price predictions
  • Risk assessment
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

## Usage

```bash
# Interactive mode
python main.py

# Analyze BTC/USDT on 1h (default)
python main.py --symbol BTC/USDT

# ETH on 4h timeframe
python main.py --symbol ETH/USDT --tf 4h

# SOL on 15m via Bybit, 300 candles
python main.py --symbol SOL/USDT --tf 15m --exchange bybit --limit 300
```

## Supported Exchanges
- Binance (default)
- Kraken
- Coinbase
- Bybit
- OKX

## AI Analysis Sections

Claude produces 6 structured sections:

| Section | Content |
|---------|---------|
| A — Market Context | Overall structure, trend phase |
| B — Signal Confluence | Top 3-5 signals, strength rating |
| C — Critical Levels | Key S/R, VWAP, POC implications |
| D — Short-Term Prediction | Primary + alternative scenarios with probabilities |
| E — Risk Assessment | Stop-loss zones, R/R ratio |
| F — Actionable Summary | Clear directional stance |

## Disclaimer

This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making any trading decisions.
