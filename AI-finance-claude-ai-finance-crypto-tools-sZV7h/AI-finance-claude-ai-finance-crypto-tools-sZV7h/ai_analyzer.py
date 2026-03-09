"""
AI Market Analyzer
Uses Claude (claude-opus-4-6) with adaptive thinking to analyze market data
and generate precise insights & predictions.
"""

import json
import anthropic
from datetime import datetime, timezone


SYSTEM_PROMPT = """You are an expert quantitative analyst and crypto/financial market specialist.
You have deep expertise in:
- Technical analysis (RSI, MACD, Bollinger Bands, EMA, Stochastic, ATR, OBV, VWAP)
- Candlestick pattern recognition and price action analysis
- Order book dynamics and market microstructure
- Volume profile and support/resistance analysis
- Market psychology and sentiment interpretation
- Risk assessment and probabilistic forecasting

Your task is to analyze the provided market data and produce a precise, structured analysis.
Be specific with numbers, price levels, and timeframes. Never give vague advice.
Structure your response in clear sections. Be direct and actionable.

Important: Always clearly distinguish between high-confidence signals and speculative predictions.
Always mention key risk levels (stop loss zones) alongside any directional bias."""


def build_analysis_prompt(market_data: dict, analysis: dict) -> str:
    ticker = market_data["ticker"]
    ob = market_data["order_book"]
    ind = analysis["latest_indicators"]
    trend = analysis["trend"]
    sr = analysis["support_resistance"]
    patterns = analysis["candlestick_patterns"]
    vol_profile = analysis["volume_profile"]
    recent = analysis["recent_candles"]

    prompt = f"""
## Market Analysis Request

**Asset:** {market_data['symbol']} on {market_data['exchange'].capitalize()}
**Timeframe:** {market_data['timeframe']} | **Candles analyzed:** {market_data['candles_fetched']}
**Analysis time:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

---

### 1. Current Price & 24h Market Context
- **Current Price:** ${ticker['price']:,.6f}
- **24h Change:** {ticker['change_24h_pct']:+.2f}%
- **24h High / Low:** ${ticker['high_24h']:,.6f} / ${ticker['low_24h']:,.6f}
- **24h Volume (base):** {ticker['volume_24h']:,.2f}

### 2. Order Book Snapshot (top {20} levels)
- **Best Bid / Ask:** ${ob['best_bid']:,.6f} / ${ob['best_ask']:,.6f}
- **Spread:** {ob['spread_pct']:.4f}%
- **Bid/Ask Imbalance:** {ob['imbalance']:+.4f} ({'more buyers' if ob['imbalance'] > 0 else 'more sellers'})
- **Bid Volume / Ask Volume:** {ob['bid_volume']:.4f} / {ob['ask_volume']:.4f}

### 3. Technical Indicators (latest candle)
**Momentum:**
- RSI(14): {ind['rsi_14']} {'(overbought >70)' if ind['rsi_14'] > 70 else '(oversold <30)' if ind['rsi_14'] < 30 else '(neutral)'}
- Stochastic K/D: {ind['stoch_k']} / {ind['stoch_d']}
- MACD Line: {ind['macd']:+.6f} | Signal: {ind['macd_signal']:+.6f} | Histogram: {ind['macd_histogram']:+.6f}

**Trend:**
- EMA 9: {ind['ema_9']:,.6f} | EMA 21: {ind['ema_21']:,.6f} | EMA 50: {ind['ema_50']:,.6f} | EMA 200: {ind['ema_200']:,.6f}
- Price vs VWAP: {'+' if ind['price'] > ind['vwap'] else '-'}{abs(ind['price'] - ind['vwap']) / ind['vwap'] * 100:.2f}% ({'above' if ind['price'] > ind['vwap'] else 'below'} VWAP)
- Trend Direction: {trend['direction'].upper()} ({trend['strength']}) | 10-candle slope: {trend['slope_10c_pct']:+.2f}%

**Volatility:**
- ATR(14): {ind['atr_14']:,.6f} ({ind['atr_14'] / ind['price'] * 100:.2f}% of price)
- BB Upper: {ind['bb_upper']:,.6f} | Middle: {ind['bb_middle']:,.6f} | Lower: {ind['bb_lower']:,.6f}
- %B: {ind['bb_pct_b']:.3f} (0=lower band, 1=upper band) | Bandwidth: {ind['bb_bandwidth']:.4f}

**Volume:**
- OBV trend: {ind['obv_trend'].upper()}
- Current volume vs 20-period avg: {ind['volume_vs_avg']:.2f}x

### 4. Support & Resistance Levels
**Resistance zones:** {', '.join([f'${r:,.6f}' for r in sr['resistance_levels']]) or 'None identified'}
**Support zones:** {', '.join([f'${s:,.6f}' for s in sr['support_levels']]) or 'None identified'}

### 5. Volume Profile
- **Point of Control (POC):** ${vol_profile['point_of_control']:,.6f}
- **High-volume price zones:** {', '.join([f'${z:,.6f}' for z in vol_profile['high_volume_zones'][:6]])}

### 6. Recent Candlestick Patterns Detected
{_format_patterns(patterns)}

### 7. Last 5 Candles
{_format_candles(recent)}

---

## Your Analysis

Please provide a comprehensive market analysis covering:

**Section A — Market Context & Trend**
Describe the overall market structure, dominant trend, and phase (accumulation/distribution/trending/ranging).

**Section B — Key Signal Confluence**
Identify the 3-5 most significant signals from the indicators above and explain what they collectively suggest. Rate the overall signal strength (weak/moderate/strong).

**Section C — Critical Price Levels**
List the most important price levels to watch (support, resistance, POC, VWAP). Explain what a break of each level would imply.

**Section D — Short-Term Prediction (next 3-10 candles)**
Based on current signals, what is the most probable price action? Give:
- Primary scenario (most likely outcome) with probability estimate
- Alternative scenario with conditions that would trigger it
- Key price targets for each scenario

**Section E — Risk Assessment**
- Recommended stop-loss zone
- Risk/reward ratio if taking a trade at current levels
- Main risk factors that could invalidate the analysis

**Section F — Actionable Summary**
One clear, concise paragraph summarizing the situation and recommended stance (bullish/bearish/neutral/wait).
"""
    return prompt


def _format_patterns(patterns: list) -> str:
    if not patterns:
        return "No significant patterns detected in recent candles."
    lines = []
    for p in patterns[-5:]:
        signal_emoji = "🟢" if "bullish" in p["signal"] or p["signal"] == "strong_bullish" else \
                       "🔴" if "bearish" in p["signal"] or p["signal"] == "strong_bearish" else "⚪"
        lines.append(f"- {signal_emoji} **{p['pattern']}** at {p['index']} → {p['signal'].replace('_', ' ').title()}")
    return "\n".join(lines)


def _format_candles(candles: list) -> str:
    lines = []
    for c in candles:
        direction = "▲" if c["bullish"] else "▼"
        lines.append(
            f"- {c['time']}: O={c['open']:.6f} H={c['high']:.6f} L={c['low']:.6f} C={c['close']:.6f} "
            f"Vol={c['volume']:.2f} {direction}"
        )
    return "\n".join(lines)


def analyze_market(
    market_data: dict,
    technical_analysis: dict,
    stream: bool = True,
) -> str:
    """
    Send market data to Claude for AI-powered analysis.

    Args:
        market_data: Output of market_data.get_market_summary()
        technical_analysis: Output of technical_analysis.full_analysis()
        stream: Whether to stream the response (recommended for long outputs)

    Returns:
        Full analysis text from Claude
    """
    client = anthropic.Anthropic()
    prompt = build_analysis_prompt(market_data, technical_analysis)

    if stream:
        full_response = ""
        with client.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream_ctx:
            for event in stream_ctx:
                if event.type == "content_block_start":
                    if event.content_block.type == "thinking":
                        pass  # Thinking is processed internally
                elif event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                        full_response += event.delta.text

            final_msg = stream_ctx.get_final_message()

        return full_response
    else:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks)
