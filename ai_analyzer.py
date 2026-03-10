"""
AI Market Analyzer
Uses Claude Haiku 4.5 to analyze market data and generate concise insights & scenarios.
"""

import anthropic
from datetime import datetime, timezone


SYSTEM_PROMPT = """You are a concise crypto market analyst. Analyze the provided data and respond ONLY with the structured format below. No preamble, no extra commentary.

Keep each section short and actionable. Use specific price levels. Be direct."""


def build_analysis_prompt(market_data: dict, analysis: dict) -> str:
    ticker = market_data["ticker"]
    ob = market_data["order_book"]
    ind = analysis["latest_indicators"]
    trend = analysis["trend"]
    sr = analysis["support_resistance"]
    patterns = analysis["candlestick_patterns"]
    vol_profile = analysis["volume_profile"]

    # Summarize patterns (last 3 only)
    pattern_str = "None"
    if patterns:
        pattern_str = ", ".join([f"{p['pattern']} ({p['signal'].replace('_', ' ')})" for p in patterns[-3:]])

    resistances = ", ".join([f"${r:,.4f}" for r in sr["resistance_levels"][:3]]) or "None"
    supports = ", ".join([f"${s:,.4f}" for s in sr["support_levels"][:3]]) or "None"

    prompt = f"""
Asset: {market_data['symbol']} | {market_data['timeframe']} | {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Price: ${ticker['price']:,.4f} | 24h: {ticker['change_24h_pct']:+.2f}% | Vol vs avg: {ind['volume_vs_avg']:.2f}x

INDICATORS:
- RSI(14): {ind['rsi_14']:.1f} | Stoch K/D: {ind['stoch_k']:.1f}/{ind['stoch_d']:.1f}
- MACD histogram: {ind['macd_histogram']:+.6f} | OBV: {ind['obv_trend']}
- Trend: {trend['direction']} ({trend['strength']}) | Slope 10c: {trend['slope_10c_pct']:+.2f}%
- BB %B: {ind['bb_pct_b']:.3f} | ATR: {ind['atr_14']:,.4f} ({ind['atr_14']/ind['price']*100:.2f}%)
- Price vs VWAP: {(ind['price']-ind['vwap'])/ind['vwap']*100:+.2f}%
- EMA 9/21/50/200: {ind['ema_9']:,.4f} / {ind['ema_21']:,.4f} / {ind['ema_50']:,.4f} / {ind['ema_200']:,.4f}

ORDER BOOK: Imbalance {ob['imbalance']:+.3f} ({'buy pressure' if ob['imbalance'] > 0 else 'sell pressure'}) | Spread: {ob['spread_pct']:.4f}%

LEVELS:
- POC: ${vol_profile['point_of_control']:,.4f}
- Resistance: {resistances}
- Support: {supports}

PATTERNS: {pattern_str}

---

Respond EXACTLY in this format:

## Signals
[2-3 lines: most significant indicator confluences and what they suggest collectively]

## Scénario Principal (probabilité estimée: XX%)
[What is most likely to happen in the next 3-10 candles]
- Cible: $XXXX
- Stop-loss: $XXXX
- R/R: X:1

## Scénario Alternatif (probabilité estimée: XX%)
[What could happen instead, and what would trigger it]
- Cible: $XXXX
- Stop-loss: $XXXX

## Verdict
[One short sentence: bullish/bearish/neutre + recommended stance]
"""
    return prompt


def analyze_market(
    market_data: dict,
    technical_analysis: dict,
    stream: bool = True,
) -> str:
    client = anthropic.Anthropic()
    prompt = build_analysis_prompt(market_data, technical_analysis)

    if stream:
        full_response = ""
        with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream_ctx:
            for event in stream_ctx:
                if event.type == "content_block_delta":
                    if event.delta.type == "text_delta":
                        print(event.delta.text, end="", flush=True)
                        full_response += event.delta.text

        return full_response
    else:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text_blocks = [b.text for b in response.content if b.type == "text"]
        return "\n".join(text_blocks)
