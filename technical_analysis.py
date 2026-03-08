"""
Technical Analysis Engine
Computes indicators and detects chart patterns from OHLCV data.
"""

import pandas as pd
import numpy as np
from typing import Optional


# ---------------------------------------------------------------------------
# Core Indicators
# ---------------------------------------------------------------------------

def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def compute_bollinger_bands(close: pd.Series, period: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    bandwidth = (upper - lower) / sma
    pct_b = (close - lower) / (upper - lower)
    return pd.DataFrame({
        "bb_upper": upper,
        "bb_middle": sma,
        "bb_lower": lower,
        "bb_bandwidth": bandwidth,
        "bb_pct_b": pct_b,
    })


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, min_periods=period).mean()


def compute_stochastic(
    high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": k, "stoch_d": d})


def compute_emas(close: pd.Series, periods: list = [9, 21, 50, 200]) -> pd.DataFrame:
    result = {}
    for p in periods:
        result[f"ema_{p}"] = close.ewm(span=p, adjust=False).mean()
    return pd.DataFrame(result)


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume-Weighted Average Price (resets each day)."""
    typical_price = (df["high"] + df["low"] + df["close"]) / 3
    cumulative_tp_vol = (typical_price * df["volume"]).cumsum()
    cumulative_vol = df["volume"].cumsum()
    return cumulative_tp_vol / cumulative_vol


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On-Balance Volume — momentum indicator using volume."""
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).cumsum()


def compute_volume_profile(df: pd.DataFrame, bins: int = 20) -> dict:
    """Identify high-volume price zones (support/resistance via volume)."""
    price_range = df["close"].max() - df["close"].min()
    bin_size = price_range / bins

    df = df.copy()
    df["price_bin"] = ((df["close"] - df["close"].min()) / bin_size).astype(int).clip(0, bins - 1)
    profile = df.groupby("price_bin")["volume"].sum()

    poc_bin = profile.idxmax()
    poc_price = df["close"].min() + poc_bin * bin_size + bin_size / 2

    high_vol_bins = profile[profile > profile.quantile(0.75)]
    support_resistance = [
        round(df["close"].min() + b * bin_size + bin_size / 2, 4)
        for b in high_vol_bins.index
    ]

    return {
        "point_of_control": round(poc_price, 4),
        "high_volume_zones": sorted(support_resistance),
    }


# ---------------------------------------------------------------------------
# Pattern Detection
# ---------------------------------------------------------------------------

def detect_candlestick_patterns(df: pd.DataFrame) -> list[dict]:
    """Detect key candlestick reversal/continuation patterns on recent candles."""
    patterns = []
    if len(df) < 3:
        return patterns

    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    body = (c - o).abs()
    avg_body = body.rolling(10).mean()

    for i in range(2, len(df)):
        idx = df.index[i]
        oi, hi, li, ci = o.iloc[i], h.iloc[i], l.iloc[i], c.iloc[i]
        op, hp, lp, cp = o.iloc[i - 1], h.iloc[i - 1], l.iloc[i - 1], c.iloc[i - 1]
        bi = body.iloc[i]
        avg = avg_body.iloc[i] if not pd.isna(avg_body.iloc[i]) else bi

        upper_wick = hi - max(oi, ci)
        lower_wick = min(oi, ci) - li

        # Doji
        if bi < 0.1 * (hi - li) and (hi - li) > 0:
            patterns.append({"index": idx, "pattern": "Doji", "signal": "indecision"})

        # Hammer (bullish reversal)
        if (cp < op and ci > oi and lower_wick >= 2 * bi and upper_wick < 0.3 * bi):
            patterns.append({"index": idx, "pattern": "Hammer", "signal": "bullish_reversal"})

        # Shooting Star (bearish reversal)
        if (cp > op and ci < oi and upper_wick >= 2 * bi and lower_wick < 0.3 * bi):
            patterns.append({"index": idx, "pattern": "Shooting Star", "signal": "bearish_reversal"})

        # Engulfing patterns
        if cp < op and ci > oi and ci > op and oi < cp:
            patterns.append({"index": idx, "pattern": "Bullish Engulfing", "signal": "bullish_reversal"})
        if cp > op and ci < oi and ci < op and oi > cp:
            patterns.append({"index": idx, "pattern": "Bearish Engulfing", "signal": "bearish_reversal"})

        # Marubozu (strong momentum)
        if bi > 1.5 * avg and upper_wick < 0.05 * bi and lower_wick < 0.05 * bi:
            signal = "strong_bullish" if ci > oi else "strong_bearish"
            patterns.append({"index": idx, "pattern": "Marubozu", "signal": signal})

    return patterns[-10:]  # Return only the 10 most recent


def detect_trend(close: pd.Series, ema_fast: pd.Series, ema_slow: pd.Series) -> dict:
    """Determine trend direction and strength."""
    price = close.iloc[-1]
    fast = ema_fast.iloc[-1]
    slow = ema_slow.iloc[-1]

    if fast > slow and price > fast:
        direction = "uptrend"
        strength = "strong"
    elif fast > slow and price < fast:
        direction = "uptrend"
        strength = "weak"
    elif fast < slow and price < fast:
        direction = "downtrend"
        strength = "strong"
    elif fast < slow and price > fast:
        direction = "downtrend"
        strength = "weak"
    else:
        direction = "sideways"
        strength = "neutral"

    slope = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10] * 100

    return {"direction": direction, "strength": strength, "slope_10c_pct": round(slope, 2)}


def detect_support_resistance(close: pd.Series, window: int = 10, tolerance: float = 0.005) -> dict:
    """Detect key support and resistance levels from local extrema."""
    highs, lows = [], []

    for i in range(window, len(close) - window):
        slice_ = close.iloc[i - window: i + window + 1]
        if close.iloc[i] == slice_.max():
            highs.append(close.iloc[i])
        if close.iloc[i] == slice_.min():
            lows.append(close.iloc[i])

    def cluster(levels):
        if not levels:
            return []
        levels = sorted(levels, reverse=True)
        clusters = [[levels[0]]]
        for lvl in levels[1:]:
            if abs(lvl - clusters[-1][0]) / clusters[-1][0] < tolerance:
                clusters[-1].append(lvl)
            else:
                clusters.append([lvl])
        return [round(sum(c) / len(c), 4) for c in clusters]

    return {
        "resistance_levels": cluster(highs)[:5],
        "support_levels": cluster(lows)[:5],
    }


# ---------------------------------------------------------------------------
# Full Analysis Pipeline
# ---------------------------------------------------------------------------

def full_analysis(df: pd.DataFrame) -> dict:
    """Run all indicators and pattern detections on OHLCV data."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    rsi = compute_rsi(close)
    macd_df = compute_macd(close)
    bb_df = compute_bollinger_bands(close)
    atr = compute_atr(high, low, close)
    stoch_df = compute_stochastic(high, low, close)
    emas_df = compute_emas(close)
    vwap = compute_vwap(df)
    obv = compute_obv(close, volume)

    latest = {
        "price": round(close.iloc[-1], 6),
        "rsi_14": round(rsi.iloc[-1], 2),
        "macd": round(macd_df["macd"].iloc[-1], 6),
        "macd_signal": round(macd_df["signal"].iloc[-1], 6),
        "macd_histogram": round(macd_df["histogram"].iloc[-1], 6),
        "bb_upper": round(bb_df["bb_upper"].iloc[-1], 6),
        "bb_middle": round(bb_df["bb_middle"].iloc[-1], 6),
        "bb_lower": round(bb_df["bb_lower"].iloc[-1], 6),
        "bb_pct_b": round(bb_df["bb_pct_b"].iloc[-1], 4),
        "bb_bandwidth": round(bb_df["bb_bandwidth"].iloc[-1], 4),
        "atr_14": round(atr.iloc[-1], 6),
        "stoch_k": round(stoch_df["stoch_k"].iloc[-1], 2),
        "stoch_d": round(stoch_df["stoch_d"].iloc[-1], 2),
        "ema_9": round(emas_df["ema_9"].iloc[-1], 6),
        "ema_21": round(emas_df["ema_21"].iloc[-1], 6),
        "ema_50": round(emas_df["ema_50"].iloc[-1], 6),
        "ema_200": round(emas_df["ema_200"].iloc[-1], 6),
        "vwap": round(vwap.iloc[-1], 6),
        "obv_trend": "rising" if obv.iloc[-1] > obv.iloc[-5] else "falling",
        "volume_vs_avg": round(volume.iloc[-1] / volume.rolling(20).mean().iloc[-1], 2),
    }

    trend = detect_trend(close, emas_df["ema_21"], emas_df["ema_50"])
    sr_levels = detect_support_resistance(close)
    patterns = detect_candlestick_patterns(df)
    vol_profile = compute_volume_profile(df)

    # Recent candle history (last 5 candles summary)
    recent_candles = []
    for i in range(-5, 0):
        row = df.iloc[i]
        recent_candles.append({
            "time": str(df.index[i]),
            "open": round(row["open"], 6),
            "high": round(row["high"], 6),
            "low": round(row["low"], 6),
            "close": round(row["close"], 6),
            "volume": round(row["volume"], 2),
            "bullish": row["close"] > row["open"],
        })

    return {
        "latest_indicators": latest,
        "trend": trend,
        "support_resistance": sr_levels,
        "candlestick_patterns": patterns,
        "volume_profile": vol_profile,
        "recent_candles": recent_candles,
    }
