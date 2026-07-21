from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .market_context import add_market_context
from .models import IndicatorSnapshot


def _safe_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False, min_periods=period).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def true_range(df: pd.DataFrame) -> pd.Series:
    previous_close = df["close"].shift()
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast_line = ema(series, fast)
    slow_line = ema(series, slow)
    macd_line = fast_line - slow_line
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return macd_line, signal_line, macd_line - signal_line


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr_smoothed = true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr_smoothed.replace(0, np.nan)
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False, min_periods=period).mean() / tr_smoothed.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_line = dx.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return adx_line, plus_di, minus_di


def stochastic(df: pd.DataFrame, period: int = 14, smooth: int = 3) -> tuple[pd.Series, pd.Series]:
    lowest = df["low"].rolling(period).min()
    highest = df["high"].rolling(period).max()
    k = 100 * (df["close"] - lowest) / (highest - lowest).replace(0, np.nan)
    d = k.rolling(smooth).mean()
    return k, d


def bollinger(series: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = series.rolling(period).mean()
    std = series.rolling(period).std(ddof=0)
    return mid + std_mult * std, mid, mid - std_mult * std


def session_vwap(df: pd.DataFrame) -> pd.Series:
    typical = (df["high"] + df["low"] + df["close"]) / 3
    volume = df["tick_volume"].replace(0, np.nan).fillna(1.0)
    dates = pd.to_datetime(df["time"], utc=True).dt.date
    pv = typical * volume
    return pv.groupby(dates).cumsum() / volume.groupby(dates).cumsum()


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0)
    return (direction * df["tick_volume"].fillna(0)).cumsum()


def choppiness(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tr_sum = true_range(df).rolling(period).sum()
    highest = df["high"].rolling(period).max()
    lowest = df["low"].rolling(period).min()
    denominator = (highest - lowest).replace(0, np.nan)
    return 100 * np.log10(tr_sum / denominator) / np.log10(period)


def supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> tuple[pd.Series, pd.Series]:
    atr_line = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    basic_upper = hl2 + multiplier * atr_line
    basic_lower = hl2 - multiplier * atr_line
    final_upper = pd.Series(np.nan, index=df.index, dtype=float)
    final_lower = pd.Series(np.nan, index=df.index, dtype=float)
    direction = pd.Series(0, index=df.index, dtype=int)
    line = pd.Series(np.nan, index=df.index, dtype=float)

    valid_positions = np.flatnonzero(atr_line.notna().to_numpy())
    if len(valid_positions) == 0:
        return line, direction
    first = int(valid_positions[0])
    final_upper.iloc[first] = basic_upper.iloc[first]
    final_lower.iloc[first] = basic_lower.iloc[first]
    direction.iloc[first] = 1
    line.iloc[first] = final_lower.iloc[first]

    for i in range(first + 1, len(df)):
        if pd.isna(atr_line.iloc[i]):
            continue
        prior_upper = final_upper.iloc[i - 1]
        prior_lower = final_lower.iloc[i - 1]
        if pd.isna(prior_upper):
            prior_upper = basic_upper.iloc[i - 1]
        if pd.isna(prior_lower):
            prior_lower = basic_lower.iloc[i - 1]
        final_upper.iloc[i] = basic_upper.iloc[i] if basic_upper.iloc[i] < prior_upper or df["close"].iloc[i - 1] > prior_upper else prior_upper
        final_lower.iloc[i] = basic_lower.iloc[i] if basic_lower.iloc[i] > prior_lower or df["close"].iloc[i - 1] < prior_lower else prior_lower

        previous_direction = direction.iloc[i - 1] or 1
        if previous_direction == -1 and df["close"].iloc[i] > final_upper.iloc[i]:
            direction.iloc[i] = 1
        elif previous_direction == 1 and df["close"].iloc[i] < final_lower.iloc[i]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = previous_direction
        line.iloc[i] = final_lower.iloc[i] if direction.iloc[i] == 1 else final_upper.iloc[i]

    return line, direction


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "tick_volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy().sort_values("time").reset_index(drop=True)
    out["ema9"] = ema(out["close"], 9)
    out["ema20"] = ema(out["close"], 20)
    out["ema50"] = ema(out["close"], 50)
    out["ema200"] = ema(out["close"], 200)
    out["rsi14"] = rsi(out["close"], 14)
    out["atr14"] = atr(out, 14)
    out["atr_pct"] = out["atr14"] / out["close"].replace(0, np.nan) * 100
    out["macd"], out["macd_signal"], out["macd_hist"] = macd(out["close"])
    out["macd_hist_slope"] = out["macd_hist"].diff(3)
    out["adx14"], out["plus_di"], out["minus_di"] = adx(out, 14)
    out["stoch_k"], out["stoch_d"] = stochastic(out)
    out["bb_upper"], out["bb_mid"], out["bb_lower"] = bollinger(out["close"])
    out["bb_width_pct"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_mid"].replace(0, np.nan) * 100
    out["vwap"] = session_vwap(out)
    out["obv"] = obv(out)
    out["obv_slope"] = out["obv"].diff(5)
    volume_mean_20 = out["tick_volume"].rolling(20).mean()
    volume_mean_50 = out["tick_volume"].rolling(50).mean()
    volume_std_50 = out["tick_volume"].rolling(50).std(ddof=0).replace(0, np.nan)
    out["volume_ratio"] = out["tick_volume"] / volume_mean_20.replace(0, np.nan)
    out["volume_zscore"] = (out["tick_volume"] - volume_mean_50) / volume_std_50
    candle_range = (out["high"] - out["low"]).replace(0, np.nan)
    out["volume_delta_proxy"] = out["tick_volume"] * ((out["close"] - out["open"]) / candle_range).clip(-1, 1).fillna(0)
    out["choppiness14"] = choppiness(out, 14)
    out["supertrend"], out["supertrend_direction"] = supertrend(out, 10, 3.0)
    out["donchian_high"] = out["high"].shift(1).rolling(20).max()
    out["donchian_low"] = out["low"].shift(1).rolling(20).min()
    out["breakout_up"] = out["close"] > out["donchian_high"]
    out["breakout_down"] = out["close"] < out["donchian_low"]
    # Faster structure and impulse fields allow the state machine to recognise
    # a real move before slow H1 EMA alignment fully catches up.
    out["structure_high_8"] = out["high"].shift(1).rolling(8).max()
    out["structure_low_8"] = out["low"].shift(1).rolling(8).min()
    out["structure_break_up"] = out["close"] > out["structure_high_8"]
    out["structure_break_down"] = out["close"] < out["structure_low_8"]
    out["impulse_1_atr"] = (out["close"] - out["close"].shift(1)) / out["atr14"].replace(0, np.nan)
    out["impulse_3_atr"] = (out["close"] - out["close"].shift(3)) / out["atr14"].replace(0, np.nan)
    candle_range_for_location = (out["high"] - out["low"]).replace(0, np.nan)
    out["close_location"] = ((out["close"] - out["low"]) / candle_range_for_location).clip(0, 1)
    out["ema20_slope_atr"] = (out["ema20"] - out["ema20"].shift(5)) / out["atr14"].replace(0, np.nan)
    out["ema50_slope_atr"] = (out["ema50"] - out["ema50"].shift(8)) / out["atr14"].replace(0, np.nan)
    width_floor = out["bb_width_pct"].rolling(100, min_periods=40).quantile(0.25)
    out["compression"] = (out["bb_width_pct"] <= width_floor) & (out["adx14"] < 19)
    out, _ = add_market_context(out)
    return out


def summarize_indicators(df: pd.DataFrame, timeframe: str) -> IndicatorSnapshot:
    row = df.iloc[-1]
    close = float(row["close"])
    atr_v = _safe_float(row.get("atr14")) or max(close * 0.001, 1e-6)
    ema9_v, ema20_v, ema50_v, ema200_v = (_safe_float(row.get(key)) for key in ("ema9", "ema20", "ema50", "ema200"))
    macd_hist_v = _safe_float(row.get("macd_hist"))
    macd_slope_v = _safe_float(row.get("macd_hist_slope"))
    plus_di_v = _safe_float(row.get("plus_di"))
    minus_di_v = _safe_float(row.get("minus_di"))
    rsi_v = _safe_float(row.get("rsi14"))
    adx_v = _safe_float(row.get("adx14"))
    ema20_slope = _safe_float(row.get("ema20_slope_atr"))
    ema50_slope = _safe_float(row.get("ema50_slope_atr"))
    avwap_v = _safe_float(row.get("avwap_active"))
    avwap_slope = _safe_float(row.get("avwap_slope_atr"))
    structure_bias = str(row.get("structure_bias", "neutral"))
    structure_state = str(row.get("structure_state", "RANGE"))
    profile_state = str(row.get("profile_state", "UNAVAILABLE"))
    profile_acceptance = str(row.get("profile_acceptance", "neutral"))
    super_dir_raw = int(row.get("supertrend_direction", 0)) if not pd.isna(row.get("supertrend_direction", np.nan)) else 0

    # Market structure and anchored VWAP are the primary trend layer. EMA is
    # retained only as a lower-weight secondary confirmation/tie-breaker.
    trend_points = 0.0
    if structure_bias == "bullish":
        trend_points += 34
    elif structure_bias == "bearish":
        trend_points -= 34
    if structure_state in {"BOS_UP", "CHOCH_UP"}:
        trend_points += 18
    elif structure_state in {"BOS_DOWN", "CHOCH_DOWN"}:
        trend_points -= 18

    if avwap_v is not None:
        distance_atr = (close - avwap_v) / max(atr_v, 1e-9)
        trend_points += max(-24, min(24, distance_atr * 15))
    if avwap_slope is not None:
        trend_points += max(-14, min(14, avwap_slope * 12))

    if profile_acceptance == "bullish":
        trend_points += 12
    elif profile_acceptance == "bearish":
        trend_points -= 12
    elif profile_state == "ABOVE_VALUE":
        trend_points += 7
    elif profile_state == "BELOW_VALUE":
        trend_points -= 7

    # EMA remains visible and useful, but cannot override structure + AVWAP.
    if None not in (ema20_v, ema50_v, ema200_v):
        if close > ema20_v > ema50_v > ema200_v:
            trend_points += 10
        elif close < ema20_v < ema50_v < ema200_v:
            trend_points -= 10
    if super_dir_raw > 0:
        trend_points += 7
    elif super_dir_raw < 0:
        trend_points -= 7

    trend = "bullish" if trend_points >= 18 else "bearish" if trend_points <= -18 else "neutral"

    momentum = "neutral"
    momentum_points = 0.0
    if macd_hist_v is not None:
        momentum_points += 16 if macd_hist_v > 0 else -16
    if macd_slope_v is not None:
        momentum_points += 8 if macd_slope_v > 0 else -8
    if plus_di_v is not None and minus_di_v is not None:
        momentum_points += 14 if plus_di_v > minus_di_v else -14
    if rsi_v is not None:
        momentum_points += max(-14, min(14, (rsi_v - 50) * 0.7))
    if momentum_points >= 10:
        momentum = "bullish"
    elif momentum_points <= -10:
        momentum = "bearish"

    direction_score = trend_points * 0.66 + momentum_points * 0.34
    if adx_v is not None:
        direction_score *= min(1.16, max(0.76, adx_v / 24))
    direction_score = max(-100, min(100, direction_score))

    timestamp = pd.to_datetime(row["time"], utc=True).isoformat()
    return IndicatorSnapshot(
        timeframe=timeframe,
        timestamp=timestamp,
        close=close,
        ema9=ema9_v,
        ema20=ema20_v,
        ema50=ema50_v,
        ema200=ema200_v,
        ema20_slope_atr=ema20_slope,
        ema50_slope_atr=ema50_slope,
        rsi14=rsi_v,
        atr14=_safe_float(row.get("atr14")),
        atr_pct=_safe_float(row.get("atr_pct")),
        macd=_safe_float(row.get("macd")),
        macd_signal=_safe_float(row.get("macd_signal")),
        macd_hist=macd_hist_v,
        macd_hist_slope=macd_slope_v,
        adx14=adx_v,
        plus_di=plus_di_v,
        minus_di=minus_di_v,
        stoch_k=_safe_float(row.get("stoch_k")),
        stoch_d=_safe_float(row.get("stoch_d")),
        bb_upper=_safe_float(row.get("bb_upper")),
        bb_mid=_safe_float(row.get("bb_mid")),
        bb_lower=_safe_float(row.get("bb_lower")),
        bb_width_pct=_safe_float(row.get("bb_width_pct")),
        vwap=_safe_float(row.get("vwap")),
        avwap_active=avwap_v,
        avwap_swing_low=_safe_float(row.get("avwap_swing_low")),
        avwap_swing_high=_safe_float(row.get("avwap_swing_high")),
        avwap_high_volume=_safe_float(row.get("avwap_high_volume")),
        avwap_slope_atr=avwap_slope,
        avwap_anchor=str(row.get("avwap_anchor", "highest_volume")),
        profile_poc=_safe_float(row.get("profile_poc")),
        profile_vah=_safe_float(row.get("profile_vah")),
        profile_val=_safe_float(row.get("profile_val")),
        profile_state=profile_state,
        profile_acceptance=profile_acceptance if profile_acceptance in {"bullish", "bearish", "neutral"} else "neutral",
        profile_hvn_above=_safe_float(row.get("profile_hvn_above")),
        profile_hvn_below=_safe_float(row.get("profile_hvn_below")),
        market_structure=structure_state,
        structure_bias=structure_bias if structure_bias in {"bullish", "bearish", "neutral"} else "neutral",
        last_swing_high=_safe_float(row.get("last_swing_high")),
        last_swing_low=_safe_float(row.get("last_swing_low")),
        obv=_safe_float(row.get("obv")),
        obv_slope=_safe_float(row.get("obv_slope")),
        volume_zscore=_safe_float(row.get("volume_zscore")),
        volume_ratio=_safe_float(row.get("volume_ratio")),
        volume_delta_proxy=_safe_float(row.get("volume_delta_proxy")),
        choppiness14=_safe_float(row.get("choppiness14")),
        supertrend=_safe_float(row.get("supertrend")),
        supertrend_direction="bullish" if super_dir_raw > 0 else "bearish" if super_dir_raw < 0 else "neutral",
        donchian_high=_safe_float(row.get("donchian_high")),
        donchian_low=_safe_float(row.get("donchian_low")),
        breakout_up=bool(row.get("breakout_up", False)),
        breakout_down=bool(row.get("breakout_down", False)),
        compression=bool(row.get("compression", False)),
        trend=trend,
        momentum=momentum,
        directional_score=round(direction_score, 2),
        impulse_1_atr=_safe_float(row.get("impulse_1_atr")),
        impulse_3_atr=_safe_float(row.get("impulse_3_atr")),
        close_location=_safe_float(row.get("close_location")),
        structure_break_up=bool(row.get("structure_break_up", False)),
        structure_break_down=bool(row.get("structure_break_down", False)),
    )
