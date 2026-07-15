from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import LiquiditySnapshot


def _finite(value: float | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _pivot_indices(series: pd.Series, left: int = 3, right: int = 3, mode: str = "high") -> list[int]:
    values = series.to_numpy(dtype=float)
    pivots: list[int] = []
    for index in range(left, len(values) - right):
        window = values[index - left : index + right + 1]
        current = values[index]
        if mode == "high" and current == np.nanmax(window) and np.sum(window == current) == 1:
            pivots.append(index)
        if mode == "low" and current == np.nanmin(window) and np.sum(window == current) == 1:
            pivots.append(index)
    return pivots


def _cluster_levels(levels: list[float], tolerance: float, minimum_touches: int = 2) -> list[dict]:
    if not levels:
        return []
    sorted_levels = sorted(float(level) for level in levels)
    clusters: list[list[float]] = [[sorted_levels[0]]]
    for level in sorted_levels[1:]:
        center = sum(clusters[-1]) / len(clusters[-1])
        if abs(level - center) <= tolerance:
            clusters[-1].append(level)
        else:
            clusters.append([level])
    zones: list[dict] = []
    for cluster in clusters:
        if len(cluster) < minimum_touches:
            continue
        center = sum(cluster) / len(cluster)
        half_width = max(tolerance * 0.55, (max(cluster) - min(cluster)) / 2)
        zones.append(
            {
                "low": round(center - half_width, 5),
                "high": round(center + half_width, 5),
                "center": round(center, 5),
                "touches": len(cluster),
            }
        )
    return zones


def _fair_value_gaps(df: pd.DataFrame, limit: int = 8) -> tuple[list[dict], list[dict]]:
    bullish: list[dict] = []
    bearish: list[dict] = []
    start = max(2, len(df) - 180)
    for i in range(start, len(df)):
        candle = df.iloc[i]
        left = df.iloc[i - 2]
        if float(candle["low"]) > float(left["high"]):
            low, high = float(left["high"]), float(candle["low"])
            filled = bool((df.iloc[i + 1 :]["low"] <= high).any()) if i + 1 < len(df) else False
            bullish.append(
                {
                    "time": pd.to_datetime(candle["time"], utc=True).isoformat(),
                    "low": round(low, 5),
                    "high": round(high, 5),
                    "filled": filled,
                }
            )
        if float(candle["high"]) < float(left["low"]):
            low, high = float(candle["high"]), float(left["low"])
            filled = bool((df.iloc[i + 1 :]["high"] >= low).any()) if i + 1 < len(df) else False
            bearish.append(
                {
                    "time": pd.to_datetime(candle["time"], utc=True).isoformat(),
                    "low": round(low, 5),
                    "high": round(high, 5),
                    "filled": filled,
                }
            )
    return bullish[-limit:], bearish[-limit:]


def _volume_profile(df: pd.DataFrame, bins: int = 48) -> tuple[float | None, float | None, float | None]:
    if df.empty:
        return None, None, None
    prices = ((df["high"] + df["low"] + df["close"]) / 3).to_numpy(dtype=float)
    volume = df["tick_volume"].fillna(0).to_numpy(dtype=float)
    low, high = float(np.nanmin(prices)), float(np.nanmax(prices))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return None, None, None
    edges = np.linspace(low, high, bins + 1)
    weighted, _ = np.histogram(prices, bins=edges, weights=volume)
    centers = (edges[:-1] + edges[1:]) / 2
    if weighted.sum() <= 0:
        return None, None, None
    poc_index = int(np.argmax(weighted))
    poc = float(centers[poc_index])
    order = np.argsort(weighted)[::-1]
    target = weighted.sum() * 0.70
    selected: list[int] = []
    running = 0.0
    for index in order:
        selected.append(int(index))
        running += float(weighted[index])
        if running >= target:
            break
    val = float(edges[min(selected)])
    vah = float(edges[max(selected) + 1])
    return poc, vah, val


def analyze_liquidity(df: pd.DataFrame, timeframe: str) -> LiquiditySnapshot:
    if len(df) < 40:
        return LiquiditySnapshot(timeframe=timeframe)

    recent = df.tail(320).copy().reset_index(drop=True)
    high_indices = _pivot_indices(recent["high"], mode="high")
    low_indices = _pivot_indices(recent["low"], mode="low")
    swing_highs = [float(recent.iloc[i]["high"]) for i in high_indices[-12:]]
    swing_lows = [float(recent.iloc[i]["low"]) for i in low_indices[-12:]]
    atr_series = recent.get("atr14", pd.Series(dtype=float)).dropna()
    atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else float((recent["high"] - recent["low"]).tail(14).mean())
    close = float(recent.iloc[-1]["close"])
    tolerance = max(atr_value * 0.16, close * 0.00018)

    high_zones = _cluster_levels(swing_highs, tolerance, minimum_touches=2)
    low_zones = _cluster_levels(swing_lows, tolerance, minimum_touches=2)
    equal_highs = [zone["center"] for zone in high_zones]
    equal_lows = [zone["center"] for zone in low_zones]

    support_candidates = low_zones + [
        {"low": round(level - tolerance * 0.45, 5), "high": round(level + tolerance * 0.45, 5), "center": round(level, 5), "touches": 1}
        for level in swing_lows[-5:]
    ]
    resistance_candidates = high_zones + [
        {"low": round(level - tolerance * 0.45, 5), "high": round(level + tolerance * 0.45, 5), "center": round(level, 5), "touches": 1}
        for level in swing_highs[-5:]
    ]
    support_zones = sorted((zone for zone in support_candidates if zone["center"] < close), key=lambda z: z["center"], reverse=True)[:5]
    resistance_zones = sorted((zone for zone in resistance_candidates if zone["center"] > close), key=lambda z: z["center"])[:5]

    utc_time = pd.to_datetime(recent["time"], utc=True)
    dates = utc_time.dt.date
    unique_dates = list(dict.fromkeys(dates.tolist()))
    previous_day_high = previous_day_low = current_day_high = current_day_low = None
    if unique_dates:
        current_mask = dates == unique_dates[-1]
        current_day_high = float(recent.loc[current_mask, "high"].max())
        current_day_low = float(recent.loc[current_mask, "low"].min())
    if len(unique_dates) >= 2:
        previous_mask = dates == unique_dates[-2]
        previous_day_high = float(recent.loc[previous_mask, "high"].max())
        previous_day_low = float(recent.loc[previous_mask, "low"].min())

    last_three = recent.tail(3)
    last = recent.iloc[-1]
    sweep_above = None
    sweep_below = None
    trap_type = "none"
    all_resistance_levels = swing_highs[:-1] + equal_highs
    all_support_levels = swing_lows[:-1] + equal_lows
    if all_resistance_levels:
        candidate = min(all_resistance_levels, key=lambda level: abs(level - close))
        if float(last_three["high"].max()) > candidate + tolerance * 0.15 and float(last["close"]) < candidate:
            sweep_above = candidate
            if float(last["close"]) < float(last["open"]):
                trap_type = "bull_trap"
    if all_support_levels:
        candidate = min(all_support_levels, key=lambda level: abs(level - close))
        if float(last_three["low"].min()) < candidate - tolerance * 0.15 and float(last["close"]) > candidate:
            sweep_below = candidate
            if float(last["close"]) > float(last["open"]):
                trap_type = "bear_trap"

    bullish_fvgs, bearish_fvgs = _fair_value_gaps(recent)
    poc, vah, val = _volume_profile(recent.tail(180))

    supports = [float(zone["center"]) for zone in support_zones]
    resistances = [float(zone["center"]) for zone in resistance_zones]
    if previous_day_low is not None and previous_day_low < close:
        supports.append(previous_day_low)
    if previous_day_high is not None and previous_day_high > close:
        resistances.append(previous_day_high)

    return LiquiditySnapshot(
        timeframe=timeframe,
        previous_day_high=_finite(previous_day_high),
        previous_day_low=_finite(previous_day_low),
        current_day_high=_finite(current_day_high),
        current_day_low=_finite(current_day_low),
        recent_swing_highs=[round(level, 5) for level in swing_highs],
        recent_swing_lows=[round(level, 5) for level in swing_lows],
        equal_highs=[round(level, 5) for level in equal_highs],
        equal_lows=[round(level, 5) for level in equal_lows],
        support_zones=support_zones,
        resistance_zones=resistance_zones,
        bullish_fvgs=bullish_fvgs,
        bearish_fvgs=bearish_fvgs,
        sweep_above=_finite(sweep_above),
        sweep_below=_finite(sweep_below),
        trap_type=trap_type,
        nearest_support=max(supports) if supports else None,
        nearest_resistance=min(resistances) if resistances else None,
        point_of_control=_finite(poc),
        value_area_high=_finite(vah),
        value_area_low=_finite(val),
    )
