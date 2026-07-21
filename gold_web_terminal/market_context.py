from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class MarketContext:
    structure_bias: str = "neutral"
    structure_state: str = "RANGE"
    last_swing_high: float | None = None
    last_swing_low: float | None = None
    avwap_anchor: str = "highest_volume"
    profile_poc: float | None = None
    profile_vah: float | None = None
    profile_val: float | None = None
    profile_state: str = "UNAVAILABLE"
    profile_acceptance: str = "neutral"
    profile_hvn_above: float | None = None
    profile_hvn_below: float | None = None


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def pivot_indices(series: pd.Series, left: int = 3, right: int = 3, mode: str = "high") -> list[int]:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    pivots: list[int] = []
    for index in range(left, len(values) - right):
        window = values[index - left : index + right + 1]
        current = values[index]
        if not math.isfinite(current):
            continue
        if mode == "high" and current == np.nanmax(window) and np.sum(window == current) == 1:
            pivots.append(index)
        elif mode == "low" and current == np.nanmin(window) and np.sum(window == current) == 1:
            pivots.append(index)
    return pivots


def anchored_vwap(df: pd.DataFrame, anchor_index: int) -> pd.Series:
    """Return HLC3 VWAP beginning at a deterministic anchor bar.

    XAU/USD CFD/forex feeds normally provide tick volume rather than centralized
    exchange volume. The same proxy is used consistently for AVWAP and the
    volume profile, and is clearly labelled as activity/tick-volume context.
    """
    result = pd.Series(np.nan, index=df.index, dtype=float)
    if df.empty:
        return result
    anchor_index = max(0, min(int(anchor_index), len(df) - 1))
    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    volume = pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    volume = volume.where(volume > 0, 1.0)
    segment_pv = (typical.iloc[anchor_index:] * volume.iloc[anchor_index:]).cumsum()
    segment_volume = volume.iloc[anchor_index:].cumsum().replace(0.0, np.nan)
    result.iloc[anchor_index:] = segment_pv / segment_volume
    return result


def _structure(df: pd.DataFrame, lookback: int = 220) -> tuple[dict[str, object], int, int, int]:
    recent = df.tail(lookback).copy()
    offset = len(df) - len(recent)
    highs = pivot_indices(recent["high"], mode="high")
    lows = pivot_indices(recent["low"], mode="low")
    high_idx = (highs[-1] + offset) if highs else max(0, len(df) - 1)
    low_idx = (lows[-1] + offset) if lows else max(0, len(df) - 1)
    high_volume_idx = int(pd.to_numeric(df["tick_volume"], errors="coerce").fillna(0).tail(160).idxmax())

    swing_highs = [(idx + offset, float(recent.iloc[idx]["high"])) for idx in highs[-3:]]
    swing_lows = [(idx + offset, float(recent.iloc[idx]["low"])) for idx in lows[-3:]]
    last_high = swing_highs[-1][1] if swing_highs else None
    last_low = swing_lows[-1][1] if swing_lows else None
    close = float(df.iloc[-1]["close"])

    higher_high = len(swing_highs) >= 2 and swing_highs[-1][1] > swing_highs[-2][1]
    lower_high = len(swing_highs) >= 2 and swing_highs[-1][1] < swing_highs[-2][1]
    higher_low = len(swing_lows) >= 2 and swing_lows[-1][1] > swing_lows[-2][1]
    lower_low = len(swing_lows) >= 2 and swing_lows[-1][1] < swing_lows[-2][1]

    prior_bias = "neutral"
    if higher_high and higher_low:
        prior_bias = "bullish"
    elif lower_high and lower_low:
        prior_bias = "bearish"

    state = "RANGE"
    bias = prior_bias
    if last_high is not None and close > last_high:
        state = "CHOCH_UP" if prior_bias == "bearish" else "BOS_UP"
        bias = "bullish"
    elif last_low is not None and close < last_low:
        state = "CHOCH_DOWN" if prior_bias == "bullish" else "BOS_DOWN"
        bias = "bearish"
    elif prior_bias == "bullish":
        state = "HH_HL"
    elif prior_bias == "bearish":
        state = "LH_LL"
    elif higher_low and not lower_high:
        state = "BUILDING_HIGHER_LOW"
        bias = "bullish"
    elif lower_high and not higher_low:
        state = "BUILDING_LOWER_HIGH"
        bias = "bearish"

    return {
        "structure_bias": bias,
        "structure_state": state,
        "last_swing_high": last_high,
        "last_swing_low": last_low,
    }, low_idx, high_idx, high_volume_idx


def volume_profile(
    df: pd.DataFrame,
    lookback: int = 180,
    bins: int = 48,
    value_area: float = 0.70,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    """Approximate volume-at-price from OHLC bars and tick/activity volume.

    Each bar's volume is distributed uniformly across the price bins touched by
    its high-low range. This is more informative than assigning the entire bar
    to HLC3, while remaining deterministic with ordinary OHLCV data.
    """
    recent = df.tail(max(40, int(lookback))).copy().reset_index(drop=True)
    low = float(pd.to_numeric(recent["low"], errors="coerce").min())
    high = float(pd.to_numeric(recent["high"], errors="coerce").max())
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        empty = np.asarray([], dtype=float)
        return {}, empty, empty

    edges = np.linspace(low, high, max(16, int(bins)) + 1)
    weights = np.zeros(len(edges) - 1, dtype=float)
    volumes = pd.to_numeric(recent["tick_volume"], errors="coerce").fillna(0.0).clip(lower=0.0)
    for (_, row), volume in zip(recent.iterrows(), volumes):
        bar_low = float(row["low"])
        bar_high = float(row["high"])
        if not (math.isfinite(bar_low) and math.isfinite(bar_high)):
            continue
        if bar_high <= bar_low:
            index = int(np.clip(np.searchsorted(edges, float(row["close"])) - 1, 0, len(weights) - 1))
            weights[index] += max(float(volume), 1.0)
            continue
        first = int(np.clip(np.searchsorted(edges, bar_low, side="right") - 1, 0, len(weights) - 1))
        last = int(np.clip(np.searchsorted(edges, bar_high, side="left"), 0, len(weights) - 1))
        count = max(1, last - first + 1)
        weights[first : last + 1] += max(float(volume), 1.0) / count

    total = float(weights.sum())
    if total <= 0:
        empty = np.asarray([], dtype=float)
        return {}, empty, empty
    centers = (edges[:-1] + edges[1:]) / 2.0
    poc_index = int(np.argmax(weights))
    poc = float(centers[poc_index])

    # Trading platforms normally expand value area around POC instead of simply
    # selecting unrelated high-volume rows. This produces a contiguous area.
    target = total * max(0.5, min(0.9, float(value_area)))
    selected = {poc_index}
    running = float(weights[poc_index])
    lower = poc_index - 1
    upper = poc_index + 1
    while running < target and (lower >= 0 or upper < len(weights)):
        lower_weight = float(weights[lower]) if lower >= 0 else -1.0
        upper_weight = float(weights[upper]) if upper < len(weights) else -1.0
        if upper_weight >= lower_weight:
            selected.add(upper)
            running += max(0.0, upper_weight)
            upper += 1
        else:
            selected.add(lower)
            running += max(0.0, lower_weight)
            lower -= 1
    val = float(edges[min(selected)])
    vah = float(edges[max(selected) + 1])

    close = float(recent.iloc[-1]["close"])
    profile_state = "ABOVE_VALUE" if close > vah else "BELOW_VALUE" if close < val else "IN_VALUE"
    last_closes = pd.to_numeric(recent["close"].tail(3), errors="coerce")
    acceptance = "neutral"
    if len(last_closes) >= 2 and bool((last_closes > vah).all()):
        acceptance = "bullish"
    elif len(last_closes) >= 2 and bool((last_closes < val).all()):
        acceptance = "bearish"
    elif profile_state == "IN_VALUE":
        acceptance = "bullish" if close > poc else "bearish" if close < poc else "neutral"

    positive = weights[weights > 0]
    node_threshold = float(np.quantile(positive, 0.72)) if len(positive) else 0.0
    hvn_centers = centers[weights >= node_threshold]
    hvn_above = min((float(x) for x in hvn_centers if x > close), default=None)
    hvn_below = max((float(x) for x in hvn_centers if x < close), default=None)

    return {
        "profile_poc": poc,
        "profile_vah": vah,
        "profile_val": val,
        "profile_state": profile_state,
        "profile_acceptance": acceptance,
        "profile_hvn_above": hvn_above,
        "profile_hvn_below": hvn_below,
    }, centers, weights


def add_market_context(df: pd.DataFrame) -> tuple[pd.DataFrame, MarketContext]:
    out = df.copy()
    structure, low_idx, high_idx, high_volume_idx = _structure(out)
    out["avwap_swing_low"] = anchored_vwap(out, low_idx)
    out["avwap_swing_high"] = anchored_vwap(out, high_idx)
    out["avwap_high_volume"] = anchored_vwap(out, high_volume_idx)

    bias = str(structure["structure_bias"])
    if bias == "bullish":
        active_key = "avwap_swing_low"
        anchor = "swing_low"
    elif bias == "bearish":
        active_key = "avwap_swing_high"
        anchor = "swing_high"
    else:
        active_key = "avwap_high_volume"
        anchor = "highest_volume"
    out["avwap_active"] = out[active_key]
    atr = pd.to_numeric(out.get("atr14"), errors="coerce").replace(0.0, np.nan)
    out["avwap_slope_atr"] = (out["avwap_active"] - out["avwap_active"].shift(5)) / atr

    profile, centers, weights = volume_profile(out)
    for key, value in structure.items():
        out[key] = value
    for key, value in profile.items():
        out[key] = value
    out["avwap_anchor"] = anchor
    context = MarketContext(
        structure_bias=bias,
        structure_state=str(structure["structure_state"]),
        last_swing_high=_finite(structure["last_swing_high"]),
        last_swing_low=_finite(structure["last_swing_low"]),
        avwap_anchor=anchor,
        profile_poc=_finite(profile.get("profile_poc")),
        profile_vah=_finite(profile.get("profile_vah")),
        profile_val=_finite(profile.get("profile_val")),
        profile_state=str(profile.get("profile_state", "UNAVAILABLE")),
        profile_acceptance=str(profile.get("profile_acceptance", "neutral")),
        profile_hvn_above=_finite(profile.get("profile_hvn_above")),
        profile_hvn_below=_finite(profile.get("profile_hvn_below")),
    )
    # Optional arrays are stored as attrs for chart renderers without expanding
    # the public model or repeating data in every row.
    out.attrs["volume_profile_centers"] = centers.tolist()
    out.attrs["volume_profile_weights"] = weights.tolist()
    return out, context
