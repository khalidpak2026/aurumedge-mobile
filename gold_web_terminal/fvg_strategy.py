from __future__ import annotations

from datetime import timedelta
import math
from typing import Iterable

import pandas as pd

from .models import FourHourFVGSignal, IndicatorSnapshot, MacroConfirmation
from .risk_engine import RiskInputs, build_position_risk_plan


# Entry-only specialist model. The strategy may be WATCH/ARMED for display, but
# TRIGGERED is reserved for a fresh, currently executable first touch of the
# entry zone. Historical confirmations never become new alerts.
MAX_GAP_AGE_BARS = 24          # 6 hours on M15
MAX_PARENT_AGE_HOURS = 12
ENTRY_TOLERANCE_ATR = 0.12
APPROACH_DISTANCE_ATR = 0.45
MISSED_DISTANCE_ATR = 0.35


def _finite(value: float | None, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _snapshot(indicators: Iterable[IndicatorSnapshot] | None, timeframe: str) -> IndicatorSnapshot | None:
    if indicators is None:
        return None
    return next((item for item in indicators if item.timeframe == timeframe), None)


def _m15_fvgs(df: pd.DataFrame, side: str, limit: int = 120) -> list[dict]:
    gaps: list[dict] = []
    start = max(2, len(df) - limit)
    for i in range(start, len(df)):
        left = df.iloc[i - 2]
        current = df.iloc[i]
        if side == "BUY" and float(current["low"]) > float(left["high"]):
            gaps.append(
                {
                    "index": i,
                    "time": pd.to_datetime(current["time"], utc=True),
                    "low": float(left["high"]),
                    "high": float(current["low"]),
                }
            )
        elif side == "SELL" and float(current["high"]) < float(left["low"]):
            gaps.append(
                {
                    "index": i,
                    "time": pd.to_datetime(current["time"], utc=True),
                    "low": float(current["high"]),
                    "high": float(left["low"]),
                }
            )
    return gaps


def _invalidated(df: pd.DataFrame, gap: dict, side: str) -> bool:
    after = df.iloc[gap["index"] + 1 :]
    if after.empty:
        return False
    if side == "BUY":
        return bool((after["close"] < float(gap["low"])).any())
    return bool((after["close"] > float(gap["high"])).any())


def _touch_indices(df: pd.DataFrame, start_index: int, low: float, high: float) -> list[int]:
    after = df.iloc[start_index + 1 :]
    if after.empty:
        return []
    mask = (after["high"] >= low) & (after["low"] <= high)
    return [int(i) for i in after.index[mask].tolist()]


def _distance_to_zone(price: float, low: float, high: float) -> float:
    if low <= price <= high:
        return 0.0
    return low - price if price < low else price - high


def detect_four_hour_fvg_signal(
    frames: dict[str, pd.DataFrame],
    indicators: list[IndicatorSnapshot] | None = None,
    macro: MacroConfirmation | None = None,
    primary_state: str | None = None,
    risk_inputs: RiskInputs | None = None,
    digits: int = 2,
) -> FourHourFVGSignal:
    """Detect an actionable 4H displacement + M15 FVG continuation entry.

    Important execution rule:
    - WATCH/ARMED are informational only.
    - TRIGGERED means the *current live price is inside or immediately beside*
      a fresh first-touch entry zone. It is never set from an old candle.
    - If price has already travelled away toward TP1, the setup is EXPIRED and
      no notification is generated.
    """
    if "H4" not in frames or "M15" not in frames or len(frames["H4"]) < 6 or len(frames["M15"]) < 30:
        return FourHourFVGSignal(warnings=["Insufficient H4/M15 history for the 4H-FVG strategy."])

    h4 = frames["H4"].copy().reset_index(drop=True)
    m15 = frames["M15"].copy().reset_index(drop=True)
    parent = h4.iloc[-2]  # most recently completed H4 candle
    parent_time = pd.to_datetime(parent["time"], utc=True)
    latest_time = pd.to_datetime(m15.iloc[-1]["time"], utc=True)
    parent_open = float(parent["open"])
    parent_high = float(parent["high"])
    parent_low = float(parent["low"])
    parent_close = float(parent["close"])
    parent_range = max(parent_high - parent_low, 1e-9)
    parent_body = abs(parent_close - parent_open)
    body_ratio = parent_body / parent_range
    h4_atr = _finite(parent.get("atr14"), float((h4["high"] - h4["low"]).tail(14).mean()))
    body_atr = parent_body / max(h4_atr, 1e-9)
    close_location = (parent_close - parent_low) / parent_range
    side = "BUY" if parent_close > parent_open else "SELL" if parent_close < parent_open else "NONE"
    valid_until = parent_time + timedelta(hours=MAX_PARENT_AGE_HOURS)

    common = dict(
        parent_candle_time=parent_time.isoformat(),
        parent_open=_round(parent_open, digits),
        parent_high=_round(parent_high, digits),
        parent_low=_round(parent_low, digits),
        parent_close=_round(parent_close, digits),
        parent_body_atr=round(body_atr, 2),
        signal_time=latest_time.isoformat(),
        valid_until=valid_until.isoformat(),
        macro_gate=macro.gate if macro else "UNAVAILABLE",
    )

    if latest_time > valid_until:
        return FourHourFVGSignal(
            state="EXPIRED", side=side if side in {"BUY", "SELL"} else "NONE",
            warnings=["The parent H4 setup is older than the permitted entry window."], **common
        )
    if side == "NONE" or body_ratio < 0.50 or body_atr < 0.55:
        return FourHourFVGSignal(
            state="NONE", side="NONE",
            warnings=["The last completed H4 candle is not a strong displacement candle."], **common
        )
    if side == "BUY" and close_location < 0.62:
        return FourHourFVGSignal(state="NONE", side="NONE", warnings=["Bullish H4 candle did not close strongly enough near its high."], **common)
    if side == "SELL" and close_location > 0.38:
        return FourHourFVGSignal(state="NONE", side="NONE", warnings=["Bearish H4 candle did not close strongly enough near its low."], **common)

    midpoint = (parent_high + parent_low) / 2.0
    parent_start = parent_time - timedelta(hours=1)
    candidates: list[dict] = []
    for gap in _m15_fvgs(m15, side):
        gap_mid = (float(gap["low"]) + float(gap["high"])) / 2.0
        age_bars = len(m15) - 1 - int(gap["index"])
        if age_bars > MAX_GAP_AGE_BARS:
            continue
        if gap["time"] < parent_start or gap["time"] > parent_time + timedelta(hours=8):
            continue
        gap_atr = _finite(m15.iloc[gap["index"]].get("atr14"), 0.0)
        min_gap_width = max(gap_atr * 0.035, 0.08)
        if float(gap["high"]) - float(gap["low"]) < min_gap_width:
            continue
        if side == "BUY" and not (parent_low <= gap_mid <= midpoint):
            continue
        if side == "SELL" and not (midpoint <= gap_mid <= parent_high):
            continue
        gap["invalidated"] = _invalidated(m15, gap, side)
        if gap["invalidated"]:
            continue
        candidates.append(gap)

    if not candidates:
        return FourHourFVGSignal(
            side=side,
            state="WATCH",
            signal_id=f"H4FVG|{side}|{parent_time.isoformat()}|WAIT",
            confidence=min(62, int(round(46 + body_ratio * 10 + min(body_atr, 1.5) * 4))),
            rationale=[
                f"The completed H4 candle is a {side.lower()} displacement candle ({body_atr:.2f} ATR body).",
                "No fresh, unused M15 FVG is currently available in the correct H4 premium/discount half.",
            ],
            **common,
        )

    # Always prefer the newest unused FVG. Old historical touches are not valid
    # new entries and must never outrank a fresh setup.
    gap = max(candidates, key=lambda item: item["time"].timestamp())
    gap_low = float(gap["low"])
    gap_high = float(gap["high"])
    gap_mid = (gap_low + gap_high) / 2.0
    last = m15.iloc[-1]
    current_price = float(last["close"])
    m15_atr = _finite(last.get("atr14"), float((m15["high"] - m15["low"]).tail(14).mean()))
    tolerance = max(ENTRY_TOLERANCE_ATR * m15_atr, 0.25)
    buffer = max(0.12 * m15_atr, 0.25)
    recent = m15.tail(16)

    if side == "BUY":
        entry_low = gap_mid
        entry_high = gap_high + 0.05 * m15_atr
        nearby_low = float(recent["low"].min())
        structural_low = nearby_low if nearby_low < gap_low and gap_low - nearby_low <= 0.55 * m15_atr else gap_low
        stop = structural_low - buffer
        entry_mid = (entry_low + entry_high) / 2.0
        risk = max(entry_mid - stop, 0.01)
        tp1 = parent_close if parent_close > entry_mid else entry_mid + 0.65 * risk
        tp2 = parent_high if parent_high > tp1 else entry_mid + 1.05 * risk
        tp3 = max(parent_high + 0.35 * parent_range, entry_mid + 1.40 * risk)
    else:
        entry_low = gap_low - 0.05 * m15_atr
        entry_high = gap_mid
        nearby_high = float(recent["high"].max())
        structural_high = nearby_high if nearby_high > gap_high and nearby_high - gap_high <= 0.55 * m15_atr else gap_high
        stop = structural_high + buffer
        entry_mid = (entry_low + entry_high) / 2.0
        risk = max(stop - entry_mid, 0.01)
        tp1 = parent_close if parent_close < entry_mid else entry_mid - 0.65 * risk
        tp2 = parent_low if parent_low < tp1 else entry_mid - 1.05 * risk
        tp3 = min(parent_low - 0.35 * parent_range, entry_mid - 1.40 * risk)

    touches = _touch_indices(m15, int(gap["index"]), entry_low, entry_high)
    first_touch = touches[0] if touches else None
    latest_index = len(m15) - 1
    touch_age = latest_index - first_touch if first_touch is not None else None
    price_near_zone = entry_low - tolerance <= current_price <= entry_high + tolerance
    current_bar_touches = float(last["high"]) >= entry_low and float(last["low"]) <= entry_high
    distance = _distance_to_zone(current_price, entry_low, entry_high)

    # A first touch is actionable only now (or at most one M15 bar ago while
    # price is still beside the zone). Any older touch means the FVG is consumed.
    first_touch_is_fresh = first_touch is not None and touch_age is not None and touch_age <= 1
    moved_to_target = (
        side == "BUY" and current_price >= tp1
    ) or (
        side == "SELL" and current_price <= tp1
    )
    moved_away_after_touch = False
    if first_touch is not None and touch_age is not None and touch_age > 1:
        if side == "BUY":
            moved_away_after_touch = current_price > entry_high + MISSED_DISTANCE_ATR * m15_atr
        else:
            moved_away_after_touch = current_price < entry_low - MISSED_DISTANCE_ATR * m15_atr

    if moved_to_target or moved_away_after_touch or (first_touch is not None and touch_age is not None and touch_age > 2):
        state = "EXPIRED"
    elif first_touch_is_fresh and price_near_zone:
        state = "TRIGGERED"  # ENTRY LIVE NOW
    elif first_touch is None and distance <= APPROACH_DISTANCE_ATR * m15_atr:
        state = "ARMED"      # approaching entry; no alert
    elif current_bar_touches and not price_near_zone:
        state = "EXPIRED"    # the bar passed through and price already left
    else:
        state = "WATCH"

    h1 = _snapshot(indicators, "H1")
    m15_snapshot = _snapshot(indicators, "M15")
    score = 48
    reasons = [
        f"Completed H4 {side.lower()} displacement: body {body_atr:.2f} ATR and {body_ratio:.0%} of candle range.",
        f"Fresh M15 {'bullish' if side == 'BUY' else 'bearish'} FVG is in the H4 {'discount' if side == 'BUY' else 'premium'} half.",
    ]
    if body_atr >= 0.9:
        score += 6
    h1_aligned = False
    if h1 is not None:
        h1_aligned = (side == "BUY" and h1.trend == "bullish") or (side == "SELL" and h1.trend == "bearish")
        h1_opposite = (side == "BUY" and h1.trend == "bearish") or (side == "SELL" and h1.trend == "bullish")
        if h1_aligned:
            score += 7
            reasons.append("H1 trend agrees with the specialist direction.")
        elif h1_opposite:
            score -= 12
            reasons.append("H1 trend opposes this FVG direction.")
    if m15_snapshot is not None:
        m15_aligned = (side == "BUY" and m15_snapshot.momentum == "bullish") or (side == "SELL" and m15_snapshot.momentum == "bearish")
        if m15_aligned:
            score += 5
            reasons.append("M15 momentum agrees with the setup.")
    if macro is not None:
        if macro.gate == "CONFIRM":
            score += 7
            reasons.append("DXY/US10Y/gold-flow gate confirms the direction.")
        elif macro.gate == "CONFLICT":
            score -= 16
            reasons.append("Macro confirmation conflicts with the setup.")
    aligns = primary_state == side
    primary_opposes = primary_state in {"BUY", "SELL"} and primary_state != side
    if aligns:
        score += 9
        reasons.append("The regular AurumEdge engine agrees with the specialist setup.")
    elif primary_opposes:
        score -= 14
        reasons.append("The regular AurumEdge engine points in the opposite direction.")

    if state == "TRIGGERED":
        score += 8
        reasons.append(f"ENTRY LIVE NOW: current price {current_price:.2f} is inside/next to the fresh first-touch zone.")
    elif state == "ARMED":
        reasons.append("Price is approaching the fresh entry zone; no notification is sent yet.")
    elif state == "EXPIRED":
        score = min(score, 55)
        reasons.append("Entry was missed/consumed or price already travelled toward the target. No late alert is allowed.")

    confidence = max(40, min(84, int(round(score))))
    # Accuracy gate: specialist-only entries without regular or macro agreement
    # are visible, but they cannot masquerade as high-confidence alerts.
    if not aligns and (macro is None or macro.gate != "CONFIRM"):
        confidence = min(confidence, 64)
    if primary_opposes or (macro is not None and macro.gate == "CONFLICT") or not h1_aligned:
        confidence = min(confidence, 59)

    risk_plan = build_position_risk_plan(entry_mid, stop, risk_inputs or RiskInputs())
    warnings = [
        "Telegram is sent only for TRIGGERED = ENTRY LIVE NOW.",
        "Do not enter if the live broker price is outside the displayed entry zone.",
    ]
    if state == "EXPIRED":
        warnings.append("This setup is historical or missed; it is not executable.")
    if risk_plan.status != "OK":
        warnings.append(risk_plan.message)

    stable_id = f"H4FVG|{side}|{parent_time.isoformat()}|{gap['time'].isoformat()}"
    return FourHourFVGSignal(
        side=side,
        state=state,
        signal_id=stable_id,
        fvg_created_time=gap["time"].isoformat(),
        fvg_low=_round(gap_low, digits),
        fvg_high=_round(gap_high, digits),
        fvg_mid=_round(gap_mid, digits),
        entry_low=_round(entry_low, digits),
        entry_high=_round(entry_high, digits),
        stop_loss=_round(stop, digits),
        take_profit_1=_round(tp1, digits),
        take_profit_2=_round(tp2, digits),
        take_profit_3=_round(tp3, digits),
        confidence=confidence,
        aligns_with_primary=aligns,
        rationale=reasons,
        warnings=warnings,
        risk_plan=risk_plan,
        **common,
    )
