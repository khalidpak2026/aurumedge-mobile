from __future__ import annotations

from datetime import timedelta
import math
from typing import Iterable

import pandas as pd

from .models import FourHourFVGSignal, IndicatorSnapshot, MacroConfirmation
from .risk_engine import RiskInputs, build_position_risk_plan


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


def _m15_fvgs(df: pd.DataFrame, side: str, limit: int = 140) -> list[dict]:
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


def _gap_status(df: pd.DataFrame, gap: dict, side: str) -> tuple[bool, int | None, bool]:
    """Return invalidated, most recent touch index and confirmation state."""
    after = df.iloc[gap["index"] + 1 :].copy()
    if after.empty:
        return False, None, False
    low = float(gap["low"])
    high = float(gap["high"])
    if side == "BUY":
        invalidated = bool((after["close"] < low).any())
        touches = after.index[(after["low"] <= high) & (after["high"] >= low)].tolist()
        touch_index = int(touches[-1]) if touches else None
        confirmed = False
        if touch_index is not None:
            recent = df.loc[touch_index:].tail(4)
            if len(recent):
                last = recent.iloc[-1]
                prior_high = float(recent.iloc[:-1]["high"].max()) if len(recent) > 1 else high
                confirmed = bool(
                    float(last["close"]) > high
                    and float(last["close"]) > float(last["open"])
                    and float(last["close"]) >= prior_high - max((high - low) * 0.20, 0.05)
                )
    else:
        invalidated = bool((after["close"] > high).any())
        touches = after.index[(after["high"] >= low) & (after["low"] <= high)].tolist()
        touch_index = int(touches[-1]) if touches else None
        confirmed = False
        if touch_index is not None:
            recent = df.loc[touch_index:].tail(4)
            if len(recent):
                last = recent.iloc[-1]
                prior_low = float(recent.iloc[:-1]["low"].min()) if len(recent) > 1 else low
                confirmed = bool(
                    float(last["close"]) < low
                    and float(last["close"]) < float(last["open"])
                    and float(last["close"]) <= prior_low + max((high - low) * 0.20, 0.05)
                )
    return invalidated, touch_index, confirmed


def detect_four_hour_fvg_signal(
    frames: dict[str, pd.DataFrame],
    indicators: list[IndicatorSnapshot] | None = None,
    macro: MacroConfirmation | None = None,
    primary_state: str | None = None,
    risk_inputs: RiskInputs | None = None,
    digits: int = 2,
) -> FourHourFVGSignal:
    """Detect a mirrored 4H impulse + M15 fair-value-gap continuation setup.

    Interpretation of the supplied strategy image:
    1. A completed 4H displacement candle defines the directional range.
    2. A fresh M15 FVG aligned with that candle must sit in the discount half
       for a BUY or the premium half for a SELL.
    3. A touch arms the setup; a close back through the FVG in the parent
       direction triggers it.
    4. Targets are the 4H close/high (BUY) or close/low (SELL), then a measured
       extension.  The mirrored SELL model is included automatically.
    """
    if "H4" not in frames or "M15" not in frames or len(frames["H4"]) < 6 or len(frames["M15"]) < 30:
        return FourHourFVGSignal(warnings=["Insufficient H4/M15 history for the 4H-FVG strategy."])

    h4 = frames["H4"].copy().reset_index(drop=True)
    m15 = frames["M15"].copy().reset_index(drop=True)
    # The newest H4 bar may still be forming. Use the most recently completed bar.
    parent = h4.iloc[-2]
    parent_time = pd.to_datetime(parent["time"], utc=True)
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

    if side == "NONE" or body_ratio < 0.50 or body_atr < 0.55:
        return FourHourFVGSignal(
            state="NONE",
            side="NONE",
            parent_candle_time=parent_time.isoformat(),
            parent_open=_round(parent_open, digits),
            parent_high=_round(parent_high, digits),
            parent_low=_round(parent_low, digits),
            parent_close=_round(parent_close, digits),
            parent_body_atr=round(body_atr, 2),
            warnings=["The last completed H4 candle is not a strong displacement candle."],
        )
    if side == "BUY" and close_location < 0.62:
        return FourHourFVGSignal(state="NONE", side="NONE", warnings=["Bullish H4 candle did not close strongly enough near its high."])
    if side == "SELL" and close_location > 0.38:
        return FourHourFVGSignal(state="NONE", side="NONE", warnings=["Bearish H4 candle did not close strongly enough near its low."])

    gaps = _m15_fvgs(m15, side)
    midpoint = (parent_high + parent_low) / 2.0
    parent_start = parent_time - timedelta(hours=1)
    candidates: list[dict] = []
    for gap in gaps:
        gap_mid = (float(gap["low"]) + float(gap["high"])) / 2.0
        # Accept gaps formed around the parent candle and the first continuation
        # phase after it. This is deliberately configurable and symmetric.
        if gap["time"] < parent_start or gap["time"] > parent_time + timedelta(hours=12):
            continue
        min_gap_width = max(_finite(m15.iloc[gap["index"]].get("atr14"), 0.0) * 0.035, 0.08)
        if float(gap["high"]) - float(gap["low"]) < min_gap_width:
            continue
        if side == "BUY" and not (parent_low <= gap_mid <= midpoint):
            continue
        if side == "SELL" and not (midpoint <= gap_mid <= parent_high):
            continue
        invalidated, touch_index, confirmed = _gap_status(m15, gap, side)
        gap.update({"invalidated": invalidated, "touch_index": touch_index, "confirmed": confirmed})
        candidates.append(gap)

    if not candidates:
        return FourHourFVGSignal(
            side=side,
            state="WATCH",
            signal_id=f"H4FVG|{side}|{parent_time.isoformat()}|WAIT",
            signal_time=pd.to_datetime(m15.iloc[-1]["time"], utc=True).isoformat(),
            parent_candle_time=parent_time.isoformat(),
            parent_open=_round(parent_open, digits),
            parent_high=_round(parent_high, digits),
            parent_low=_round(parent_low, digits),
            parent_close=_round(parent_close, digits),
            parent_body_atr=round(body_atr, 2),
            confidence=min(64, int(round(48 + body_ratio * 12 + min(body_atr, 1.5) * 5))),
            macro_gate=macro.gate if macro else "UNAVAILABLE",
            valid_until=(parent_time + timedelta(hours=16)).isoformat(),
            rationale=[
                f"The completed H4 candle is a {side.lower()} displacement candle ({body_atr:.2f} ATR body).",
                "Waiting for a fresh M15 FVG in the correct discount/premium half of the H4 range.",
            ],
        )

    # Prefer an actually triggered setup, then an armed/touched setup, then the
    # latest untouched watch candidate. This prevents a newer minor FVG from
    # hiding a valid entry that has just confirmed.
    def candidate_rank(item: dict) -> tuple[int, float]:
        if item.get("invalidated"):
            rank = 0
        elif item.get("confirmed"):
            rank = 3
        elif item.get("touch_index") is not None:
            rank = 2
        else:
            rank = 1
        return rank, item["time"].timestamp()

    gap = max(candidates, key=candidate_rank)
    gap_low = float(gap["low"])
    gap_high = float(gap["high"])
    gap_mid = (gap_low + gap_high) / 2.0
    last = m15.iloc[-1]
    current_price = float(last["close"])
    m15_atr = _finite(last.get("atr14"), float((m15["high"] - m15["low"]).tail(14).mean()))
    touch_index = gap.get("touch_index")

    if gap["invalidated"]:
        state = "INVALIDATED"
    elif gap.get("confirmed"):
        state = "TRIGGERED"
    elif touch_index is not None or (gap_low <= current_price <= gap_high):
        state = "ARMED"
    else:
        state = "WATCH"

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

    h1 = _snapshot(indicators, "H1")
    score = 50
    reasons = [
        f"Completed H4 {side.lower()} displacement: body {body_atr:.2f} ATR and {body_ratio:.0%} of candle range.",
        f"Fresh M15 {'bullish' if side == 'BUY' else 'bearish'} FVG lies in the H4 {'discount' if side == 'BUY' else 'premium'} half.",
    ]
    if body_atr >= 0.9:
        score += 7
    if state == "ARMED":
        score += 5
        reasons.append("Price has entered the FVG entry zone; confirmation is still required.")
    if state == "TRIGGERED":
        score += 12
        reasons.append("M15 closed back through the FVG in the H4 candle direction.")
    if h1 is not None:
        aligned = (side == "BUY" and h1.trend == "bullish") or (side == "SELL" and h1.trend == "bearish")
        opposite = (side == "BUY" and h1.trend == "bearish") or (side == "SELL" and h1.trend == "bullish")
        if aligned:
            score += 6
            reasons.append("H1 trend agrees with the 4H-FVG direction.")
        elif opposite:
            score -= 7
    if macro is not None:
        if macro.gate == "CONFIRM":
            score += 5
            reasons.append("DXY/US10Y/gold-flow gate confirms the direction.")
        elif macro.gate == "CONFLICT":
            score -= 10
            reasons.append("Macro confirmation conflicts with the strategy direction.")
    aligns = primary_state == side
    if aligns:
        score += 5
        reasons.append("The regular AurumEdge decision agrees with this specialist setup.")

    confidence = max(45, min(82, int(round(score))))
    risk_plan = build_position_risk_plan(entry_mid, stop, risk_inputs or RiskInputs())
    warnings = [
        "This is a separate specialist setup. Do not treat WATCH or ARMED as an executed entry.",
        "TRIGGERED still requires live spread and broker-price verification.",
    ]
    if macro is not None and macro.gate == "CONFLICT":
        warnings.append("Macro conflict: the setup is visible but should not be executed without stronger confirmation.")
    if risk_plan.status != "OK":
        warnings.append(risk_plan.message)

    signal_time = pd.to_datetime(m15.iloc[-1]["time"], utc=True)
    return FourHourFVGSignal(
        side=side,
        state=state,
        signal_id=f"H4FVG|{side}|{parent_time.isoformat()}|{gap['time'].isoformat()}|{state}",
        signal_time=signal_time.isoformat(),
        parent_candle_time=parent_time.isoformat(),
        parent_open=_round(parent_open, digits),
        parent_high=_round(parent_high, digits),
        parent_low=_round(parent_low, digits),
        parent_close=_round(parent_close, digits),
        parent_body_atr=round(body_atr, 2),
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
        macro_gate=macro.gate if macro else "UNAVAILABLE",
        valid_until=(parent_time + timedelta(hours=16)).isoformat(),
        rationale=reasons,
        warnings=warnings,
        risk_plan=risk_plan,
    )
