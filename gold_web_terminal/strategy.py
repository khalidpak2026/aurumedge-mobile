from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any

from .models import (
    AdaptiveLearningSummary,
    IndicatorSnapshot,
    LiquiditySnapshot,
    MacroConfirmation,
    TechnicalReport,
    TradeSetup,
)
from .risk_engine import RiskInputs, build_position_risk_plan


TF_WEIGHTS = {"M5": 0.08, "M15": 0.22, "H1": 0.34, "H4": 0.26, "D1": 0.10}
DEFAULT_ADAPTIVE_WEIGHTS = {
    "market_structure": 1.0,
    "anchored_vwap": 1.0,
    "volume_profile": 1.0,
    "entry_quality": 1.0,
}


def _round_price(value: float, digits: int) -> float:
    return round(float(value), digits)


def _rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    return 0.0 if risk <= 0 else round(abs(target - entry) / risk, 2)


def _select_snapshot(indicators: list[IndicatorSnapshot], timeframe: str) -> IndicatorSnapshot:
    return next((item for item in indicators if item.timeframe == timeframe), indicators[0])


def _weight(weights: dict[str, float], key: str) -> float:
    return max(0.70, min(1.30, float(weights.get(key, 1.0))))


def _pillar_vote(item: IndicatorSnapshot) -> tuple[int, int, int]:
    """Return structure, anchored-VWAP and volume-profile votes (-1/0/+1)."""
    structure = 0
    if item.market_structure in {"BOS_UP", "CHOCH_UP"} or item.structure_bias == "bullish":
        structure = 1
    elif item.market_structure in {"BOS_DOWN", "CHOCH_DOWN"} or item.structure_bias == "bearish":
        structure = -1

    avwap = 0
    if item.avwap_active is not None:
        slope = float(item.avwap_slope_atr or 0.0)
        if item.close > float(item.avwap_active) and slope >= -0.03:
            avwap = 1
        elif item.close < float(item.avwap_active) and slope <= 0.03:
            avwap = -1

    profile = 0
    if item.profile_acceptance == "bullish" or item.profile_state == "ABOVE_VALUE":
        profile = 1
    elif item.profile_acceptance == "bearish" or item.profile_state == "BELOW_VALUE":
        profile = -1
    elif item.profile_poc is not None:
        distance = item.close - float(item.profile_poc)
        atr = max(float(item.atr14 or 0.0), 1e-9)
        if distance >= atr * 0.10:
            profile = 1
        elif distance <= -atr * 0.10:
            profile = -1
    return structure, avwap, profile


def _weighted_market_scores(
    indicators: list[IndicatorSnapshot], adaptive_weights: dict[str, float]
) -> tuple[float, float, list[str], list[str]]:
    """Score only the three requested pillars.

    Market structure carries 45%, anchored VWAP 35%, and volume profile 20%.
    No EMA, MACD, RSI, ADX, liquidity-sweep or macro vote can change direction.
    """
    bullish = 0.0
    bearish = 0.0
    buy_reasons: list[str] = []
    sell_reasons: list[str] = []
    for item in indicators:
        tf_weight = TF_WEIGHTS.get(item.timeframe, 0.1)
        structure, avwap, profile = _pillar_vote(item)
        components = (
            (structure, 45.0 * tf_weight * _weight(adaptive_weights, "market_structure"), "market structure"),
            (avwap, 35.0 * tf_weight * _weight(adaptive_weights, "anchored_vwap"), "anchored VWAP"),
            (profile, 20.0 * tf_weight * _weight(adaptive_weights, "volume_profile"), "volume profile"),
        )
        for vote, points, label in components:
            if vote > 0:
                bullish += points
                buy_reasons.append(f"{item.timeframe}: {label} confirms buyers")
            elif vote < 0:
                bearish += points
                sell_reasons.append(f"{item.timeframe}: {label} confirms sellers")
    return bullish, bearish, buy_reasons, sell_reasons


def _market_stats(indicators: list[IndicatorSnapshot]) -> dict[str, float | int | bool]:
    """Price-action statistics used for timing/risk, not directional voting."""
    m15 = _select_snapshot(indicators, "M15")
    h1 = _select_snapshot(indicators, "H1")
    structure_up = bool(
        m15.structure_break_up
        or m15.market_structure in {"BOS_UP", "CHOCH_UP"}
        or h1.market_structure == "BOS_UP"
    )
    structure_down = bool(
        m15.structure_break_down
        or m15.market_structure in {"BOS_DOWN", "CHOCH_DOWN"}
        or h1.market_structure == "BOS_DOWN"
    )
    return {
        "breakout_up": structure_up,
        "breakout_down": structure_down,
    }


def _timeframe_alignment(indicators: list[IndicatorSnapshot]) -> dict[str, int]:
    bullish = 0
    bearish = 0
    neutral = 0
    for timeframe in ("M15", "H1", "H4", "D1"):
        item = next((row for row in indicators if row.timeframe == timeframe), None)
        if item is None:
            continue
        votes = _pillar_vote(item)
        score = sum(votes)
        if score >= 1:
            bullish += 1
        elif score <= -1:
            bearish += 1
        else:
            neutral += 1
    return {"bullish": bullish, "bearish": bearish, "neutral": neutral}


def _liquidity_context(liquidity: list[LiquiditySnapshot], price: float) -> dict[str, Any]:
    supports: list[float] = []
    resistances: list[float] = []
    bull_traps: list[str] = []
    bear_traps: list[str] = []
    for item in liquidity:
        for zone in item.support_zones:
            for key in ("upper", "center", "lower"):
                value = zone.get(key)
                if value is not None and float(value) < price:
                    supports.append(float(value))
        for zone in item.resistance_zones:
            for key in ("lower", "center", "upper"):
                value = zone.get(key)
                if value is not None and float(value) > price:
                    resistances.append(float(value))
        for level in (item.previous_day_low, item.value_area_low, item.point_of_control, *item.recent_swing_lows):
            if level is not None and level < price:
                supports.append(float(level))
        for level in (item.previous_day_high, item.value_area_high, item.point_of_control, *item.recent_swing_highs):
            if level is not None and level > price:
                resistances.append(float(level))
        if item.trap_type == "bull_trap" or item.sweep_above is not None:
            bull_traps.append(item.timeframe)
        if item.trap_type == "bear_trap" or item.sweep_below is not None:
            bear_traps.append(item.timeframe)
    active_timeframes = {"M15", "H1"}

    def _fresh(item: LiquiditySnapshot, side: str) -> bool:
        age = item.sweep_above_age if side == "above" else item.sweep_below_age
        if age is None:
            return False
        # M15 traps may affect the current and immediately following candle.
        # H1 traps affect only the current H1 candle.  Older sweeps remain chart
        # context but must not freeze execution for hours.
        return age <= (1 if item.timeframe == "M15" else 0)

    bull_trap_levels = {
        item.timeframe: float(item.sweep_above)
        for item in liquidity
        if item.timeframe in active_timeframes and item.sweep_above is not None and _fresh(item, "above")
    }
    bear_trap_levels = {
        item.timeframe: float(item.sweep_below)
        for item in liquidity
        if item.timeframe in active_timeframes and item.sweep_below is not None and _fresh(item, "below")
    }
    bull_trap_ages = {
        item.timeframe: int(item.sweep_above_age)
        for item in liquidity
        if item.timeframe in bull_trap_levels and item.sweep_above_age is not None
    }
    bear_trap_ages = {
        item.timeframe: int(item.sweep_below_age)
        for item in liquidity
        if item.timeframe in bear_trap_levels and item.sweep_below_age is not None
    }
    two_sided_timeframes = sorted(set(bull_trap_levels) & set(bear_trap_levels))
    return {
        "supports": sorted(set(round(level, 5) for level in supports), reverse=True),
        "resistances": sorted(set(round(level, 5) for level in resistances)),
        "nearest_support": max(supports) if supports else None,
        "nearest_resistance": min(resistances) if resistances else None,
        # All-timeframe lists remain useful as context and chart labels.  The
        # execution classifier below only treats fresh M15/H1 sweeps as a trap.
        "bull_traps": bull_traps,
        "bear_traps": bear_traps,
        "bull_trap_levels": bull_trap_levels,
        "bear_trap_levels": bear_trap_levels,
        "bull_trap_ages": bull_trap_ages,
        "bear_trap_ages": bear_trap_ages,
        "two_sided_timeframes": two_sided_timeframes,
    }


def _entry_candidate(
    side: str,
    price: float,
    m15: IndicatorSnapshot,
    h1: IndicatorSnapshot,
    supports: list[float],
    resistances: list[float],
    regime: str,
) -> tuple[float, str]:
    h1_atr = float(h1.atr14 or m15.atr14 or price * 0.002)
    m15_atr = float(m15.atr14 or h1_atr * 0.5)
    if "breakout" in regime:
        return price, "LIVE THREE-PILLAR BREAKOUT"

    m15_votes = _pillar_vote(m15)
    h1_votes = _pillar_vote(h1)
    side_sign = 1 if side == "BUY" else -1
    m15_confirmations = sum(vote == side_sign for vote in m15_votes)
    h1_confirmations = sum(vote == side_sign for vote in h1_votes)
    reference = m15.avwap_active or m15.profile_poc or h1.avwap_active or h1.profile_poc
    distance_from_value = abs(price - float(reference)) if reference is not None else 0.0

    # This is the responsive continuation path. It prevents the terminal from
    # waiting through an entire $50-$100 move when all three pillars have
    # already aligned. It is still blocked when price is excessively extended.
    if m15_confirmations >= 2 and h1_confirmations >= 2 and distance_from_value <= m15_atr * 1.20:
        return price, "LIVE THREE-PILLAR CONTINUATION"

    if side == "BUY":
        candidates = [level for level in supports if price - h1_atr * 0.85 <= level <= price]
        for level in (m15.avwap_active, h1.avwap_active, m15.profile_poc, h1.profile_poc, m15.profile_val):
            if level is not None and price - h1_atr * 0.85 <= level <= price:
                candidates.append(float(level))
        if candidates:
            level = max(candidates)
            if price - level <= m15_atr * 0.35:
                return price, "LIVE THREE-PILLAR VALUE RECLAIM"
            return level + m15_atr * 0.05, "PULLBACK TO AVWAP / PROFILE VALUE"
    else:
        candidates = [level for level in resistances if price <= level <= price + h1_atr * 0.85]
        for level in (m15.avwap_active, h1.avwap_active, m15.profile_poc, h1.profile_poc, m15.profile_vah):
            if level is not None and price <= level <= price + h1_atr * 0.85:
                candidates.append(float(level))
        if candidates:
            level = min(candidates)
            if level - price <= m15_atr * 0.35:
                return price, "LIVE THREE-PILLAR VALUE REJECTION"
            return level - m15_atr * 0.05, "PULLBACK TO AVWAP / PROFILE VALUE"
    return price, "LIVE THREE-PILLAR MARKET ZONE"


def _nearest_below(levels: list[float], price: float) -> float | None:
    eligible = [level for level in levels if level < price]
    return max(eligible) if eligible else None


def _nearest_above(levels: list[float], price: float) -> float | None:
    eligible = [level for level in levels if level > price]
    return min(eligible) if eligible else None


def _target_before_level(side: str, raw: float, levels: list[float], entry: float, buffer: float) -> float:
    if side == "BUY":
        level = _nearest_above(levels, entry)
        if level is not None and level < raw:
            return max(entry + buffer, level - buffer)
        return raw
    level = _nearest_below(levels, entry)
    if level is not None and level > raw:
        return min(entry - buffer, level + buffer)
    return raw


def _build_setup(
    side: str,
    price: float,
    h1_atr: float,
    m15_atr: float,
    confidence: int,
    supports: list[float],
    resistances: list[float],
    digits: int,
    active: bool,
    reasons: list[str],
    data_source: str,
    m15: IndicatorSnapshot,
    h1: IndicatorSnapshot,
    regime: str,
    risk_inputs: RiskInputs,
    target_multipliers: dict[str, float],
) -> TradeSetup:
    entry_mid, entry_type = _entry_candidate(side, price, m15, h1, supports, resistances, regime)
    entry_half_width = max(0.06 * m15_atr, 0.05)
    entry_low = entry_mid - entry_half_width
    entry_high = entry_mid + entry_half_width
    min_stop_distance = max(risk_inputs.minimum_stop_atr * m15_atr, risk_inputs.spread_price * 2.2, 0.40)
    max_normal_distance = max(risk_inputs.maximum_stop_atr * h1_atr, min_stop_distance)
    buffer = max(0.14 * m15_atr, risk_inputs.spread_price * 1.5, 0.25)
    common_warnings = [
        "This is indicative decision support, not an executable broker quote.",
        "The lot recommendation assumes the configured contract size; verify the broker symbol specification.",
    ]
    if data_source == "DEMO":
        common_warnings.append("Demo data is synthetic and must not be used for live trading.")

    if side == "BUY":
        structural = _nearest_below(supports, entry_mid)
        if structural is not None:
            stop = structural - buffer
            stop_basis = f"Below the nearest valid support/liquidity structure at {structural:.2f}, plus a volatility/spread buffer."
        else:
            stop = entry_mid - max(0.85 * h1_atr, min_stop_distance)
            stop_basis = "No close support was available; stop uses a conservative ATR invalidation distance."
        if entry_mid - stop < min_stop_distance:
            stop = entry_mid - min_stop_distance
        stop_distance = entry_mid - stop
        if stop_distance > max_normal_distance:
            common_warnings.append(
                "The nearest structural invalidation is unusually far from entry. The engine keeps the structural stop and reduces lot size instead of placing a false tight stop."
            )
        risk = max(stop_distance, 1e-9)
        distances = {
            "tp1": min(risk * target_multipliers["tp1"], h1_atr * 0.90),
            "tp2": min(risk * target_multipliers["tp2"], h1_atr * 1.35),
            "tp3": min(risk * target_multipliers["tp3"], h1_atr * 1.85),
        }
        tp1 = _target_before_level("BUY", entry_mid + distances["tp1"], resistances, entry_mid, buffer * 0.45)
        tp2 = _target_before_level("BUY", entry_mid + distances["tp2"], [x for x in resistances if x > tp1], entry_mid, buffer * 0.45)
        tp3 = _target_before_level("BUY", entry_mid + distances["tp3"], [x for x in resistances if x > tp2], entry_mid, buffer * 0.45)
        tp2 = max(tp2, tp1 + max(m15_atr * 0.25, 0.50))
        tp3 = max(tp3, tp2 + max(m15_atr * 0.30, 0.70))
        invalidation = f"M15 closes below {_round_price(stop, digits)}; H1 structure must also be reviewed."
    else:
        structural = _nearest_above(resistances, entry_mid)
        if structural is not None:
            stop = structural + buffer
            stop_basis = f"Above the nearest valid resistance/liquidity structure at {structural:.2f}, plus a volatility/spread buffer."
        else:
            stop = entry_mid + max(0.85 * h1_atr, min_stop_distance)
            stop_basis = "No close resistance was available; stop uses a conservative ATR invalidation distance."
        if stop - entry_mid < min_stop_distance:
            stop = entry_mid + min_stop_distance
        stop_distance = stop - entry_mid
        if stop_distance > max_normal_distance:
            common_warnings.append(
                "The nearest structural invalidation is unusually far from entry. The engine keeps the structural stop and reduces lot size instead of placing a false tight stop."
            )
        risk = max(stop_distance, 1e-9)
        distances = {
            "tp1": min(risk * target_multipliers["tp1"], h1_atr * 0.90),
            "tp2": min(risk * target_multipliers["tp2"], h1_atr * 1.35),
            "tp3": min(risk * target_multipliers["tp3"], h1_atr * 1.85),
        }
        tp1 = _target_before_level("SELL", entry_mid - distances["tp1"], supports, entry_mid, buffer * 0.45)
        tp2 = _target_before_level("SELL", entry_mid - distances["tp2"], [x for x in supports if x < tp1], entry_mid, buffer * 0.45)
        tp3 = _target_before_level("SELL", entry_mid - distances["tp3"], [x for x in supports if x < tp2], entry_mid, buffer * 0.45)
        tp2 = min(tp2, tp1 - max(m15_atr * 0.25, 0.50))
        tp3 = min(tp3, tp2 - max(m15_atr * 0.30, 0.70))
        invalidation = f"M15 closes above {_round_price(stop, digits)}; H1 structure must also be reviewed."

    risk_plan = build_position_risk_plan(entry_mid, stop, risk_inputs)
    status_active = active and risk_plan.status != "NO_TRADE"
    if risk_plan.status == "REDUCE_LOT":
        common_warnings.append(risk_plan.message)
    elif risk_plan.status == "NO_TRADE":
        common_warnings.append(risk_plan.message)

    valid_until = (datetime.now(timezone.utc) + timedelta(minutes=45)).strftime("%Y-%m-%d %H:%M UTC")
    target_basis = (
        f"Targets use adaptive reachable-range multipliers {target_multipliers['tp1']:.2f}R / "
        f"{target_multipliers['tp2']:.2f}R / {target_multipliers['tp3']:.2f}R, capped by H1 ATR and placed before nearby structure/profile levels."
    )
    management = [
        "Do not chase outside the entry zone; recalculate if price moves more than 0.35 M15 ATR away.",
        "At TP1, take partial profit. Move the stop to breakeven only after an M15 close confirms continuation, not on a wick touch.",
        "TP3 is a runner target; cancel it when market structure breaks against the trade or price loses anchored VWAP/value acceptance.",
    ]
    if risk_plan.status == "REDUCE_LOT":
        management.insert(0, f"Use approximately {risk_plan.recommended_lot:.2f} lot or less; the requested lot exceeds the risk budget.")

    return TradeSetup(
        side=side,  # type: ignore[arg-type]
        status="ENTER" if status_active else "NO_TRADE",
        confidence=confidence,
        entry_low=_round_price(entry_low, digits),
        entry_high=_round_price(entry_high, digits),
        entry_type=entry_type if status_active else "DIRECTIONAL SCENARIO / RISK GATE ACTIVE",
        stop_loss=_round_price(stop, digits),
        take_profit_1=_round_price(tp1, digits),
        take_profit_2=_round_price(tp2, digits),
        take_profit_3=_round_price(tp3, digits),
        risk_reward_1=_rr(entry_mid, stop, tp1),
        risk_reward_2=_rr(entry_mid, stop, tp2),
        risk_reward_3=_rr(entry_mid, stop, tp3),
        valid_until=valid_until,
        invalidation=invalidation,
        stop_basis=stop_basis,
        target_basis=target_basis,
        management_plan=management,
        risk_plan=risk_plan,
        rationale=reasons[:12],
        warnings=common_warnings,
    )


def build_technical_report(
    symbol: str,
    data_time: str,
    price: float,
    indicators: list[IndicatorSnapshot],
    liquidity: list[LiquiditySnapshot],
    data_source: str,
    digits: int = 2,
    point: float = 0.01,
    extra_notes: list[str] | None = None,
    adaptive_weights: dict[str, float] | None = None,
    target_multipliers: dict[str, float] | None = None,
    adaptive_summary: AdaptiveLearningSummary | None = None,
    risk_inputs: RiskInputs | None = None,
    macro: MacroConfirmation | None = None,
    macro_required_for_entry: bool = False,
    previous_state: str | None = None,
    trap_anchor_price: float | None = None,
    trap_age: int = 0,
) -> TechnicalReport:
    """Build a three-pillar XAU/USD decision.

    Direction is produced exclusively by market structure, anchored VWAP and
    volume profile. DXY and US10Y remain display-only context and never block a
    technical entry. ATR is retained only for risk geometry and target sizing.
    """
    del macro_required_for_entry, previous_state, trap_anchor_price, trap_age
    if not indicators:
        raise ValueError("At least one indicator snapshot is required.")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("The supplied market price is invalid.")

    weights = {**DEFAULT_ADAPTIVE_WEIGHTS, **(adaptive_weights or {})}
    targets = {"tp1": 0.65, "tp2": 1.05, "tp3": 1.40, **(target_multipliers or {})}
    risk = risk_inputs or RiskInputs()

    bullish, bearish, buy_reasons, sell_reasons = _weighted_market_scores(indicators, weights)
    stats = _market_stats(indicators)
    alignment = _timeframe_alignment(indicators)
    liq = _liquidity_context(liquidity, price)
    m15 = _select_snapshot(indicators, "M15")
    h1 = _select_snapshot(indicators, "H1")
    h4 = _select_snapshot(indicators, "H4")

    h1_atr = max(float(h1.atr14 or m15.atr14 or h4.atr14 or price * 0.002), point * 10)
    m15_atr = max(float(m15.atr14 or h1_atr * 0.45), point * 8)
    atr_pct = float(h1.atr_pct or (h1_atr / price * 100.0))

    net = bullish - bearish
    buy_score = int(round(max(5.0, min(95.0, 50.0 + net * 0.58))))
    sell_score = 100 - buy_score
    bullish_tf = int(alignment["bullish"])
    bearish_tf = int(alignment["bearish"])

    m15_votes = _pillar_vote(m15)
    h1_votes = _pillar_vote(h1)
    h4_votes = _pillar_vote(h4)
    m15_bull = sum(1 for vote in m15_votes if vote > 0)
    m15_bear = sum(1 for vote in m15_votes if vote < 0)
    h1_bull = sum(1 for vote in h1_votes if vote > 0)
    h1_bear = sum(1 for vote in h1_votes if vote < 0)
    h4_bull = sum(1 for vote in h4_votes if vote > 0)
    h4_bear = sum(1 for vote in h4_votes if vote < 0)

    breakout_up = bool(stats["breakout_up"] and m15_bull >= 2 and h1_bear <= 1)
    breakout_down = bool(stats["breakout_down"] and m15_bear >= 2 and h1_bull <= 1)

    buy_confirmed = bool(
        breakout_up
        or (
            buy_score >= 60
            and bullish_tf >= 2
            and m15_bull >= 2
            and h1_bull >= 2
            and h4_bear <= 1
        )
        or (buy_score >= 68 and bullish_tf >= 3 and h1_bear <= 1)
    )
    sell_confirmed = bool(
        breakout_down
        or (
            sell_score >= 60
            and bearish_tf >= 2
            and m15_bear >= 2
            and h1_bear >= 2
            and h4_bull <= 1
        )
        or (sell_score >= 68 and bearish_tf >= 3 and h1_bull <= 1)
    )

    # When both sides appear confirmed, prefer the side with the larger score;
    # otherwise classify as balanced rather than inventing a trap state.
    if buy_confirmed and sell_confirmed:
        if buy_score >= sell_score + 10:
            sell_confirmed = False
        elif sell_score >= buy_score + 10:
            buy_confirmed = False
        else:
            buy_confirmed = sell_confirmed = False

    if atr_pct >= 0.85:
        volatility_state = "extreme"
    elif atr_pct >= 0.48:
        volatility_state = "high"
    elif atr_pct <= 0.16:
        volatility_state = "low"
    else:
        volatility_state = "normal"

    trap_reason = ""
    if buy_confirmed:
        market_state = "BUY"
        regime = "breakout_up" if breakout_up else (
            "volatile_bullish" if volatility_state in {"high", "extreme"} else "bullish_trend"
        )
        signal_label = "ENTER BUY · THREE-PILLAR BREAKOUT" if breakout_up else "ENTER BUY · THREE-PILLAR TREND"
    elif sell_confirmed:
        market_state = "SELL"
        regime = "breakout_down" if breakout_down else (
            "volatile_bearish" if volatility_state in {"high", "extreme"} else "bearish_trend"
        )
        signal_label = "ENTER SELL · THREE-PILLAR BREAKOUT" if breakout_down else "ENTER SELL · THREE-PILLAR TREND"
    else:
        market_state = "STUCK"
        regime = "stuck_range"
        signal_label = "NO TRADE · THREE PILLARS NOT ALIGNED"
        trap_reason = (
            "Market structure, anchored VWAP and volume profile do not yet agree on one direction. "
            "DXY and US10Y are informational only and did not block a signal."
        )

    edge = max(buy_score, sell_score) - 50
    aligned_tf = bullish_tf if market_state == "BUY" else bearish_tf if market_state == "SELL" else max(bullish_tf, bearish_tf)
    confidence = int(round(max(52, min(88, 54 + edge * 0.48 + aligned_tf * 3.5))))
    trend_strength = int(round(max(0, min(100, abs(net) * 1.15 + aligned_tf * 12))))

    active_buy = market_state == "BUY"
    active_sell = market_state == "SELL"
    buy_setup = _build_setup(
        "BUY", price, h1_atr, m15_atr, buy_score, liq["supports"], liq["resistances"], digits,
        active_buy, buy_reasons, data_source, m15, h1, str(regime), risk, targets,
    )
    sell_setup = _build_setup(
        "SELL", price, h1_atr, m15_atr, sell_score, liq["supports"], liq["resistances"], digits,
        active_sell, sell_reasons, data_source, m15, h1, str(regime), risk, targets,
    )
    active_setup = buy_setup if active_buy else sell_setup if active_sell else None

    if active_setup is not None and active_setup.risk_plan is not None:
        if active_setup.risk_plan.status == "REDUCE_LOT":
            signal_label += " · REDUCE LOT"
        elif active_setup.risk_plan.status == "NO_TRADE":
            signal_label = f"{market_state} BIAS · RISK TOO HIGH"

    if active_setup is not None and active_setup.status == "ENTER":
        entry_tolerance = max(0.20 * m15_atr, 0.35)
        entry_live = active_setup.entry_low - entry_tolerance <= price <= active_setup.entry_high + entry_tolerance
        if entry_live:
            signal_label += " · ENTRY LIVE NOW"
        else:
            signal_label += " · WAIT FOR ENTRY ZONE"
            active_setup.warnings.append(
                "Direction is active, but Telegram remains silent until live price reaches the entry zone."
            )

    notes = [
        "Directional decisions use only market structure, anchored VWAP and volume profile.",
        "DXY and US10Y are display-only context; they cannot block or reverse a signal.",
        "ATR is used only to place realistic stops, targets and entry tolerances; it does not vote on direction.",
        "The FVG specialist strategy and EMA/MACD/RSI/ADX decision layers are disabled in this build.",
    ]
    notes.extend(extra_notes or [])

    return TechnicalReport(
        symbol=symbol,
        data_time=data_time,
        last_price=_round_price(price, digits),
        data_source=data_source,  # type: ignore[arg-type]
        regime=regime,  # type: ignore[arg-type]
        market_state=market_state,  # type: ignore[arg-type]
        recommendation=market_state,  # type: ignore[arg-type]
        signal_label=signal_label,
        confidence=confidence,
        buy_score=buy_score,
        sell_score=sell_score,
        trend_strength=trend_strength,
        volatility_state=volatility_state,  # type: ignore[arg-type]
        trap_reason=trap_reason,
        indicators=indicators,
        liquidity=liquidity,
        active_setup=active_setup,
        buy_setup=buy_setup,
        sell_setup=sell_setup,
        macro=macro,
        adaptive=adaptive_summary,
        special_signals=[],
        data_quality_notes=notes,
    )
