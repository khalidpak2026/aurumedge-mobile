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


TF_WEIGHTS = {"M5": 0.06, "M15": 0.20, "H1": 0.32, "H4": 0.31, "D1": 0.11}
DEFAULT_ADAPTIVE_WEIGHTS = {
    "ema_trend": 1.0,
    "momentum": 1.0,
    "adx_dmi": 1.0,
    "vwap": 1.0,
    "volume": 1.0,
    "liquidity": 1.0,
    "breakout": 1.0,
    "macro": 1.0,
    "entry_quality": 1.0,
    "market_structure": 1.0,
    "anchored_vwap": 1.0,
    "volume_profile": 1.0,
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


def _weighted_market_scores(
    indicators: list[IndicatorSnapshot], adaptive_weights: dict[str, float]
) -> tuple[float, float, list[str], list[str]]:
    bullish = 0.0
    bearish = 0.0
    buy_reasons: list[str] = []
    sell_reasons: list[str] = []
    for item in indicators:
        tf_weight = TF_WEIGHTS.get(item.timeframe, 0.1)

        structure_weight = _weight(adaptive_weights, "market_structure")
        trend_component = abs(item.directional_score) * tf_weight * structure_weight
        if item.directional_score > 0:
            bullish += trend_component
        elif item.directional_score < 0:
            bearish += trend_component
        if item.structure_bias == "bullish":
            buy_reasons.append(f"{item.timeframe}: market structure {item.market_structure} favors buyers")
        elif item.structure_bias == "bearish":
            sell_reasons.append(f"{item.timeframe}: market structure {item.market_structure} favors sellers")

        avwap_points = 9 * tf_weight * _weight(adaptive_weights, "anchored_vwap")
        if item.avwap_active is not None:
            if item.close > item.avwap_active and (item.avwap_slope_atr or 0) >= -0.05:
                bullish += avwap_points
                buy_reasons.append(f"{item.timeframe}: price holds above {item.avwap_anchor.replace('_',' ')} anchored VWAP")
            elif item.close < item.avwap_active and (item.avwap_slope_atr or 0) <= 0.05:
                bearish += avwap_points
                sell_reasons.append(f"{item.timeframe}: price holds below {item.avwap_anchor.replace('_',' ')} anchored VWAP")

        profile_points = 9 * tf_weight * _weight(adaptive_weights, "volume_profile")
        if item.profile_acceptance == "bullish" or item.profile_state == "ABOVE_VALUE":
            bullish += profile_points
            buy_reasons.append(f"{item.timeframe}: volume profile accepts price above value")
        elif item.profile_acceptance == "bearish" or item.profile_state == "BELOW_VALUE":
            bearish += profile_points
            sell_reasons.append(f"{item.timeframe}: volume profile accepts price below value")
        elif item.profile_poc is not None:
            if item.close > item.profile_poc:
                bullish += profile_points * 0.35
            elif item.close < item.profile_poc:
                bearish += profile_points * 0.35

        momentum_points = 10 * tf_weight * _weight(adaptive_weights, "momentum")
        if item.momentum == "bullish":
            bullish += momentum_points
            buy_reasons.append(f"{item.timeframe}: MACD and RSI momentum are aligned up")
        elif item.momentum == "bearish":
            bearish += momentum_points
            sell_reasons.append(f"{item.timeframe}: MACD and RSI momentum are aligned down")

        dmi_points = 7 * tf_weight * _weight(adaptive_weights, "adx_dmi")
        if (item.adx14 or 0) >= 18:
            if (item.plus_di or 0) > (item.minus_di or 0):
                bullish += dmi_points
            elif (item.minus_di or 0) > (item.plus_di or 0):
                bearish += dmi_points

        # EMA remains a low-weight secondary confirmation only. It cannot
        # override market structure, anchored VWAP, or volume-profile context.
        ema_points = 2.5 * tf_weight * _weight(adaptive_weights, "ema_trend")
        if item.ema20 is not None and item.ema50 is not None:
            if item.close > item.ema20 > item.ema50:
                bullish += ema_points
            elif item.close < item.ema20 < item.ema50:
                bearish += ema_points

        breakout_points = 9 * tf_weight * _weight(adaptive_weights, "breakout")
        if item.breakout_up or item.market_structure in {"BOS_UP", "CHOCH_UP"}:
            bullish += breakout_points
            buy_reasons.append(f"{item.timeframe}: market structure broke upward")
        if item.breakout_down or item.market_structure in {"BOS_DOWN", "CHOCH_DOWN"}:
            bearish += breakout_points
            sell_reasons.append(f"{item.timeframe}: market structure broke downward")

    return bullish, bearish, buy_reasons, sell_reasons



def _market_stats(indicators: list[IndicatorSnapshot]) -> dict[str, float | int | bool]:
    weighted_adx = 0.0
    weighted_chop = 0.0
    weight_adx = 0.0
    weight_chop = 0.0
    compressions = 0
    breakout_up = False
    breakout_down = False
    for item in indicators:
        weight = TF_WEIGHTS.get(item.timeframe, 0.1)
        if item.adx14 is not None:
            weighted_adx += item.adx14 * weight
            weight_adx += weight
        if item.choppiness14 is not None:
            weighted_chop += item.choppiness14 * weight
            weight_chop += weight
        if item.compression and item.timeframe in {"M15", "H1", "H4"}:
            compressions += 1
        if item.timeframe in {"M15", "H1"}:
            breakout_up = breakout_up or item.breakout_up
            breakout_down = breakout_down or item.breakout_down
    return {
        "average_adx": weighted_adx / weight_adx if weight_adx else 0.0,
        "average_chop": weighted_chop / weight_chop if weight_chop else 50.0,
        "compressions": compressions,
        "breakout_up": breakout_up,
        "breakout_down": breakout_down,
    }


def _timeframe_alignment(indicators: list[IndicatorSnapshot]) -> dict[str, int]:
    """Count directional agreement once per important timeframe.

    The old engine could let one fresh liquidity sweep overrule four aligned
    timeframes.  This helper deliberately evaluates the *stack* first.  It is
    not a signal by itself; it is used to decide whether a sweep is a genuine
    reversal warning or only a temporary event inside an established trend.
    """

    bullish = 0
    bearish = 0
    neutral = 0
    for timeframe in ("M15", "H1", "H4", "D1"):
        item = next((row for row in indicators if row.timeframe == timeframe), None)
        if item is None:
            continue

        bull_votes = 0
        bear_votes = 0
        if item.trend == "bullish":
            bull_votes += 2
        elif item.trend == "bearish":
            bear_votes += 2
        if item.structure_bias == "bullish" or item.market_structure in {"BOS_UP", "CHOCH_UP"}:
            bull_votes += 2
        elif item.structure_bias == "bearish" or item.market_structure in {"BOS_DOWN", "CHOCH_DOWN"}:
            bear_votes += 2
        if item.momentum == "bullish":
            bull_votes += 1
        elif item.momentum == "bearish":
            bear_votes += 1
        if item.directional_score >= 12:
            bull_votes += 1
        elif item.directional_score <= -12:
            bear_votes += 1
        if item.avwap_active is not None:
            if item.close > item.avwap_active:
                bull_votes += 1
            elif item.close < item.avwap_active:
                bear_votes += 1

        if bull_votes >= bear_votes + 2:
            bullish += 1
        elif bear_votes >= bull_votes + 2:
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
        return price, "BREAKOUT RETEST / CURRENT ZONE"

    if side == "BUY":
        candidates = [level for level in supports if price - h1_atr * 0.85 <= level <= price]
        for level in (m15.avwap_active, h1.avwap_active, m15.profile_poc, h1.profile_poc, m15.vwap):
            if level is not None and price - h1_atr * 0.85 <= level <= price:
                candidates.append(float(level))
        if candidates:
            level = max(candidates)
            if price - level <= m15_atr * 0.25:
                return price, "CURRENT PRICE AT SUPPORT/VALUE"
            return level + m15_atr * 0.05, "PULLBACK TO NEAREST SUPPORT/VALUE"
    else:
        candidates = [level for level in resistances if price <= level <= price + h1_atr * 0.85]
        for level in (m15.avwap_active, h1.avwap_active, m15.profile_poc, h1.profile_poc, m15.vwap):
            if level is not None and price <= level <= price + h1_atr * 0.85:
                candidates.append(float(level))
        if candidates:
            level = min(candidates)
            if level - price <= m15_atr * 0.25:
                return price, "CURRENT PRICE AT RESISTANCE/VALUE"
            return level - m15_atr * 0.05, "PULLBACK TO NEAREST RESISTANCE/VALUE"
    return price, "CURRENT MARKET ZONE"


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
        f"{target_multipliers['tp2']:.2f}R / {target_multipliers['tp3']:.2f}R, capped by H1 ATR and placed before nearby liquidity."
    )
    management = [
        "Do not chase outside the entry zone; recalculate if price moves more than 0.35 M15 ATR away.",
        "At TP1, take partial profit. Move the stop to breakeven only after an M15 close confirms continuation, not on a wick touch.",
        "TP3 is a runner target; cancel it when momentum or macro confirmation reverses.",
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
    macro_required_for_entry: bool = True,
    previous_state: str | None = None,
    trap_anchor_price: float | None = None,
    trap_age: int = 0,
) -> TechnicalReport:
    if not indicators:
        raise ValueError("At least one indicator snapshot is required.")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("The supplied market price is invalid.")

    weights = {**DEFAULT_ADAPTIVE_WEIGHTS, **(adaptive_weights or {})}
    targets = {"tp1": 0.65, "tp2": 1.05, "tp3": 1.40, **(target_multipliers or {})}
    risk = risk_inputs or RiskInputs()

    bullish, bearish, buy_reasons, sell_reasons = _weighted_market_scores(indicators, weights)
    stats = _market_stats(indicators)
    liq = _liquidity_context(liquidity, price)
    h1 = _select_snapshot(indicators, "H1")
    h4 = _select_snapshot(indicators, "H4")
    m15 = _select_snapshot(indicators, "M15")
    h1_atr = max(float(h1.atr14 or m15.atr14 or h4.atr14 or price * 0.002), point * 10)
    m15_atr = max(float(m15.atr14 or h1_atr * 0.45), point * 8)
    atr_pct = h1.atr_pct or (h1_atr / price * 100)

    liquidity_weight = _weight(weights, "liquidity")
    if liq["bear_traps"] and h1.momentum == "bullish":
        bullish += 9 * liquidity_weight
        buy_reasons.append("Sell-side liquidity was swept and reclaimed")
    if liq["bull_traps"] and h1.momentum == "bearish":
        bearish += 9 * liquidity_weight
        sell_reasons.append("Buy-side liquidity was swept and rejected")

    if macro is not None:
        macro_weight = _weight(weights, "macro")
        macro_points = 14 * macro_weight * max(0.55, macro.coverage_score / 100.0)
        if macro.macro_bias == "BULLISH_GOLD":
            bullish += macro_points
            buy_reasons.extend(macro.reasons[:3])
        elif macro.macro_bias == "BEARISH_GOLD":
            bearish += macro_points
            sell_reasons.extend(macro.conflicts[:3])

    average_adx = float(stats["average_adx"])
    average_chop = float(stats["average_chop"])
    compression_count = int(stats["compressions"])
    alignment = _timeframe_alignment(indicators)
    bullish_tf_count = int(alignment["bullish"])
    bearish_tf_count = int(alignment["bearish"])
    breakout_up = bool(stats["breakout_up"] and (m15.profile_acceptance == "bullish" or m15.profile_state == "ABOVE_VALUE" or m15.close > float(m15.profile_poc or m15.close)))
    breakout_down = bool(stats["breakout_down"] and (m15.profile_acceptance == "bearish" or m15.profile_state == "BELOW_VALUE" or m15.close < float(m15.profile_poc or m15.close)))
    net = bullish - bearish

    # Directional resolution is evaluated before the trap label.  A liquidity
    # sweep is an event, not a market regime.  Strong displacement, a local
    # structure break and aligned momentum must immediately release the engine
    # into BUY or SELL even while slower H1/H4 EMAs are still catching up.
    h1_adx = float(h1.adx14 or 0.0)
    bull_dmi = h1_adx >= 18 and float(h1.plus_di or 0) > float(h1.minus_di or 0)
    bear_dmi = h1_adx >= 18 and float(h1.minus_di or 0) > float(h1.plus_di or 0)
    bull_momentum = h1.momentum == "bullish" and (m15.momentum == "bullish" or breakout_up or m15.structure_break_up)
    bear_momentum = h1.momentum == "bearish" and (m15.momentum == "bearish" or breakout_down or m15.structure_break_down)
    h1_above_value = h1.close > float(h1.avwap_active or h1.vwap or h1.ema20 or h1.close)
    h1_below_value = h1.close < float(h1.avwap_active or h1.vwap or h1.ema20 or h1.close)
    h1_above_fast = h1.close > float(h1.avwap_active or h1.ema9 or h1.ema20 or h1.close)
    h1_below_fast = h1.close < float(h1.avwap_active or h1.ema9 or h1.ema20 or h1.close)

    m15_bull_impulse = bool(
        ((m15.impulse_1_atr or 0) >= 0.42 or (m15.impulse_3_atr or 0) >= 0.78)
        and (m15.close_location or 0.5) >= 0.62
    )
    m15_bear_impulse = bool(
        ((m15.impulse_1_atr or 0) <= -0.42 or (m15.impulse_3_atr or 0) <= -0.78)
        and (m15.close_location or 0.5) <= 0.38
    )
    h1_bull_impulse = bool(
        (h1.impulse_1_atr or 0) >= 0.30
        or (h1.impulse_3_atr or 0) >= 0.60
        or ((h1.macd_hist_slope or 0) > 0 and h1_above_fast)
    )
    h1_bear_impulse = bool(
        (h1.impulse_1_atr or 0) <= -0.30
        or (h1.impulse_3_atr or 0) <= -0.60
        or ((h1.macd_hist_slope or 0) < 0 and h1_below_fast)
    )

    fast_break_up = bool(breakout_up or m15.structure_break_up)
    fast_break_down = bool(breakout_down or m15.structure_break_down)

    m15_above_value = m15.close > float(m15.avwap_active or m15.profile_poc or m15.vwap or m15.ema20 or m15.close)
    m15_below_value = m15.close < float(m15.avwap_active or m15.profile_poc or m15.vwap or m15.ema20 or m15.close)
    bullish_reversal_evidence = bool(
        fast_break_up
        and (m15_bull_impulse or m15.momentum == "bullish")
        and (m15_above_value or m15.market_structure == "CHOCH_UP")
    )
    bearish_reversal_evidence = bool(
        fast_break_down
        and (m15_bear_impulse or m15.momentum == "bearish")
        and (m15_below_value or m15.market_structure == "CHOCH_DOWN")
    )

    strong_bull_stack = bool(
        bullish_tf_count >= 3
        and bearish_tf_count <= 1
        and h1.trend == "bullish"
        and h4.trend in {"bullish", "neutral"}
        and net >= 18
        and average_adx >= 22
    )
    strong_bear_stack = bool(
        bearish_tf_count >= 3
        and bullish_tf_count <= 1
        and h1.trend == "bearish"
        and h4.trend in {"bearish", "neutral"}
        and net <= -18
        and average_adx >= 22
    )
    bullish_resolution = bool(
        m15_bull_impulse
        and fast_break_up
        and (h1_bull_impulse or bull_dmi or h1_above_fast)
        and net >= -14
    )
    bearish_resolution = bool(
        m15_bear_impulse
        and fast_break_down
        and (h1_bear_impulse or bear_dmi or h1_below_fast)
        and net <= 14
    )

    bullish_confirmation = bool(
        bullish_resolution
        or (
            net >= 10
            and (
                (h1.trend == "bullish" and h4.trend in {"bullish", "neutral"} and (bull_momentum or bull_dmi))
                or (fast_break_up and (bull_momentum or m15_bull_impulse) and (h1_above_value or h1_bull_impulse))
                or (net >= 24 and bull_momentum and h1_above_value)
            )
        )
    )
    bearish_confirmation = bool(
        bearish_resolution
        or (
            net <= -10
            and (
                (h1.trend == "bearish" and h4.trend in {"bearish", "neutral"} and (bear_momentum or bear_dmi))
                or (fast_break_down and (bear_momentum or m15_bear_impulse) and (h1_below_value or h1_bear_impulse))
                or (net <= -24 and bear_momentum and h1_below_value)
            )
        )
    )

    # A strong multi-timeframe trend should remain a directional regime while
    # the engine waits for a safe entry.  A sweep alone is not enough to convert
    # four aligned bearish/bullish timeframes into TRAP.  The opposite side must
    # also produce a real M15 structure reversal.
    if strong_bull_stack and not bearish_reversal_evidence:
        bullish_confirmation = True
        bearish_confirmation = False
        buy_reasons.append(
            f"{bullish_tf_count}/4 key timeframes are bullish with strong ADX; the trend stack overrides an unconfirmed sweep."
        )
    elif strong_bear_stack and not bullish_reversal_evidence:
        bearish_confirmation = True
        bullish_confirmation = False
        sell_reasons.append(
            f"{bearish_tf_count}/4 key timeframes are bearish with strong ADX; the trend stack overrides an unconfirmed sweep."
        )

    bull_trap_levels: dict[str, float] = liq.get("bull_trap_levels", {})
    bear_trap_levels: dict[str, float] = liq.get("bear_trap_levels", {})
    bull_trap_near = any(abs(price - level) <= h1_atr * 0.45 for level in bull_trap_levels.values())
    bear_trap_near = any(abs(price - level) <= h1_atr * 0.45 for level in bear_trap_levels.values())
    two_sided_fresh = bool(liq.get("two_sided_timeframes"))

    # Same-side sweeps support the opposite direction.  They only create a
    # temporary no-trade TRAP when price has not yet resolved and the attempted
    # move is directly contradicted.  A confirmed directional impulse always
    # overrides the trap label.
    liquidity_conflict = bool(
        (
            bull_trap_near
            and bullish_confirmation
            and bearish_reversal_evidence
            and not bullish_resolution
        )
        or (
            bear_trap_near
            and bearish_confirmation
            and bullish_reversal_evidence
            and not bearish_resolution
        )
    )
    unresolved_two_sided = bool(
        two_sided_fresh
        and abs(net) < 36
        and not bullish_resolution
        and not bearish_resolution
        and not bullish_confirmation
        and not bearish_confirmation
    )
    trap = bool(liquidity_conflict or unresolved_two_sided)

    trap_release_side: str | None = None
    if bullish_resolution:
        trap = False
        bullish_confirmation = True
        bearish_confirmation = False
        if previous_state == "TRAP":
            trap_release_side = "BUY"
            buy_reasons.append("The prior liquidity event resolved into a confirmed bullish structure break")
    elif bearish_resolution:
        trap = False
        bearish_confirmation = True
        bullish_confirmation = False
        if previous_state == "TRAP":
            trap_release_side = "SELL"
            sell_reasons.append("The prior liquidity event resolved into a confirmed bearish structure break")
    elif trap and previous_state == "TRAP" and trap_anchor_price is not None:
        displacement_r = (price - float(trap_anchor_price)) / h1_atr
        if trap_age >= 2:
            # A trap may block at most two refresh cycles without fresh evidence.
            trap = False
        elif displacement_r >= 0.45 and net >= 4 and (m15_bull_impulse or bull_momentum or bull_dmi):
            trap = False
            bullish_confirmation = True
            bearish_confirmation = False
            trap_release_side = "BUY"
            buy_reasons.append("Price displaced above the prior trap anchor and held bullish momentum")
        elif displacement_r <= -0.45 and net <= -4 and (m15_bear_impulse or bear_momentum or bear_dmi):
            trap = False
            bearish_confirmation = True
            bullish_confirmation = False
            trap_release_side = "SELL"
            sell_reasons.append("Price displaced below the prior trap anchor and held bearish momentum")

    stuck = (
        not bullish_confirmation
        and not bearish_confirmation
        and (
            (average_adx < 17.5 and average_chop > 61.0)
            or (compression_count >= 2 and average_adx < 20)
            or (h1.trend == "neutral" and h4.trend == "neutral" and abs(net) < 8)
        )
    )
    if atr_pct >= 0.85:
        volatility_state = "extreme"
    elif atr_pct >= 0.48:
        volatility_state = "high"
    elif atr_pct <= 0.16:
        volatility_state = "low"
    else:
        volatility_state = "normal"

    trap_reason = ""
    if trap:
        market_state = "TRAP"
        regime = "liquidity_trap"
        signal_label = "NO TRADE · FRESH LIQUIDITY CONFLICT"
        if unresolved_two_sided:
            trap_reason = "Both sides were swept in the same active timeframe and direction has not yet resolved."
        elif bull_trap_near:
            trap_reason = "A fresh buy-side sweep conflicts with the attempted bullish move."
        else:
            trap_reason = "A fresh sell-side sweep conflicts with the attempted bearish move."
    elif bullish_confirmation:
        market_state = "BUY"
        regime = "breakout_up" if breakout_up or trap_release_side == "BUY" else (
            "volatile_bullish" if volatility_state in {"high", "extreme"} else "bullish_trend"
        )
        if trap_release_side == "BUY":
            signal_label = "ENTER BUY · TRAP RESOLVED"
        elif breakout_up:
            signal_label = "ENTER BUY · BREAKOUT CONFIRMED"
            bullish += 7
        else:
            signal_label = "ENTER BUY · TREND CONFIRMED"
    elif bearish_confirmation:
        market_state = "SELL"
        regime = "breakout_down" if breakout_down or trap_release_side == "SELL" else (
            "volatile_bearish" if volatility_state in {"high", "extreme"} else "bearish_trend"
        )
        if trap_release_side == "SELL":
            signal_label = "ENTER SELL · TRAP RESOLVED"
        elif breakout_down:
            signal_label = "ENTER SELL · BREAKOUT CONFIRMED"
            bearish += 7
        else:
            signal_label = "ENTER SELL · TREND CONFIRMED"
    elif stuck:
        market_state = "STUCK"
        regime = "stuck_range"
        signal_label = "NO TRADE · MARKET STUCK"
        trap_reason = "Trend strength is weak and price is compressed or balanced."
    elif breakout_up and net >= 4:
        market_state = "BUY"
        regime = "breakout_up"
        signal_label = "ENTER BUY · EARLY BREAKOUT"
        bullish += 5
    elif breakout_down and net <= -4:
        market_state = "SELL"
        regime = "breakout_down"
        signal_label = "ENTER SELL · EARLY BREAKOUT"
        bearish += 5
    elif net >= 8:
        market_state = "BUY"
        regime = "volatile_bullish" if volatility_state in {"high", "extreme"} else "bullish_trend"
        signal_label = "ENTER BUY · DIRECTIONAL BIAS"
    elif net <= -8:
        market_state = "SELL"
        regime = "volatile_bearish" if volatility_state in {"high", "extreme"} else "bearish_trend"
        signal_label = "ENTER SELL · DIRECTIONAL BIAS"
    else:
        market_state = "STUCK"
        regime = "stuck_range"
        signal_label = "NO TRADE · INSUFFICIENT EDGE"
        trap_reason = "Directional evidence is too balanced to justify a trade."

    # Macro is an execution gate, not a decorative score. When enabled and
    # required, incomplete or conflicting macro data cannot produce a high-
    # confidence entry. Mixed macro can only pass an exceptionally strong,
    # aligned technical trend.
    if market_state in {"BUY", "SELL"} and macro is not None:
        technical_alignment = (
            (market_state == "BUY" and h1.trend == "bullish" and h4.trend == "bullish")
            or (market_state == "SELL" and h1.trend == "bearish" and h4.trend == "bearish")
        )
        if macro_required_for_entry and (macro.gate == "UNAVAILABLE" or macro.coverage_score < 80):
            trap_reason = "DXY/yield/gold-flow coverage is incomplete, so the execution gate blocks a directional entry."
            market_state = "STUCK"
            regime = "stuck_range"
            signal_label = "NO TRADE · MACRO DATA INCOMPLETE"
        elif macro.gate == "CONFLICT":
            trap_reason = f"Technical {market_state} conflicts with DXY/US10Y/gold-flow confirmation."
            market_state = "STUCK"
            regime = "stuck_range"
            signal_label = "NO TRADE · MACRO CONFLICT"
        elif macro.gate == "NEUTRAL" and macro_required_for_entry:
            strong_stack = strong_bull_stack if market_state == "BUY" else strong_bear_stack
            # Mixed DXY/yield data should reduce confidence, not erase a clear
            # 3-to-4 timeframe trend.  It still blocks weak or poorly aligned
            # entries, but no longer turns an obvious 93/7 directional stack
            # into STUCK merely because macro is neutral.
            if (not strong_stack) and (abs(net) < 30 or average_adx < 24 or not technical_alignment):
                trap_reason = "DXY and yield are mixed; technical strength is not exceptional enough to justify execution."
                market_state = "STUCK"
                regime = "stuck_range"
                signal_label = "NO TRADE · MACRO MIXED"

    net = bullish - bearish
    buy_score = int(round(50 + math.tanh(net / 28.0) * 45))
    buy_score = max(5, min(95, buy_score))
    sell_score = 100 - buy_score
    trend_strength = int(round(max(0, min(100, average_adx * 2.4 + abs(net) * 0.30))))
    if market_state in {"STUCK", "TRAP"}:
        confidence = int(round(max(55, min(88, 57 + average_chop * 0.25 + compression_count * 4))))
    else:
        directional_score = buy_score if market_state == "BUY" else sell_score
        confidence = int(round(max(52, min(88, 42 + directional_score * 0.34 + average_adx * 0.45))))
        if macro is not None and macro.gate == "CONFIRM":
            confidence = min(92, confidence + 8)
        elif macro is not None and macro.gate == "NEUTRAL":
            confidence = min(74, max(50, confidence - 5))
        elif macro is not None and macro.gate in {"CONFLICT", "UNAVAILABLE"}:
            confidence = max(50, confidence - 16)

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

    # The terminal now separates directional regime from execution timing.
    # BUY/SELL can remain visible instead of falling back to TRAP/STUCK, while
    # Telegram remains silent until the current live price reaches the entry.
    if active_setup is not None and active_setup.status == "ENTER":
        entry_tolerance = max(0.12 * m15_atr, 0.25)
        entry_live = (
            active_setup.entry_low - entry_tolerance
            <= price
            <= active_setup.entry_high + entry_tolerance
        )
        if entry_live:
            signal_label = f"{signal_label} · ENTRY LIVE NOW"
        else:
            signal_label = f"{signal_label} · WAIT FOR ENTRY ZONE"
            active_setup.warnings.append(
                "Directional setup is active, but no alert is sent until the live price reaches the displayed entry zone."
            )

    notes = [
        "The decision engine prioritizes multi-timeframe market structure, anchored VWAP and volume-profile acceptance, then confirms with momentum, DMI/ADX, ATR, liquidity and the DXY/US10Y/gold-flow execution gate.",
        "Stops remain outside structural invalidation. Position size is reduced when that stop is too expensive; the engine does not invent a dangerously tight stop to accommodate a large lot.",
        "Targets begin with a conservative 0.65R / 1.05R / 1.40R ladder and adapt only from clean pre-exit excursion evidence.",
        "Adaptive learning is bounded, evidence-weighted and based on completed signals; it cannot guarantee future accuracy.",
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
        data_quality_notes=notes,
    )
