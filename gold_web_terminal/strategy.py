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

        trend_component = abs(item.directional_score) * tf_weight * _weight(adaptive_weights, "ema_trend")
        if item.directional_score > 0:
            bullish += trend_component
        elif item.directional_score < 0:
            bearish += trend_component
        if item.trend == "bullish":
            buy_reasons.append(f"{item.timeframe}: EMA structure and slope favor buyers")
        elif item.trend == "bearish":
            sell_reasons.append(f"{item.timeframe}: EMA structure and slope favor sellers")

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

        super_points = 6 * tf_weight * _weight(adaptive_weights, "ema_trend")
        if item.supertrend_direction == "bullish":
            bullish += super_points
        elif item.supertrend_direction == "bearish":
            bearish += super_points

        vwap_points = 5 * tf_weight * _weight(adaptive_weights, "vwap")
        if item.vwap is not None:
            if item.close > item.vwap:
                bullish += vwap_points
            else:
                bearish += vwap_points

        if item.volume_ratio is not None and item.volume_ratio >= 1.05:
            volume_points = min(7, 3.5 * item.volume_ratio) * tf_weight * _weight(adaptive_weights, "volume")
            if (item.volume_delta_proxy or 0) > 0:
                bullish += volume_points
                buy_reasons.append(f"{item.timeframe}: activity expanded with positive candle pressure")
            elif (item.volume_delta_proxy or 0) < 0:
                bearish += volume_points
                sell_reasons.append(f"{item.timeframe}: activity expanded with negative candle pressure")

        breakout_points = 9 * tf_weight * _weight(adaptive_weights, "breakout")
        if item.breakout_up:
            bullish += breakout_points
            buy_reasons.append(f"{item.timeframe}: price broke the prior 20-bar high")
        if item.breakout_down:
            bearish += breakout_points
            sell_reasons.append(f"{item.timeframe}: price broke the prior 20-bar low")

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
    return {
        "supports": sorted(set(round(level, 5) for level in supports), reverse=True),
        "resistances": sorted(set(round(level, 5) for level in resistances)),
        "nearest_support": max(supports) if supports else None,
        "nearest_resistance": min(resistances) if resistances else None,
        "bull_traps": bull_traps,
        "bear_traps": bear_traps,
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
        for level in (m15.ema20, m15.vwap, h1.ema20):
            if level is not None and price - h1_atr * 0.85 <= level <= price:
                candidates.append(float(level))
        if candidates:
            level = max(candidates)
            if price - level <= m15_atr * 0.25:
                return price, "CURRENT PRICE AT SUPPORT/VALUE"
            return level + m15_atr * 0.05, "PULLBACK TO NEAREST SUPPORT/VALUE"
    else:
        candidates = [level for level in resistances if price <= level <= price + h1_atr * 0.85]
        for level in (m15.ema20, m15.vwap, h1.ema20):
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
) -> TechnicalReport:
    if not indicators:
        raise ValueError("At least one indicator snapshot is required.")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("The supplied market price is invalid.")

    weights = {**DEFAULT_ADAPTIVE_WEIGHTS, **(adaptive_weights or {})}
    targets = {"tp1": 0.80, "tp2": 1.30, "tp3": 1.80, **(target_multipliers or {})}
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
    strong_bull_trap = bool(liq["bull_traps"] and h1.momentum != "bullish")
    strong_bear_trap = bool(liq["bear_traps"] and h1.momentum != "bearish")
    both_sides_swept = bool(liq["bull_traps"] and liq["bear_traps"])

    stuck = (
        (average_adx < 17.5 and average_chop > 61.0)
        or (compression_count >= 2 and average_adx < 20)
        or (h1.trend == "neutral" and h4.trend == "neutral" and abs(bullish - bearish) < 8)
    )
    trap = both_sides_swept or (strong_bull_trap and strong_bear_trap)
    breakout_up = bool(stats["breakout_up"] and (m15.volume_ratio or 0) >= 1.05)
    breakout_down = bool(stats["breakout_down"] and (m15.volume_ratio or 0) >= 1.05)
    net = bullish - bearish

    if not trap and strong_bull_trap and net > -5:
        trap = True
    if not trap and strong_bear_trap and net < 5:
        trap = True

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
        signal_label = "NO TRADE · LIQUIDITY TRAP"
        if both_sides_swept:
            trap_reason = "Liquidity was taken on both sides; the apparent direction is vulnerable to stop-hunting."
        elif strong_bull_trap:
            trap_reason = "Price swept buy-side liquidity/resistance and failed back below it."
        else:
            trap_reason = "Price swept sell-side liquidity/support and failed to hold the break."
    elif stuck:
        market_state = "STUCK"
        regime = "stuck_range"
        signal_label = "NO TRADE · MARKET STUCK"
        trap_reason = "ADX is weak and choppiness/compression indicate a range."
    elif breakout_up and net >= -4:
        market_state = "BUY"
        regime = "breakout_up"
        signal_label = "ENTER BUY · BREAKOUT"
        bullish += 7
    elif breakout_down and net <= 4:
        market_state = "SELL"
        regime = "breakout_down"
        signal_label = "ENTER SELL · BREAKOUT"
        bearish += 7
    elif net >= 0:
        market_state = "BUY"
        regime = "volatile_bullish" if volatility_state in {"high", "extreme"} else "bullish_trend"
        signal_label = "ENTER BUY · VOLATILE TREND" if "volatile" in regime else "ENTER BUY · TREND"
    else:
        market_state = "SELL"
        regime = "volatile_bearish" if volatility_state in {"high", "extreme"} else "bearish_trend"
        signal_label = "ENTER SELL · VOLATILE TREND" if "volatile" in regime else "ENTER SELL · TREND"

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
            market_state = "TRAP"
            regime = "liquidity_trap"
            signal_label = "NO TRADE · TECHNICAL/MACRO CONFLICT"
        elif macro.gate == "NEUTRAL" and macro_required_for_entry:
            if abs(net) < 38 or average_adx < 27 or not technical_alignment:
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

    notes = [
        "The decision engine combines multi-timeframe trend, momentum, DMI/ADX, ATR, VWAP, volume/activity, support/resistance, liquidity and a required DXY/US10Y/gold-flow execution gate.",
        "Stops remain outside structural invalidation. Position size is reduced when that stop is too expensive; the engine does not invent a dangerously tight stop to accommodate a large lot.",
        "Targets are capped by reachable ATR range and adaptive historical maximum-favourable-excursion statistics.",
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
