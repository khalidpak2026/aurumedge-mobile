from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from .models import IndicatorSnapshot, LiquiditySnapshot, TechnicalReport, TradeSetup


TF_WEIGHTS = {"M5": 0.08, "M15": 0.18, "H1": 0.31, "H4": 0.31, "D1": 0.12}


def _round_price(value: float, digits: int) -> float:
    return round(float(value), digits)


def _rr(entry: float, stop: float, target: float) -> float:
    risk = abs(entry - stop)
    return 0.0 if risk <= 0 else round(abs(target - entry) / risk, 2)


def _select_snapshot(indicators: list[IndicatorSnapshot], timeframe: str) -> IndicatorSnapshot:
    return next((item for item in indicators if item.timeframe == timeframe), indicators[0])


def _liquidity_by_tf(liquidity: list[LiquiditySnapshot], timeframe: str) -> LiquiditySnapshot | None:
    return next((item for item in liquidity if item.timeframe == timeframe), None)


def _weighted_market_scores(indicators: list[IndicatorSnapshot]) -> tuple[float, float, list[str], list[str]]:
    bullish = 0.0
    bearish = 0.0
    buy_reasons: list[str] = []
    sell_reasons: list[str] = []
    for item in indicators:
        weight = TF_WEIGHTS.get(item.timeframe, 0.1)
        raw = item.directional_score
        if raw > 0:
            bullish += raw * weight
        elif raw < 0:
            bearish += abs(raw) * weight

        if item.trend == "bullish":
            buy_reasons.append(f"{item.timeframe}: EMA structure and slope favor buyers")
        elif item.trend == "bearish":
            sell_reasons.append(f"{item.timeframe}: EMA structure and slope favor sellers")

        if item.momentum == "bullish":
            bullish += 10 * weight
            buy_reasons.append(f"{item.timeframe}: MACD, RSI and DMI momentum are aligned up")
        elif item.momentum == "bearish":
            bearish += 10 * weight
            sell_reasons.append(f"{item.timeframe}: MACD, RSI and DMI momentum are aligned down")

        if item.supertrend_direction == "bullish":
            bullish += 7 * weight
        elif item.supertrend_direction == "bearish":
            bearish += 7 * weight

        if item.close and item.vwap is not None:
            if item.close > item.vwap:
                bullish += 5 * weight
            else:
                bearish += 5 * weight

        if item.volume_ratio is not None and item.volume_ratio >= 1.15:
            if item.volume_delta_proxy is not None and item.volume_delta_proxy > 0:
                bullish += min(7, 4 * item.volume_ratio) * weight
                buy_reasons.append(f"{item.timeframe}: activity expanded with positive candle pressure")
            elif item.volume_delta_proxy is not None and item.volume_delta_proxy < 0:
                bearish += min(7, 4 * item.volume_ratio) * weight
                sell_reasons.append(f"{item.timeframe}: activity expanded with negative candle pressure")

        if item.breakout_up:
            bullish += 10 * weight
            buy_reasons.append(f"{item.timeframe}: price broke the prior 20-bar high")
        if item.breakout_down:
            bearish += 10 * weight
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


def _liquidity_context(liquidity: list[LiquiditySnapshot], price: float) -> dict:
    supports: list[float] = []
    resistances: list[float] = []
    bull_traps: list[str] = []
    bear_traps: list[str] = []
    for item in liquidity:
        for zone in item.support_zones:
            center = float(zone.get("center", 0))
            if center < price:
                supports.append(center)
        for zone in item.resistance_zones:
            center = float(zone.get("center", 0))
            if center > price:
                resistances.append(center)
        for level in (item.previous_day_low, item.value_area_low, item.point_of_control):
            if level is not None and level < price:
                supports.append(float(level))
        for level in (item.previous_day_high, item.value_area_high, item.point_of_control):
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


def _target_from_levels(levels: list[float], minimum: float, fallback: float, direction: str) -> float:
    if direction == "up":
        eligible = [level for level in levels if level >= minimum]
        return min(eligible) if eligible else fallback
    eligible = [level for level in levels if level <= minimum]
    return max(eligible) if eligible else fallback


def _build_setup(
    side: str,
    price: float,
    atr_value: float,
    confidence: int,
    supports: list[float],
    resistances: list[float],
    digits: int,
    active: bool,
    reasons: list[str],
    data_source: str,
) -> TradeSetup:
    entry_mid = price
    entry_half_width = atr_value * 0.08
    entry_low = entry_mid - entry_half_width
    entry_high = entry_mid + entry_half_width
    common_warnings = [
        "This is an indicative web-data setup, not an executable broker quote.",
        "Spread, slippage and the broker's XAUUSD contract specification are not included.",
    ]
    if data_source == "DEMO":
        common_warnings.append("Demo data is synthetic and must not be used for live trading.")

    if side == "BUY":
        support = max((level for level in supports if level < price), default=None)
        structural_stop = support - atr_value * 0.22 if support is not None else price - atr_value * 1.25
        volatility_stop = price - atr_value * 1.20
        stop = min(structural_stop, volatility_stop)
        stop = max(stop, price - atr_value * 2.10)
        risk = max(price - stop, atr_value * 0.85)
        tp1 = _target_from_levels(resistances, price + risk * 0.80, price + risk * 1.05, "up")
        tp1 = max(tp1, price + risk * 0.80)
        tp2 = _target_from_levels(resistances, price + risk * 1.55, price + risk * 1.85, "up")
        tp2 = max(tp2, tp1 + risk * 0.45)
        tp3 = _target_from_levels(resistances, price + risk * 2.25, price + risk * 2.75, "up")
        tp3 = max(tp3, tp2 + risk * 0.55)
        invalidation = f"Signal is invalid if H1 closes below {_round_price(stop, digits)}."
    else:
        resistance = min((level for level in resistances if level > price), default=None)
        structural_stop = resistance + atr_value * 0.22 if resistance is not None else price + atr_value * 1.25
        volatility_stop = price + atr_value * 1.20
        stop = max(structural_stop, volatility_stop)
        stop = min(stop, price + atr_value * 2.10)
        risk = max(stop - price, atr_value * 0.85)
        tp1 = _target_from_levels(supports, price - risk * 0.80, price - risk * 1.05, "down")
        tp1 = min(tp1, price - risk * 0.80)
        tp2 = _target_from_levels(supports, price - risk * 1.55, price - risk * 1.85, "down")
        tp2 = min(tp2, tp1 - risk * 0.45)
        tp3 = _target_from_levels(supports, price - risk * 2.25, price - risk * 2.75, "down")
        tp3 = min(tp3, tp2 - risk * 0.55)
        invalidation = f"Signal is invalid if H1 closes above {_round_price(stop, digits)}."

    valid_until = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M UTC")
    return TradeSetup(
        side=side,  # type: ignore[arg-type]
        status="ENTER" if active else "NO_TRADE",
        confidence=confidence,
        entry_low=_round_price(entry_low, digits),
        entry_high=_round_price(entry_high, digits),
        entry_type="CURRENT MARKET ZONE" if active else "INACTIVE SCENARIO",
        stop_loss=_round_price(stop, digits),
        take_profit_1=_round_price(tp1, digits),
        take_profit_2=_round_price(tp2, digits),
        take_profit_3=_round_price(tp3, digits),
        risk_reward_1=_rr(entry_mid, stop, tp1),
        risk_reward_2=_rr(entry_mid, stop, tp2),
        risk_reward_3=_rr(entry_mid, stop, tp3),
        valid_until=valid_until,
        invalidation=invalidation,
        rationale=reasons[:10],
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
) -> TechnicalReport:
    if not indicators:
        raise ValueError("At least one indicator snapshot is required.")
    if not math.isfinite(price) or price <= 0:
        raise ValueError("The supplied market price is invalid.")

    bullish, bearish, buy_reasons, sell_reasons = _weighted_market_scores(indicators)
    stats = _market_stats(indicators)
    liq = _liquidity_context(liquidity, price)
    h1 = _select_snapshot(indicators, "H1")
    h4 = _select_snapshot(indicators, "H4")
    m15 = _select_snapshot(indicators, "M15")
    atr_value = h1.atr14 or m15.atr14 or h4.atr14 or price * 0.002
    atr_value = max(float(atr_value), point * 10)
    atr_pct = h1.atr_pct or (atr_value / price * 100)

    # Liquidity confirmation and trap penalties.
    if liq["bear_traps"] and h1.momentum == "bullish":
        bullish += 9
        buy_reasons.append("Sell-side liquidity was swept and reclaimed")
    if liq["bull_traps"] and h1.momentum == "bearish":
        bearish += 9
        sell_reasons.append("Buy-side liquidity was swept and rejected")

    average_adx = float(stats["average_adx"])
    average_chop = float(stats["average_chop"])
    compression_count = int(stats["compressions"])
    strong_bull_trap = bool(liq["bull_traps"] and h1.momentum != "bullish")
    strong_bear_trap = bool(liq["bear_traps"] and h1.momentum != "bearish")
    both_sides_swept = bool(liq["bull_traps"] and liq["bear_traps"])

    stuck = (
        (average_adx < 18.5 and average_chop > 59.5)
        or (compression_count >= 2 and average_adx < 21)
        or (h1.trend == "neutral" and h4.trend == "neutral" and abs(bullish - bearish) < 9)
    )
    trap = both_sides_swept or (strong_bull_trap and strong_bear_trap)

    breakout_up = bool(stats["breakout_up"] and m15.volume_ratio is not None and m15.volume_ratio >= 1.05)
    breakout_down = bool(stats["breakout_down"] and m15.volume_ratio is not None and m15.volume_ratio >= 1.05)
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
            trap_reason = "Liquidity was taken on both sides, indicating stop-hunting and unreliable direction."
        elif strong_bull_trap:
            trap_reason = "Price swept resistance/buy-side liquidity and failed back below it."
        else:
            trap_reason = "Price swept support/sell-side liquidity and failed to hold the break."
    elif stuck:
        market_state = "STUCK"
        regime = "stuck_range"
        signal_label = "NO TRADE · MARKET STUCK"
        trap_reason = "ADX is weak and/or choppiness and volatility compression indicate a range."
    elif breakout_up and net >= -4:
        market_state = "BUY"
        regime = "breakout_up"
        signal_label = "ENTER BUY · BREAKOUT"
        bullish += 8
        buy_reasons.append("M15/H1 breakout is supported by expanding activity")
    elif breakout_down and net <= 4:
        market_state = "SELL"
        regime = "breakout_down"
        signal_label = "ENTER SELL · BREAKOUT"
        bearish += 8
        sell_reasons.append("M15/H1 breakdown is supported by expanding activity")
    elif net >= 0:
        market_state = "BUY"
        if volatility_state in {"high", "extreme"}:
            regime = "volatile_bullish"
            signal_label = "ENTER BUY · VOLATILE TREND"
        else:
            regime = "bullish_trend"
            signal_label = "ENTER BUY · TREND"
    else:
        market_state = "SELL"
        if volatility_state in {"high", "extreme"}:
            regime = "volatile_bearish"
            signal_label = "ENTER SELL · VOLATILE TREND"
        else:
            regime = "bearish_trend"
            signal_label = "ENTER SELL · TREND"

    # Normalize the directional scores after all adjustments.
    scale = max(1.0, bullish + bearish)
    buy_score = int(round(max(0, min(100, 50 + (bullish - bearish) / scale * 50))))
    sell_score = 100 - buy_score
    trend_strength = int(round(max(0, min(100, average_adx * 2.7 + abs(net) * 0.35))))
    if market_state in {"STUCK", "TRAP"}:
        confidence = int(round(max(55, min(95, 58 + average_chop * 0.35 + compression_count * 5))))
    else:
        confidence = int(round(max(51, min(94, 52 + abs(net) * 0.75 + average_adx * 0.45))))

    active_buy = market_state == "BUY"
    active_sell = market_state == "SELL"
    buy_setup = _build_setup(
        "BUY",
        price,
        atr_value,
        buy_score,
        liq["supports"],
        liq["resistances"],
        digits,
        active_buy,
        buy_reasons,
        data_source,
    )
    sell_setup = _build_setup(
        "SELL",
        price,
        atr_value,
        sell_score,
        liq["supports"],
        liq["resistances"],
        digits,
        active_sell,
        sell_reasons,
        data_source,
    )
    active_setup = buy_setup if active_buy else sell_setup if active_sell else None

    notes = [
        "The signal engine combines multi-timeframe EMA alignment, slopes, Supertrend, MACD, RSI, ADX/DMI, ATR, VWAP, activity/volume, Choppiness, Donchian breakout, support/resistance, volume profile and liquidity sweeps.",
        "BUY or SELL is produced when the market is directional. STUCK or TRAP is produced only when range/compression or false-break conditions dominate.",
        "No indicator combination can guarantee a perfect entry or eliminate losses.",
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
        data_quality_notes=notes,
    )
