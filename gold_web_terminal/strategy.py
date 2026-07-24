from __future__ import annotations

"""AurumEdge v5.8.1 three-pillar decision engine.

Only market structure, anchored VWAP and volume profile vote BUY or SELL.
ATR is used only for entry tolerance, invalidation, targets and risk geometry.
Macro, EMA, RSI, MACD, ADX/DMI, raw volume, FVG and liquidity labels cannot
block, reverse or create a directional signal in this module.
"""

from datetime import datetime, timedelta, timezone
import math
from statistics import median
from typing import Any, Iterable

try:  # Existing repository model; kept untouched by this patch.
    from .models import TechnicalReport
except Exception:  # pragma: no cover - compatibility fallback for validation.
    TechnicalReport = None  # type: ignore[assignment]


PILLARS = ("market_structure", "anchored_vwap", "volume_profile")
LEGACY_FEATURES = (
    "ema_trend",
    "rsi",
    "macd",
    "adx_dmi",
    "ordinary_volume",
    "supertrend",
    "liquidity",
    "fvg",
    "macro",
)


class AttrMap(dict):
    """JSON-friendly mapping with attribute access for old UI/model code."""

    __getattr__ = dict.get

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def _finite(value: Any, default: float | None = None) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, dict) and name in obj:
            value = obj[name]
        else:
            value = getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _number(obj: Any, *names: str, default: float | None = None) -> float | None:
    return _finite(_get(obj, *names, default=default), default)


def _text(obj: Any, *names: str, default: str = "") -> str:
    value = _get(obj, *names, default=default)
    return str(value or default).strip()


def _snapshot(items: Iterable[Any], timeframe: str) -> Any | None:
    tf = timeframe.upper()
    return next((item for item in items if _text(item, "timeframe").upper() == tf), None)


def _construct_report(values: dict[str, Any]) -> Any:
    """Build the repository TechnicalReport without requiring model changes."""
    if TechnicalReport is None:
        return AttrMap(values)
    try:
        if hasattr(TechnicalReport, "model_construct"):
            report = TechnicalReport.model_construct(**values)
        else:  # Pydantic v1
            report = TechnicalReport.construct(**values)
        for key, value in values.items():
            try:
                object.__setattr__(report, key, value)
            except Exception:
                try:
                    setattr(report, key, value)
                except Exception:
                    pass
        return report
    except Exception:
        return AttrMap(values)


def _structure_vote(item: Any) -> int:
    bias = _text(item, "structure_bias", "swing_bias", "market_bias").lower()
    state = _text(item, "market_structure", "structure_state", "structure").upper()
    if bias in {"bullish", "buy", "up"} or any(token in state for token in ("BOS_UP", "CHOCH_UP", "HH", "HL", "BULL")):
        return 1
    if bias in {"bearish", "sell", "down"} or any(token in state for token in ("BOS_DOWN", "CHOCH_DOWN", "LH", "LL", "BEAR")):
        return -1
    return 0


def _avwap_value(item: Any) -> float | None:
    return _number(
        item,
        "anchored_vwap",
        "active_anchored_vwap",
        "active_avwap",
        "avwap",
        "swing_anchored_vwap",
        "vwap",
    )


def _avwap_slope(item: Any) -> float:
    return float(
        _number(
            item,
            "anchored_vwap_slope",
            "active_avwap_slope",
            "avwap_slope",
            "vwap_slope",
            default=0.0,
        )
        or 0.0
    )


def _avwap_vote(item: Any, price: float) -> int:
    explicit = _text(item, "anchored_vwap_bias", "avwap_bias").lower()
    if explicit in {"bullish", "buy", "above", "up"}:
        return 1
    if explicit in {"bearish", "sell", "below", "down"}:
        return -1
    value = _avwap_value(item)
    if value is None:
        return 0
    slope = _avwap_slope(item)
    tolerance = max(abs(price) * 0.00003, 0.05)
    if price > value + tolerance and slope >= -tolerance * 0.1:
        return 1
    if price < value - tolerance and slope <= tolerance * 0.1:
        return -1
    if slope > tolerance * 0.1:
        return 1
    if slope < -tolerance * 0.1:
        return -1
    return 0


def _profile_values(item: Any) -> tuple[float | None, float | None, float | None]:
    poc = _number(item, "profile_poc", "volume_profile_poc", "poc", "value_poc")
    vah = _number(item, "profile_vah", "volume_profile_vah", "vah", "value_area_high")
    val = _number(item, "profile_val", "volume_profile_val", "val", "value_area_low")
    return poc, vah, val


def _profile_vote(item: Any, price: float) -> int:
    explicit = _text(item, "profile_bias", "volume_profile_bias", "profile_acceptance").lower()
    if any(token in explicit for token in ("above", "bull", "buy", "up acceptance")):
        return 1
    if any(token in explicit for token in ("below", "bear", "sell", "down acceptance")):
        return -1
    poc, vah, val = _profile_values(item)
    if vah is not None and price > vah:
        return 1
    if val is not None and price < val:
        return -1
    if poc is not None:
        tolerance = max(abs(price) * 0.00003, 0.05)
        if price > poc + tolerance:
            return 1
        if price < poc - tolerance:
            return -1
    return 0


def _timeframe_votes(item: Any, price: float) -> dict[str, int]:
    if item is None:
        return {name: 0 for name in PILLARS}
    return {
        "market_structure": _structure_vote(item),
        "anchored_vwap": _avwap_vote(item, price),
        "volume_profile": _profile_vote(item, price),
    }


def _aggregate_pillars(indicators: list[Any], price: float) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    per_tf = {
        tf: _timeframe_votes(_snapshot(indicators, tf), price)
        for tf in ("M15", "H1")
    }
    result: dict[str, int] = {}
    for pillar in PILLARS:
        m15 = per_tf["M15"][pillar]
        h1 = per_tf["H1"][pillar]
        total = m15 + h1
        if total > 0:
            result[pillar] = 1
        elif total < 0:
            result[pillar] = -1
        else:
            result[pillar] = h1 or m15
    return result, per_tf


def _adaptive_weight(weights: dict[str, float] | None, pillar: str) -> float:
    if not weights:
        return 1.0
    value = _finite(weights.get(pillar), 1.0)
    return max(0.70, min(1.30, float(value or 1.0)))


def _direction(pillars: dict[str, int], weights: dict[str, float] | None) -> tuple[str, float, float]:
    buy = sum(_adaptive_weight(weights, key) for key, vote in pillars.items() if vote > 0)
    sell = sum(_adaptive_weight(weights, key) for key, vote in pillars.items() if vote < 0)
    buy_count = sum(vote > 0 for vote in pillars.values())
    sell_count = sum(vote < 0 for vote in pillars.values())
    if buy_count >= 2 and buy > sell:
        return "BUY", buy, sell
    if sell_count >= 2 and sell > buy:
        return "SELL", buy, sell
    return "STUCK", buy, sell


def _atr(indicators: list[Any], price: float) -> float:
    for tf in ("M15", "H1", "M5"):
        item = _snapshot(indicators, tf)
        value = _number(item, "atr14", "atr", "average_true_range") if item is not None else None
        if value and value > 0:
            return float(value)
    return max(abs(price) * 0.00035, 0.75)


def _level_candidates(item: Any) -> dict[str, float]:
    if item is None:
        return {}
    candidates: dict[str, float] = {}
    mapping = {
        "avwap": ("anchored_vwap", "active_anchored_vwap", "active_avwap", "avwap", "vwap"),
        "poc": ("profile_poc", "volume_profile_poc", "poc"),
        "vah": ("profile_vah", "volume_profile_vah", "vah", "value_area_high"),
        "val": ("profile_val", "volume_profile_val", "val", "value_area_low"),
        "swing_high": ("last_swing_high", "swing_high", "structure_high", "recent_swing_high"),
        "swing_low": ("last_swing_low", "swing_low", "structure_low", "recent_swing_low"),
    }
    for name, aliases in mapping.items():
        value = _number(item, *aliases)
        if value is not None:
            candidates[name] = float(value)
    return candidates


def _risk_plan(side: str, entry: float, stop: float, risk_inputs: Any | None) -> AttrMap:
    distance = max(abs(entry - stop), 0.01)
    balance = float(_number(risk_inputs, "account_balance", default=10000.0) or 10000.0)
    risk_percent = float(_number(risk_inputs, "risk_percent", default=1.0) or 1.0)
    requested = float(_number(risk_inputs, "requested_lot", default=0.10) or 0.10)
    contract = float(_number(risk_inputs, "contract_size", default=100.0) or 100.0)
    step = max(float(_number(risk_inputs, "lot_step", default=0.01) or 0.01), 0.001)
    minimum = max(float(_number(risk_inputs, "min_lot", default=0.01) or 0.01), step)
    max_dollars = float(_number(risk_inputs, "maximum_risk_dollars", default=0.0) or 0.0)
    risk_budget = balance * risk_percent / 100.0
    if max_dollars > 0:
        risk_budget = min(risk_budget, max_dollars)
    one_lot_loss = distance * contract
    raw_lot = risk_budget / one_lot_loss if one_lot_loss > 0 else 0.0
    recommended = math.floor(raw_lot / step + 1e-9) * step
    recommended = round(max(0.0, recommended), 4)
    estimated = requested * one_lot_loss
    if recommended < minimum:
        status = "BLOCK"
        recommended = 0.0
    elif requested <= recommended + step * 0.25:
        status = "OK"
    else:
        status = "REDUCE_LOT"
    return AttrMap(
        status=status,
        side=side,
        requested_lot=round(requested, 4),
        recommended_lot=recommended,
        estimated_loss_requested_lot=round(estimated, 2),
        risk_budget=round(risk_budget, 2),
        stop_distance=round(distance, 4),
    )


def _entry_geometry(
    side: str,
    price: float,
    indicators: list[Any],
    pillars: dict[str, int],
    per_tf: dict[str, dict[str, int]],
    risk_inputs: Any | None,
    targets: dict[str, float] | None,
    digits: int,
) -> AttrMap:
    m15 = _snapshot(indicators, "M15")
    h1 = _snapshot(indicators, "H1")
    atr = _atr(indicators, price)
    m15_levels = _level_candidates(m15)
    h1_levels = _level_candidates(h1)
    levels = {**h1_levels, **m15_levels}
    sign = 1 if side == "BUY" else -1
    aligned = sum(vote == sign for vote in pillars.values())
    tf_alignment = sum(per_tf[tf][pillar] == sign for tf in per_tf for pillar in PILLARS)
    avwap = m15_levels.get("avwap") or h1_levels.get("avwap")
    poc = m15_levels.get("poc") or h1_levels.get("poc")
    reference = median([value for value in (avwap, poc) if value is not None]) if any(value is not None for value in (avwap, poc)) else price
    extension = abs(price - reference) / max(atr, 1e-9)
    structure_text = _text(m15, "market_structure", "structure_state").upper()
    fresh_break = (side == "BUY" and any(token in structure_text for token in ("BOS_UP", "CHOCH_UP"))) or (
        side == "SELL" and any(token in structure_text for token in ("BOS_DOWN", "CHOCH_DOWN"))
    )
    continuation_live = aligned == 3 or (aligned >= 2 and tf_alignment >= 4 and extension <= 0.85)
    immediate = fresh_break or continuation_live
    zone_half = max(atr * 0.12, abs(price) * 0.00004, 0.12)
    if immediate:
        entry_mid = price
        setup_type = "BREAKOUT" if fresh_break else "CONTINUATION"
    else:
        if side == "BUY":
            supports = [
                value
                for key, value in levels.items()
                if key in {"avwap", "poc", "vah", "val", "swing_high", "swing_low"}
                and value <= price + atr * 0.20
            ]
            entry_mid = max(supports) if supports else price - atr * 0.35
        else:
            resistances = [
                value
                for key, value in levels.items()
                if key in {"avwap", "poc", "vah", "val", "swing_high", "swing_low"}
                and value >= price - atr * 0.20
            ]
            entry_mid = min(resistances) if resistances else price + atr * 0.35
        setup_type = "PULLBACK"
    entry_low = entry_mid - zone_half
    entry_high = entry_mid + zone_half

    if side == "BUY":
        below = [
            value
            for key, value in m15_levels.items()
            if key in {"avwap", "val", "poc", "swing_low"} and value < entry_mid
        ]
        structural = max(below) - atr * 0.10 if below else entry_mid - atr * 0.85
        raw_distance = entry_mid - structural
    else:
        above = [
            value
            for key, value in m15_levels.items()
            if key in {"avwap", "vah", "poc", "swing_high"} and value > entry_mid
        ]
        structural = min(above) + atr * 0.10 if above else entry_mid + atr * 0.85
        raw_distance = structural - entry_mid

    minimum_atr = float(_number(risk_inputs, "minimum_stop_atr", default=0.55) or 0.55)
    maximum_atr = float(_number(risk_inputs, "maximum_stop_atr", default=1.60) or 1.60)
    minimum_atr = max(0.45, minimum_atr)
    maximum_atr = max(minimum_atr + 0.10, min(1.80, maximum_atr))
    stop_distance = min(max(raw_distance, atr * minimum_atr), atr * maximum_atr)
    stop = entry_mid - stop_distance if side == "BUY" else entry_mid + stop_distance

    multipliers = {"tp1": 0.65, "tp2": 1.05, "tp3": 1.40}
    if targets:
        for key in multipliers:
            value = _finite(targets.get(key), multipliers[key])
            if value is not None:
                multipliers[key] = float(value)
    multipliers["tp1"] = max(0.60, min(0.90, multipliers["tp1"]))
    multipliers["tp2"] = max(multipliers["tp1"] + 0.20, min(1.35, multipliers["tp2"]))
    multipliers["tp3"] = max(multipliers["tp2"] + 0.20, min(1.80, multipliers["tp3"]))
    tp1 = entry_mid + sign * stop_distance * multipliers["tp1"]
    tp2 = entry_mid + sign * stop_distance * multipliers["tp2"]
    tp3 = entry_mid + sign * stop_distance * multipliers["tp3"]
    risk_plan = _risk_plan(side, entry_mid, stop, risk_inputs)
    entry_live = entry_low <= price <= entry_high and risk_plan.status != "BLOCK"
    near_tolerance = max(atr * 0.22, zone_half)
    near_entry = entry_low - near_tolerance <= price <= entry_high + near_tolerance
    status = "ENTER" if entry_live else "WAIT"
    now = datetime.now(timezone.utc)
    return AttrMap(
        side=side,
        status=status,
        setup_type=setup_type,
        entry_low=round(entry_low, digits),
        entry_high=round(entry_high, digits),
        entry_price=round(entry_mid, digits),
        stop_loss=round(stop, digits),
        take_profit_1=round(tp1, digits),
        take_profit_2=round(tp2, digits),
        take_profit_3=round(tp3, digits),
        risk_reward_1=round(multipliers["tp1"], 2),
        risk_reward_2=round(multipliers["tp2"], 2),
        risk_reward_3=round(multipliers["tp3"], 2),
        entry_live=entry_live,
        near_entry=near_entry,
        entry_tolerance=round(near_tolerance, digits),
        risk_plan=risk_plan,
        invalidation=(
            "M15 structure / AVWAP / value support failed"
            if side == "BUY"
            else "M15 structure / AVWAP / value resistance failed"
        ),
        valid_until=(now + timedelta(minutes=90)).isoformat(),
        atr=round(atr, digits),
        pillar_votes=dict(pillars),
    )


def derive_feature_votes(
    indicators: list[Any],
    liquidity: list[Any] | None = None,
    macro: Any | None = None,
    market_state: str | None = None,
) -> dict[str, int]:
    """Return only the three directional feature votes; legacy votes are neutral."""
    price = 0.0
    for tf in ("M15", "H1", "M5"):
        item = _snapshot(indicators, tf)
        price = float(_number(item, "last_price", "close", "price", default=0.0) or 0.0)
        if price:
            break
    pillars, _ = _aggregate_pillars(indicators, price)
    votes = {name: int(pillars[name]) for name in PILLARS}
    votes.update({name: 0 for name in LEGACY_FEATURES})
    return votes


def build_technical_report(
    symbol: str,
    data_time: str,
    price: float,
    indicators: list[Any],
    liquidity: list[Any] | None = None,
    data_source: str = "LIVE",
    digits: int = 2,
    adaptive_weights: dict[str, float] | None = None,
    target_multipliers: dict[str, float] | None = None,
    adaptive_summary: Any | None = None,
    risk_inputs: Any | None = None,
    macro: Any | None = None,
    macro_required_for_entry: bool = False,
    **_: Any,
) -> Any:
    """Build a report whose direction is decided exclusively by the three pillars."""
    last_price = float(price)
    pillars, per_tf = _aggregate_pillars(indicators, last_price)
    state, buy_weight, sell_weight = _direction(pillars, adaptive_weights)
    active_setup = None
    reasons: list[str] = []
    for pillar in PILLARS:
        vote = pillars[pillar]
        reasons.append(f"{pillar.replace('_', ' ').title()}: {'BUY' if vote > 0 else 'SELL' if vote < 0 else 'NEUTRAL'}")
    aligned = sum(vote != 0 and ((state == "BUY" and vote > 0) or (state == "SELL" and vote < 0)) for vote in pillars.values())
    tf_aligned = 0
    if state in {"BUY", "SELL"}:
        sign = 1 if state == "BUY" else -1
        tf_aligned = sum(per_tf[tf][pillar] == sign for tf in per_tf for pillar in PILLARS)
        active_setup = _entry_geometry(
            state,
            last_price,
            indicators,
            pillars,
            per_tf,
            risk_inputs,
            target_multipliers,
            digits,
        )
    confidence = 35 if state == "STUCK" else min(92, 52 + aligned * 10 + tf_aligned * 3)
    if active_setup is not None and active_setup.risk_plan.status == "BLOCK":
        confidence = min(confidence, 68)
    buy_score = round(buy_weight / 3.0 * 100)
    sell_score = round(sell_weight / 3.0 * 100)
    trap_reason = "Three pillars are not sufficiently aligned." if state == "STUCK" else ""
    execution_label = (
        "NO TRADE · THREE PILLARS NOT ALIGNED"
        if state == "STUCK"
        else f"ENTER {state} · ENTRY LIVE NOW"
        if active_setup and active_setup.status == "ENTER"
        else f"{state} TREND · WAIT FOR ENTRY"
    )
    values = {
        "symbol": symbol,
        "data_time": data_time,
        "last_price": round(last_price, digits),
        "market_state": state,
        "confidence": int(confidence),
        "buy_score": int(buy_score),
        "sell_score": int(sell_score),
        "regime": f"{state}_THREE_PILLAR" if state != "STUCK" else "THREE_PILLAR_MIXED",
        "volatility_state": "normal",
        "trap_reason": trap_reason,
        "active_setup": active_setup,
        "data_source": data_source,
        "reasons": reasons,
        "indicator_snapshots": indicators,
        "indicators": indicators,
        "liquidity_snapshots": liquidity or [],
        "liquidity": liquidity or [],
        "macro": macro,
        "special_signals": [],
        "pillar_votes": pillars,
        "pillar_timeframe_votes": per_tf,
        "execution_label": execution_label,
        "entry_live": bool(active_setup and active_setup.entry_live),
        "near_entry": bool(active_setup and active_setup.near_entry),
        "adaptive_summary": adaptive_summary,
        "macro_required_for_entry": False,
    }
    return _construct_report(values)
