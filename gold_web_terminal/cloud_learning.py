from __future__ import annotations

"""Persistent 24/7 learning helpers for the scheduled AurumEdge watcher.

This module deliberately keeps specialist (4H + M15 FVG) statistics separate
from the regular AurumEdge model.  It stores both groups in the same
``adaptive_state.json`` file without changing the public models used by the
Streamlit application.
"""

from copy import deepcopy
from datetime import datetime, timezone
import math
from typing import Any

import numpy as np
import pandas as pd

from .adaptive_engine import AdaptiveEngine
from .models import FourHourFVGSignal, IndicatorSnapshot, MacroConfirmation


FVG_FEATURES = (
    "h4_displacement",
    "fvg_size",
    "fvg_location",
    "entry_quality",
    "m15_confirmation",
    "h1_alignment",
    "market_structure",
    "anchored_vwap",
    "volume_profile",
    "macro",
    "primary_alignment",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _snapshot(items: list[IndicatorSnapshot], timeframe: str) -> IndicatorSnapshot | None:
    return next((item for item in items if item.timeframe == timeframe), None)


def _feature_defaults() -> dict[str, dict[str, float | int]]:
    return {
        name: {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
        for name in FVG_FEATURES
    }


def _specialist_defaults() -> dict[str, Any]:
    return {
        "version": 1,
        "features": _feature_defaults(),
        "signals": [],
        "reviews": [],
        "target_mfe_r": [],
        "counts": {"wins": 0, "losses": 0, "timeouts": 0},
        "last_review": "No completed 4H-FVG signals have been reviewed yet.",
    }


def derive_fvg_quality_votes(
    signal: FourHourFVGSignal,
    indicators: list[IndicatorSnapshot],
    macro: MacroConfirmation | None,
    primary_votes: dict[str, int],
    current_price: float,
) -> dict[str, int]:
    """Return quality votes for one FVG setup.

    Unlike the regular model, these are quality flags rather than directional
    votes: +1 supports the setup, -1 warns against it, and 0 is unknown.
    """
    votes = {name: 0 for name in FVG_FEATURES}
    side_sign = 1 if signal.side == "BUY" else -1

    body_atr = _finite(signal.parent_body_atr)
    if body_atr >= 0.80:
        votes["h4_displacement"] = 1
    elif body_atr < 0.55:
        votes["h4_displacement"] = -1

    m15 = _snapshot(indicators, "M15")
    h1 = _snapshot(indicators, "H1")
    gap_size = abs(_finite(signal.fvg_high) - _finite(signal.fvg_low))
    m15_atr = _finite(m15.atr14 if m15 is not None else None)
    if m15_atr > 0:
        gap_atr = gap_size / m15_atr
        if 0.08 <= gap_atr <= 0.85:
            votes["fvg_size"] = 1
        elif gap_atr < 0.03 or gap_atr > 1.25:
            votes["fvg_size"] = -1

    parent_mid = (_finite(signal.parent_high) + _finite(signal.parent_low)) / 2.0
    fvg_mid = _finite(signal.fvg_mid, (_finite(signal.fvg_low) + _finite(signal.fvg_high)) / 2.0)
    correct_location = (signal.side == "BUY" and fvg_mid <= parent_mid) or (
        signal.side == "SELL" and fvg_mid >= parent_mid
    )
    votes["fvg_location"] = 1 if correct_location else -1

    entry_low = _finite(signal.entry_low)
    entry_high = _finite(signal.entry_high)
    entry_mid = (entry_low + entry_high) / 2.0
    stop = _finite(signal.stop_loss)
    risk = abs(entry_mid - stop)
    if entry_low <= current_price <= entry_high:
        votes["entry_quality"] = 1
    elif risk > 0:
        extension_r = (current_price - entry_mid) / risk * side_sign
        if extension_r > 0.85:
            votes["entry_quality"] = -1
        elif extension_r <= 0.35:
            votes["entry_quality"] = 1

    votes["m15_confirmation"] = 1 if signal.state == "TRIGGERED" else 0

    if h1 is not None:
        h1_score = 0
        if h1.trend == "bullish":
            h1_score += 1
        elif h1.trend == "bearish":
            h1_score -= 1
        if h1.structure_bias == "bullish" or h1.market_structure in {"BOS_UP", "CHOCH_UP"}:
            h1_score += 1
        elif h1.structure_bias == "bearish" or h1.market_structure in {"BOS_DOWN", "CHOCH_DOWN"}:
            h1_score -= 1
        votes["h1_alignment"] = 1 if h1_score * side_sign > 0 else -1 if h1_score * side_sign < 0 else 0

    for source, target in (
        ("market_structure", "market_structure"),
        ("anchored_vwap", "anchored_vwap"),
        ("volume_profile", "volume_profile"),
    ):
        value = int(primary_votes.get(source, 0))
        votes[target] = 1 if value == side_sign else -1 if value == -side_sign else 0

    if macro is not None:
        if macro.macro_bias == "BULLISH_GOLD":
            macro_sign = 1
        elif macro.macro_bias == "BEARISH_GOLD":
            macro_sign = -1
        else:
            macro_sign = 0
        votes["macro"] = 1 if macro_sign == side_sign else -1 if macro_sign == -side_sign else 0

    votes["primary_alignment"] = 1 if signal.aligns_with_primary else 0
    return votes


class CloudLearningCoordinator:
    """Coordinates regular and FVG learning for the scheduled cloud watcher."""

    def __init__(self, adaptive: AdaptiveEngine) -> None:
        self.adaptive = adaptive
        specialist = self.adaptive.state.setdefault("specialist_learning", _specialist_defaults())
        defaults = _specialist_defaults()
        specialist["features"] = {**defaults["features"], **specialist.get("features", {})}
        for key in ("signals", "reviews", "target_mfe_r"):
            specialist.setdefault(key, deepcopy(defaults[key]))
        specialist.setdefault("counts", deepcopy(defaults["counts"]))
        specialist.setdefault("last_review", defaults["last_review"])
        specialist["version"] = 1

    @property
    def specialist(self) -> dict[str, Any]:
        return self.adaptive.state["specialist_learning"]

    def specialist_counts(self) -> dict[str, int]:
        raw = self.specialist.get("counts", {})
        return {
            "wins": int(raw.get("wins", 0)),
            "losses": int(raw.get("losses", 0)),
            "timeouts": int(raw.get("timeouts", 0)),
        }

    def specialist_confidence_cap(self) -> int:
        counts = self.specialist_counts()
        decided = counts["wins"] + counts["losses"]
        if decided < 5:
            return 65
        if decided < 10:
            return 70
        if decided < 20:
            return 76
        if decided < 40:
            return 82
        return 88

    def specialist_weights(self) -> dict[str, float]:
        return {
            name: float(self.specialist.get("features", {}).get(name, {}).get("weight", 1.0))
            for name in FVG_FEATURES
        }

    def specialist_target_multipliers(self) -> dict[str, float]:
        rows = [
            row for row in self.specialist.get("signals", [])
            if row.get("status") in {"WIN", "LOSS", "TIMEOUT"}
        ]
        clean_all: list[float] = []
        clean_wins: list[float] = []
        for row in rows:
            value = _finite(row.get("pre_exit_mfe_r", row.get("mfe_r")), -1.0)
            if value < 0:
                continue
            value = max(0.0, min(5.0, value))
            clean_all.append(value)
            if row.get("status") == "WIN":
                runner = _finite(row.get("runner_mfe_r", row.get("horizon_mfe_r", value)), value)
                clean_wins.append(max(value, min(5.0, runner)))
        if len(clean_all) < self.adaptive.minimum_samples:
            return {"tp1": 0.60, "tp2": 0.95, "tp3": 1.30}
        all_values = np.asarray(clean_all[-250:], dtype=float)
        win_values = np.asarray((clean_wins or clean_all)[-250:], dtype=float)
        tp1 = max(0.50, min(0.85, float(np.quantile(all_values, 0.45))))
        tp2 = max(tp1 + 0.20, min(1.25, float(np.quantile(win_values, 0.55))))
        tp3 = max(tp2 + 0.20, min(1.70, float(np.quantile(win_values, 0.75))))
        return {"tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2)}

    def review_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, list[dict[str, Any]]]:
        primary = self.adaptive.review_pending(frames)
        specialist = self.review_pending_specialist(frames)
        return {"primary": primary, "fvg": specialist}

    def review_pending_specialist(self, frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        if not self.adaptive.enabled:
            return []
        completed: list[dict[str, Any]] = []
        changed = False
        for signal in self.specialist.get("signals", []):
            if signal.get("status") != "PENDING":
                continue
            frame = frames.get(signal.get("timeframe", "M15"))
            if frame is None:
                continue
            review = AdaptiveEngine._evaluate_signal(signal, frame, self.adaptive.horizon_bars)
            if not review:
                continue
            if review.get("interim"):
                message = str(review.get("message", ""))
                if message and signal.get("interim_review") != message:
                    signal["interim_review"] = message
                    changed = True
                continue
            signal["status"] = review["outcome"]
            signal["reviewed_at"] = _utc_now()
            signal.update({key: value for key, value in review.items() if key != "interim"})
            self._update_specialist_from_completed(signal)
            completed.append(deepcopy(signal))
            changed = True
        if changed:
            self.adaptive.save()
        return completed

    def _update_specialist_from_completed(self, signal: dict[str, Any]) -> None:
        outcome = signal["status"]
        counts = self.specialist.setdefault("counts", {"wins": 0, "losses": 0, "timeouts": 0})
        key = "wins" if outcome == "WIN" else "losses" if outcome == "LOSS" else "timeouts"
        counts[key] = int(counts.get(key, 0)) + 1

        mfe_r = _finite(signal.get("pre_exit_mfe_r", signal.get("mfe_r")), -1.0)
        if mfe_r >= 0:
            self.specialist.setdefault("target_mfe_r", []).append(round(max(0.0, min(5.0, mfe_r)), 4))
            self.specialist["target_mfe_r"] = self.specialist["target_mfe_r"][-500:]

        mistakes: list[str] = []
        if outcome in {"WIN", "LOSS"}:
            for feature in FVG_FEATURES:
                vote = int(signal.get("feature_votes", {}).get(feature, 0))
                if vote == 0:
                    continue
                correct = (outcome == "WIN" and vote > 0) or (outcome == "LOSS" and vote < 0)
                item = self.specialist.setdefault("features", {}).setdefault(
                    feature, {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
                )
                item["samples"] = int(item.get("samples", 0)) + 1
                if correct:
                    item["alpha"] = float(item.get("alpha", 5.0)) + 1.0
                else:
                    item["beta"] = float(item.get("beta", 5.0)) + 1.0
                    if outcome == "LOSS" and vote > 0:
                        mistakes.append(feature.replace("_", " "))
                samples = int(item["samples"])
                posterior = float(item["alpha"]) / max(1e-9, float(item["alpha"]) + float(item["beta"]))
                evidence = min(1.0, samples / self.adaptive.minimum_samples)
                desired = 1.0 + (posterior - 0.5) * 1.4 * evidence
                desired = max(0.70, min(1.30, desired))
                old = float(item.get("weight", 1.0))
                delta = max(
                    -self.adaptive.max_weight_change,
                    min(self.adaptive.max_weight_change, desired - old),
                )
                item["weight"] = round(old + delta, 4)

        text = (
            f"4H-FVG {signal['side']} closed as {outcome}; "
            f"MFE {float(signal.get('mfe_r', 0)):.2f}R, MAE {float(signal.get('mae_r', 0)):.2f}R."
        )
        if mistakes:
            text += " Conditions that supported the failed setup: " + ", ".join(mistakes[:6]) + "."
        self.specialist["last_review"] = text
        self.specialist.setdefault("reviews", []).append(
            {
                "reviewed_at": _utc_now(),
                "signal_id": signal.get("id"),
                "outcome": outcome,
                "mfe_r": signal.get("mfe_r"),
                "mae_r": signal.get("mae_r"),
                "summary": text,
            }
        )
        self.specialist["reviews"] = self.specialist["reviews"][-300:]

    def apply_specialist_learning_gate(
        self,
        signal: FourHourFVGSignal | None,
        feature_votes: dict[str, int],
    ) -> FourHourFVGSignal | None:
        if signal is None or signal.side not in {"BUY", "SELL"}:
            return signal

        cap = self.specialist_confidence_cap()
        weights = self.specialist_weights()
        adjustment = sum((weights[name] - 1.0) * 20.0 * int(feature_votes.get(name, 0)) for name in FVG_FEATURES)
        signal.confidence = max(40, min(cap, int(round(signal.confidence + adjustment))))

        counts = self.specialist_counts()
        decided = counts["wins"] + counts["losses"]
        blocking: list[str] = []
        if signal.state == "TRIGGERED" and decided < self.adaptive.minimum_samples:
            for required in ("h4_displacement", "fvg_location", "m15_confirmation"):
                if int(feature_votes.get(required, 0)) <= 0:
                    blocking.append(required.replace("_", " "))
            if int(feature_votes.get("macro", 0)) < 0:
                blocking.append("macro conflict")
            confirmations = sum(
                1 for name in ("h1_alignment", "market_structure", "anchored_vwap", "volume_profile", "primary_alignment")
                if int(feature_votes.get(name, 0)) > 0
            )
            if confirmations < 2:
                blocking.append("fewer than two structure/value confirmations")
            if int(feature_votes.get("entry_quality", 0)) < 0:
                blocking.append("extended entry")

        if blocking:
            signal.confidence = min(signal.confidence, 59)
            signal.warnings.append(
                "24/7 learning gate withheld the alert: " + ", ".join(dict.fromkeys(blocking)) + "."
            )

        # Never expand existing structural targets.  Learning may only pull
        # them closer to distances that have actually been achieved.
        if signal.entry_low is not None and signal.entry_high is not None and signal.stop_loss is not None:
            entry = (float(signal.entry_low) + float(signal.entry_high)) / 2.0
            risk = abs(entry - float(signal.stop_loss))
            mult = self.specialist_target_multipliers()
            if risk > 0 and signal.take_profit_1 is not None:
                if signal.side == "BUY":
                    signal.take_profit_1 = round(min(float(signal.take_profit_1), entry + mult["tp1"] * risk), 2)
                    signal.take_profit_2 = round(min(float(signal.take_profit_2), entry + mult["tp2"] * risk), 2)
                    signal.take_profit_3 = round(min(float(signal.take_profit_3), entry + mult["tp3"] * risk), 2)
                else:
                    signal.take_profit_1 = round(max(float(signal.take_profit_1), entry - mult["tp1"] * risk), 2)
                    signal.take_profit_2 = round(max(float(signal.take_profit_2), entry - mult["tp2"] * risk), 2)
                    signal.take_profit_3 = round(max(float(signal.take_profit_3), entry - mult["tp3"] * risk), 2)
        return signal

    def register_specialist_signal(
        self,
        signal: FourHourFVGSignal | None,
        feature_votes: dict[str, int],
        symbol: str = "XAU/USD",
        timeframe: str = "M15",
    ) -> bool:
        if not self.adaptive.enabled or signal is None or signal.state != "TRIGGERED" or signal.side not in {"BUY", "SELL"}:
            return False
        if signal.confidence < 60:
            return False
        existing = {row.get("id") for row in self.specialist.get("signals", [])}
        signal_id = signal.signal_id or f"H4FVG|{signal.side}|{signal.parent_candle_time}|{signal.fvg_created_time}"
        if signal_id in existing:
            return False
        if None in (signal.entry_low, signal.entry_high, signal.stop_loss, signal.take_profit_1):
            return False
        entry = (float(signal.entry_low) + float(signal.entry_high)) / 2.0
        risk = abs(entry - float(signal.stop_loss))
        if risk <= 0:
            return False
        self.specialist.setdefault("signals", []).append(
            {
                "id": signal_id,
                "strategy_id": signal.strategy_id,
                "created_at": _utc_now(),
                "signal_time": pd.to_datetime(signal.signal_time, utc=True).isoformat(),
                "symbol": symbol,
                "timeframe": timeframe,
                "side": signal.side,
                "entry": entry,
                "stop": float(signal.stop_loss),
                "tp1": float(signal.take_profit_1),
                "tp2": float(signal.take_profit_2) if signal.take_profit_2 is not None else float(signal.take_profit_1),
                "tp3": float(signal.take_profit_3) if signal.take_profit_3 is not None else float(signal.take_profit_1),
                "risk": risk,
                "confidence": int(signal.confidence),
                "feature_votes": {name: int(feature_votes.get(name, 0)) for name in FVG_FEATURES},
                "status": "PENDING",
                "interim_review": "",
                "parent_candle_time": signal.parent_candle_time,
                "fvg_created_time": signal.fvg_created_time,
            }
        )
        self.specialist["signals"] = self.specialist["signals"][-500:]
        self.adaptive.save()
        return True
