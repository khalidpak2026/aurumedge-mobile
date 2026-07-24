from __future__ import annotations

"""Bounded three-pillar adaptive learning for AurumEdge v5.8.1.

The learner starts updating after the first completed delivered trade, while a
Bayesian prior and per-review change cap prevent one outcome from dominating.
Unknown legacy sections in adaptive_state.json are preserved during migration.
"""

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd


PILLARS = ("market_structure", "anchored_vwap", "volume_profile")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _set(obj: Any, name: str, value: Any) -> None:
    if isinstance(obj, dict):
        obj[name] = value
        return
    try:
        setattr(obj, name, value)
    except Exception:
        try:
            object.__setattr__(obj, name, value)
        except Exception:
            pass


def _feature_default() -> dict[str, Any]:
    return {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0, "wins": 0, "losses": 0}


def _defaults() -> dict[str, Any]:
    return {
        "version": 3,
        "features": {name: _feature_default() for name in PILLARS},
        "signals": [],
        "reviews": [],
        "counts": {"wins": 0, "losses": 0, "timeouts": 0},
        "target_multipliers": {"tp1": 0.65, "tp2": 1.05, "tp3": 1.40},
        "target_mfe_r": [],
        "patterns": {},
        "last_review": "No completed delivered trades have been reviewed yet.",
    }


def derive_feature_votes(
    indicators: list[Any],
    liquidity: list[Any] | None = None,
    macro: Any | None = None,
    market_state: str | None = None,
) -> dict[str, int]:
    """Compatibility wrapper around the three-pillar strategy vote function."""
    from .strategy import derive_feature_votes as strategy_votes

    return strategy_votes(indicators, liquidity, macro, market_state)


class AdaptiveEngine:
    def __init__(
        self,
        state_path: str | Path,
        enabled: bool = True,
        minimum_samples: int = 20,
        horizon_bars: int = 12,
        max_weight_change: float = 0.05,
    ) -> None:
        self.state_path = Path(state_path)
        self.enabled = bool(enabled)
        self.minimum_samples = max(1, int(minimum_samples))
        self.horizon_bars = max(3, int(horizon_bars))
        self.max_weight_change = max(0.005, min(0.10, float(max_weight_change)))
        self.state = self._load_and_migrate()

    def _load_and_migrate(self) -> dict[str, Any]:
        state: dict[str, Any] = {}
        try:
            if self.state_path.exists():
                parsed = json.loads(self.state_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    state = parsed
        except (OSError, json.JSONDecodeError):
            state = {}
        defaults = _defaults()
        for key, value in defaults.items():
            state.setdefault(key, deepcopy(value))
        raw_features = state.setdefault("features", {})
        for name in PILLARS:
            old = raw_features.get(name, {})
            merged = _feature_default()
            if isinstance(old, dict):
                merged.update(old)
            merged["weight"] = max(0.70, min(1.30, _finite(merged.get("weight"), 1.0)))
            raw_features[name] = merged
        state["counts"] = {**defaults["counts"], **(state.get("counts") or {})}
        state["target_multipliers"] = {
            **defaults["target_multipliers"],
            **(state.get("target_multipliers") or {}),
        }
        for key in ("signals", "reviews", "target_mfe_r"):
            if not isinstance(state.get(key), list):
                state[key] = []
        if not isinstance(state.get("patterns"), dict):
            state["patterns"] = {}
        state["version"] = 3
        return state

    def save(self) -> None:
        if not self.enabled:
            return
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.state, indent=2, default=str), encoding="utf-8")
        temporary.replace(self.state_path)

    def weights(self) -> dict[str, float]:
        return {
            name: max(0.70, min(1.30, _finite(self.state["features"].get(name, {}).get("weight"), 1.0)))
            for name in PILLARS
        }

    def target_multipliers(self) -> dict[str, float]:
        raw = self.state.get("target_multipliers", {})
        tp1 = max(0.60, min(0.90, _finite(raw.get("tp1"), 0.65)))
        tp2 = max(tp1 + 0.20, min(1.35, _finite(raw.get("tp2"), 1.05)))
        tp3 = max(tp2 + 0.20, min(1.80, _finite(raw.get("tp3"), 1.40)))
        return {"tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2)}

    def summary(self) -> Any:
        counts = self.state.get("counts", {})
        pending = sum(1 for row in self.state.get("signals", []) if row.get("status") == "PENDING")
        reviewed = int(counts.get("wins", 0)) + int(counts.get("losses", 0)) + int(counts.get("timeouts", 0))
        return SimpleNamespace(
            reviewed_signals=reviewed,
            pending_signals=pending,
            wins=int(counts.get("wins", 0)),
            losses=int(counts.get("losses", 0)),
            timeouts=int(counts.get("timeouts", 0)),
            weights=self.weights(),
            target_multipliers=self.target_multipliers(),
            last_review=str(self.state.get("last_review", "")),
        )

    def apply_capital_preservation(
        self,
        report: Any,
        signal_time: Any | None = None,
        feature_votes: dict[str, int] | None = None,
    ) -> Any:
        """Keep risk blocking separate from direction and force live-zone truth.

        A historical cooldown or pending-record flag is deliberately not used
        here. If current price is inside the displayed zone and its risk plan is
        not BLOCK, the setup is ENTRY LIVE.
        """
        setup = _get(report, "active_setup")
        state = str(_get(report, "market_state", "STUCK"))
        if setup is None or state not in {"BUY", "SELL"}:
            return report
        price = _finite(_get(report, "last_price"))
        low = _finite(_get(setup, "entry_low"))
        high = _finite(_get(setup, "entry_high"))
        risk_plan = _get(setup, "risk_plan")
        risk_status = str(_get(risk_plan, "status", "OK"))
        inside = low <= price <= high
        if inside and risk_status != "BLOCK":
            _set(setup, "status", "ENTER")
            _set(setup, "entry_live", True)
            _set(report, "entry_live", True)
            _set(report, "execution_label", f"ENTER {state} · ENTRY LIVE NOW")
        elif risk_status == "BLOCK":
            _set(setup, "status", "WAIT")
            _set(setup, "entry_live", False)
            _set(report, "entry_live", False)
        return report

    def register_signal(
        self,
        report: Any,
        signal_time: Any,
        feature_votes: dict[str, int],
        timeframe: str = "M15",
        delivery_event_id: str | None = None,
        **_: Any,
    ) -> bool:
        """Register only an executable entry; caller controls delivery proof."""
        if not self.enabled:
            return False
        state = str(_get(report, "market_state", ""))
        setup = _get(report, "active_setup")
        if state not in {"BUY", "SELL"} or setup is None:
            return False
        if str(_get(setup, "status", "WAIT")) != "ENTER" or not bool(_get(setup, "entry_live", False)):
            return False
        entry = _finite(_get(setup, "entry_price"), (_finite(_get(setup, "entry_low")) + _finite(_get(setup, "entry_high"))) / 2.0)
        stop = _finite(_get(setup, "stop_loss"))
        tp1 = _finite(_get(setup, "take_profit_1"))
        if entry <= 0 or stop <= 0 or tp1 <= 0 or abs(entry - stop) < 1e-9:
            return False
        stamp = pd.to_datetime(signal_time, utc=True, errors="coerce")
        stamp_text = stamp.isoformat() if not pd.isna(stamp) else str(signal_time)
        raw_key = f"{_get(report, 'symbol', 'XAU/USD')}|{state}|{stamp_text}|{entry:.2f}|{stop:.2f}|{tp1:.2f}"
        signal_id = hashlib.sha1(raw_key.encode("utf-8")).hexdigest()[:20]
        if any(row.get("signal_id") == signal_id for row in self.state.get("signals", [])):
            return False
        # Do not create overlapping duplicate pending trades on the same side.
        for row in self.state.get("signals", []):
            if row.get("status") == "PENDING" and row.get("side") == state:
                if abs(_finite(row.get("entry")) - entry) <= max(abs(entry - stop) * 0.25, 0.20):
                    return False
        record = {
            "signal_id": signal_id,
            "delivery_event_id": delivery_event_id,
            "symbol": str(_get(report, "symbol", "XAU/USD")),
            "side": state,
            "status": "PENDING",
            "signal_time": stamp_text,
            "opened_at": _utc_now(),
            "timeframe": timeframe,
            "entry": round(entry, 6),
            "entry_low": round(_finite(_get(setup, "entry_low"), entry), 6),
            "entry_high": round(_finite(_get(setup, "entry_high"), entry), 6),
            "stop": round(stop, 6),
            "tp1": round(tp1, 6),
            "tp2": round(_finite(_get(setup, "take_profit_2"), tp1), 6),
            "tp3": round(_finite(_get(setup, "take_profit_3"), tp1), 6),
            "feature_votes": {name: int(feature_votes.get(name, 0)) for name in PILLARS},
            "setup_type": str(_get(setup, "setup_type", "THREE_PILLAR")),
            "confidence": int(_finite(_get(report, "confidence"), 0)),
            "last_price": round(_finite(_get(report, "last_price"), entry), 6),
        }
        self.state.setdefault("signals", []).append(record)
        self.state["signals"] = self.state["signals"][-1000:]
        self.save()
        return True

    @staticmethod
    def _evaluate_signal(signal: dict[str, Any], frame: pd.DataFrame, horizon_bars: int) -> dict[str, Any] | None:
        if frame is None or frame.empty:
            return None
        required = {"time", "high", "low", "close"}
        if not required.issubset(frame.columns):
            return None
        data = frame.copy()
        data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
        data = data.dropna(subset=["time"]).sort_values("time")
        start = pd.to_datetime(signal.get("signal_time") or signal.get("opened_at"), utc=True, errors="coerce")
        if pd.isna(start):
            return None
        future = data[data["time"] > start].head(max(1, int(horizon_bars)))
        if future.empty:
            return None
        side = str(signal.get("side", "BUY"))
        entry = _finite(signal.get("entry"))
        stop = _finite(signal.get("stop"))
        tp1 = _finite(signal.get("tp1"))
        risk = abs(entry - stop)
        if entry <= 0 or risk <= 0 or tp1 <= 0:
            return None
        mfe = 0.0
        mae = 0.0
        outcome: str | None = None
        exit_time: str | None = None
        exit_price: float | None = None
        bars_seen = 0
        for _, candle in future.iterrows():
            bars_seen += 1
            high = _finite(candle["high"])
            low = _finite(candle["low"])
            if side == "BUY":
                mfe = max(mfe, (high - entry) / risk)
                mae = max(mae, (entry - low) / risk)
                stop_hit = low <= stop
                target_hit = high >= tp1
            else:
                mfe = max(mfe, (entry - low) / risk)
                mae = max(mae, (high - entry) / risk)
                stop_hit = high >= stop
                target_hit = low <= tp1
            # When one candle contains both levels, choose the conservative result.
            if stop_hit:
                outcome = "LOSS"
                exit_price = stop
            elif target_hit:
                outcome = "WIN"
                exit_price = tp1
            if outcome:
                exit_time = pd.Timestamp(candle["time"]).isoformat()
                break
        if outcome is None and len(future) >= max(1, int(horizon_bars)):
            outcome = "TIMEOUT"
            last = future.iloc[-1]
            exit_price = _finite(last["close"], entry)
            exit_time = pd.Timestamp(last["time"]).isoformat()
        if outcome is None:
            return {
                "interim": True,
                "message": f"PENDING after {bars_seen} bar(s); MFE {mfe:.2f}R, MAE {mae:.2f}R",
                "mfe_r": round(max(0.0, mfe), 4),
                "mae_r": round(max(0.0, mae), 4),
            }
        return {
            "interim": False,
            "outcome": outcome,
            "exit_time": exit_time,
            "exit_price": round(float(exit_price or entry), 6),
            "bars_held": bars_seen,
            "mfe_r": round(max(0.0, mfe), 4),
            "mae_r": round(max(0.0, mae), 4),
            "pre_exit_mfe_r": round(max(0.0, mfe), 4),
            "runner_mfe_r": round(max(0.0, mfe), 4),
            "horizon_mfe_r": round(max(0.0, mfe), 4),
            "message": f"{outcome}: MFE {mfe:.2f}R, MAE {mae:.2f}R after {bars_seen} bar(s)",
        }

    def review_pending(self, frames: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        completed: list[dict[str, Any]] = []
        changed = False
        for signal in self.state.get("signals", []):
            if signal.get("status") != "PENDING":
                continue
            # M5 gives better TP/SL ordering resolution; M15 remains fallback.
            review = None
            m5 = frames.get("M5")
            if m5 is not None:
                review = self._evaluate_signal(signal, m5, self.horizon_bars * 3)
            if not review or review.get("interim"):
                m15 = frames.get(signal.get("timeframe", "M15"))
                if m15 is None:
                    m15 = frames.get("M15")
                if m15 is not None:
                    m15_review = self._evaluate_signal(signal, m15, self.horizon_bars)
                    if m15_review and not m15_review.get("interim"):
                        review = m15_review
                    elif review is None:
                        review = m15_review
            if not review:
                continue
            if review.get("interim"):
                message = str(review.get("message", ""))
                if signal.get("interim_review") != message:
                    signal["interim_review"] = message
                    signal["mfe_r"] = review.get("mfe_r", 0.0)
                    signal["mae_r"] = review.get("mae_r", 0.0)
                    changed = True
                continue
            signal["status"] = str(review["outcome"])
            signal["reviewed_at"] = _utc_now()
            signal.update({key: value for key, value in review.items() if key != "interim"})
            self._learn_from_completed(signal)
            completed.append(deepcopy(signal))
            changed = True
        if changed:
            self.save()
        return completed

    def _learn_from_completed(self, signal: dict[str, Any]) -> None:
        outcome = str(signal.get("status", "TIMEOUT"))
        count_key = "wins" if outcome == "WIN" else "losses" if outcome == "LOSS" else "timeouts"
        counts = self.state.setdefault("counts", {"wins": 0, "losses": 0, "timeouts": 0})
        counts[count_key] = int(counts.get(count_key, 0)) + 1
        successful: list[str] = []
        failed: list[str] = []
        if outcome in {"WIN", "LOSS"}:
            for pillar in PILLARS:
                vote = int(signal.get("feature_votes", {}).get(pillar, 0))
                if vote == 0:
                    continue
                item = self.state.setdefault("features", {}).setdefault(pillar, _feature_default())
                item["samples"] = int(item.get("samples", 0)) + 1
                correct = outcome == "WIN"  # all stored votes support the delivered side
                if correct:
                    item["alpha"] = _finite(item.get("alpha"), 5.0) + 1.0
                    item["wins"] = int(item.get("wins", 0)) + 1
                    successful.append(pillar)
                else:
                    item["beta"] = _finite(item.get("beta"), 5.0) + 1.0
                    item["losses"] = int(item.get("losses", 0)) + 1
                    failed.append(pillar)
                samples = int(item["samples"])
                alpha = _finite(item.get("alpha"), 5.0)
                beta = _finite(item.get("beta"), 5.0)
                posterior = alpha / max(alpha + beta, 1e-9)
                # Learning begins immediately, but first-sample evidence is small.
                evidence = max(0.15, min(1.0, samples / self.minimum_samples))
                desired = 1.0 + (posterior - 0.5) * 1.20 * evidence
                desired = max(0.70, min(1.30, desired))
                old = _finite(item.get("weight"), 1.0)
                delta = max(-self.max_weight_change, min(self.max_weight_change, desired - old))
                item["weight"] = round(max(0.70, min(1.30, old + delta)), 4)
        mfe = max(0.0, min(5.0, _finite(signal.get("mfe_r"))))
        mae = max(0.0, min(5.0, _finite(signal.get("mae_r"))))
        self.state.setdefault("target_mfe_r", []).append(mfe)
        self.state["target_mfe_r"] = self.state["target_mfe_r"][-500:]
        self._adapt_targets(mfe, outcome)
        pattern_key = f"{signal.get('side', 'NA')}|" + ",".join(
            f"{name}:{int(signal.get('feature_votes', {}).get(name, 0))}" for name in PILLARS
        )
        pattern = self.state.setdefault("patterns", {}).setdefault(
            pattern_key,
            {"samples": 0, "wins": 0, "losses": 0, "timeouts": 0, "mfe_sum": 0.0, "mae_sum": 0.0},
        )
        pattern["samples"] = int(pattern.get("samples", 0)) + 1
        pattern[count_key] = int(pattern.get(count_key, 0)) + 1
        pattern["mfe_sum"] = round(_finite(pattern.get("mfe_sum")) + mfe, 4)
        pattern["mae_sum"] = round(_finite(pattern.get("mae_sum")) + mae, 4)
        review = {
            "signal_id": signal.get("signal_id"),
            "reviewed_at": signal.get("reviewed_at"),
            "outcome": outcome,
            "side": signal.get("side"),
            "setup_type": signal.get("setup_type"),
            "mfe_r": mfe,
            "mae_r": mae,
            "successful_pillars": successful,
            "failed_pillars": failed,
            "pattern_key": pattern_key,
            "weights_after": self.weights(),
            "targets_after": self.target_multipliers(),
        }
        self.state.setdefault("reviews", []).append(review)
        self.state["reviews"] = self.state["reviews"][-1000:]
        self.state["last_review"] = (
            f"{outcome} {signal.get('side')} · MFE {mfe:.2f}R · MAE {mae:.2f}R · "
            f"successful: {', '.join(successful) or 'none'} · failed: {', '.join(failed) or 'none'}"
        )

    def _adapt_targets(self, mfe: float, outcome: str) -> None:
        current = self.target_multipliers()
        if outcome == "WIN":
            desired_tp1 = max(0.60, min(0.85, mfe * 0.85))
            desired_tp2 = max(desired_tp1 + 0.20, min(1.25, mfe * 1.10))
            desired_tp3 = max(desired_tp2 + 0.20, min(1.70, mfe * 1.35))
        else:
            desired_tp1 = max(0.60, min(current["tp1"], max(0.60, mfe * 0.90)))
            desired_tp2 = max(desired_tp1 + 0.20, min(current["tp2"], max(desired_tp1 + 0.20, mfe * 1.10)))
            desired_tp3 = max(desired_tp2 + 0.20, min(current["tp3"], max(desired_tp2 + 0.20, mfe * 1.30)))
        desired = {"tp1": desired_tp1, "tp2": desired_tp2, "tp3": desired_tp3}
        updated: dict[str, float] = {}
        for key in ("tp1", "tp2", "tp3"):
            delta = max(-0.05, min(0.05, desired[key] - current[key]))
            updated[key] = round(current[key] + delta, 2)
        updated["tp1"] = max(0.60, min(0.90, updated["tp1"]))
        updated["tp2"] = max(updated["tp1"] + 0.20, min(1.35, updated["tp2"]))
        updated["tp3"] = max(updated["tp2"] + 0.20, min(1.80, updated["tp3"]))
        self.state["target_multipliers"] = updated

    def pending_near_exit(self, current_price: float, fraction: float = 0.25) -> bool:
        price = float(current_price)
        for row in self.state.get("signals", []):
            if row.get("status") != "PENDING":
                continue
            entry = _finite(row.get("entry"))
            stop = _finite(row.get("stop"))
            tp1 = _finite(row.get("tp1"))
            risk = abs(entry - stop)
            if risk <= 0:
                continue
            threshold = max(risk * max(0.10, fraction), 0.20)
            if abs(price - stop) <= threshold or abs(price - tp1) <= threshold:
                return True
        return False

    def pending_trade_count(self) -> int:
        return sum(1 for row in self.state.get("signals", []) if row.get("status") == "PENDING")
