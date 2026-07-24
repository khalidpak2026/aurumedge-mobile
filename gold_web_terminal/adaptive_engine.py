from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import AdaptiveLearningSummary, IndicatorSnapshot, LiquiditySnapshot, MacroConfirmation, TechnicalReport


FEATURES = (
    "market_structure",
    "anchored_vwap",
    "volume_profile",
    "momentum",
    "adx_dmi",
    "liquidity",
    "breakout",
    "macro",
    "entry_quality",
    # Legacy features remain loadable so existing adaptive-state files migrate
    # cleanly, but they no longer drive the primary decision.
    "ema_trend",
    "vwap",
    "volume",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "version": 4,
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "features": {
            name: {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
            for name in FEATURES
        },
        "signals": [],
        "reviews": [],
        "target_mfe_r": [],
        "counts": {"wins": 0, "losses": 0, "timeouts": 0},
        "patterns": {},
        "last_review": "No completed signals have been reviewed yet.",
    }


def _safe_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _snapshot(items: list[IndicatorSnapshot], timeframe: str) -> IndicatorSnapshot:
    return next((item for item in items if item.timeframe == timeframe), items[0])


def derive_feature_votes(
    indicators: list[IndicatorSnapshot],
    liquidity: list[LiquiditySnapshot],
    macro: MacroConfirmation | None,
    side_hint: str | None = None,
) -> dict[str, int]:
    """Return votes for the three active pillars plus entry quality.

    Legacy feature keys remain present only so old adaptive-state files load
    without migration errors. They are always neutral in this build.
    """
    del liquidity, macro
    votes: dict[str, int] = {name: 0 for name in FEATURES}
    if not indicators:
        return votes
    h1 = _snapshot(indicators, "H1")
    h4 = _snapshot(indicators, "H4")
    m15 = _snapshot(indicators, "M15")

    structure_score = 0
    avwap_score = 0
    profile_score = 0.0
    for item in (m15, h1, h4):
        if item.market_structure in {"BOS_UP", "CHOCH_UP"} or item.structure_bias == "bullish":
            structure_score += 1
        elif item.market_structure in {"BOS_DOWN", "CHOCH_DOWN"} or item.structure_bias == "bearish":
            structure_score -= 1

        if item.avwap_active is not None:
            slope = float(item.avwap_slope_atr or 0.0)
            if item.close > float(item.avwap_active) and slope >= -0.03:
                avwap_score += 1
            elif item.close < float(item.avwap_active) and slope <= 0.03:
                avwap_score -= 1

        if item.profile_acceptance == "bullish" or item.profile_state == "ABOVE_VALUE":
            profile_score += 1
        elif item.profile_acceptance == "bearish" or item.profile_state == "BELOW_VALUE":
            profile_score -= 1
        elif item.profile_poc is not None:
            profile_score += 0.25 if item.close > float(item.profile_poc) else -0.25

    votes["market_structure"] = 1 if structure_score >= 2 else -1 if structure_score <= -2 else 0
    votes["anchored_vwap"] = 1 if avwap_score >= 2 else -1 if avwap_score <= -2 else 0
    votes["volume_profile"] = 1 if profile_score >= 1.5 else -1 if profile_score <= -1.5 else 0

    # Entry quality is timing only. It never decides BUY versus SELL.
    atr = max(float(m15.atr14 or h1.atr14 or 0.0), 1e-9)
    references = [value for value in (m15.avwap_active, m15.profile_poc, m15.profile_val, m15.profile_vah) if value is not None]
    if references:
        distance = min(abs(float(m15.close) - float(value)) for value in references)
        if distance <= atr * 0.65:
            votes["entry_quality"] = 1
        elif distance > atr * 1.25:
            votes["entry_quality"] = -1
    if side_hint == "BUY" and (m15.structure_break_up or m15.market_structure in {"BOS_UP", "CHOCH_UP"}):
        votes["entry_quality"] = max(votes["entry_quality"], 0)
    elif side_hint == "SELL" and (m15.structure_break_down or m15.market_structure in {"BOS_DOWN", "CHOCH_DOWN"}):
        votes["entry_quality"] = max(votes["entry_quality"], 0)
    return votes


class AdaptiveEngine:
    """Controlled online learner.

    It never rewrites source code and never changes weights from a single adverse candle.
    Completed signals update bounded feature reliabilities after a minimum evidence threshold.
    """

    def __init__(
        self,
        path: str | Path,
        enabled: bool = True,
        minimum_samples: int = 20,
        horizon_bars: int = 12,
        max_weight_change: float = 0.05,
    ) -> None:
        self.path = Path(path)
        self.enabled = enabled
        self.minimum_samples = max(8, int(minimum_samples))
        self.horizon_bars = max(4, int(horizon_bars))
        self.max_weight_change = max(0.01, min(0.20, float(max_weight_change)))
        self.state = self._load()

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return _default_state()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            base = _default_state()
            base.update(data if isinstance(data, dict) else {})
            base["features"] = {**_default_state()["features"], **base.get("features", {})}
            # v2 fixes entry_quality semantics (+1 good, -1 poor). Rebuild that
            # feature from completed signals so older directional treatment does
            # not contaminate the new capital-preservation gate.
            loaded_version = int(base.get("version", 1))
            if loaded_version < 2:
                quality = {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
                for signal in base.get("signals", []):
                    outcome = signal.get("status")
                    vote = int(signal.get("feature_votes", {}).get("entry_quality", 0))
                    if outcome not in {"WIN", "LOSS"} or vote == 0:
                        continue
                    quality["samples"] += 1
                    correct = (outcome == "WIN" and vote > 0) or (outcome == "LOSS" and vote < 0)
                    quality["alpha" if correct else "beta"] += 1.0
                base["features"]["entry_quality"] = quality
            if not isinstance(base.get("patterns"), dict):
                base["patterns"] = {}
            if loaded_version < 4:
                # Do not replay historical close alerts during migration. Only
                # outcomes completed by this build start with close_alert_sent=False.
                for signal in base.get("signals", []):
                    if signal.get("status") in {"WIN", "LOSS", "TIMEOUT"}:
                        signal.setdefault("close_alert_sent", True)
            base["version"] = 4
            return base
        except Exception:
            backup = self.path.with_suffix(self.path.suffix + ".corrupt")
            try:
                self.path.replace(backup)
            except OSError:
                pass
            return _default_state()

    def save(self) -> None:
        if not self.enabled:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state["updated_at"] = _utc_now()
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(json.dumps(self.state, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(self.path)

    def weights(self) -> dict[str, float]:
        return {
            name: float(self.state.get("features", {}).get(name, {}).get("weight", 1.0))
            for name in FEATURES
        }

    def target_multipliers(self) -> dict[str, float]:
        """Return conservative, reachable targets until clean evidence exists.

        Early capital-preservation defaults are intentionally closer than the
        old 0.80R/1.30R/1.80R ladder. Full adaptation starts only after the
        configured evidence threshold.
        """
        completed = [row for row in self.state.get("signals", []) if row.get("status") in {"WIN", "LOSS", "TIMEOUT"}]
        clean_all = []
        clean_wins = []
        for row in completed:
            value = _safe_number(row.get("pre_exit_mfe_r", row.get("mfe_r")))
            if value is None:
                continue
            value = max(0.0, min(5.0, value))
            clean_all.append(value)
            if row.get("status") == "WIN":
                runner = _safe_number(row.get("runner_mfe_r", row.get("horizon_mfe_r", value)))
                clean_wins.append(max(value, min(5.0, runner if runner is not None else value)))

        # Protect capital while the sample is small. These levels reflect the
        # uploaded history: TP1 was reachable around 0.8R, while larger targets
        # were inconsistent.
        if len(clean_all) < self.minimum_samples:
            return {"tp1": 0.65, "tp2": 1.05, "tp3": 1.40}

        all_values = np.asarray(clean_all[-250:], dtype=float)
        win_values = np.asarray((clean_wins or clean_all)[-250:], dtype=float)
        tp1 = float(np.quantile(all_values, 0.45))
        tp2 = float(np.quantile(win_values, 0.55))
        tp3 = float(np.quantile(win_values, 0.75))
        tp1 = max(0.60, min(0.90, tp1))
        tp2 = max(tp1 + 0.25, min(1.35, tp2))
        tp3 = max(tp2 + 0.25, min(1.80, tp3))
        return {"tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2)}

    def _decided_count(self) -> int:
        counts = self.state.get("counts", {})
        return int(counts.get("wins", 0)) + int(counts.get("losses", 0))

    def confidence_cap(self) -> int:
        decided = self._decided_count()
        if decided < 5:
            return 68
        if decided < 10:
            return 72
        if decided < 20:
            return 78
        if decided < 40:
            return 84
        return 90

    def _same_side_guard(self, side: str, signal_time: Any) -> str | None:
        now = pd.to_datetime(signal_time, utc=True)
        rows = [row for row in self.state.get("signals", []) if row.get("side") == side]
        pending = [row for row in rows if row.get("status") == "PENDING"]
        for row in pending:
            prior = pd.to_datetime(row.get("signal_time"), utc=True, errors="coerce")
            if pd.notna(prior) and now > prior:
                return f"An earlier {side} signal is still pending; duplicate entries are blocked until it resolves."

        completed = [row for row in rows if row.get("status") in {"WIN", "LOSS"}]
        completed.sort(key=lambda row: str(row.get("outcome_time") or row.get("signal_time") or ""))
        if completed:
            last = completed[-1]
            last_time = pd.to_datetime(last.get("outcome_time") or last.get("signal_time"), utc=True, errors="coerce")
            if pd.notna(last_time):
                elapsed = now - last_time
                cooldown = timedelta(minutes=60 if last.get("status") == "LOSS" else 30)
                if timedelta(0) <= elapsed < cooldown:
                    return f"{side} cooldown is active after the previous {last.get('status', '').lower()} signal."
        recent_losses = [row for row in completed[-3:] if row.get("status") == "LOSS"]
        if len(recent_losses) >= 2:
            last_loss_time = pd.to_datetime(recent_losses[-1].get("outcome_time") or recent_losses[-1].get("signal_time"), utc=True, errors="coerce")
            first_loss_time = pd.to_datetime(recent_losses[-2].get("outcome_time") or recent_losses[-2].get("signal_time"), utc=True, errors="coerce")
            if pd.notna(last_loss_time) and pd.notna(first_loss_time):
                if last_loss_time - first_loss_time <= timedelta(hours=6) and now - last_loss_time < timedelta(hours=2):
                    return f"Two recent {side} losses triggered a two-hour capital-preservation lock."
        return None

    @staticmethod
    def _tf_direction(item: IndicatorSnapshot) -> int:
        score = 0
        if item.market_structure in {"BOS_UP", "CHOCH_UP"} or item.structure_bias == "bullish":
            score += 2
        elif item.market_structure in {"BOS_DOWN", "CHOCH_DOWN"} or item.structure_bias == "bearish":
            score -= 2
        if item.avwap_active is not None:
            if item.close > float(item.avwap_active):
                score += 1
            elif item.close < float(item.avwap_active):
                score -= 1
        if item.profile_acceptance == "bullish" or item.profile_state == "ABOVE_VALUE":
            score += 1
        elif item.profile_acceptance == "bearish" or item.profile_state == "BELOW_VALUE":
            score -= 1
        return 1 if score > 0 else -1 if score < 0 else 0

    def apply_capital_preservation(
        self,
        report: TechnicalReport,
        signal_time: Any,
        feature_votes: dict[str, int],
    ) -> TechnicalReport:
        """Calibrate confidence without reintroducing discarded indicators."""
        cap = self.confidence_cap()
        report.confidence = min(int(report.confidence), cap)
        if report.active_setup is not None:
            report.active_setup.confidence = min(int(report.active_setup.confidence), cap)

        if report.market_state not in {"BUY", "SELL"} or report.active_setup is None:
            return report

        side = report.market_state
        side_sign = 1 if side == "BUY" else -1
        reasons: list[str] = []
        repeat_reason = self._same_side_guard(side, signal_time)
        if repeat_reason:
            reasons.append(repeat_reason)

        if self._decided_count() < self.minimum_samples:
            confirmations = sum(
                int(feature_votes.get(name, 0)) == side_sign
                for name in ("market_structure", "anchored_vwap", "volume_profile")
            )
            if confirmations < 2:
                reasons.append("Fewer than two of the three active pillars confirm the direction.")
            if int(feature_votes.get("entry_quality", 0)) < 0:
                reasons.append("Price is extended from anchored VWAP/value; wait for a better live entry.")

            tf = {item.timeframe: self._tf_direction(item) for item in report.indicators}
            opposite = -side_sign
            if tf.get("M15") == opposite and tf.get("H1") == opposite:
                reasons.append("M15 and H1 three-pillar direction both oppose the setup.")

        if reasons:
            setup = report.active_setup
            setup.status = "NO_TRADE"
            setup.entry_type = "WAIT FOR THREE-PILLAR ENTRY"
            setup.warnings = list(dict.fromkeys(reasons + setup.warnings))
            report.active_setup = setup
            report.recommendation = report.market_state
            report.signal_label = f"{side} TREND · WAIT FOR LIVE ENTRY"
            report.trap_reason = " ".join(reasons)
            report.data_quality_notes.append(
                "Adaptive safety withheld execution without changing the three-pillar directional bias."
            )
        return report

    def register_signal(
        self,
        report: TechnicalReport,
        signal_time: Any,
        feature_votes: dict[str, int],
        timeframe: str = "M15",
        delivery_event_id: str = "",
    ) -> bool:
        if not self.enabled or report.active_setup is None or report.market_state not in {"BUY", "SELL"}:
            return False
        if self._same_side_guard(report.market_state, signal_time):
            return False
        setup = report.active_setup
        signal_time_iso = pd.to_datetime(signal_time, utc=True).isoformat()
        signal_id = f"{report.symbol}|{timeframe}|{signal_time_iso}|{setup.side}"
        existing = {row.get("id") for row in self.state.get("signals", [])}
        if signal_id in existing:
            return False
        entry = (setup.entry_low + setup.entry_high) / 2
        risk = abs(entry - setup.stop_loss)
        if risk <= 0:
            return False
        self.state.setdefault("signals", []).append(
            {
                "id": signal_id,
                "created_at": _utc_now(),
                "signal_time": signal_time_iso,
                "symbol": report.symbol,
                "timeframe": timeframe,
                "side": setup.side,
                "entry": entry,
                "stop": setup.stop_loss,
                "tp1": setup.take_profit_1,
                "tp2": setup.take_profit_2,
                "tp3": setup.take_profit_3,
                "risk": risk,
                "confidence": report.confidence,
                "delivery_event_id": delivery_event_id,
                "feature_votes": {name: int(feature_votes.get(name, 0)) for name in FEATURES},
                "status": "PENDING",
                "close_alert_sent": False,
                "interim_review": "",
            }
        )
        self.state["signals"] = self.state["signals"][-500:]
        self.save()
        return True


    def pending_trade_count(self) -> int:
        return sum(1 for row in self.state.get("signals", []) if row.get("status") == "PENDING")


    def pending_close_reviews(self) -> list[dict[str, Any]]:
        return [
            deepcopy(row)
            for row in self.state.get("signals", [])
            if row.get("status") in {"WIN", "LOSS", "TIMEOUT"}
            and not bool(row.get("close_alert_sent", False))
        ]

    def mark_close_alerted(self, signal_ids: list[str]) -> bool:
        wanted = {str(value) for value in signal_ids if value}
        if not wanted:
            return False
        changed = False
        for row in self.state.get("signals", []):
            if str(row.get("id", "")) in wanted and not bool(row.get("close_alert_sent", False)):
                row["close_alert_sent"] = True
                row["close_alerted_at"] = _utc_now()
                changed = True
        if changed:
            self.save()
        return changed

    def pending_near_exit(self, current_price: float, fraction: float = 0.25) -> bool:
        """Return True when a tracked trade is close to TP1 or SL.

        This is used only to increase watcher/UI refresh frequency; it does not
        alter trade direction, stops, targets or learning outcomes.
        """
        price = _safe_number(current_price)
        if price is None:
            return False
        threshold_fraction = max(0.10, min(0.60, float(fraction)))
        for row in self.state.get("signals", []):
            if row.get("status") != "PENDING":
                continue
            entry = _safe_number(row.get("entry"))
            stop = _safe_number(row.get("stop"))
            tp1 = _safe_number(row.get("tp1"))
            risk = _safe_number(row.get("risk"))
            if None in {entry, stop, tp1, risk} or risk is None or risk <= 0:
                continue
            threshold = max(0.35, risk * threshold_fraction)
            if min(abs(price - stop), abs(price - tp1)) <= threshold:
                return True
        return False

    @staticmethod
    def _evaluate_signal(signal: dict[str, Any], bars: pd.DataFrame, horizon: int) -> dict[str, Any] | None:
        if bars.empty:
            return None
        frame = bars.copy()
        frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        start = pd.to_datetime(signal["signal_time"], utc=True)
        future = frame[frame["time"] > start].head(horizon)
        if future.empty:
            return None
        side = signal["side"]
        entry = float(signal["entry"])
        stop = float(signal["stop"])
        tp1 = float(signal["tp1"])
        risk = max(1e-9, float(signal["risk"]))
        mfe = 0.0
        mae = 0.0
        outcome: str | None = None
        outcome_time = ""
        outcome_price = None
        bars_observed = 0

        for index, (_, row) in enumerate(future.iterrows(), start=1):
            high = float(row["high"])
            low = float(row["low"])
            if side == "BUY":
                stop_hit = low <= stop
                tp_hit = high >= tp1
                bar_mfe = max(0.0, high - entry)
                bar_mae = max(0.0, entry - low)
            else:
                stop_hit = high >= stop
                tp_hit = low <= tp1
                bar_mfe = max(0.0, entry - low)
                bar_mae = max(0.0, high - entry)

            # Same-bar ambiguity is conservative. Crucially, excursion used for
            # learning is capped at the first executable exit rather than the
            # full candle extreme after the trade would already be closed.
            if stop_hit:
                mae = max(mae, risk)
                outcome = "LOSS"
                outcome_price = stop
                outcome_time = pd.to_datetime(row["time"], utc=True).isoformat()
                bars_observed = index
                break
            if tp_hit:
                mfe = max(mfe, abs(tp1 - entry))
                outcome = "WIN"
                outcome_price = tp1
                outcome_time = pd.to_datetime(row["time"], utc=True).isoformat()
                bars_observed = index
                break
            mfe = max(mfe, bar_mfe)
            mae = max(mae, bar_mae)

        horizon_mfe = 0.0
        horizon_mae = 0.0
        if side == "BUY":
            horizon_mfe = max(0.0, float(future["high"].max()) - entry)
            horizon_mae = max(0.0, entry - float(future["low"].min()))
        else:
            horizon_mfe = max(0.0, entry - float(future["low"].min()))
            horizon_mae = max(0.0, float(future["high"].max()) - entry)

        if outcome is None and len(future) >= horizon:
            outcome = "TIMEOUT"
            outcome_time = pd.to_datetime(future.iloc[-1]["time"], utc=True).isoformat()
            outcome_price = float(future.iloc[-1]["close"])
            bars_observed = len(future)
        if outcome is None:
            adverse_r = mae / risk
            if adverse_r >= 0.50:
                return {
                    "interim": True,
                    "message": f"Adverse excursion reached {adverse_r:.2f}R before resolution; learning weights are not changed until the signal closes.",
                }
            return None
        pre_mfe_r = round(mfe / risk, 4)
        pre_mae_r = round(mae / risk, 4)
        return {
            "interim": False,
            "outcome": outcome,
            "outcome_time": outcome_time,
            "outcome_price": outcome_price,
            "mfe_r": pre_mfe_r,
            "mae_r": pre_mae_r,
            "pre_exit_mfe_r": pre_mfe_r,
            "pre_exit_mae_r": pre_mae_r,
            "horizon_mfe_r": round(horizon_mfe / risk, 4),
            "horizon_mae_r": round(horizon_mae / risk, 4),
            "runner_mfe_r": round(horizon_mfe / risk, 4) if outcome == "WIN" else pre_mfe_r,
            "bars_observed": bars_observed,
        }

    def review_pending(self, bars_by_timeframe: dict[str, pd.DataFrame]) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        completed: list[dict[str, Any]] = []
        changed = False
        for signal in self.state.get("signals", []):
            if signal.get("status") != "PENDING":
                continue
            frame = bars_by_timeframe.get(signal.get("timeframe", "M15"))
            if frame is None:
                continue
            review = self._evaluate_signal(signal, frame, self.horizon_bars)
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
            signal["close_alert_sent"] = False
            signal.update({key: value for key, value in review.items() if key != "interim"})
            self._update_from_completed(signal)
            completed.append(deepcopy(signal))
            changed = True
        if changed:
            self.save()
        return completed

    def _update_from_completed(self, signal: dict[str, Any]) -> None:
        outcome = signal["status"]
        counts = self.state.setdefault("counts", {"wins": 0, "losses": 0, "timeouts": 0})
        if outcome == "WIN":
            counts["wins"] = int(counts.get("wins", 0)) + 1
        elif outcome == "LOSS":
            counts["losses"] = int(counts.get("losses", 0)) + 1
        else:
            counts["timeouts"] = int(counts.get("timeouts", 0)) + 1
        mfe_r = _safe_number(signal.get("pre_exit_mfe_r", signal.get("mfe_r")))
        if mfe_r is not None:
            self.state.setdefault("target_mfe_r", []).append(round(max(0.0, min(5.0, mfe_r)), 4))
            self.state["target_mfe_r"] = self.state["target_mfe_r"][-500:]

        side_sign = 1 if signal["side"] == "BUY" else -1
        mistake_parts: list[str] = []
        successful_parts: list[str] = []
        active_votes = signal.get("feature_votes", {})
        if outcome in {"WIN", "LOSS"}:
            for feature in FEATURES:
                vote = int(active_votes.get(feature, 0))
                if vote == 0:
                    continue
                if feature == "entry_quality":
                    # Entry quality is non-directional: +1 good, -1 poor.
                    feature_correct = (outcome == "WIN" and vote > 0) or (outcome == "LOSS" and vote < 0)
                    aligned = vote > 0
                else:
                    aligned = vote == side_sign
                    feature_correct = (outcome == "WIN" and aligned) or (outcome == "LOSS" and not aligned)
                item = self.state.setdefault("features", {}).setdefault(
                    feature, {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
                )
                item["samples"] = int(item.get("samples", 0)) + 1
                if feature_correct:
                    item["alpha"] = float(item.get("alpha", 5.0)) + 1.0
                    if outcome == "WIN" and aligned:
                        successful_parts.append(feature.replace("_", " "))
                else:
                    item["beta"] = float(item.get("beta", 5.0)) + 1.0
                    if outcome == "LOSS" and aligned:
                        mistake_parts.append(feature.replace("_", " "))
                samples = int(item["samples"])
                posterior = float(item["alpha"]) / max(1e-9, float(item["alpha"]) + float(item["beta"]))
                evidence = min(1.0, samples / self.minimum_samples)
                desired = 1.0 + (posterior - 0.5) * 1.4 * evidence
                desired = max(0.70, min(1.30, desired))
                old = float(item.get("weight", 1.0))
                delta = max(-self.max_weight_change, min(self.max_weight_change, desired - old))
                item["weight"] = round(old + delta, 4)

        directional_pillars = ("market_structure", "anchored_vwap", "volume_profile")
        pattern_bits = [f"{name}:{int(active_votes.get(name, 0))}" for name in directional_pillars]
        pattern_key = f"{signal.get('side', '')}|" + "|".join(pattern_bits)
        pattern = self.state.setdefault("patterns", {}).setdefault(
            pattern_key, {"wins": 0, "losses": 0, "timeouts": 0, "samples": 0, "last_outcome": ""}
        )
        pattern["samples"] = int(pattern.get("samples", 0)) + 1
        bucket = "wins" if outcome == "WIN" else "losses" if outcome == "LOSS" else "timeouts"
        pattern[bucket] = int(pattern.get(bucket, 0)) + 1
        pattern["last_outcome"] = outcome
        pattern["updated_at"] = _utc_now()
        signal["successful_features"] = successful_parts
        signal["failed_features"] = mistake_parts
        signal["pattern_key"] = pattern_key

        review_text = (
            f"{signal['side']} signal closed as {outcome}; MFE {float(signal.get('mfe_r', 0)):.2f}R, "
            f"MAE {float(signal.get('mae_r', 0)):.2f}R."
        )
        if successful_parts:
            review_text += " Successful groups: " + ", ".join(successful_parts[:5]) + "."
        if mistake_parts:
            review_text += " Aligned groups that failed: " + ", ".join(mistake_parts[:5]) + "."
        elif outcome == "LOSS":
            review_text += " No single pillar is blamed; the loss is treated as a combined three-pillar error."
        self.state["last_review"] = review_text
        self.state.setdefault("reviews", []).append(
            {
                "reviewed_at": _utc_now(),
                "signal_id": signal.get("id"),
                "outcome": outcome,
                "mfe_r": signal.get("mfe_r"),
                "mae_r": signal.get("mae_r"),
                "successful_features": successful_parts,
                "failed_features": mistake_parts,
                "pattern_key": pattern_key,
                "summary": review_text,
            }
        )
        self.state["reviews"] = self.state["reviews"][-300:]

    def summary(self) -> AdaptiveLearningSummary:
        counts = self.state.get("counts", {})
        wins = int(counts.get("wins", 0))
        losses = int(counts.get("losses", 0))
        timeouts = int(counts.get("timeouts", 0))
        reviewed = wins + losses + timeouts
        decided = wins + losses
        samples = {
            name: int(self.state.get("features", {}).get(name, {}).get("samples", 0))
            for name in FEATURES
        }
        return AdaptiveLearningSummary(
            enabled=self.enabled,
            reviewed_signals=reviewed,
            wins=wins,
            losses=losses,
            timeouts=timeouts,
            win_rate=round((wins / decided * 100) if decided else 0.0, 2),
            indicator_weights=self.weights(),
            indicator_samples=samples,
            target_r_multipliers=self.target_multipliers(),
            last_review=str(self.state.get("last_review", "")),
            safeguards=[
                f"Feature weights remain at prior values until evidence accumulates; full adaptation begins near {self.minimum_samples} observations per feature.",
                f"A single completed signal can move a feature weight by no more than {self.max_weight_change:.2f}.",
                "Adverse movement is reviewed immediately, but weights change only after TP1, SL, or the fixed outcome horizon.",
                "Weights are bounded between 0.70 and 1.30; the program never rewrites its own source code.",
                f"Directional confidence is capped at {self.confidence_cap()}% while the reviewed sample is small.",
                "Duplicate pending signals, post-loss cooldowns and weak entries without two-of-three pillar confirmation are blocked.",
                "Target learning uses pre-exit excursion; full candle movement after an exit cannot inflate future targets.",
            ],
        )

    def reset(self) -> None:
        self.state = _default_state()
        self.save()
