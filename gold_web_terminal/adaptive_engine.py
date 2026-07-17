from __future__ import annotations

import json
import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .models import AdaptiveLearningSummary, IndicatorSnapshot, LiquiditySnapshot, MacroConfirmation, TechnicalReport


FEATURES = (
    "ema_trend",
    "momentum",
    "adx_dmi",
    "vwap",
    "volume",
    "liquidity",
    "breakout",
    "macro",
    "entry_quality",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
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
    """Return -1 bearish, 0 neutral, +1 bullish votes for stable feature groups."""
    if not indicators:
        return {name: 0 for name in FEATURES}
    h1 = _snapshot(indicators, "H1")
    h4 = _snapshot(indicators, "H4")
    m15 = _snapshot(indicators, "M15")

    votes: dict[str, int] = {name: 0 for name in FEATURES}

    trend_values = [h1.trend, h4.trend]
    if trend_values.count("bullish") >= 2 or (h1.trend == "bullish" and (h1.ema20_slope_atr or 0) > 0):
        votes["ema_trend"] = 1
    elif trend_values.count("bearish") >= 2 or (h1.trend == "bearish" and (h1.ema20_slope_atr or 0) < 0):
        votes["ema_trend"] = -1

    momentum_score = 0
    for item in (m15, h1):
        if item.momentum == "bullish":
            momentum_score += 1
        elif item.momentum == "bearish":
            momentum_score -= 1
        if (item.macd_hist_slope or 0) > 0:
            momentum_score += 0.5
        elif (item.macd_hist_slope or 0) < 0:
            momentum_score -= 0.5
    votes["momentum"] = 1 if momentum_score >= 1.5 else -1 if momentum_score <= -1.5 else 0

    dmi_score = 0
    for item in (h1, h4):
        if (item.adx14 or 0) >= 18:
            if (item.plus_di or 0) > (item.minus_di or 0):
                dmi_score += 1
            elif (item.minus_di or 0) > (item.plus_di or 0):
                dmi_score -= 1
    votes["adx_dmi"] = 1 if dmi_score > 0 else -1 if dmi_score < 0 else 0

    vwap_score = 0
    for item in (m15, h1):
        if item.vwap is None:
            continue
        vwap_score += 1 if item.close > item.vwap else -1
    votes["vwap"] = 1 if vwap_score > 0 else -1 if vwap_score < 0 else 0

    volume_score = 0
    for item in (m15, h1):
        if (item.volume_ratio or 0) >= 1.05:
            if (item.volume_delta_proxy or 0) > 0:
                volume_score += 1
            elif (item.volume_delta_proxy or 0) < 0:
                volume_score -= 1
    votes["volume"] = 1 if volume_score > 0 else -1 if volume_score < 0 else 0

    bull_sweep = any(item.sweep_below is not None or item.trap_type == "bear_trap" for item in liquidity)
    bear_sweep = any(item.sweep_above is not None or item.trap_type == "bull_trap" for item in liquidity)
    if bull_sweep and not bear_sweep:
        votes["liquidity"] = 1
    elif bear_sweep and not bull_sweep:
        votes["liquidity"] = -1

    breakout_score = 0
    for item in (m15, h1):
        if item.breakout_up:
            breakout_score += 1
        if item.breakout_down:
            breakout_score -= 1
    votes["breakout"] = 1 if breakout_score > 0 else -1 if breakout_score < 0 else 0

    if macro is not None:
        if macro.macro_bias == "BULLISH_GOLD":
            votes["macro"] = 1
        elif macro.macro_bias == "BEARISH_GOLD":
            votes["macro"] = -1

    # Entry-quality vote is deliberately conservative: it penalizes chasing an extended candle.
    atr = m15.atr14 or h1.atr14 or 0.0
    distance_from_ema = abs(m15.close - (m15.ema20 or m15.close))
    if atr > 0 and distance_from_ema > atr * 1.15:
        if side_hint == "BUY" and m15.close > (m15.ema20 or m15.close):
            votes["entry_quality"] = -1
        elif side_hint == "SELL" and m15.close < (m15.ema20 or m15.close):
            votes["entry_quality"] = 1
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
        samples = [float(x) for x in self.state.get("target_mfe_r", []) if _safe_number(x) is not None]
        if len(samples) < self.minimum_samples:
            return {"tp1": 0.80, "tp2": 1.30, "tp3": 1.80}
        values = np.asarray(samples[-250:], dtype=float)
        # Chosen to target progressively lower hit rates while remaining bounded and realistic.
        tp1 = float(np.quantile(values, 0.40))
        tp2 = float(np.quantile(values, 0.65))
        tp3 = float(np.quantile(values, 0.82))
        tp1 = max(0.60, min(1.00, tp1))
        tp2 = max(tp1 + 0.25, min(1.55, tp2))
        tp3 = max(tp2 + 0.25, min(2.10, tp3))
        return {"tp1": round(tp1, 2), "tp2": round(tp2, 2), "tp3": round(tp3, 2)}

    def register_signal(
        self,
        report: TechnicalReport,
        signal_time: Any,
        feature_votes: dict[str, int],
        timeframe: str = "M15",
    ) -> bool:
        if not self.enabled or report.active_setup is None or report.market_state not in {"BUY", "SELL"}:
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
                "feature_votes": {name: int(feature_votes.get(name, 0)) for name in FEATURES},
                "status": "PENDING",
                "interim_review": "",
            }
        )
        self.state["signals"] = self.state["signals"][-500:]
        self.save()
        return True

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

        for _, row in future.iterrows():
            high = float(row["high"])
            low = float(row["low"])
            if side == "BUY":
                mfe = max(mfe, high - entry)
                mae = max(mae, entry - low)
                stop_hit = low <= stop
                tp_hit = high >= tp1
            else:
                mfe = max(mfe, entry - low)
                mae = max(mae, high - entry)
                stop_hit = high >= stop
                tp_hit = low <= tp1
            # Same-bar ambiguity is treated conservatively as a loss.
            if stop_hit:
                outcome = "LOSS"
                outcome_price = stop
                outcome_time = pd.to_datetime(row["time"], utc=True).isoformat()
                break
            if tp_hit:
                outcome = "WIN"
                outcome_price = tp1
                outcome_time = pd.to_datetime(row["time"], utc=True).isoformat()
                break

        if outcome is None and len(future) >= horizon:
            outcome = "TIMEOUT"
            outcome_time = pd.to_datetime(future.iloc[-1]["time"], utc=True).isoformat()
            outcome_price = float(future.iloc[-1]["close"])
        if outcome is None:
            adverse_r = mae / risk
            if adverse_r >= 0.50:
                return {
                    "interim": True,
                    "message": f"Adverse excursion reached {adverse_r:.2f}R before resolution; learning weights are not changed until the signal closes.",
                }
            return None
        return {
            "interim": False,
            "outcome": outcome,
            "outcome_time": outcome_time,
            "outcome_price": outcome_price,
            "mfe_r": round(mfe / risk, 4),
            "mae_r": round(mae / risk, 4),
            "bars_observed": len(future),
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
        mfe_r = _safe_number(signal.get("mfe_r"))
        if mfe_r is not None:
            self.state.setdefault("target_mfe_r", []).append(round(max(0.0, min(5.0, mfe_r)), 4))
            self.state["target_mfe_r"] = self.state["target_mfe_r"][-500:]

        side_sign = 1 if signal["side"] == "BUY" else -1
        mistake_parts: list[str] = []
        if outcome in {"WIN", "LOSS"}:
            for feature in FEATURES:
                vote = int(signal.get("feature_votes", {}).get(feature, 0))
                if vote == 0:
                    continue
                aligned = vote == side_sign
                feature_correct = (outcome == "WIN" and aligned) or (outcome == "LOSS" and not aligned)
                item = self.state.setdefault("features", {}).setdefault(
                    feature, {"alpha": 5.0, "beta": 5.0, "weight": 1.0, "samples": 0}
                )
                item["samples"] = int(item.get("samples", 0)) + 1
                if feature_correct:
                    item["alpha"] = float(item.get("alpha", 5.0)) + 1.0
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

        review_text = (
            f"{signal['side']} signal closed as {outcome}; MFE {float(signal.get('mfe_r', 0)):.2f}R, "
            f"MAE {float(signal.get('mae_r', 0)):.2f}R."
        )
        if mistake_parts:
            review_text += " Aligned feature groups that failed: " + ", ".join(mistake_parts[:5]) + "."
        elif outcome == "LOSS":
            review_text += " No single indicator is blamed; the loss is treated as a combined-model error."
        self.state["last_review"] = review_text
        self.state.setdefault("reviews", []).append(
            {
                "reviewed_at": _utc_now(),
                "signal_id": signal.get("id"),
                "outcome": outcome,
                "mfe_r": signal.get("mfe_r"),
                "mae_r": signal.get("mae_r"),
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
            ],
        )

    def reset(self) -> None:
        self.state = _default_state()
        self.save()
