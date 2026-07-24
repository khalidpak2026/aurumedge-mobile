from __future__ import annotations

"""AurumEdge Mobile v5.8.1 scheduled watcher.

Normal GitHub scheduling remains five minutes. When an entry or TP1/SL is near,
--burst performs up to four checks approximately 45 seconds apart inside the
same workflow job.
"""

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

from dotenv import load_dotenv

from gold_web_terminal.adaptive_engine import AdaptiveEngine, derive_feature_votes
from gold_web_terminal.alerts import (
    AlertConfig,
    AlertState,
    build_alert_events,
    dispatch_events,
    record_non_alert_states,
    send_test_alert,
)
from gold_web_terminal.config import Settings
from gold_web_terminal.indicators import add_indicators, summarize_indicators
from gold_web_terminal.liquidity import analyze_liquidity
from gold_web_terminal.market_data import TwelveDataClient
from gold_web_terminal.risk_engine import RiskInputs
from gold_web_terminal.strategy import build_technical_report

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]


def _setting(settings: Any, name: str, default: Any) -> Any:
    value = getattr(settings, name, default)
    return default if value is None else value


def risk_inputs(settings: Settings) -> RiskInputs:
    values = {
        "account_balance": _setting(settings, "account_balance", 10000.0),
        "risk_percent": _setting(settings, "risk_percent", 1.0),
        "requested_lot": _setting(settings, "requested_lot", 0.10),
        "contract_size": _setting(settings, "contract_size", 100.0),
        "lot_step": _setting(settings, "lot_step", 0.01),
        "min_lot": _setting(settings, "min_lot", 0.01),
        "maximum_risk_dollars": _setting(settings, "maximum_risk_dollars", 0.0),
        "spread_price": _setting(settings, "spread_price", 0.0),
        "slippage_price": _setting(settings, "slippage_price", 0.0),
        "minimum_stop_atr": _setting(settings, "minimum_stop_atr", 0.55),
        "maximum_stop_atr": _setting(settings, "maximum_stop_atr", 1.60),
    }
    try:
        return RiskInputs(**values)
    except TypeError:
        # Compatibility with an older risk model exposing fewer fields.
        supported = getattr(RiskInputs, "model_fields", None) or getattr(RiskInputs, "__fields__", {})
        return RiskInputs(**{key: value for key, value in values.items() if key in supported})


def _resolve_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else APP_DIR / path


def _text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return ""


def _write_change_marker(path: Path, name: str, changed: bool) -> None:
    marker = path.parent / name
    marker.parent.mkdir(parents=True, exist_ok=True)
    if changed:
        marker.write_text("true", encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def _fetch_macro_context(settings: Settings, frames: dict[str, Any], preliminary_state: str) -> Any | None:
    if not bool(_setting(settings, "macro_enabled", True)):
        return None
    api_key = str(_setting(settings, "twelve_data_api_key", ""))
    dxy_symbol = str(_setting(settings, "dxy_symbol", "DXY"))
    us10y_symbol = str(_setting(settings, "us10y_symbol", "US10Y"))
    gold_h1 = frames["H1"][["time", "close"]].copy()
    try:
        from gold_web_terminal.macro_data import fetch_macro_confirmation

        return fetch_macro_confirmation(api_key, dxy_symbol, us10y_symbol, gold_h1, preliminary_state)
    except Exception:
        try:
            from gold_web_terminal.macro_mobile_v542 import fetch_macro_confirmation

            return fetch_macro_confirmation(api_key, dxy_symbol, us10y_symbol, gold_h1, preliminary_state)
        except Exception:
            return None


def _setup_value(report: Any, name: str, default: Any = None) -> Any:
    setup = getattr(report, "active_setup", None)
    if isinstance(setup, dict):
        return setup.get(name, default)
    return getattr(setup, name, default) if setup is not None else default


def run_once(dry_run: bool = False, emit: bool = True) -> dict[str, Any]:
    settings = Settings.from_env()
    api_key = str(_setting(settings, "twelve_data_api_key", "")).strip()
    if not api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is required for the signal watcher.")

    symbol = str(_setting(settings, "market_symbol", "XAU/USD"))
    bars = int(_setting(settings, "bars_per_timeframe", 500))
    bundle = TwelveDataClient(api_key).fetch_bundle(symbol, TIMEFRAMES, bars)
    frames = {tf: add_indicators(bundle.frames[tf]) for tf in TIMEFRAMES}
    indicators = [summarize_indicators(frames[tf], tf) for tf in TIMEFRAMES]
    liquidity = [analyze_liquidity(frames[tf], tf) for tf in ["M15", "H1", "H4", "D1"]]

    adaptive_path = _resolve_path(str(_setting(settings, "adaptive_state_path", "data/adaptive_state.json")))
    alert_state_path = _resolve_path(str(_setting(settings, "alert_state_path", "data/alert_state.json")))
    before_adaptive = _text_or_empty(adaptive_path)
    before_alert = _text_or_empty(alert_state_path)
    adaptive = AdaptiveEngine(
        adaptive_path,
        enabled=bool(_setting(settings, "adaptive_learning", True)),
        minimum_samples=int(_setting(settings, "adaptive_min_samples", 20)),
        horizon_bars=int(_setting(settings, "adaptive_horizon_bars", 12)),
        max_weight_change=float(_setting(settings, "adaptive_max_weight_change", 0.05)),
    )

    review_outcomes = [] if dry_run else adaptive.review_pending(frames)
    risk = risk_inputs(settings)
    preliminary = build_technical_report(
        symbol=bundle.symbol,
        data_time=bundle.data_time,
        price=bundle.last_price,
        indicators=indicators,
        liquidity=liquidity,
        data_source=bundle.source,
        digits=2,
        adaptive_weights=adaptive.weights(),
        target_multipliers=adaptive.target_multipliers(),
        adaptive_summary=adaptive.summary(),
        risk_inputs=risk,
        macro_required_for_entry=False,
    )
    macro = _fetch_macro_context(settings, frames, str(getattr(preliminary, "market_state", "STUCK")))
    report = build_technical_report(
        symbol=bundle.symbol,
        data_time=bundle.data_time,
        price=bundle.last_price,
        indicators=indicators,
        liquidity=liquidity,
        data_source=bundle.source,
        digits=2,
        adaptive_weights=adaptive.weights(),
        target_multipliers=adaptive.target_multipliers(),
        adaptive_summary=adaptive.summary(),
        risk_inputs=risk,
        macro=macro,
        macro_required_for_entry=False,  # DXY/US10Y are display-only.
    )
    signal_time = frames["M15"].iloc[-1]["time"]
    feature_votes = dict(getattr(report, "pillar_votes", {}) or {})
    if not feature_votes:
        feature_votes = derive_feature_votes(indicators, liquidity, macro, getattr(report, "market_state", "STUCK"))
    report = adaptive.apply_capital_preservation(report, signal_time, feature_votes)

    config = AlertConfig.from_env()
    alert_state = AlertState(alert_state_path)
    events = build_alert_events(report, review_outcomes=review_outcomes)
    if dry_run:
        sent, errors = [], []
        registered = False
    else:
        sent, errors = dispatch_events(events, config, alert_state)
        record_non_alert_states(report, None, config, alert_state)
        entry_events = [event for event in events if event.kind == "ENTRY"]
        delivered_entry = next(
            (
                event
                for event in entry_events
                if event.event_id in sent or alert_state.was_sent(event.event_id)
            ),
            None,
        )
        registered = False
        if delivered_entry is not None:
            registered = adaptive.register_signal(
                report,
                signal_time,
                feature_votes,
                timeframe="M15",
                delivery_event_id=delivered_entry.event_id,
            )
        adaptive.save()

    price = float(getattr(report, "last_price", bundle.last_price))
    state = str(getattr(report, "market_state", "STUCK"))
    entry_live = bool(getattr(report, "entry_live", False) or _setup_value(report, "entry_live", False))
    near_entry = bool(getattr(report, "near_entry", False) or _setup_value(report, "near_entry", False))
    near_exit = adaptive.pending_near_exit(price)
    recommended = 30 if entry_live or near_exit else 45 if near_entry else 120 if state in {"BUY", "SELL"} else 300

    after_adaptive = _text_or_empty(adaptive_path)
    after_alert = _text_or_empty(alert_state_path)
    adaptive_changed = before_adaptive != after_adaptive
    alert_changed = before_alert != after_alert
    _write_change_marker(adaptive_path, ".adaptive_state_changed", adaptive_changed)
    _write_change_marker(alert_state_path, ".alert_state_changed", alert_changed)

    output = {
        "build": "5.8.1-mobile-entry-lifecycle",
        "data_time": bundle.data_time,
        "price": price,
        "primary_state": state,
        "primary_confidence": int(getattr(report, "confidence", 0)),
        "execution_label": str(getattr(report, "execution_label", "")),
        "entry_live": entry_live,
        "near_entry": near_entry,
        "near_exit": near_exit,
        "recommended_poll_seconds": recommended,
        "events": [event.event_id for event in events],
        "sent": sent,
        "errors": errors,
        "registered_delivered_entry": registered,
        "pending_trades": adaptive.pending_trade_count(),
        "review_outcomes": review_outcomes,
        "last_review": str(adaptive.state.get("last_review", "")),
        "pillar_votes": getattr(report, "pillar_votes", {}),
        "macro_context": {
            "dxy": getattr(getattr(macro, "dxy", None), "direction", "UNAVAILABLE") if macro is not None else "UNAVAILABLE",
            "us10y": getattr(getattr(macro, "us10y", None), "direction", "UNAVAILABLE") if macro is not None else "UNAVAILABLE",
            "blocks_signal": False,
        },
        "alert_channels": {"telegram": config.telegram_enabled, "email": config.email_enabled},
        "adaptive_state_changed": adaptive_changed,
        "alert_state_changed": alert_changed,
    }
    if emit:
        print(json.dumps(output, indent=2, default=str))
    return output


def run_burst(dry_run: bool = False) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    maximum_checks = max(1, min(4, int(os.getenv("BURST_MAX_CHECKS", "4"))))
    interval = max(30, min(90, int(os.getenv("BURST_INTERVAL_SECONDS", "45"))))
    for index in range(maximum_checks):
        result = run_once(dry_run=dry_run, emit=True)
        results.append(result)
        should_continue = bool(result["near_entry"] or result["entry_live"] or result["near_exit"])
        if not should_continue or index >= maximum_checks - 1:
            break
        time.sleep(interval)
    final = dict(results[-1])
    final["burst_runs"] = len(results)
    final["burst_interval_seconds"] = interval
    final["burst_history"] = [
        {
            "data_time": item["data_time"],
            "price": item["price"],
            "entry_live": item["entry_live"],
            "near_entry": item["near_entry"],
            "near_exit": item["near_exit"],
            "sent": item["sent"],
        }
        for item in results
    ]
    print(json.dumps({"burst_summary": final}, indent=2, default=str))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="AurumEdge v5.8.1 XAU/USD signal watcher")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without alerts or learning changes")
    parser.add_argument("--test-alert", action="store_true", help="Send one notification-channel test")
    parser.add_argument("--burst", action="store_true", help="Use adaptive near-entry/near-exit burst checks")
    args = parser.parse_args()
    if args.test_alert:
        ok, errors = send_test_alert(AlertConfig.from_env())
        print(json.dumps({"delivered": ok, "errors": errors}, indent=2))
        raise SystemExit(0 if ok else 1)
    if args.burst:
        run_burst(dry_run=args.dry_run)
    else:
        run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
