from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from gold_web_terminal.adaptive_engine import AdaptiveEngine, derive_feature_votes
from gold_web_terminal.alerts import (
    AlertConfig,
    AlertState,
    build_alert_events,
    dispatch_events,
    is_primary_entry_live,
    record_non_alert_states,
    send_test_alert,
)
from gold_web_terminal.config import Settings
from gold_web_terminal.indicators import add_indicators, summarize_indicators
from gold_web_terminal.liquidity import analyze_liquidity
from gold_web_terminal.macro_data import fetch_macro_confirmation
from gold_web_terminal.market_data import TwelveDataClient
from gold_web_terminal.risk_engine import RiskInputs
from gold_web_terminal.strategy import build_technical_report

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]


def risk_inputs(settings: Settings) -> RiskInputs:
    return RiskInputs(
        account_balance=settings.account_balance,
        risk_percent=settings.risk_percent,
        requested_lot=settings.requested_lot,
        contract_size=settings.contract_size,
        lot_step=settings.lot_step,
        min_lot=settings.min_lot,
        maximum_risk_dollars=settings.maximum_risk_dollars,
        spread_price=settings.spread_price,
        slippage_price=settings.slippage_price,
        minimum_stop_atr=settings.minimum_stop_atr,
        maximum_stop_atr=settings.maximum_stop_atr,
    )


def _resolve_path(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else APP_DIR / path


def _text_or_empty(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8") if path.exists() else ""
    except OSError:
        return ""


def _write_change_marker(path: Path, name: str, changed: bool) -> None:
    marker = path.parent / name
    if changed:
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("true", encoding="utf-8")
    elif marker.exists():
        marker.unlink()


def run_once(dry_run: bool = False) -> dict:
    settings = Settings.from_env()
    if not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is required for the signal watcher.")

    bundle = TwelveDataClient(settings.twelve_data_api_key).fetch_bundle(
        settings.market_symbol, TIMEFRAMES, settings.bars_per_timeframe
    )
    frames = {tf: add_indicators(bundle.frames[tf]) for tf in TIMEFRAMES}
    indicators = [summarize_indicators(frames[tf], tf) for tf in TIMEFRAMES]
    liquidity = [analyze_liquidity(frames[tf], tf) for tf in ["M15", "H1", "H4", "D1"]]

    adaptive_path = _resolve_path(settings.adaptive_state_path)
    before_adaptive = _text_or_empty(adaptive_path)
    adaptive = AdaptiveEngine(
        adaptive_path,
        enabled=settings.adaptive_learning,
        minimum_samples=settings.adaptive_min_samples,
        horizon_bars=settings.adaptive_horizon_bars,
        max_weight_change=settings.adaptive_max_weight_change,
    )
    completed = adaptive.review_pending(frames) if not dry_run else []
    summary = adaptive.summary()
    risk = risk_inputs(settings)

    # First calculate the three-pillar direction. Macro is fetched afterward
    # only for display in Telegram/app; it can never stop or reverse a signal.
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
        adaptive_summary=summary,
        risk_inputs=risk,
        macro_required_for_entry=False,
    )

    macro = None
    macro_error = ""
    if settings.macro_enabled:
        try:
            gold_h1 = frames["H1"][["time", "close"]].copy()
            macro = fetch_macro_confirmation(
                settings.twelve_data_api_key,
                settings.dxy_symbol,
                settings.us10y_symbol,
                gold_h1,
                preliminary.market_state,
            )
        except Exception as exc:  # display-only data must not fail the watcher
            macro_error = f"{exc.__class__.__name__}: {exc}"

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
        adaptive_summary=summary,
        risk_inputs=risk,
        macro=macro,
        macro_required_for_entry=False,
    )

    signal_time = frames["M15"].iloc[-1]["time"]
    feature_votes = derive_feature_votes(indicators, liquidity, macro, report.market_state)
    report = adaptive.apply_capital_preservation(report, signal_time, feature_votes)

    config = AlertConfig.from_env()
    alert_state_path = _resolve_path(settings.alert_state_path)
    before_alert = _text_or_empty(alert_state_path)
    alert_state = AlertState(alert_state_path)
    events = build_alert_events(report, None)
    entry_live = is_primary_entry_live(report)

    if dry_run:
        sent, errors = [], []
        registered = False
    else:
        sent, errors = dispatch_events(events, config, alert_state)
        record_non_alert_states(report, None, config, alert_state)
        eligible = entry_live and report.confidence >= config.minimum_confidence
        registered = adaptive.register_signal(report, signal_time, feature_votes, timeframe="M15") if eligible else False
        adaptive.save()

    after_alert = _text_or_empty(alert_state_path)
    after_adaptive = _text_or_empty(adaptive_path)
    alert_changed = before_alert != after_alert
    adaptive_changed = before_adaptive != after_adaptive
    _write_change_marker(alert_state_path, ".alert_state_changed", alert_changed)
    _write_change_marker(adaptive_path, ".adaptive_state_changed", adaptive_changed)

    macro_context = {
        "dxy": macro.dxy.direction if macro is not None else "UNAVAILABLE",
        "us10y": macro.us10y.direction if macro is not None else "UNAVAILABLE",
        "non_blocking": True,
        "error": macro_error,
    }
    output = {
        "engine": "THREE_PILLAR",
        "data_time": bundle.data_time,
        "price": report.last_price,
        "state": report.market_state,
        "confidence": report.confidence,
        "entry_live": entry_live,
        "events": [event.event_id for event in events],
        "sent": sent,
        "errors": errors,
        "macro_context": macro_context,
        "learning": {
            "reviews_this_run": len(completed),
            "registered": registered,
            "counts": adaptive.state.get("counts", {}),
            "state_changed": adaptive_changed,
        },
        "alert_state_changed": alert_changed,
    }
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AurumEdge three-pillar XAU/USD signal watcher")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without sending alerts or updating learning")
    parser.add_argument("--test-alert", action="store_true", help="Send a test alert using configured channels")
    args = parser.parse_args()
    if args.test_alert:
        ok, errors = send_test_alert(AlertConfig.from_env())
        print(json.dumps({"delivered": ok, "errors": errors}, indent=2))
        raise SystemExit(0 if ok else 1)
    run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
