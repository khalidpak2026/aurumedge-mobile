from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from gold_web_terminal.adaptive_engine import AdaptiveEngine, derive_feature_votes
from gold_web_terminal.alerts import AlertConfig, AlertState, build_alert_events, dispatch_events, record_non_alert_states, send_test_alert
from gold_web_terminal.config import Settings
from gold_web_terminal.fvg_strategy import detect_four_hour_fvg_signal
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

    adaptive_path = Path(settings.adaptive_state_path)
    if not adaptive_path.is_absolute():
        adaptive_path = APP_DIR / adaptive_path
    adaptive = AdaptiveEngine(
        adaptive_path,
        enabled=settings.adaptive_learning,
        minimum_samples=settings.adaptive_min_samples,
        horizon_bars=settings.adaptive_horizon_bars,
        max_weight_change=settings.adaptive_max_weight_change,
    )
    summary = adaptive.summary()
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
        adaptive_summary=summary,
        risk_inputs=risk,
        macro_required_for_entry=False,
    )
    gold_h1 = frames["H1"][["time", "close"]].copy()
    macro = fetch_macro_confirmation(
        settings.twelve_data_api_key,
        settings.dxy_symbol,
        settings.us10y_symbol,
        gold_h1,
        preliminary.market_state,
    ) if settings.macro_enabled else None

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
        macro_required_for_entry=settings.macro_required_for_entry,
    )
    signal_time = frames["M15"].iloc[-1]["time"]
    votes = derive_feature_votes(indicators, liquidity, macro, report.market_state)
    report = adaptive.apply_capital_preservation(report, signal_time, votes)

    special = detect_four_hour_fvg_signal(
        frames,
        indicators=indicators,
        macro=macro,
        primary_state=report.market_state,
        risk_inputs=risk,
        digits=2,
    ) if settings.h4_fvg_strategy_enabled else None
    if special is not None:
        report.special_signals = [special]

    config = AlertConfig.from_env()
    state_path = Path(settings.alert_state_path)
    if not state_path.is_absolute():
        state_path = APP_DIR / state_path
    state = AlertState(state_path)
    before = state.path.read_text(encoding="utf-8") if state.path.exists() else ""
    events = build_alert_events(report, special)
    if dry_run:
        sent, errors = [], []
    else:
        sent, errors = dispatch_events(events, config, state)
        record_non_alert_states(report, special, config, state)
    after = state.path.read_text(encoding="utf-8") if state.path.exists() else ""
    changed = before != after
    marker = state.path.parent / ".alert_state_changed"
    if changed:
        marker.write_text("true", encoding="utf-8")
    elif marker.exists():
        marker.unlink()

    output = {
        "data_time": bundle.data_time,
        "price": report.last_price,
        "primary_state": report.market_state,
        "primary_confidence": report.confidence,
        "special_state": special.state if special else "DISABLED",
        "special_side": special.side if special else "NONE",
        "special_confidence": special.confidence if special else 0,
        "events": [event.event_id for event in events],
        "sent": sent,
        "errors": errors,
        "alert_channels": {
            "telegram": config.telegram_enabled,
            "email": config.email_enabled,
        },
        "state_changed": changed,
    }
    print(json.dumps(output, indent=2))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AurumEdge scheduled XAU/USD signal watcher")
    parser.add_argument("--dry-run", action="store_true", help="Calculate signals but do not send alerts")
    parser.add_argument("--test-alert", action="store_true", help="Send a test alert using configured channels")
    args = parser.parse_args()
    if args.test_alert:
        ok, errors = send_test_alert(AlertConfig.from_env())
        print(json.dumps({"delivered": ok, "errors": errors}, indent=2))
        raise SystemExit(0 if ok else 1)
    run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
