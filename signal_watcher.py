from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

from gold_web_terminal.config import Settings
from gold_web_terminal.market_data import TwelveDataClient
from gold_web_terminal.sma18_ema_strategy import (
    evaluate_sma18_strategy,
    process_sma18_alerts,
    send_telegram_message,
    snapshot_to_dict,
)

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
BUILD_VERSION = "5.9.0-sma18-m15-telegram"


def _state_path() -> Path:
    raw = os.getenv("SMA18_STATE_PATH", "data/sma18_ema_state.json")
    path = Path(raw)
    return path if path.is_absolute() else APP_DIR / path


def _write_change_marker(path: Path, changed: bool) -> None:
    if not changed:
        return
    marker = path.parent / ".sma18_state_changed"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("true", encoding="utf-8")


def _clear_change_marker(path: Path) -> None:
    marker = path.parent / ".sma18_state_changed"
    if marker.exists():
        try:
            marker.unlink()
        except OSError:
            pass


def run_once(*, dry_run: bool = False, emit: bool = True) -> dict:
    settings = Settings.from_env()
    if not settings.twelve_data_api_key:
        raise RuntimeError("TWELVE_DATA_API_KEY is required for the M15 signal watcher.")

    bundle = TwelveDataClient(settings.twelve_data_api_key).fetch_bundle(
        settings.market_symbol,
        TIMEFRAMES,
        settings.bars_per_timeframe,
    )
    snapshot = evaluate_sma18_strategy(bundle.frames, symbol=bundle.symbol)
    state_path = _state_path()
    delivery = process_sma18_alerts(
        snapshot,
        state_path,
        dry_run=dry_run,
        max_alert_age_minutes=max(15, int(os.getenv("SMA18_MAX_ALERT_AGE_MINUTES", "20"))),
    )
    _write_change_marker(state_path, delivery.state_changed)

    output = {
        "build": BUILD_VERSION,
        "engine": "SMA18_EMA_M15",
        "data_time": bundle.data_time,
        "price": snapshot.latest_price,
        "state": snapshot.current_signal.side if snapshot.current_signal else "WAIT",
        "signal_confirmed": snapshot.current_signal is not None,
        "sent": delivery.sent_signal_ids,
        "errors": delivery.errors,
        "state_changed": delivery.state_changed,
        "strategy": snapshot_to_dict(snapshot),
    }
    if emit:
        print(json.dumps(output, indent=2, default=str))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="AurumEdge M15 SMA18/EMA Telegram signal watcher")
    parser.add_argument("--dry-run", action="store_true", help="Calculate without sending Telegram or writing state")
    parser.add_argument("--test-alert", action="store_true", help="Send a Telegram test notification")
    # Kept for compatibility with the previous GitHub workflow. M15 signals are
    # evaluated only after candle close, so burst mode performs one normal pass.
    parser.add_argument("--burst", action="store_true", help="Compatibility alias for one normal M15 check")
    args = parser.parse_args()

    state_path = _state_path()
    _clear_change_marker(state_path)

    if args.test_alert:
        delivered, error = send_telegram_message(
            "🟡 <b>AURUMEDGE TEST</b>\n\nM15 SMA18/EMA Telegram delivery is configured."
        )
        print(json.dumps({"delivered": delivered, "errors": [] if delivered else [error]}, indent=2))
        raise SystemExit(0 if delivered else 1)

    run_once(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
