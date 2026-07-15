from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import TechnicalReport


FIELDS = [
    "saved_at_utc",
    "symbol",
    "data_time",
    "data_source",
    "market_state",
    "signal_label",
    "regime",
    "confidence",
    "buy_score",
    "sell_score",
    "selected_side",
    "setup_json",
    "notes",
]


def save_report(path: str, report: TechnicalReport, selected_side: str, notes: str = "") -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    setup = report.buy_setup if selected_side.upper() == "BUY" else report.sell_setup
    row: dict[str, Any] = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": report.symbol,
        "data_time": report.data_time,
        "data_source": report.data_source,
        "market_state": report.market_state,
        "signal_label": report.signal_label,
        "regime": report.regime,
        "confidence": report.confidence,
        "buy_score": report.buy_score,
        "sell_score": report.sell_score,
        "selected_side": selected_side.upper(),
        "setup_json": json.dumps(setup.model_dump(mode="json"), ensure_ascii=False),
        "notes": notes,
    }
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return target
