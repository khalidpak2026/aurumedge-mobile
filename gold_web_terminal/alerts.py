from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
import json
import math
import os
from pathlib import Path
import smtplib
import ssl
from typing import Any, Iterable

import requests

from .models import FourHourFVGSignal, TechnicalReport


@dataclass(slots=True)
class AlertConfig:
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""
    minimum_confidence: int = 65
    forming_alerts: bool = False  # retained for backward compatibility

    @classmethod
    def from_env(cls) -> "AlertConfig":
        try:
            port = int(os.getenv("SMTP_PORT", "587"))
        except ValueError:
            port = 587
        try:
            confidence = int(os.getenv("ALERT_MIN_CONFIDENCE", "65"))
        except ValueError:
            confidence = 65
        return cls(
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            smtp_host=os.getenv("SMTP_HOST", "").strip(),
            smtp_port=port,
            smtp_username=os.getenv("SMTP_USERNAME", "").strip(),
            smtp_password=os.getenv("SMTP_APP_PASSWORD", os.getenv("SMTP_PASSWORD", "")).strip(),
            email_from=os.getenv("ALERT_EMAIL_FROM", os.getenv("SMTP_USERNAME", "")).strip(),
            email_to=os.getenv("ALERT_EMAIL_TO", "").strip(),
            minimum_confidence=max(50, min(90, confidence)),
            forming_alerts=os.getenv("ALERT_FORMING_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"},
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.smtp_username and self.smtp_password and self.email_to)

    @property
    def enabled(self) -> bool:
        return self.telegram_enabled or self.email_enabled


@dataclass(slots=True)
class AlertEvent:
    event_id: str
    title: str
    message: str
    category: str
    confidence: int
    transition_key: str = ""
    transition_value: str = ""
    symbol: str = "XAU/USD"
    side: str = ""
    signal_id: str = ""


class AlertState:
    """Backward-compatible alert state with entry/close lifecycle tracking."""

    def __init__(self, path: str | Path = "data/alert_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {
            "version": 2,
            "sent": {},
            "transitions": {},
            "open_entries": {},
            "updated_at": "",
        }
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except Exception:
                pass
        if not isinstance(self.data.get("sent"), dict):
            self.data["sent"] = {}
        if not isinstance(self.data.get("transitions"), dict):
            self.data["transitions"] = {}
        if not isinstance(self.data.get("open_entries"), dict):
            self.data["open_entries"] = {}

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)

    def was_sent(self, event_id: str) -> bool:
        return event_id in self.data.get("sent", {})

    def transition(self, key: str) -> str:
        return str(self.data.get("transitions", {}).get(key, ""))

    def has_open_entry(self, symbol: str = "XAU/USD") -> bool:
        return str(symbol) in self.data.get("open_entries", {})

    def set_transition(self, key: str, value: str) -> None:
        transitions = self.data.setdefault("transitions", {})
        if transitions.get(key) == value:
            return
        transitions[key] = value
        self._save()

    def mark_sent(self, event: AlertEvent) -> None:
        sent = self.data.setdefault("sent", {})
        sent[event.event_id] = {
            "title": event.title,
            "category": event.category,
            "confidence": event.confidence,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(sent) > 500:
            ordered = sorted(sent.items(), key=lambda item: item[1].get("sent_at", ""), reverse=True)[:350]
            self.data["sent"] = dict(ordered)
        if event.transition_key:
            self.data.setdefault("transitions", {})[event.transition_key] = event.transition_value
        if event.category == "THREE_PILLAR_ENTRY_LIVE":
            self.data.setdefault("open_entries", {})[event.symbol] = {
                "side": event.side,
                "signal_id": event.signal_id,
                "event_id": event.event_id,
                "sent_at": datetime.now(timezone.utc).isoformat(),
            }
        elif event.category.startswith("TRADE_CLOSE_"):
            self.data.setdefault("open_entries", {}).pop(event.symbol, None)
            self.data.setdefault("transitions", {})["primary_entry_live"] = "WAIT"
        self._save()


def _finite(value: object, fallback: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return fallback
    return number if math.isfinite(number) else fallback


def _m15_atr(report: TechnicalReport) -> float:
    item = next((row for row in report.indicators if row.timeframe == "M15"), None)
    return max(_finite(getattr(item, "atr14", None), 2.5), 0.5)


def _inside_live_zone(price: float, low: float, high: float, atr: float) -> bool:
    tolerance = max(0.20 * atr, 0.35)
    return float(low) - tolerance <= float(price) <= float(high) + tolerance


def _distance_to_zone(price: float, low: float, high: float) -> float:
    if low <= price <= high:
        return 0.0
    return low - price if price < low else price - high


def is_primary_entry_live(report: TechnicalReport) -> bool:
    setup = report.active_setup
    if report.market_state not in {"BUY", "SELL"} or setup is None or setup.status != "ENTER":
        return False
    price = float(report.last_price)
    if not _inside_live_zone(price, setup.entry_low, setup.entry_high, _m15_atr(report)):
        return False
    if report.market_state == "BUY" and price >= setup.take_profit_1:
        return False
    if report.market_state == "SELL" and price <= setup.take_profit_1:
        return False
    return True


def is_primary_entry_near(report: TechnicalReport, atr_fraction: float = 0.65) -> bool:
    """True when an executable setup is close enough to justify fast refresh."""
    setup = report.active_setup
    if report.market_state not in {"BUY", "SELL"} or setup is None or setup.status != "ENTER":
        return False
    price = float(report.last_price)
    if report.market_state == "BUY" and price >= setup.take_profit_1:
        return False
    if report.market_state == "SELL" and price <= setup.take_profit_1:
        return False
    threshold = max(_m15_atr(report) * max(0.20, atr_fraction), 0.75)
    return _distance_to_zone(price, float(setup.entry_low), float(setup.entry_high)) <= threshold


def is_fvg_entry_live(report: TechnicalReport, special: FourHourFVGSignal | None) -> bool:
    if special is None or special.side not in {"BUY", "SELL"} or special.state != "TRIGGERED":
        return False
    if special.entry_low is None or special.entry_high is None:
        return False
    price = float(report.last_price)
    if not _inside_live_zone(price, special.entry_low, special.entry_high, _m15_atr(report)):
        return False
    if special.take_profit_1 is not None:
        if special.side == "BUY" and price >= float(special.take_profit_1):
            return False
        if special.side == "SELL" and price <= float(special.take_profit_1):
            return False
    return True


def _macro_line(report: TechnicalReport) -> str:
    macro = report.macro
    if macro is None:
        return "DXY/US10Y context: unavailable (not a signal gate)"
    return f"Context only — DXY {macro.dxy.direction} | US10Y {macro.us10y.direction}; neither can block this signal"


def _pillar_line(report: TechnicalReport) -> str:
    rows = {item.timeframe: item for item in report.indicators}
    parts: list[str] = []
    for tf in ("M15", "H1", "H4"):
        item = rows.get(tf)
        if item is None:
            continue
        structure = item.structure_bias.upper()
        avwap = "ABOVE" if item.avwap_active is not None and item.close > item.avwap_active else "BELOW" if item.avwap_active is not None else "N/A"
        parts.append(f"{tf}: structure {structure}, AVWAP {avwap}, profile {item.profile_state}")
    return " | ".join(parts)


def _setup_lines(report: TechnicalReport) -> list[str]:
    setup = report.active_setup
    if setup is None:
        return []
    return [
        f"ENTRY NOW {setup.entry_low:.2f}–{setup.entry_high:.2f}",
        f"SL {setup.stop_loss:.2f}",
        f"TP {setup.take_profit_1:.2f} / {setup.take_profit_2:.2f} / {setup.take_profit_3:.2f}",
        f"Risk status {setup.risk_plan.status if setup.risk_plan else 'UNAVAILABLE'}",
    ]


def _entry_event(report: TechnicalReport) -> AlertEvent | None:
    if not is_primary_entry_live(report):
        return None
    setup = report.active_setup
    assert setup is not None
    candle_time = next((item.timestamp for item in report.indicators if item.timeframe == "M15"), report.data_time)
    setup_key = f"{report.market_state}|{setup.entry_low:.2f}|{setup.entry_high:.2f}|{setup.stop_loss:.2f}"
    signal_id = f"{report.symbol}|M15|{candle_time}|{report.market_state}"
    lines = [
        "ENTRY PRICE HAS ARRIVED",
        f"XAU/USD live {report.last_price:.2f} | {report.market_state} | confidence {report.confidence}%",
        *_setup_lines(report),
        _pillar_line(report),
        _macro_line(report),
        "Signal basis: market structure + anchored VWAP + volume profile only.",
        "Valid only while live price remains inside/next to this entry zone. Do not chase later.",
    ]
    return AlertEvent(
        event_id=f"THREE_PILLAR_ENTRY|{setup_key}|{candle_time}",
        title=f"XAU/USD {report.market_state} — ENTRY LIVE NOW",
        message="\n".join(lines),
        category="THREE_PILLAR_ENTRY_LIVE",
        confidence=report.confidence,
        transition_key="primary_entry_live",
        transition_value=setup_key,
        symbol=report.symbol,
        side=report.market_state,
        signal_id=signal_id,
    )


def _close_events(review_outcomes: Iterable[dict[str, Any]] | None) -> list[AlertEvent]:
    events: list[AlertEvent] = []
    for review in review_outcomes or []:
        outcome = str(review.get("status") or review.get("outcome") or "").upper()
        if outcome not in {"WIN", "LOSS", "TIMEOUT"}:
            continue
        signal_id = str(review.get("id") or review.get("signal_id") or "unknown")
        symbol = str(review.get("symbol") or "XAU/USD")
        side = str(review.get("side") or "")
        outcome_time = str(review.get("outcome_time") or review.get("reviewed_at") or "")
        result_label = "TP1 HIT" if outcome == "WIN" else "STOP LOSS HIT" if outcome == "LOSS" else "TIMEOUT CLOSED"
        event_id = f"TRADE_CLOSE|{signal_id}|{outcome}|{outcome_time}"
        message = "\n".join(
            [
                f"{symbol} {side} — {result_label}",
                f"Outcome price: {_finite(review.get('outcome_price'), 0.0):.2f}",
                f"MFE: {_finite(review.get('mfe_r'), 0.0):.2f}R | MAE: {_finite(review.get('mae_r'), 0.0):.2f}R",
                f"Bars observed: {int(_finite(review.get('bars_observed'), 0.0))}",
                "The adaptive brain has recorded this completed trade and updated only bounded three-pillar statistics.",
            ]
        )
        events.append(
            AlertEvent(
                event_id=event_id,
                title=f"{symbol} {side} — {result_label}",
                message=message,
                category=f"TRADE_CLOSE_{outcome}",
                confidence=100,
                transition_key=f"trade_close:{signal_id}",
                transition_value=outcome,
                symbol=symbol,
                side=side,
                signal_id=signal_id,
            )
        )
    return events


def build_alert_events(
    report: TechnicalReport,
    special: FourHourFVGSignal | None = None,
    review_outcomes: Iterable[dict[str, Any]] | None = None,
) -> list[AlertEvent]:
    """Build close alerts plus one deduplicated live three-pillar entry alert."""
    del special
    events = _close_events(review_outcomes)
    entry = _entry_event(report)
    if entry is not None:
        events.append(entry)
    return events


def send_telegram(config: AlertConfig, event: AlertEvent) -> None:
    url = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    response = requests.post(
        url,
        json={"chat_id": config.telegram_chat_id, "text": f"{event.title}\n\n{event.message}"},
        timeout=20,
    )
    response.raise_for_status()


def send_email(config: AlertConfig, event: AlertEvent) -> None:
    msg = EmailMessage()
    msg["Subject"] = event.title
    msg["From"] = config.email_from or config.smtp_username
    msg["To"] = config.email_to
    msg.set_content(event.message)
    context = ssl.create_default_context()
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=25) as server:
        server.ehlo()
        server.starttls(context=context)
        server.ehlo()
        server.login(config.smtp_username, config.smtp_password)
        server.send_message(msg)


def dispatch_events(
    events: Iterable[AlertEvent],
    config: AlertConfig,
    state: AlertState,
) -> tuple[list[str], list[str]]:
    sent: list[str] = []
    errors: list[str] = []
    for event in events:
        if event.confidence < config.minimum_confidence:
            continue
        if state.was_sent(event.event_id):
            continue
        if event.transition_key and state.transition(event.transition_key) == event.transition_value:
            continue
        delivered = False
        if config.telegram_enabled:
            try:
                send_telegram(config, event)
                delivered = True
            except Exception as exc:
                errors.append(f"Telegram {event.event_id}: {exc.__class__.__name__}")
        if config.email_enabled:
            try:
                send_email(config, event)
                delivered = True
            except Exception as exc:
                errors.append(f"Email {event.event_id}: {exc.__class__.__name__}")
        if delivered:
            state.mark_sent(event)
            sent.append(event.event_id)
    return sent, errors


def record_non_alert_states(
    report: TechnicalReport,
    special: FourHourFVGSignal | None,
    config: AlertConfig,
    state: AlertState,
) -> None:
    del special, config
    if not is_primary_entry_live(report):
        state.set_transition("primary_entry_live", "WAIT")


def send_test_alert(config: AlertConfig) -> tuple[bool, list[str]]:
    event = AlertEvent(
        event_id=f"TEST|{datetime.now(timezone.utc).isoformat()}",
        title="AurumEdge test notification",
        message="Notification delivery is configured correctly. No trade signal is attached to this test.",
        category="TEST",
        confidence=100,
    )
    errors: list[str] = []
    delivered = False
    if config.telegram_enabled:
        try:
            send_telegram(config, event)
            delivered = True
        except Exception as exc:
            errors.append(f"Telegram: {exc}")
    if config.email_enabled:
        try:
            send_email(config, event)
            delivered = True
        except Exception as exc:
            errors.append(f"Email: {exc}")
    return delivered, errors
