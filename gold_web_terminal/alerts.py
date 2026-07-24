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
from typing import Iterable

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
    forming_alerts: bool = False  # retained for backward compatibility; entry-only alerts ignore it

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


class AlertState:
    def __init__(self, path: str | Path = "data/alert_state.json") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {"sent": {}, "transitions": {}, "updated_at": ""}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
            except Exception:
                pass

    def was_sent(self, event_id: str) -> bool:
        return event_id in self.data.get("sent", {})

    def transition(self, key: str) -> str:
        return str(self.data.get("transitions", {}).get(key, ""))

    def set_transition(self, key: str, value: str) -> None:
        self.data.setdefault("transitions", {})[key] = value
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def mark_sent(self, event: AlertEvent) -> None:
        sent = self.data.setdefault("sent", {})
        sent[event.event_id] = {
            "title": event.title,
            "category": event.category,
            "confidence": event.confidence,
            "sent_at": datetime.now(timezone.utc).isoformat(),
        }
        if len(sent) > 250:
            ordered = sorted(sent.items(), key=lambda item: item[1].get("sent_at", ""), reverse=True)[:250]
            self.data["sent"] = dict(ordered)
        if event.transition_key:
            self.data.setdefault("transitions", {})[event.transition_key] = event.transition_value
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


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
    tolerance = max(0.10 * atr, 0.25)
    return float(low) - tolerance <= float(price) <= float(high) + tolerance


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
        return "Macro: unavailable"
    dxy = "—" if macro.dxy.value is None else f"{macro.dxy.value:.3f} {macro.dxy.direction}"
    yld = "—" if macro.us10y.value is None else f"{macro.us10y.value:.3f}% {macro.us10y.direction}"
    gold4 = "—" if macro.gold_change_4h is None else f"{macro.gold_change_4h:+.2f} {macro.gold_direction}"
    return f"DXY {dxy} | US10Y {yld} | Gold 4H {gold4} | gate {macro.gate}"


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


def build_alert_events(report: TechnicalReport, special: FourHourFVGSignal | None) -> list[AlertEvent]:
    """Build only immediately executable, live-price entry alerts.

    No WATCH, ARMED, historical TRIGGERED, bias-only, or already-missed setup is
    eligible. This is deliberately redundant with the strategy state so a
    stale detector can never send a late price alert.
    """
    events: list[AlertEvent] = []
    candle_time = next((item.timestamp for item in report.indicators if item.timeframe == "M15"), report.data_time)

    if is_primary_entry_live(report):
        setup = report.active_setup
        assert setup is not None
        setup_key = f"{report.market_state}|{setup.entry_low:.2f}|{setup.entry_high:.2f}|{setup.stop_loss:.2f}"
        event_id = f"PRIMARY_ENTRY|{setup_key}|{candle_time}"
        lines = [
            "ENTRY PRICE HAS ARRIVED",
            f"XAU/USD live {report.last_price:.2f} | {report.market_state} | confidence {report.confidence}%",
            *_setup_lines(report),
            _macro_line(report),
            "Valid only while the live broker price remains inside/next to this entry zone.",
            "Do not chase if price has already left the zone.",
        ]
        events.append(
            AlertEvent(
                event_id,
                f"XAU/USD {report.market_state} — ENTRY LIVE NOW",
                "\n".join(lines),
                "PRIMARY_ENTRY_LIVE",
                report.confidence,
                transition_key="primary_entry_live",
                transition_value=setup_key,
            )
        )

    if is_fvg_entry_live(report, special):
        assert special is not None
        setup_key = f"{special.side}|{special.parent_candle_time}|{special.fvg_created_time}"
        lines = [
            "ENTRY PRICE HAS ARRIVED",
            f"4H + M15 FVG {special.side} | live {report.last_price:.2f} | confidence {special.confidence}%",
            f"Entry now {special.entry_low:.2f}–{special.entry_high:.2f}",
            f"FVG {special.fvg_low:.2f}–{special.fvg_high:.2f}" if special.fvg_low is not None else "FVG unavailable",
            f"SL {special.stop_loss:.2f}",
            f"TP {special.take_profit_1:.2f} / {special.take_profit_2:.2f} / {special.take_profit_3:.2f}",
            f"Regular engine agreement: {'YES' if special.aligns_with_primary else 'NO'} | macro gate {special.macro_gate}",
            "This is a fresh first-touch entry. Do not enter later if price leaves the zone.",
        ]
        events.append(
            AlertEvent(
                f"H4_FVG_ENTRY|{setup_key}",
                f"XAU/USD H4-FVG {special.side} — ENTRY LIVE NOW",
                "\n".join(lines),
                "H4_FVG_ENTRY_LIVE",
                special.confidence,
                transition_key="h4_fvg_entry_live",
                transition_value=setup_key,
            )
        )
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
    """Reset entry transitions when price is no longer in an executable zone."""
    if not is_primary_entry_live(report):
        state.set_transition("primary_entry_live", "WAIT")
    if not is_fvg_entry_live(report, special):
        state.set_transition("h4_fvg_entry_live", "WAIT")


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
