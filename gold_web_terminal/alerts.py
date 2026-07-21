from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from email.message import EmailMessage
import json
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
    forming_alerts: bool = False

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
        self.data = {"sent": {}, "updated_at": ""}
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
        # Keep the state compact.
        if len(sent) > 200:
            ordered = sorted(sent.items(), key=lambda item: item[1].get("sent_at", ""), reverse=True)[:200]
            self.data["sent"] = dict(ordered)
        if event.transition_key:
            self.data.setdefault("transitions", {})[event.transition_key] = event.transition_value
        self.data["updated_at"] = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")


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
        f"Entry {setup.entry_low:.2f}–{setup.entry_high:.2f}",
        f"SL {setup.stop_loss:.2f}",
        f"TP {setup.take_profit_1:.2f} / {setup.take_profit_2:.2f} / {setup.take_profit_3:.2f}",
        f"Risk status {setup.risk_plan.status if setup.risk_plan else 'UNAVAILABLE'}",
    ]


def build_alert_events(report: TechnicalReport, special: FourHourFVGSignal | None) -> list[AlertEvent]:
    events: list[AlertEvent] = []
    if report.market_state in {"BUY", "SELL"} and report.active_setup is not None and report.active_setup.status == "ENTER":
        candle_time = next((item.timestamp for item in report.indicators if item.timeframe == "M15"), report.data_time)
        event_id = f"PRIMARY|{report.market_state}|{candle_time}|{report.signal_label}"
        lines = [
            f"AurumEdge regular signal: {report.signal_label}",
            f"XAU/USD {report.last_price:.2f} | confidence {report.confidence}%",
            *_setup_lines(report),
            _macro_line(report),
            "Verify the live broker spread before execution.",
        ]
        events.append(AlertEvent(
            event_id, f"XAU/USD {report.market_state} signal", "\n".join(lines), "PRIMARY", report.confidence,
            transition_key="primary_state", transition_value=report.market_state,
        ))

    if special is not None and special.side in {"BUY", "SELL"} and special.state in {"ARMED", "TRIGGERED"}:
        title_state = "TRIGGERED" if special.state == "TRIGGERED" else "FORMING"
        lines = [
            f"4H Candle + M15 FVG strategy {title_state}",
            f"Side {special.side} | confidence {special.confidence}%",
            f"FVG {special.fvg_low:.2f}–{special.fvg_high:.2f}" if special.fvg_low is not None else "FVG pending",
        ]
        if special.entry_low is not None:
            lines.extend(
                [
                    f"Entry {special.entry_low:.2f}–{special.entry_high:.2f}",
                    f"SL {special.stop_loss:.2f}",
                    f"TP {special.take_profit_1:.2f} / {special.take_profit_2:.2f} / {special.take_profit_3:.2f}",
                ]
            )
        lines.append(f"Regular engine agreement: {'YES' if special.aligns_with_primary else 'NO'} | macro gate {special.macro_gate}")
        lines.append("ARMED means wait for confirmation; TRIGGERED means the defined M15 confirmation closed.")
        events.append(
            AlertEvent(
                special.signal_id,
                f"XAU/USD H4-FVG {special.side} {title_state}",
                "\n".join(lines),
                f"H4_FVG_{special.state}",
                special.confidence,
                transition_key="h4_fvg_state",
                transition_value=f"{special.side}|{special.state}|{special.parent_candle_time}|{special.fvg_created_time}",
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
        if event.category == "H4_FVG_ARMED" and not config.forming_alerts:
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
    """Persist neutral states so only the start of a new signal is alerted."""
    if report.market_state not in {"BUY", "SELL"}:
        state.set_transition("primary_state", report.market_state)
    if special is None:
        state.set_transition("h4_fvg_state", "NONE")
    elif special.state not in {"ARMED", "TRIGGERED"}:
        state.set_transition(
            "h4_fvg_state",
            f"{special.side}|{special.state}|{special.parent_candle_time}|{special.fvg_created_time}",
        )
    elif special.state == "ARMED" and not config.forming_alerts:
        state.set_transition(
            "h4_fvg_state",
            f"{special.side}|ARMED|{special.parent_candle_time}|{special.fvg_created_time}",
        )


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
