from __future__ import annotations

"""Entry-live and trade-lifecycle alerts for AurumEdge v5.8.1."""

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import smtplib
from email.message import EmailMessage
from typing import Any, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _float(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(_get(obj, name, default))
    except (TypeError, ValueError):
        return default


def _bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class AlertConfig:
    minimum_confidence: int = 65
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_from: str = ""
    email_to: str = ""

    @classmethod
    def from_env(cls) -> "AlertConfig":
        try:
            minimum = int(os.getenv("ALERT_MIN_CONFIDENCE", "65"))
        except ValueError:
            minimum = 65
        try:
            port = int(os.getenv("SMTP_PORT", "587"))
        except ValueError:
            port = 587
        return cls(
            minimum_confidence=max(0, min(100, minimum)),
            telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", "").strip(),
            telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", "").strip(),
            smtp_host=os.getenv("SMTP_HOST", "").strip(),
            smtp_port=port,
            smtp_username=os.getenv("SMTP_USERNAME", "").strip(),
            smtp_password=os.getenv("SMTP_APP_PASSWORD", os.getenv("SMTP_PASSWORD", "")).strip(),
            email_from=os.getenv("ALERT_EMAIL_FROM", "").strip(),
            email_to=os.getenv("ALERT_EMAIL_TO", "").strip(),
        )

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    @property
    def email_enabled(self) -> bool:
        return bool(self.smtp_host and self.email_from and self.email_to)


@dataclass(slots=True)
class AlertEvent:
    event_id: str
    kind: str
    title: str
    message: str
    confidence: int = 100
    symbol: str = "XAU/USD"
    side: str = ""
    created_at: str = ""

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = _utc_now()


class AlertState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data = {"version": 2, "sent": {}, "last_states": {}, "last_event": None, "open_entries": {}}
        try:
            if self.path.exists():
                parsed = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    self.data.update(parsed)
        except (OSError, json.JSONDecodeError):
            pass
        if not isinstance(self.data.get("sent"), dict):
            self.data["sent"] = {}
        if not isinstance(self.data.get("open_entries"), dict):
            self.data["open_entries"] = {}
        last = self.data.get("last_event")
        if not self.data["open_entries"] and isinstance(last, dict) and last.get("kind") == "ENTRY":
            symbol = str(last.get("symbol") or "XAU/USD")
            self.data["open_entries"][symbol] = {
                "side": str(last.get("side") or ""),
                "event_id": str(last.get("event_id") or "legacy-entry"),
                "sent_at": str(last.get("created_at") or _utc_now()),
            }

    def was_sent(self, event_id: str) -> bool:
        return event_id in self.data.get("sent", {})

    def has_open_entry(self, symbol: str) -> bool:
        return str(symbol) in self.data.get("open_entries", {})

    def mark_sent(self, event: AlertEvent, channels: list[str]) -> None:
        self.data.setdefault("sent", {})[event.event_id] = {
            "sent_at": _utc_now(),
            "kind": event.kind,
            "channels": channels,
            "title": event.title,
        }
        # Bound state growth while retaining enough history for deduplication.
        sent = self.data["sent"]
        if len(sent) > 1500:
            newest = list(sent.items())[-1000:]
            self.data["sent"] = dict(newest)
        if event.kind == "ENTRY":
            self.data.setdefault("open_entries", {})[event.symbol] = {
                "side": event.side,
                "event_id": event.event_id,
                "sent_at": _utc_now(),
            }
        elif event.kind == "CLOSE":
            self.data.setdefault("open_entries", {}).pop(event.symbol, None)
        self.data["last_event"] = asdict(event)
        self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        temporary.replace(self.path)


def _entry_event(report: Any) -> AlertEvent | None:
    state = str(_get(report, "market_state", ""))
    setup = _get(report, "active_setup")
    if state not in {"BUY", "SELL"} or setup is None:
        return None
    risk_plan = _get(setup, "risk_plan")
    if str(_get(risk_plan, "status", "OK")) == "BLOCK":
        return None
    price = _float(report, "last_price")
    low = _float(setup, "entry_low")
    high = _float(setup, "entry_high")
    atr = max(_float(setup, "atr", 1.0), 0.01)
    adjacent = max(min(_float(setup, "entry_tolerance", atr * 0.12), atr * 0.12), 0.08)
    entry_live = bool(_get(setup, "entry_live", False)) or low <= price <= high
    immediately_beside = low - adjacent <= price <= high + adjacent
    if not (entry_live or immediately_beside):
        return None
    symbol = str(_get(report, "symbol", "XAU/USD"))
    entry_mid = _float(setup, "entry_price", (low + high) / 2.0)
    stop = _float(setup, "stop_loss")
    tp1 = _float(setup, "take_profit_1")
    confidence = int(_float(report, "confidence", 0))
    # Rounded geometry keeps one live setup deduplicated across small feed changes.
    key = f"entry:{symbol}:{state}:{entry_mid:.1f}:{stop:.1f}:{tp1:.1f}"
    pillars = _get(report, "pillar_votes", {}) or _get(setup, "pillar_votes", {}) or {}
    pillar_text = ", ".join(
        f"{name.replace('_', ' ').title()}={'BUY' if int(pillars.get(name, 0)) > 0 else 'SELL' if int(pillars.get(name, 0)) < 0 else 'NEUTRAL'}"
        for name in ("market_structure", "anchored_vwap", "volume_profile")
    )
    message = (
        f"ENTRY LIVE — {state} {symbol}\n"
        f"Current price: {price:.2f}\n"
        f"Entry zone: {low:.2f} – {high:.2f}\n"
        f"Stop loss: {stop:.2f}\n"
        f"TP1: {tp1:.2f}\n"
        f"Confidence: {confidence}%\n"
        f"{pillar_text}\n"
        "DXY and US10Y are display-only context. Verify the broker quote before execution."
    )
    return AlertEvent(
        event_id=key,
        kind="ENTRY",
        title=f"AurumEdge ENTRY LIVE — {state} {symbol}",
        message=message,
        confidence=confidence,
        symbol=symbol,
        side=state,
    )


def _close_events(review_outcomes: Iterable[dict[str, Any]] | None) -> list[AlertEvent]:
    events: list[AlertEvent] = []
    for review in review_outcomes or []:
        outcome = str(review.get("outcome") or review.get("status") or "").upper()
        if outcome not in {"WIN", "LOSS", "TIMEOUT"}:
            continue
        signal_id = str(review.get("signal_id") or "unknown")
        side = str(review.get("side") or "")
        symbol = str(review.get("symbol") or "XAU/USD")
        mfe = float(review.get("mfe_r") or 0.0)
        mae = float(review.get("mae_r") or 0.0)
        exit_price = float(review.get("exit_price") or 0.0)
        label = "TP1 HIT" if outcome == "WIN" else "STOP LOSS HIT" if outcome == "LOSS" else "TRADE TIMEOUT"
        message = (
            f"{label} — {side} {symbol}\n"
            f"Outcome: {outcome}\n"
            f"Exit/last price: {exit_price:.2f}\n"
            f"MFE: {mfe:.2f}R · MAE: {mae:.2f}R\n"
            f"Bars held: {int(review.get('bars_held') or 0)}\n"
            "The adaptive brain stored this result and updated bounded three-pillar statistics."
        )
        events.append(
            AlertEvent(
                event_id=f"close:{signal_id}:{outcome}",
                kind="CLOSE",
                title=f"AurumEdge {label} — {side} {symbol}",
                message=message,
                confidence=100,
                symbol=symbol,
                side=side,
            )
        )
    return events


def build_alert_events(
    report: Any,
    special: Any | None = None,
    review_outcomes: Iterable[dict[str, Any]] | None = None,
) -> list[AlertEvent]:
    """Return current entry plus completed-trade events. FVG is intentionally ignored."""
    events = _close_events(review_outcomes)
    entry = _entry_event(report)
    if entry is not None:
        events.append(entry)
    return events


def _send_telegram(config: AlertConfig, event: AlertEvent) -> None:
    endpoint = f"https://api.telegram.org/bot{config.telegram_bot_token}/sendMessage"
    body = urlencode({"chat_id": config.telegram_chat_id, "text": f"{event.title}\n\n{event.message}"}).encode("utf-8")
    request = Request(endpoint, data=body, method="POST")
    with urlopen(request, timeout=20) as response:  # noqa: S310 - fixed Telegram endpoint
        payload = json.loads(response.read().decode("utf-8"))
    if not payload.get("ok"):
        raise RuntimeError(str(payload.get("description") or "Telegram delivery failed"))


def _send_email(config: AlertConfig, event: AlertEvent) -> None:
    message = EmailMessage()
    message["Subject"] = event.title
    message["From"] = config.email_from
    message["To"] = config.email_to
    message.set_content(event.message)
    with smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=25) as server:
        server.ehlo()
        if config.smtp_port != 25 or _bool_env("SMTP_STARTTLS", True):
            server.starttls()
            server.ehlo()
        if config.smtp_username:
            server.login(config.smtp_username, config.smtp_password)
        server.send_message(message)


def dispatch_events(
    events: Iterable[AlertEvent],
    config: AlertConfig,
    state: AlertState,
) -> tuple[list[str], list[str]]:
    sent: list[str] = []
    errors: list[str] = []
    for event in events:
        if event.kind == "ENTRY" and event.confidence < config.minimum_confidence:
            continue
        if event.kind == "ENTRY" and state.has_open_entry(event.symbol):
            continue
        if state.was_sent(event.event_id):
            continue
        delivered_channels: list[str] = []
        if config.telegram_enabled:
            try:
                _send_telegram(config, event)
                delivered_channels.append("telegram")
            except Exception as exc:  # network/provider error should not expose secrets
                errors.append(f"{event.event_id}: Telegram: {exc}")
        if config.email_enabled:
            try:
                _send_email(config, event)
                delivered_channels.append("email")
            except Exception as exc:
                errors.append(f"{event.event_id}: Email: {exc}")
        if delivered_channels:
            state.mark_sent(event, delivered_channels)
            sent.append(event.event_id)
    return sent, errors


def record_non_alert_states(
    report: Any,
    special: Any | None,
    config: AlertConfig,
    state: AlertState,
) -> None:
    state.data.setdefault("last_states", {})[str(_get(report, "symbol", "XAU/USD"))] = {
        "market_state": str(_get(report, "market_state", "STUCK")),
        "entry_live": bool(_get(report, "entry_live", False)),
        "updated_at": _utc_now(),
    }
    state.save()


def send_test_alert(config: AlertConfig) -> tuple[bool, list[str]]:
    event = AlertEvent(
        event_id=f"test:{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}",
        kind="TEST",
        title="AurumEdge notification test",
        message="The AurumEdge Mobile v5.8.1 notification channel is working.",
    )
    temporary = AlertState(Path(os.getenv("ALERT_STATE_PATH", "data/alert_state.json")))
    sent, errors = dispatch_events([event], config, temporary)
    if not config.telegram_enabled and not config.email_enabled:
        errors.append("No Telegram or email channel is configured.")
    return bool(sent), errors
