from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field
from datetime import timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib import parse, request

import pandas as pd


@dataclass(frozen=True)
class Sma18Signal:
    signal_id: str
    timeframe: str
    side: str
    setup: str
    candle_time: str
    confirmed_time: str
    entry: float
    stop_loss: float
    win_level: float
    take_profit: float
    risk: float
    support_4h: float | None
    resistance_4h: float | None
    daily_bias: str
    profile: str = "Balanced"


@dataclass(frozen=True)
class TradeOutcome:
    signal_id: str
    side: str
    status: str
    entry: float
    stop_loss: float
    win_level: float
    resolved_time: str | None = None


@dataclass
class Sma18Stats:
    total: int = 0
    wins: int = 0
    losses: int = 0
    open: int = 0

    @property
    def win_rate(self) -> float:
        return self.wins * 100.0 / self.total if self.total else 0.0


@dataclass
class Sma18Snapshot:
    symbol: str
    latest_price: float
    latest_candle_time: str
    current_signal: Sma18Signal | None
    latest_signal: Sma18Signal | None
    signals: list[Sma18Signal] = field(default_factory=list)
    outcomes: list[TradeOutcome] = field(default_factory=list)
    stats: Sma18Stats = field(default_factory=Sma18Stats)
    support_4h: float | None = None
    resistance_4h: float | None = None
    daily_bias: str = "NEUTRAL"
    errors: list[str] = field(default_factory=list)


@dataclass
class DeliveryResult:
    sent_signal_ids: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    state_changed: bool = False


# These defaults reproduce the latest TradingView indicator's Balanced profile.
FAST_EMA_LENGTH = 9
SLOW_EMA_LENGTH = 21
BREAKOUT_LENGTH = 20
PULLBACK_TOLERANCE_PCT = 0.18
DMI_LENGTH = 14
ADX_SMOOTHING = 14
ATR_LENGTH = 14
MINIMUM_ADX = 14.0
MINIMUM_EMA_SEPARATION_PCT = 0.003
MINIMUM_4H_ROOM_PCT = 0.03
COUNTER_TREND_ADX = 18.0
DAILY_SMA_LENGTH = 18
SR_LOOKBACK_4H = 30
PIVOT_LEFT = 3
PIVOT_RIGHT = 3
FALLBACK_SWING_LOOKBACK = 12
SAME_SIDE_SPACING_BARS = 8
RISK_REWARD = 2.0
WIN_MOVE_R = 0.8
DEFAULT_TICK_SIZE = 0.01
MAX_STATE_SIGNAL_IDS = 600


def _frame(raw: pd.DataFrame, name: str) -> pd.DataFrame:
    if raw is None or raw.empty:
        raise ValueError(f"{name} candle frame is empty")
    required = {"time", "open", "high", "low", "close"}
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"{name} candle frame is missing: {', '.join(sorted(missing))}")
    frame = raw.copy()
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    for column in ["open", "high", "low", "close"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["time", "open", "high", "low", "close"])
    frame = frame.sort_values("time").drop_duplicates("time", keep="last").reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{name} candle frame contains no valid rows")
    return frame


def _rma(series: pd.Series, length: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float).to_numpy()
    result = [math.nan] * len(values)
    if len(values) < length:
        return pd.Series(result, index=series.index, dtype=float)
    seed_values = values[:length]
    if any(math.isnan(value) for value in seed_values):
        return pd.Series(result, index=series.index, dtype=float)
    result[length - 1] = float(sum(seed_values) / length)
    alpha = 1.0 / length
    for index in range(length, len(values)):
        value = values[index]
        previous = result[index - 1]
        if math.isnan(value) or math.isnan(previous):
            result[index] = previous
        else:
            result[index] = alpha * value + (1.0 - alpha) * previous
    return pd.Series(result, index=series.index, dtype=float)


def _add_engine_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fast_ema"] = out["close"].ewm(span=FAST_EMA_LENGTH, adjust=False).mean()
    out["slow_ema"] = out["close"].ewm(span=SLOW_EMA_LENGTH, adjust=False).mean()
    out["prior_resistance"] = out["high"].rolling(BREAKOUT_LENGTH, min_periods=BREAKOUT_LENGTH).max().shift(1)
    out["prior_support"] = out["low"].rolling(BREAKOUT_LENGTH, min_periods=BREAKOUT_LENGTH).min().shift(1)

    previous_close = out["close"].shift(1)
    true_range = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr"] = _rma(true_range, ATR_LENGTH)

    up_move = out["high"].diff()
    down_move = -out["low"].diff()
    plus_dm = pd.Series(0.0, index=out.index)
    minus_dm = pd.Series(0.0, index=out.index)
    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move[(up_move > down_move) & (up_move > 0)]
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move[(down_move > up_move) & (down_move > 0)]
    plus_smoothed = _rma(plus_dm, DMI_LENGTH)
    minus_smoothed = _rma(minus_dm, DMI_LENGTH)
    atr_dmi = _rma(true_range, DMI_LENGTH).replace(0.0, math.nan)
    out["plus_di"] = 100.0 * plus_smoothed / atr_dmi
    out["minus_di"] = 100.0 * minus_smoothed / atr_dmi
    di_sum = (out["plus_di"] + out["minus_di"]).replace(0.0, math.nan)
    dx = 100.0 * (out["plus_di"] - out["minus_di"]).abs() / di_sum
    out["adx"] = _rma(dx.fillna(0.0), ADX_SMOOTHING)

    separation = (out["fast_ema"] - out["slow_ema"]).abs() / out["close"].replace(0.0, math.nan) * 100.0
    out["long_trend"] = (out["fast_ema"] > out["slow_ema"]) & (separation >= MINIMUM_EMA_SEPARATION_PCT)
    out["short_trend"] = (out["fast_ema"] < out["slow_ema"]) & (separation >= MINIMUM_EMA_SEPARATION_PCT)

    long_threshold = out["prior_resistance"]
    short_threshold = out["prior_support"]
    bullish = out["close"] > out["open"]
    bearish = out["close"] < out["open"]
    out["long_breakout"] = (
        (out["close"] > long_threshold)
        & (out["close"].shift(1) <= long_threshold.shift(1))
        & bullish
    )
    out["short_breakout"] = (
        (out["close"] < short_threshold)
        & (out["close"].shift(1) >= short_threshold.shift(1))
        & bearish
    )

    tolerance = PULLBACK_TOLERANCE_PCT / 100.0
    long_touch = (out["low"] <= out["fast_ema"] * (1.0 + tolerance)) & (
        out["high"] >= out["fast_ema"] * (1.0 - tolerance)
    )
    short_touch = (out["high"] >= out["fast_ema"] * (1.0 - tolerance)) & (
        out["low"] <= out["fast_ema"] * (1.0 + tolerance)
    )
    out["long_pullback"] = (
        out["long_trend"]
        & long_touch
        & (out["close"] > out["fast_ema"])
        & (out["close"] > out["slow_ema"])
        & (out["close"] > out["high"].shift(1))
        & bullish
    )
    out["short_pullback"] = (
        out["short_trend"]
        & short_touch
        & (out["close"] < out["fast_ema"])
        & (out["close"] < out["slow_ema"])
        & (out["close"] < out["low"].shift(1))
        & bearish
    )

    out["long_combo"] = out["long_breakout"] & out["long_pullback"]
    out["short_combo"] = out["short_breakout"] & out["short_pullback"]
    out["long_standalone_breakout"] = out["long_breakout"] & ~out["long_pullback"]
    out["short_standalone_breakout"] = out["short_breakout"] & ~out["short_pullback"]
    out["long_standalone_pullback"] = out["long_pullback"] & ~out["long_breakout"]
    out["short_standalone_pullback"] = out["short_pullback"] & ~out["short_breakout"]
    out["long_candidate"] = out["long_combo"] | out["long_standalone_breakout"] | out["long_standalone_pullback"]
    out["short_candidate"] = out["short_combo"] | out["short_standalone_breakout"] | out["short_standalone_pullback"]
    previous_long_candidate = out["long_candidate"].shift(1).fillna(False).astype(bool)
    previous_short_candidate = out["short_candidate"].shift(1).fillna(False).astype(bool)
    out["long_event"] = out["long_candidate"] & ~previous_long_candidate
    out["short_event"] = out["short_candidate"] & ~previous_short_candidate
    return out


def _completed_index(frame: pd.DataFrame, minutes: int, now: pd.Timestamp | None = None) -> int:
    now = now or pd.Timestamp.now(tz="UTC")
    for index in range(len(frame) - 1, -1, -1):
        start = frame.iloc[index]["time"]
        if start + pd.Timedelta(minutes=minutes) <= now:
            return index
    raise ValueError("No completed candle is available")


def _daily_context(daily: pd.DataFrame, bar_close_time: pd.Timestamp) -> tuple[bool, bool, str]:
    frame = daily.copy()
    frame["sma18"] = frame["close"].rolling(DAILY_SMA_LENGTH, min_periods=DAILY_SMA_LENGTH).mean()
    completed = frame[(frame["time"] + pd.Timedelta(days=1)) <= bar_close_time]
    if completed.empty:
        return False, False, "NEUTRAL"
    latest = completed.iloc[-1]
    if pd.isna(latest["sma18"]):
        return False, False, "NEUTRAL"
    long_one = float(latest["close"]) > float(latest["sma18"])
    short_one = float(latest["close"]) < float(latest["sma18"])
    return long_one, short_one, "BUY" if long_one else "SELL" if short_one else "NEUTRAL"


def _four_hour_levels(h4: pd.DataFrame, bar_close_time: pd.Timestamp) -> tuple[float | None, float | None]:
    completed = h4[(h4["time"] + pd.Timedelta(hours=4)) <= bar_close_time]
    if len(completed) < SR_LOOKBACK_4H:
        return None, None
    window = completed.iloc[-SR_LOOKBACK_4H:]
    resistance = float(window["high"].max())
    support = float(window["low"].min())
    if not math.isfinite(resistance) or not math.isfinite(support) or resistance <= support:
        return None, None
    return resistance, support


def _latest_confirmed_pivot(frame: pd.DataFrame, index: int, side: str) -> float | None:
    latest: float | None = None
    values = frame["low"] if side == "LOW" else frame["high"]
    final_center = index - PIVOT_RIGHT
    for center in range(PIVOT_LEFT, final_center + 1):
        window = values.iloc[center - PIVOT_LEFT : center + PIVOT_RIGHT + 1]
        value = float(values.iloc[center])
        if side == "LOW" and value <= float(window.min()):
            latest = value
        elif side == "HIGH" and value >= float(window.max()):
            latest = value
    return latest


def _risk_levels(frame: pd.DataFrame, index: int, side: str, tick_size: float) -> tuple[float, float, float, float] | None:
    close = float(frame.iloc[index]["close"])
    fallback_start = max(0, index - FALLBACK_SWING_LOOKBACK)
    previous = frame.iloc[fallback_start:index]
    if previous.empty:
        return None
    if side == "BUY":
        pivot = _latest_confirmed_pivot(frame, index, "LOW")
        fallback = float(previous["low"].min())
        swing = pivot if pivot is not None and pivot < close else fallback
        stop = swing - (2.0 * tick_size)
        risk = close - stop
        if not math.isfinite(risk) or risk <= tick_size:
            return None
        return close, stop, close + risk * WIN_MOVE_R, close + risk * RISK_REWARD
    pivot = _latest_confirmed_pivot(frame, index, "HIGH")
    fallback = float(previous["high"].max())
    swing = pivot if pivot is not None and pivot > close else fallback
    stop = swing + (2.0 * tick_size)
    risk = stop - close
    if not math.isfinite(risk) or risk <= tick_size:
        return None
    return close, stop, close - risk * WIN_MOVE_R, close - risk * RISK_REWARD


def _setup_name(row: pd.Series, side: str) -> str:
    prefix = "long" if side == "BUY" else "short"
    if bool(row[f"{prefix}_combo"]):
        return "Pullback + Breakout"
    if bool(row[f"{prefix}_standalone_breakout"]):
        return "Breakout"
    return "Pullback"


def _build_signals(
    m15: pd.DataFrame,
    h4: pd.DataFrame,
    daily: pd.DataFrame,
    symbol: str,
    tick_size: float,
    now: pd.Timestamp,
) -> tuple[list[Sma18Signal], int, float | None, float | None, str]:
    engine = _add_engine_columns(m15)
    completed_index = _completed_index(engine, 15, now)
    signals: list[Sma18Signal] = []
    buy_armed = True
    sell_armed = True
    last_buy_index: int | None = None
    last_sell_index: int | None = None
    latest_resistance: float | None = None
    latest_support: float | None = None
    latest_daily_bias = "NEUTRAL"

    start = max(BREAKOUT_LENGTH + 2, DMI_LENGTH + ADX_SMOOTHING + 2, FALLBACK_SWING_LOOKBACK + 2)
    for index in range(start, completed_index + 1):
        row = engine.iloc[index]
        close_time = row["time"] + pd.Timedelta(minutes=15)
        daily_long, daily_short, daily_bias = _daily_context(daily, close_time)
        resistance, support = _four_hour_levels(h4, close_time)
        if index == completed_index:
            latest_resistance, latest_support, latest_daily_bias = resistance, support, daily_bias

        long_trend = bool(row["long_trend"])
        short_trend = bool(row["short_trend"])
        if not buy_armed and (
            float(row["close"]) < float(row["fast_ema"])
            or float(row["low"]) <= float(row["slow_ema"])
            or not long_trend
        ):
            buy_armed = True
        if not sell_armed and (
            float(row["close"]) > float(row["fast_ema"])
            or float(row["high"]) >= float(row["slow_ema"])
            or not short_trend
        ):
            sell_armed = True

        plus_di = float(row["plus_di"]) if pd.notna(row["plus_di"]) else math.nan
        minus_di = float(row["minus_di"]) if pd.notna(row["minus_di"]) else math.nan
        adx = float(row["adx"]) if pd.notna(row["adx"]) else math.nan
        long_dmi = math.isfinite(adx) and adx >= MINIMUM_ADX and plus_di > minus_di
        short_dmi = math.isfinite(adx) and adx >= MINIMUM_ADX and minus_di > plus_di

        long_reversal = bool(row["long_breakout"]) and long_trend and plus_di > minus_di and adx >= COUNTER_TREND_ADX
        short_reversal = bool(row["short_breakout"]) and short_trend and minus_di > plus_di and adx >= COUNTER_TREND_ADX
        long_daily_allowed = daily_long or long_reversal
        short_daily_allowed = daily_short or short_reversal

        close = float(row["close"])
        room_resistance = (
            (resistance - close) / close * 100.0
            if resistance is not None and resistance > close
            else None
        )
        room_support = (
            (close - support) / close * 100.0
            if support is not None and support < close
            else None
        )
        long_location = room_resistance is None or room_resistance >= MINIMUM_4H_ROOM_PCT
        short_location = room_support is None or room_support >= MINIMUM_4H_ROOM_PCT

        long_setup = (
            long_daily_allowed
            and long_trend
            and long_dmi
            and long_location
            and bool(row["long_event"])
        )
        short_setup = (
            short_daily_allowed
            and short_trend
            and short_dmi
            and short_location
            and bool(row["short_event"])
        )
        buy_spacing = last_buy_index is None or index - last_buy_index >= SAME_SIDE_SPACING_BARS
        sell_spacing = last_sell_index is None or index - last_sell_index >= SAME_SIDE_SPACING_BARS

        side: str | None = None
        if long_setup and not short_setup and buy_armed and buy_spacing:
            side = "BUY"
        elif short_setup and not long_setup and sell_armed and sell_spacing:
            side = "SELL"
        if side is None:
            continue

        levels = _risk_levels(engine, index, side, tick_size)
        if levels is None:
            continue
        entry, stop, win_level, target = levels
        candle_time = pd.Timestamp(row["time"]).tz_convert("UTC")
        confirmed_time = candle_time + pd.Timedelta(minutes=15)
        setup = _setup_name(row, side)
        signal_id = f"SMA18-M15:{candle_time.isoformat()}:{side}:{setup.replace(' ', '_')}"
        signals.append(
            Sma18Signal(
                signal_id=signal_id,
                timeframe="M15",
                side=side,
                setup=setup,
                candle_time=candle_time.isoformat(),
                confirmed_time=confirmed_time.isoformat(),
                entry=round(entry, 6),
                stop_loss=round(stop, 6),
                win_level=round(win_level, 6),
                take_profit=round(target, 6),
                risk=round(abs(entry - stop), 6),
                support_4h=round(support, 6) if support is not None else None,
                resistance_4h=round(resistance, 6) if resistance is not None else None,
                daily_bias=daily_bias,
            )
        )
        if side == "BUY":
            last_buy_index = index
            buy_armed = False
            sell_armed = True
        else:
            last_sell_index = index
            sell_armed = False
            buy_armed = True

    return signals, completed_index, latest_resistance, latest_support, latest_daily_bias


def _resolve_outcomes(signals: Iterable[Sma18Signal], m5: pd.DataFrame, m15: pd.DataFrame) -> tuple[list[TradeOutcome], Sma18Stats]:
    outcomes: list[TradeOutcome] = []
    wins = losses = open_count = 0
    for signal in signals:
        confirmed = pd.Timestamp(signal.confirmed_time)
        future = m5[m5["time"] >= confirmed]
        if future.empty:
            future = m15[m15["time"] >= confirmed]
        status = "OPEN"
        resolved_time: str | None = None
        for _, candle in future.iterrows():
            stop_hit = float(candle["low"]) <= signal.stop_loss if signal.side == "BUY" else float(candle["high"]) >= signal.stop_loss
            win_hit = float(candle["high"]) >= signal.win_level if signal.side == "BUY" else float(candle["low"]) <= signal.win_level
            if stop_hit or win_hit:
                # Conservative same-candle rule: SL first.
                status = "LOSS" if stop_hit else "WIN"
                resolved_time = pd.Timestamp(candle["time"]).tz_convert("UTC").isoformat()
                break
        if status == "WIN":
            wins += 1
        elif status == "LOSS":
            losses += 1
        else:
            open_count += 1
        outcomes.append(
            TradeOutcome(
                signal_id=signal.signal_id,
                side=signal.side,
                status=status,
                entry=signal.entry,
                stop_loss=signal.stop_loss,
                win_level=signal.win_level,
                resolved_time=resolved_time,
            )
        )
    return outcomes, Sma18Stats(total=wins + losses, wins=wins, losses=losses, open=open_count)


def evaluate_sma18_strategy(
    frames: dict[str, pd.DataFrame],
    symbol: str = "XAU/USD",
    *,
    now: pd.Timestamp | None = None,
    tick_size: float | None = None,
) -> Sma18Snapshot:
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")
    tick_size = float(tick_size or os.getenv("SMA18_TICK_SIZE", DEFAULT_TICK_SIZE))
    m5 = _frame(frames["M5"], "M5")
    m15 = _frame(frames["M15"], "M15")
    h4 = _frame(frames["H4"], "H4")
    daily = _frame(frames["D1"], "D1")
    signals, completed_index, resistance, support, daily_bias = _build_signals(
        m15, h4, daily, symbol, tick_size, now
    )
    outcomes, stats = _resolve_outcomes(signals, m5, m15)
    latest_signal = signals[-1] if signals else None
    completed_row = m15.iloc[completed_index]
    latest_time = pd.Timestamp(completed_row["time"]).tz_convert("UTC")
    current_signal = latest_signal if latest_signal and pd.Timestamp(latest_signal.candle_time) == latest_time else None
    return Sma18Snapshot(
        symbol=symbol,
        latest_price=float(completed_row["close"]),
        latest_candle_time=latest_time.isoformat(),
        current_signal=current_signal,
        latest_signal=latest_signal,
        signals=signals,
        outcomes=outcomes,
        stats=stats,
        support_4h=support,
        resistance_4h=resistance,
        daily_bias=daily_bias,
    )


def _load_state(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _telegram_credentials() -> tuple[str, str]:
    token = (
        os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        or os.getenv("TELEGRAM_TOKEN", "").strip()
        or os.getenv("BOT_TOKEN", "").strip()
    )
    chat_id = (
        os.getenv("TELEGRAM_CHAT_ID", "").strip()
        or os.getenv("TELEGRAM_TARGET_CHAT_ID", "").strip()
        or os.getenv("CHAT_ID", "").strip()
    )
    return token, chat_id


def format_telegram_message(signal: Sma18Signal, symbol: str) -> str:
    icon = "🟢" if signal.side == "BUY" else "🔴"
    support = "N/A" if signal.support_4h is None else f"{signal.support_4h:,.2f}"
    resistance = "N/A" if signal.resistance_4h is None else f"{signal.resistance_4h:,.2f}"
    return (
        f"{icon} <b>AURUMEDGE {signal.side} SIGNAL</b>\n\n"
        f"Symbol: <b>{symbol}</b>\n"
        f"Timeframe: <b>M15</b>\n"
        f"Setup: <b>{signal.setup}</b>\n"
        f"Entry: <b>{signal.entry:,.2f}</b>\n"
        f"Stop Loss: <b>{signal.stop_loss:,.2f}</b>\n"
        f"0.8R Win Level: <b>{signal.win_level:,.2f}</b>\n"
        f"Take Profit (2R): <b>{signal.take_profit:,.2f}</b>\n"
        f"4H Support: {support}\n"
        f"4H Resistance: {resistance}\n"
        f"Daily SMA18 bias: {signal.daily_bias}\n"
        f"Confirmed: {signal.confirmed_time.replace('+00:00', ' UTC')}\n\n"
        "Signal is confirmed only after the 15-minute candle closes."
    )


def send_telegram_message(message: str) -> tuple[bool, str]:
    token, chat_id = _telegram_credentials()
    if not token or not chat_id:
        return False, "TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID are required"
    payload = parse.urlencode(
        {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": "true",
        }
    ).encode("utf-8")
    endpoint = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        with request.urlopen(request.Request(endpoint, data=payload, method="POST"), timeout=20) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not body.get("ok"):
            return False, str(body)
        return True, ""
    except Exception as exc:  # Network/provider errors are returned to the watcher output.
        return False, f"{exc.__class__.__name__}: {exc}"


def process_sma18_alerts(
    snapshot: Sma18Snapshot,
    state_path: str | Path,
    *,
    dry_run: bool = False,
    max_alert_age_minutes: int = 20,
    now: pd.Timestamp | None = None,
) -> DeliveryResult:
    path = Path(state_path)
    state = _load_state(path)
    sent_ids = list(state.get("sent_signal_ids", []))
    sent_set = set(str(item) for item in sent_ids)
    result = DeliveryResult()
    now = now or pd.Timestamp.now(tz="UTC")
    if now.tzinfo is None:
        now = now.tz_localize("UTC")
    else:
        now = now.tz_convert("UTC")

    eligible: list[Sma18Signal] = []
    for signal in snapshot.signals:
        confirmed = pd.Timestamp(signal.confirmed_time)
        age = now - confirmed
        if signal.signal_id not in sent_set and timedelta(0) <= age.to_pytimedelta() <= timedelta(minutes=max_alert_age_minutes):
            eligible.append(signal)

    for signal in eligible:
        if dry_run:
            result.sent_signal_ids.append(signal.signal_id)
            continue
        ok, error = send_telegram_message(format_telegram_message(signal, snapshot.symbol))
        if ok:
            sent_set.add(signal.signal_id)
            result.sent_signal_ids.append(signal.signal_id)
        else:
            result.errors.append(error)

    # Persist only after a Telegram delivery succeeds. This avoids committing a
    # state-file change on every scheduled candle check.
    previous_sent_set = set(str(item) for item in state.get("sent_signal_ids", []))
    if not dry_run and sent_set != previous_sent_set:
        new_state = {
            "version": 1,
            "engine": "SMA18_EMA_M15",
            "last_signal_id": result.sent_signal_ids[-1] if result.sent_signal_ids else state.get("last_signal_id", ""),
            "sent_signal_ids": sorted(sent_set)[-MAX_STATE_SIGNAL_IDS:],
            "updated_at": now.isoformat(),
        }
        _save_state(path, new_state)
        result.state_changed = True
    return result


def snapshot_to_dict(snapshot: Sma18Snapshot) -> dict[str, Any]:
    return {
        "symbol": snapshot.symbol,
        "latest_price": snapshot.latest_price,
        "latest_candle_time": snapshot.latest_candle_time,
        "current_signal": asdict(snapshot.current_signal) if snapshot.current_signal else None,
        "latest_signal": asdict(snapshot.latest_signal) if snapshot.latest_signal else None,
        "stats": {
            "total": snapshot.stats.total,
            "wins": snapshot.stats.wins,
            "losses": snapshot.stats.losses,
            "open": snapshot.stats.open,
            "win_rate": snapshot.stats.win_rate,
        },
        "support_4h": snapshot.support_4h,
        "resistance_4h": snapshot.resistance_4h,
        "daily_bias": snapshot.daily_bias,
        "errors": snapshot.errors,
    }
