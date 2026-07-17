from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import requests


INTERVAL_MAP = {
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
}


class MarketDataError(RuntimeError):
    """Safe market-data error that never includes an API key or request URL."""


class MarketDataRateLimit(MarketDataError):
    """Raised when the provider returns HTTP 429 or a quota message."""


@dataclass(slots=True)
class MarketBundle:
    frames: dict[str, pd.DataFrame]
    symbol: str
    last_price: float
    data_time: str
    source: str
    notes: list[str] = field(default_factory=list)


def _activity_proxy(df: pd.DataFrame) -> pd.Series:
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - df["close"].shift()).abs(),
            (df["low"] - df["close"].shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    baseline = true_range.rolling(50, min_periods=10).median().replace(0, np.nan)
    proxy = (true_range / baseline).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    return (proxy.clip(0.05, 8.0) * 1000).round().astype(int)


def normalize_ohlcv(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    aliases = {column.lower().strip(): column for column in df.columns}
    required = ["time", "open", "high", "low", "close"]
    missing = [key for key in required if key not in aliases]
    if missing:
        if "datetime" in aliases and "time" in missing:
            aliases["time"] = aliases["datetime"]
            missing.remove("time")
    if missing:
        raise ValueError(f"Missing OHLC columns: {', '.join(missing)}")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(df[aliases["time"]], utc=True, errors="coerce"),
            "open": pd.to_numeric(df[aliases["open"]], errors="coerce"),
            "high": pd.to_numeric(df[aliases["high"]], errors="coerce"),
            "low": pd.to_numeric(df[aliases["low"]], errors="coerce"),
            "close": pd.to_numeric(df[aliases["close"]], errors="coerce"),
        }
    )
    volume_column = None
    for candidate in ("volume", "tick_volume", "real_volume"):
        if candidate in aliases:
            volume_column = aliases[candidate]
            break
    has_provider_volume = volume_column is not None
    if volume_column is not None:
        volume = pd.to_numeric(df[volume_column], errors="coerce").fillna(0)
    else:
        volume = pd.Series(np.zeros(len(out)), index=out.index)

    out = out.dropna(subset=["time", "open", "high", "low", "close"]).copy()
    if out.empty:
        raise ValueError("No valid OHLC rows were found.")
    volume = volume.loc[out.index]
    out["tick_volume"] = volume.to_numpy()
    if not has_provider_volume or float(out["tick_volume"].abs().sum()) == 0:
        out["tick_volume"] = _activity_proxy(out)
        has_provider_volume = False
    out["real_volume"] = 0
    out["spread"] = 0
    out = out.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    return out, has_provider_volume


def resample_ohlcv(frame: pd.DataFrame, rule: str, limit: int | None = None) -> pd.DataFrame:
    """Derive a higher timeframe without spending another provider API credit."""
    source = frame.copy().set_index("time").sort_index()
    aggregation: dict[str, str] = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "tick_volume": "sum",
        "real_volume": "sum",
        "spread": "max",
    }
    derived = source.resample(rule, label="left", closed="left").agg(aggregation)
    derived = derived.dropna(subset=["open", "high", "low", "close"]).reset_index()
    if limit is not None:
        derived = derived.tail(int(limit)).reset_index(drop=True)
    return derived


class TwelveDataClient:
    def __init__(self, api_key: str, base_url: str = "https://api.twelvedata.com") -> None:
        if not api_key:
            raise ValueError("TWELVE_DATA_API_KEY is missing.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.last_credits_left: str | None = None

    @staticmethod
    def _payload_message(payload: dict[str, Any], fallback: str) -> str:
        value = payload.get("message") or payload.get("code") or fallback
        return str(value).replace("\n", " ")[:300]

    def fetch(self, symbol: str, timeframe: str, outputsize: int = 500) -> tuple[pd.DataFrame, bool]:
        if timeframe not in INTERVAL_MAP:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
        interval = INTERVAL_MAP[timeframe]
        try:
            response = requests.get(
                f"{self.base_url}/time_series",
                params={
                    "symbol": symbol,
                    "interval": interval,
                    "outputsize": max(220, min(int(outputsize), 5000)),
                    "timezone": "UTC",
                    "format": "JSON",
                    "apikey": self.api_key,
                },
                timeout=25,
            )
        except requests.RequestException as exc:
            raise MarketDataError(f"Market-data connection failed: {exc.__class__.__name__}") from exc

        self.last_credits_left = response.headers.get("api-credits-left")
        try:
            payload: dict[str, Any] = response.json()
        except ValueError:
            payload = {}

        if response.status_code == 429:
            raise MarketDataRateLimit(
                "Twelve Data quota/rate limit reached (HTTP 429). Wait for the provider quota to reset; no demo trade signal will be substituted."
            )
        if response.status_code >= 400:
            message = self._payload_message(payload, f"HTTP {response.status_code}")
            raise MarketDataError(f"Twelve Data request failed: {message}")
        if payload.get("status") == "error" or "values" not in payload:
            message = self._payload_message(payload, "Unknown Twelve Data response")
            if "credit" in message.lower() or "limit" in message.lower() or "quota" in message.lower():
                raise MarketDataRateLimit(f"Twelve Data quota/rate limit reached: {message}")
            raise MarketDataError(f"Twelve Data returned an error: {message}")

        raw = pd.DataFrame(payload["values"])
        if "datetime" in raw.columns:
            raw = raw.rename(columns={"datetime": "time"})
        return normalize_ohlcv(raw)

    def fetch_bundle(self, symbol: str, timeframes: list[str], outputsize: int = 500) -> MarketBundle:
        """Fetch two source series and derive M15, H4 and D1 locally.

        Direct provider calls:
        - M5 (M15 is derived)
        - H1 with enough history to derive H4 and at least 200 D1 bars

        This limits each all-timeframe gold synchronization to two provider
        series. Macro assets are loaded through their independent cached source
        hierarchy, so changing the visible timeframe never spends extra candle
        credits.
        """
        requested = list(dict.fromkeys(timeframes))
        invalid = [tf for tf in requested if tf not in INTERVAL_MAP]
        if invalid:
            raise ValueError(f"Unsupported timeframe(s): {', '.join(invalid)}")
        target = max(240, min(int(outputsize), 500))

        frames: dict[str, pd.DataFrame] = {}
        volume_flags: dict[str, bool] = {}
        provider_calls: list[str] = []

        if any(tf in requested for tf in ("M5", "M15")):
            m5_size = min(5000, target * 3 + 30)
            m5, has_volume = self.fetch(symbol, "M5", m5_size)
            provider_calls.append("M5")
            frames["M5"] = m5.tail(target).reset_index(drop=True)
            volume_flags["M5"] = has_volume
            if "M15" in requested:
                frames["M15"] = resample_ohlcv(m5, "15min", target)
                volume_flags["M15"] = has_volume

        if any(tf in requested for tf in ("H1", "H4", "D1")):
            # 5,000 hourly bars produce roughly 208 daily bars, enough for
            # EMA200 plus the core daily trend filters without a third request.
            required_h1 = 5000 if "D1" in requested else min(5000, target * 4 + 30)
            h1, has_volume = self.fetch(symbol, "H1", required_h1)
            provider_calls.append("H1")
            frames["H1"] = h1.tail(target).reset_index(drop=True)
            volume_flags["H1"] = has_volume
            if "H4" in requested:
                frames["H4"] = resample_ohlcv(h1, "4h", target)
                volume_flags["H4"] = has_volume
            if "D1" in requested:
                d1_limit = min(target, 208)
                frames["D1"] = resample_ohlcv(h1, "1D", d1_limit)
                volume_flags["D1"] = has_volume
                if len(frames["D1"]) < 200:
                    raise MarketDataError(
                        "The H1 feed did not return enough history to build a reliable D1 EMA200 series. Refresh later or reduce provider restrictions."
                    )

        frames = {tf: frames[tf] for tf in requested}
        reference_tf = next(tf for tf in ("M5", "M15", "H1", "H4", "D1") if tf in frames)
        reference = frames[reference_tf]
        latest = reference.iloc[-1]
        notes = [
            "Price and candles are indicative web-market data, not an executable broker quote.",
            f"Quota-safe all-timeframe plan: {', '.join(provider_calls)} requested from Twelve Data; M15, H4 and D1 are derived locally.",
            "Changing the visible timeframe uses the synchronized in-memory snapshot and makes no new provider request.",
        ]
        if self.last_credits_left is not None:
            notes.append(f"Provider-reported API credits left after the latest candle request: {self.last_credits_left}.")
        if not all(volume_flags.get(tf, False) for tf in requested):
            notes.append(
                "Spot XAU/USD has no centralized exchange volume in this feed. Where volume is absent, the terminal uses a candle-range activity proxy and labels it accordingly."
            )
        return MarketBundle(
            frames=frames,
            symbol=symbol,
            last_price=float(latest["close"]),
            data_time=pd.to_datetime(latest["time"], utc=True).isoformat(),
            source="TWELVE_DATA",
            notes=notes,
        )


def bundle_from_csv(file_bytes: bytes, symbol: str = "XAU/USD") -> MarketBundle:
    raw = pd.read_csv(BytesIO(file_bytes))
    frame, has_volume = normalize_ohlcv(raw)
    latest = frame.iloc[-1]
    notes = ["CSV mode analyzes the uploaded timeframe only; it is copied into each analysis slot."]
    if not has_volume:
        notes.append("The CSV had no usable volume column, so candle-range activity is used as a proxy.")
    frames = {tf: frame.copy() for tf in INTERVAL_MAP}
    return MarketBundle(
        frames=frames,
        symbol=symbol,
        last_price=float(latest["close"]),
        data_time=pd.to_datetime(latest["time"], utc=True).isoformat(),
        source="CSV_UPLOAD",
        notes=notes,
    )
