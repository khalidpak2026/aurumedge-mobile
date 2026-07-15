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


class TwelveDataClient:
    def __init__(self, api_key: str, base_url: str = "https://api.twelvedata.com") -> None:
        if not api_key:
            raise ValueError("TWELVE_DATA_API_KEY is missing.")
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")

    def fetch(self, symbol: str, timeframe: str, outputsize: int = 500) -> tuple[pd.DataFrame, bool]:
        interval = INTERVAL_MAP[timeframe]
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
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        if payload.get("status") == "error" or "values" not in payload:
            message = payload.get("message") or payload.get("code") or "Unknown Twelve Data response"
            raise RuntimeError(str(message))
        raw = pd.DataFrame(payload["values"])
        if "datetime" in raw.columns:
            raw = raw.rename(columns={"datetime": "time"})
        return normalize_ohlcv(raw)

    def fetch_bundle(self, symbol: str, timeframes: list[str], outputsize: int = 500) -> MarketBundle:
        frames: dict[str, pd.DataFrame] = {}
        volume_flags: list[bool] = []
        for timeframe in timeframes:
            frame, has_volume = self.fetch(symbol, timeframe, outputsize)
            frames[timeframe] = frame
            volume_flags.append(has_volume)
        reference = frames["M5"] if "M5" in frames else frames[timeframes[0]]
        latest = reference.iloc[-1]
        notes = ["Price and candles are indicative web-market data, not an executable broker quote."]
        if not all(volume_flags):
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
