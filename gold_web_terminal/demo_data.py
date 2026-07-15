from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd


_TIMEFRAME_MINUTES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240, "D1": 1440}


def generate_demo_bars(timeframe: str, count: int = 650, anchor: float = 3300.0) -> pd.DataFrame:
    minutes = _TIMEFRAME_MINUTES[timeframe]
    seed = int(hashlib.sha256(timeframe.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    end = pd.Timestamp.now(tz="UTC").floor(f"{minutes}min")
    times = pd.date_range(end=end, periods=count, freq=f"{minutes}min", tz="UTC")
    scale = {"M5": 0.7, "M15": 1.1, "H1": 2.0, "H4": 4.0, "D1": 9.0}[timeframe]
    drift = {"M5": 0.01, "M15": 0.015, "H1": 0.025, "H4": 0.05, "D1": 0.12}[timeframe]
    returns = rng.normal(drift, scale, count)
    close = anchor + np.cumsum(returns)
    open_ = np.r_[close[0] - returns[0], close[:-1]]
    wick = np.abs(rng.normal(scale * 0.7, scale * 0.35, count)) + 0.05
    high = np.maximum(open_, close) + wick
    low = np.minimum(open_, close) - wick
    volume = rng.integers(150, 1400, count)
    spread = rng.integers(15, 45, count)
    return pd.DataFrame(
        {
            "time": times,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
            "spread": spread,
            "real_volume": np.zeros(count, dtype=int),
        }
    )
