import pandas as pd

from gold_web_terminal.market_data import normalize_ohlcv


def test_normalize_generates_activity_proxy_without_volume():
    raw = pd.DataFrame(
        {
            "datetime": pd.date_range("2026-01-01", periods=40, freq="h"),
            "open": range(100, 140),
            "high": range(102, 142),
            "low": range(99, 139),
            "close": range(101, 141),
        }
    )
    frame, has_volume = normalize_ohlcv(raw)
    assert not has_volume
    assert len(frame) == 40
    assert frame["tick_volume"].sum() > 0
