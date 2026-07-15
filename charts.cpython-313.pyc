from __future__ import annotations

import html
import math
from typing import Iterable

import numpy as np
import pandas as pd

from .models import IndicatorSnapshot, LiquiditySnapshot, TradeSetup

BG = "#07101d"
PANEL = "#0b1627"
GRID = "#203047"
TEXT = "#eaf0f8"
MUTED = "#8fa0b7"
GREEN = "#22cfa0"
RED = "#ff5d7d"
GOLD = "#f4c85b"
CYAN = "#59d5f5"
PURPLE = "#a98bff"
WHITE = "#f8fafc"


def _finite(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _points(values: Iterable[object], xs: list[float], y_map) -> str:
    pts: list[str] = []
    for x, raw in zip(xs, values):
        value = _finite(raw)
        if value is not None:
            pts.append(f"{x:.1f},{y_map(value):.1f}")
    return " ".join(pts)


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _price_label(x: float, y: float, label: str, value: float, color: str) -> str:
    text = f"{label} {value:,.2f}"
    width = max(72, 7.1 * len(text) + 14)
    return (
        f'<rect x="{x-width:.1f}" y="{y-10:.1f}" width="{width:.1f}" height="20" rx="5" fill="{color}" opacity=".92"/>'
        f'<text x="{x-6:.1f}" y="{y+4:.1f}" text-anchor="end" fill="#07101d" font-size="10" font-weight="800">{_esc(text)}</text>'
    )


def mobile_market_map_html(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    liquidity: LiquiditySnapshot | None = None,
    active_setup: TradeSetup | None = None,
    bars: int = 92,
) -> str:
    required = {"time", "open", "high", "low", "close"}
    if df is None or df.empty or not required.issubset(df.columns):
        return '<div class="svg-chart-error">Chart data is unavailable.</div>'

    view = df.tail(max(36, bars)).copy().reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        view[col] = pd.to_numeric(view[col], errors="coerce")
    view = view.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    if len(view) < 12:
        return '<div class="svg-chart-error">Not enough candles to draw the chart.</div>'

    W, H = 1000, 700
    L, R = 58, 888
    T, B = 58, 500
    VT, VB = 530, 635
    n = len(view)
    xs = np.linspace(L + 5, R - 5, n).tolist()
    step = (R - L) / max(1, n)
    candle_w = max(2.2, min(7.0, step * 0.58))

    extra_prices: list[float] = []
    if liquidity:
        for value in (
            liquidity.previous_day_high,
            liquidity.previous_day_low,
            liquidity.point_of_control,
            liquidity.value_area_high,
            liquidity.value_area_low,
            liquidity.nearest_support,
            liquidity.nearest_resistance,
        ):
            if _finite(value) is not None:
                extra_prices.append(float(value))
        for zone in liquidity.support_zones[:3] + liquidity.resistance_zones[:3]:
            extra_prices.extend([float(zone["low"]), float(zone["high"])])
    if active_setup:
        extra_prices.extend([
            active_setup.entry_low, active_setup.entry_high, active_setup.stop_loss,
            active_setup.take_profit_1, active_setup.take_profit_2, active_setup.take_profit_3,
        ])

    low = float(np.nanmin(np.r_[view["low"].to_numpy(float), np.asarray(extra_prices or [np.nan])]))
    high = float(np.nanmax(np.r_[view["high"].to_numpy(float), np.asarray(extra_prices or [np.nan])]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return '<div class="svg-chart-error">Price scale could not be calculated.</div>'
    pad = max((high - low) * 0.07, float(view["close"].iloc[-1]) * 0.0007)
    low -= pad
    high += pad

    def y_map(value: float) -> float:
        return B - (value - low) / (high - low) * (B - T)

    parts: list[str] = []
    parts.append('<div class="svg-chart-shell">')
    parts.append(f'<svg class="mobile-market-svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="{_esc(symbol)} {timeframe} market chart">')
    parts.append(f'<rect width="{W}" height="{H}" rx="18" fill="{BG}"/>')
    parts.append(f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" rx="8" fill="{PANEL}"/>')
    parts.append(f'<rect x="{L}" y="{VT}" width="{R-L}" height="{VB-VT}" rx="8" fill="{PANEL}"/>')

    # Horizontal grid and labels.
    for i in range(6):
        price = low + (high - low) * i / 5
        y = y_map(price)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{GRID}" stroke-width="1" opacity=".65"/>')
        parts.append(f'<text x="{R+12}" y="{y+4:.1f}" fill="{MUTED}" font-size="10">{price:,.1f}</text>')
    for i in range(7):
        x = L + (R - L) * i / 6
        parts.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{B}" stroke="{GRID}" stroke-width="1" opacity=".35"/>')

    # Supply and demand zones.
    if liquidity:
        for zone in liquidity.support_zones[:3]:
            y1, y2 = y_map(float(zone["high"])), y_map(float(zone["low"]))
            parts.append(f'<rect x="{L}" y="{min(y1,y2):.1f}" width="{R-L}" height="{abs(y2-y1):.1f}" fill="{GREEN}" opacity=".10" stroke="{GREEN}" stroke-opacity=".35"/>')
            parts.append(f'<text x="{L+7}" y="{min(y1,y2)+13:.1f}" fill="{GREEN}" font-size="9" font-weight="700">DEMAND · {int(zone.get("touches",1))} touch</text>')
        for zone in liquidity.resistance_zones[:3]:
            y1, y2 = y_map(float(zone["high"])), y_map(float(zone["low"]))
            parts.append(f'<rect x="{L}" y="{min(y1,y2):.1f}" width="{R-L}" height="{abs(y2-y1):.1f}" fill="{RED}" opacity=".085" stroke="{RED}" stroke-opacity=".30"/>')
            parts.append(f'<text x="{L+7}" y="{min(y1,y2)+13:.1f}" fill="{RED}" font-size="9" font-weight="700">SUPPLY · {int(zone.get("touches",1))} touch</text>')

    # Entry and target overlays.
    if active_setup:
        y1, y2 = y_map(active_setup.entry_high), y_map(active_setup.entry_low)
        parts.append(f'<rect x="{L}" y="{min(y1,y2):.1f}" width="{R-L}" height="{abs(y2-y1):.1f}" fill="{GOLD}" opacity=".12" stroke="{GOLD}" stroke-width="1.3"/>')
        parts.append(f'<text x="{L+8}" y="{min(y1,y2)+14:.1f}" fill="{GOLD}" font-size="10" font-weight="800">{active_setup.side} ENTRY</text>')

    # Candles.
    for i, row in view.iterrows():
        x = xs[i]
        o, h, lo, c = map(float, (row["open"], row["high"], row["low"], row["close"]))
        color = GREEN if c >= o else RED
        parts.append(f'<line x1="{x:.1f}" y1="{y_map(h):.1f}" x2="{x:.1f}" y2="{y_map(lo):.1f}" stroke="{color}" stroke-width="1.15" opacity=".95"/>')
        top = min(y_map(o), y_map(c))
        height = max(1.5, abs(y_map(o) - y_map(c)))
        parts.append(f'<rect x="{x-candle_w/2:.1f}" y="{top:.1f}" width="{candle_w:.1f}" height="{height:.1f}" rx=".7" fill="{color}" opacity=".95"/>')

    # Indicator paths.
    path_specs = [
        ("ema20", "EMA20", CYAN, 1.8, ""),
        ("ema50", "EMA50", GOLD, 1.9, ""),
        ("ema200", "EMA200", PURPLE, 2.0, ""),
        ("vwap", "VWAP", WHITE, 1.3, 'stroke-dasharray="5 4"'),
        ("supertrend", "SUPERTREND", GREEN, 1.3, 'stroke-dasharray="7 5"'),
    ]
    for key, _, color, width, extra in path_specs:
        if key in view.columns:
            pts = _points(view[key], xs, y_map)
            if pts:
                parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" {extra} opacity=".92"/>')

    # Named levels and setup levels.
    levels: list[tuple[float | None, str, str, str]] = []
    if liquidity:
        levels.extend([
            (liquidity.previous_day_high, "PDH", "#e589cf", "4 4"),
            (liquidity.previous_day_low, "PDL", "#6da8ff", "4 4"),
            (liquidity.point_of_control, "POC", GOLD, "7 4"),
            (liquidity.value_area_high, "VAH", "#dcae46", "3 4"),
            (liquidity.value_area_low, "VAL", "#dcae46", "3 4"),
            (liquidity.nearest_support, "SUPPORT", GREEN, "5 5"),
            (liquidity.nearest_resistance, "RESIST", RED, "5 5"),
        ])
    if active_setup:
        levels.extend([
            (active_setup.stop_loss, "SL", RED, "7 4"),
            (active_setup.take_profit_1, "TP1", GREEN, "4 4"),
            (active_setup.take_profit_2, "TP2", GREEN, "4 4"),
            (active_setup.take_profit_3, "TP3", GREEN, "4 4"),
        ])
    used_y: list[float] = []
    for raw, label, color, dash in levels:
        value = _finite(raw)
        if value is None or value < low or value > high:
            continue
        y = y_map(value)
        if any(abs(y - prev) < 15 for prev in used_y):
            y += 13
        used_y.append(y)
        parts.append(f'<line x1="{L}" y1="{y:.1f}" x2="{R}" y2="{y:.1f}" stroke="{color}" stroke-width="1" stroke-dasharray="{dash}" opacity=".72"/>')
        parts.append(_price_label(R + 104, y, label, value, color))

    last = float(view["close"].iloc[-1])
    last_y = y_map(last)
    parts.append(f'<line x1="{L}" y1="{last_y:.1f}" x2="{R}" y2="{last_y:.1f}" stroke="{WHITE}" stroke-width="1" opacity=".60"/>')
    parts.append(_price_label(R + 104, last_y, "LAST", last, WHITE))

    # Volume bars.
    vol_col = "tick_volume" if "tick_volume" in view.columns else None
    if vol_col:
        vols = pd.to_numeric(view[vol_col], errors="coerce").fillna(0).to_numpy(float)
        vmax = max(float(np.nanmax(vols)), 1.0)
        for i, vol in enumerate(vols):
            x = xs[i]
            bar_h = (vol / vmax) * (VB - VT - 10)
            color = GREEN if float(view.iloc[i]["close"]) >= float(view.iloc[i]["open"]) else RED
            parts.append(f'<rect x="{x-candle_w/2:.1f}" y="{VB-bar_h:.1f}" width="{candle_w:.1f}" height="{bar_h:.1f}" fill="{color}" opacity=".48"/>')

    # Date/time labels.
    times = pd.to_datetime(view["time"], utc=True, errors="coerce")
    for idx in np.linspace(0, n - 1, 5).astype(int):
        if pd.isna(times.iloc[idx]):
            continue
        label = times.iloc[idx].strftime("%d %b\n%H:%M")
        date, clock = label.split("\n")
        x = xs[idx]
        parts.append(f'<text x="{x:.1f}" y="{VB+22}" text-anchor="middle" fill="{MUTED}" font-size="9">{date}</text>')
        parts.append(f'<text x="{x:.1f}" y="{VB+34}" text-anchor="middle" fill="{MUTED}" font-size="8">{clock}</text>')

    # Title and legend.
    atr = _finite(view["atr14"].iloc[-1]) if "atr14" in view.columns else None
    rsi = _finite(view["rsi14"].iloc[-1]) if "rsi14" in view.columns else None
    adx = _finite(view["adx14"].iloc[-1]) if "adx14" in view.columns else None
    stats = " · ".join(filter(None, [f"ATR {atr:.2f}" if atr else "", f"RSI {rsi:.1f}" if rsi else "", f"ADX {adx:.1f}" if adx else ""]))
    parts.append(f'<text x="{L}" y="28" fill="{TEXT}" font-size="17" font-weight="800">{_esc(symbol)} · {_esc(timeframe)}</text>')
    parts.append(f'<text x="{L}" y="45" fill="{MUTED}" font-size="10">{_esc(stats)}</text>')
    legend = [(CYAN, "EMA20"), (GOLD, "EMA50"), (PURPLE, "EMA200"), (WHITE, "VWAP"), (GREEN, "Supertrend")]
    lx = 500
    for color, label in legend:
        parts.append(f'<line x1="{lx}" y1="31" x2="{lx+18}" y2="31" stroke="{color}" stroke-width="2"/>')
        parts.append(f'<text x="{lx+23}" y="35" fill="{MUTED}" font-size="9">{label}</text>')
        lx += 94
    parts.append(f'<text x="{L}" y="{VT-8}" fill="{MUTED}" font-size="10" font-weight="700">VOLUME / ACTIVITY</text>')
    parts.append('</svg></div>')
    return "".join(parts)


def mobile_regime_html(snapshots: list[IndicatorSnapshot]) -> str:
    W, H = 720, 250
    rows = snapshots[:5]
    left, right = 105, 665
    top = 32
    row_h = 39
    out = [f'<div class="svg-chart-shell"><svg class="mobile-mini-svg" viewBox="0 0 {W} {H}" role="img" aria-label="Multi timeframe directional strength">', f'<rect width="{W}" height="{H}" rx="16" fill="{BG}"/>']
    out.append(f'<text x="20" y="24" fill="{TEXT}" font-size="14" font-weight="800">MULTI-TIMEFRAME DIRECTION</text>')
    out.append(f'<line x1="{(left+right)/2}" y1="30" x2="{(left+right)/2}" y2="{H-18}" stroke="{GRID}" stroke-width="1"/>')
    for i, s in enumerate(rows):
        y = top + i * row_h + 17
        score = max(-100.0, min(100.0, float(s.directional_score)))
        center = (left + right) / 2
        width = abs(score) / 100 * (right-left)/2
        x = center if score >= 0 else center - width
        color = GREEN if score >= 0 else RED
        out.append(f'<text x="20" y="{y+4}" fill="{TEXT}" font-size="11" font-weight="800">{_esc(s.timeframe)}</text>')
        out.append(f'<rect x="{left}" y="{y-9}" width="{right-left}" height="18" rx="9" fill="{PANEL}"/>')
        out.append(f'<rect x="{x:.1f}" y="{y-9}" width="{max(width,1):.1f}" height="18" rx="9" fill="{color}" opacity=".78"/>')
        out.append(f'<text x="{right+10}" y="{y+4}" fill="{color}" font-size="10" font-weight="800">{score:+.0f}</text>')
    out.append('</svg></div>')
    return "".join(out)


def mobile_macd_html(df: pd.DataFrame, timeframe: str, bars: int = 90) -> str:
    needed = {"time", "macd", "macd_signal", "macd_hist"}
    if df is None or df.empty or not needed.issubset(df.columns):
        return '<div class="svg-chart-error">MACD data is unavailable.</div>'
    view = df.tail(bars).copy().reset_index(drop=True)
    W, H = 820, 300
    L, R, T, B = 44, 790, 45, 245
    xs = np.linspace(L, R, len(view)).tolist()
    values = pd.concat([pd.to_numeric(view[c], errors="coerce") for c in ("macd", "macd_signal", "macd_hist")]).dropna()
    if values.empty:
        return '<div class="svg-chart-error">MACD values are unavailable.</div>'
    vmin, vmax = float(values.min()), float(values.max())
    span = max(vmax-vmin, 1e-6)
    vmin -= span*.12; vmax += span*.12
    def y(v: float) -> float:
        return B - (v-vmin)/(vmax-vmin)*(B-T)
    zero = y(0.0)
    out = [f'<div class="svg-chart-shell"><svg class="mobile-mini-svg" viewBox="0 0 {W} {H}" role="img" aria-label="{_esc(timeframe)} MACD">', f'<rect width="{W}" height="{H}" rx="16" fill="{BG}"/>', f'<rect x="{L}" y="{T}" width="{R-L}" height="{B-T}" rx="8" fill="{PANEL}"/>']
    out.append(f'<text x="{L}" y="26" fill="{TEXT}" font-size="14" font-weight="800">{_esc(timeframe)} MACD</text>')
    out.append(f'<line x1="{L}" y1="{zero:.1f}" x2="{R}" y2="{zero:.1f}" stroke="{GRID}" stroke-width="1"/>')
    bw = max(2.0, (R-L)/len(view)*.58)
    for i, raw in enumerate(view["macd_hist"]):
        value = _finite(raw)
        if value is None: continue
        yy = y(value)
        color = GREEN if value >= 0 else RED
        out.append(f'<rect x="{xs[i]-bw/2:.1f}" y="{min(yy,zero):.1f}" width="{bw:.1f}" height="{max(1,abs(yy-zero)):.1f}" fill="{color}" opacity=".58"/>')
    for key, color, label in (("macd", CYAN, "MACD"), ("macd_signal", GOLD, "SIGNAL")):
        pts = _points(view[key], xs, y)
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"/>')
    out.append(f'<line x1="{R-150}" y1="25" x2="{R-130}" y2="25" stroke="{CYAN}" stroke-width="2"/><text x="{R-125}" y="29" fill="{MUTED}" font-size="9">MACD</text>')
    out.append(f'<line x1="{R-75}" y1="25" x2="{R-55}" y2="25" stroke="{GOLD}" stroke-width="2"/><text x="{R-50}" y="29" fill="{MUTED}" font-size="9">SIGNAL</text>')
    out.append('</svg></div>')
    return "".join(out)
