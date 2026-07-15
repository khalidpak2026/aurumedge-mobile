from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .models import LiquiditySnapshot, TradeSetup


DARK_BG = "#09111f"
PANEL_BG = "#0a1322"
GRID = "rgba(148,163,184,0.09)"
TEXT = "#dce5f0"
MUTED = "#7f8da2"
GOLD = "#f4c85b"
GREEN = "#22cfa0"
RED = "#ff5d7d"
CYAN = "#5dd6f4"
PURPLE = "#a98bff"


def _safe_last(df: pd.DataFrame, key: str) -> float | None:
    if key not in df.columns or df.empty:
        return None
    value = df[key].iloc[-1]
    return None if pd.isna(value) else float(value)


def _add_price_level(
    fig: go.Figure,
    level: float | None,
    label: str,
    color: str,
    dash: str = "dot",
    width: float = 1.0,
    opacity: float = 0.78,
) -> None:
    if level is None:
        return
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=level,
        y1=level,
        yref="y",
        line={"color": color, "width": width, "dash": dash},
        opacity=opacity,
        layer="above",
    )
    fig.add_annotation(
        x=1,
        xref="paper",
        y=level,
        yref="y",
        text=f" {label}  {level:,.2f} ",
        showarrow=False,
        xanchor="right",
        yanchor="bottom",
        bgcolor="rgba(6,9,19,.88)",
        bordercolor=color,
        borderwidth=1,
        borderpad=3,
        font={"color": color, "size": 9, "family": "Inter, Segoe UI, Arial"},
    )


def professional_market_chart(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    liquidity: LiquiditySnapshot | None = None,
    active_setup: TradeSetup | None = None,
) -> go.Figure:
    view = df.tail(260).copy()
    view["volume_ma20"] = view["tick_volume"].rolling(20, min_periods=1).mean()
    atr_last = _safe_last(view, "atr14")
    rsi_last = _safe_last(view, "rsi14")
    adx_last = _safe_last(view, "adx14")
    macd_last = _safe_last(view, "macd_hist")

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.62, 0.13, 0.13, 0.12],
        specs=[[{}], [{}], [{}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Candlestick(
            x=view["time"],
            open=view["open"],
            high=view["high"],
            low=view["low"],
            close=view["close"],
            name=symbol,
            increasing_line_color=GREEN,
            decreasing_line_color=RED,
            increasing_fillcolor=GREEN,
            decreasing_fillcolor=RED,
            whiskerwidth=0.35,
            hoverlabel={"bgcolor": "#0b1423"},
        ),
        row=1,
        col=1,
    )

    line_specs = (
        ("ema9", "EMA 9", "#7dd3fc", 1.0, "solid"),
        ("ema20", "EMA 20", CYAN, 1.45, "solid"),
        ("ema50", "EMA 50", GOLD, 1.55, "solid"),
        ("ema200", "EMA 200", PURPLE, 1.75, "solid"),
        ("vwap", "VWAP", "#f8fafc", 1.05, "dot"),
        ("supertrend", "Supertrend", "#35d399", 1.15, "dash"),
    )
    for key, label, color, width, dash in line_specs:
        if key in view.columns:
            fig.add_trace(
                go.Scatter(
                    x=view["time"],
                    y=view[key],
                    mode="lines",
                    name=label,
                    line={"color": color, "width": width, "dash": dash},
                    opacity=0.88,
                    hovertemplate=f"{label}: %{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if liquidity:
        for zone in liquidity.support_zones[:3]:
            fig.add_hrect(
                y0=zone["low"],
                y1=zone["high"],
                row=1,
                col=1,
                fillcolor="rgba(34,207,160,0.095)",
                line_color="rgba(34,207,160,.28)",
                line_width=0.7,
                annotation_text=f"DEMAND · {zone.get('touches', 1)} touches",
                annotation_position="bottom left",
                annotation_font_color="#66e6bd",
                annotation_font_size=9,
            )
        for zone in liquidity.resistance_zones[:3]:
            fig.add_hrect(
                y0=zone["low"],
                y1=zone["high"],
                row=1,
                col=1,
                fillcolor="rgba(255,93,125,0.085)",
                line_color="rgba(255,93,125,.26)",
                line_width=0.7,
                annotation_text=f"SUPPLY · {zone.get('touches', 1)} touches",
                annotation_position="top left",
                annotation_font_color="#ff91a8",
                annotation_font_size=9,
            )

        _add_price_level(fig, liquidity.previous_day_high, "PDH", "#ee8ed1", "dot", 0.8, 0.62)
        _add_price_level(fig, liquidity.previous_day_low, "PDL", "#6ca8ff", "dot", 0.8, 0.62)
        _add_price_level(fig, liquidity.point_of_control, "POC", GOLD, "dash", 1.15, 0.8)
        _add_price_level(fig, liquidity.value_area_high, "VAH", "#dcae46", "dot", 0.75, 0.55)
        _add_price_level(fig, liquidity.value_area_low, "VAL", "#dcae46", "dot", 0.75, 0.55)

        for level in liquidity.equal_highs[-2:]:
            _add_price_level(fig, level, "BUY-SIDE LIQ", RED, "dot", 0.75, 0.52)
        for level in liquidity.equal_lows[-2:]:
            _add_price_level(fig, level, "SELL-SIDE LIQ", GREEN, "dot", 0.75, 0.52)

        if liquidity.sweep_above is not None:
            marker_time = view["time"].iloc[-4] if len(view) > 4 else view["time"].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[marker_time],
                    y=[liquidity.sweep_above],
                    mode="markers+text",
                    text=["LIQUIDITY SWEEP"],
                    textposition="top center",
                    marker={"symbol": "triangle-down", "size": 13, "color": RED, "line": {"width": 1, "color": "#ffd0da"}},
                    textfont={"color": "#ff91a8", "size": 9},
                    name="Buy-side sweep",
                    hovertemplate="Buy-side liquidity sweep<extra></extra>",
                ),
                row=1,
                col=1,
            )
        if liquidity.sweep_below is not None:
            marker_time = view["time"].iloc[-4] if len(view) > 4 else view["time"].iloc[-1]
            fig.add_trace(
                go.Scatter(
                    x=[marker_time],
                    y=[liquidity.sweep_below],
                    mode="markers+text",
                    text=["LIQUIDITY SWEEP"],
                    textposition="bottom center",
                    marker={"symbol": "triangle-up", "size": 13, "color": GREEN, "line": {"width": 1, "color": "#c7ffed"}},
                    textfont={"color": "#66e6bd", "size": 9},
                    name="Sell-side sweep",
                    hovertemplate="Sell-side liquidity sweep<extra></extra>",
                ),
                row=1,
                col=1,
            )

        for gap in [item for item in liquidity.bullish_fvgs if not item.get("filled")][-2:]:
            fig.add_hrect(
                y0=gap["low"], y1=gap["high"], row=1, col=1,
                fillcolor="rgba(93,214,244,.055)", line_color="rgba(93,214,244,.20)", line_width=.6,
            )
        for gap in [item for item in liquidity.bearish_fvgs if not item.get("filled")][-2:]:
            fig.add_hrect(
                y0=gap["low"], y1=gap["high"], row=1, col=1,
                fillcolor="rgba(169,139,255,.05)", line_color="rgba(169,139,255,.18)", line_width=.6,
            )

    if active_setup:
        setup_color = GREEN if active_setup.side == "BUY" else RED
        fig.add_hrect(
            y0=active_setup.entry_low,
            y1=active_setup.entry_high,
            row=1,
            col=1,
            fillcolor="rgba(244,200,91,0.13)",
            line_color=GOLD,
            line_width=1.0,
            annotation_text=f"{active_setup.side} ENTRY ZONE",
            annotation_position="top right",
            annotation_font_color="#ffe59b",
            annotation_font_size=9,
        )
        _add_price_level(fig, active_setup.stop_loss, "STOP", RED, "dash", 1.3, 0.95)
        _add_price_level(fig, active_setup.take_profit_1, "TP1", setup_color, "dot", 1.0, 0.83)
        _add_price_level(fig, active_setup.take_profit_2, "TP2", setup_color, "dot", 1.0, 0.75)
        _add_price_level(fig, active_setup.take_profit_3, "TP3", setup_color, "dot", 1.0, 0.67)

    last_price = float(view["close"].iloc[-1])
    _add_price_level(fig, last_price, "LAST", "#f2f5f8", "solid", 0.8, 0.56)

    volume_colors = [GREEN if close >= open_ else RED for open_, close in zip(view["open"], view["close"])]
    fig.add_trace(
        go.Bar(
            x=view["time"], y=view["tick_volume"], name="Volume / activity",
            marker_color=volume_colors, opacity=0.6, hovertemplate="Activity: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=view["time"], y=view["volume_ma20"], mode="lines", name="Volume MA20",
            line={"color": "#c7d2e1", "width": 1.0}, opacity=.7,
            hovertemplate="Volume MA20: %{y:,.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    macd_colors = [GREEN if value >= 0 else RED for value in view["macd_hist"]]
    fig.add_trace(
        go.Bar(x=view["time"], y=view["macd_hist"], name="MACD histogram", marker_color=macd_colors, opacity=.62),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view["time"], y=view["macd"], mode="lines", name="MACD", line={"color": CYAN, "width": 1.1}),
        row=3, col=1,
    )
    fig.add_trace(
        go.Scatter(x=view["time"], y=view["macd_signal"], mode="lines", name="MACD signal", line={"color": GOLD, "width": 1.0}),
        row=3, col=1,
    )

    fig.add_trace(
        go.Scatter(x=view["time"], y=view["rsi14"], mode="lines", name="RSI 14", line={"color": CYAN, "width": 1.25}),
        row=4, col=1, secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=view["time"], y=view["adx14"], mode="lines", name="ADX 14", line={"color": GOLD, "width": 1.15}),
        row=4, col=1, secondary_y=True,
    )
    for level in (30, 50, 70):
        fig.add_hline(y=level, row=4, col=1, line_dash="dot", line_color="rgba(148,163,184,.22)", line_width=.7)
    fig.add_hline(y=20, row=4, col=1, secondary_y=True, line_dash="dash", line_color="rgba(244,200,91,.34)", line_width=.75)

    stats = []
    if atr_last is not None:
        stats.append(f"ATR {atr_last:,.2f}")
    if rsi_last is not None:
        stats.append(f"RSI {rsi_last:.1f}")
    if adx_last is not None:
        stats.append(f"ADX {adx_last:.1f}")
    if macd_last is not None:
        stats.append(f"MACD H {macd_last:+.2f}")

    fig.update_layout(
        title={
            "text": f"<b>{symbol}</b>  ·  {timeframe} &nbsp;&nbsp; <span style='font-size:11px;color:{MUTED}'>{'  ·  '.join(stats)}</span>",
            "x": .012, "xanchor": "left", "y": .975,
        },
        height=760,
        paper_bgcolor=DARK_BG,
        plot_bgcolor=PANEL_BG,
        font={"color": TEXT, "family": "Inter, Segoe UI, Arial", "size": 10},
        margin={"l": 46, "r": 78, "t": 60, "b": 24},
        legend={
            "orientation": "h", "yanchor": "bottom", "y": 1.015,
            "xanchor": "right", "x": 1, "font": {"size": 9, "color": "#aeb9c8"},
            "bgcolor": "rgba(6,9,19,.55)", "bordercolor": "rgba(148,163,184,.08)", "borderwidth": 1,
        },
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        hoverlabel={"bgcolor": "#0b1423", "bordercolor": "rgba(148,163,184,.18)", "font": {"color": TEXT}},
        dragmode="pan",
        modebar={"bgcolor": "rgba(9,17,31,.78)", "color": MUTED, "activecolor": GOLD},
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, rangeslider_visible=False, showline=False, tickfont={"color": MUTED, "size": 9})
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, tickfont={"color": MUTED, "size": 9})
    fig.update_yaxes(title_text="PRICE", title_font={"size": 9, "color": MUTED}, row=1, col=1, side="right")
    fig.update_yaxes(title_text="VOL", title_font={"size": 8, "color": MUTED}, row=2, col=1, side="right")
    fig.update_yaxes(title_text="MACD", title_font={"size": 8, "color": MUTED}, row=3, col=1, side="right")
    fig.update_yaxes(title_text="RSI", range=[0, 100], title_font={"size": 8, "color": MUTED}, row=4, col=1, secondary_y=False)
    fig.update_yaxes(title_text="ADX", range=[0, 70], title_font={"size": 8, "color": MUTED}, row=4, col=1, secondary_y=True, side="right")
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    view = df.tail(220)
    colors = [GREEN if value >= 0 else RED for value in view["macd_hist"]]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=view["time"], y=view["macd_hist"], name="Histogram", marker_color=colors, opacity=.7))
    fig.add_trace(go.Scatter(x=view["time"], y=view["macd"], mode="lines", name="MACD", line={"color": CYAN, "width":1.3}))
    fig.add_trace(go.Scatter(x=view["time"], y=view["macd_signal"], mode="lines", name="Signal", line={"color": GOLD, "width":1.2}))
    fig.update_layout(
        height=315, paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font={"color": TEXT, "family":"Inter"},
        margin={"l": 30, "r": 20, "t": 40, "b": 25}, xaxis_rangeslider_visible=False,
        legend={"orientation":"h", "y":1.08, "x":1, "xanchor":"right", "font":{"size":9}},
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor=GRID, tickfont={"color":MUTED, "size":9})
    fig.update_yaxes(gridcolor=GRID, tickfont={"color":MUTED, "size":9}, zerolinecolor="rgba(148,163,184,.20)")
    return fig


def regime_strength_chart(snapshots: list) -> go.Figure:
    labels = [item.timeframe for item in snapshots]
    values = [item.directional_score for item in snapshots]
    colors = [GREEN if value >= 0 else RED for value in values]
    fig = go.Figure(
        go.Bar(
            x=labels, y=values, marker_color=colors,
            text=[f"{value:+.0f}" for value in values], textposition="outside",
            marker_line={"color":"rgba(255,255,255,.08)","width":1},
        )
    )
    fig.add_hrect(y0=-14, y1=14, fillcolor="rgba(244,200,91,.05)", line_width=0, annotation_text="NEUTRAL", annotation_position="top left", annotation_font_color=GOLD)
    fig.add_hline(y=0, line_color="rgba(226,232,240,.35)")
    fig.update_yaxes(range=[-110, 110], title="Directional score", gridcolor=GRID, tickfont={"color":MUTED})
    fig.update_layout(
        height=315, paper_bgcolor=DARK_BG, plot_bgcolor=PANEL_BG, font={"color": TEXT, "family":"Inter"},
        margin={"l": 45, "r": 20, "t": 32, "b": 28}, showlegend=False,
    )
    fig.update_xaxes(gridcolor=GRID, tickfont={"color":MUTED})
    return fig
