from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from gold_web_terminal.ai_engine import research_gold_news, synthesize_ai_analysis
from gold_web_terminal.charts import macd_chart, professional_market_chart, regime_strength_chart
from gold_web_terminal.config import Settings
from gold_web_terminal.demo_data import generate_demo_bars
from gold_web_terminal.indicators import add_indicators, summarize_indicators
from gold_web_terminal.liquidity import analyze_liquidity
from gold_web_terminal.market_data import MarketBundle, TwelveDataClient
from gold_web_terminal.models import AIAnalysis, TechnicalReport
from gold_web_terminal.strategy import build_technical_report
from gold_web_terminal.ui import level_ladder_html, signal_panel_html, timeframe_cards_html

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

BUILD_VERSION = "5.0.0-mobile"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
TV_INTERVALS = {"M5": "5", "M15": "15", "H1": "60", "H4": "240", "D1": "D"}

st.set_page_config(
    page_title="AurumEdge Mobile",
    page_icon="🟡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

MOBILE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');
:root{--bg:#060a12;--panel:#0d1625;--panel2:#101c2f;--line:rgba(148,163,184,.15);--text:#f4f7fb;--muted:#8e9bb0;--gold:#f4c85b;--green:#22cfa0;--red:#ff5d7d;--purple:#a98bff;--amber:#ffb94a;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;}
.stApp{background:radial-gradient(circle at 100% -10%,rgba(244,200,91,.12),transparent 28%),linear-gradient(180deg,#050811,#08101d 52%,#050810);color:var(--text);}
header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:760px!important;padding:calc(.5rem + env(safe-area-inset-top)) .72rem calc(5.8rem + env(safe-area-inset-bottom))!important;}
.mobile-header{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:8px 2px 12px;border-bottom:1px solid var(--line);margin-bottom:10px;}
.mobile-brand{display:flex;align-items:center;gap:10px;min-width:0}.mobile-logo{width:39px;height:39px;border-radius:13px;display:grid;place-items:center;font-family:'Manrope';font-weight:800;color:#171106;background:linear-gradient(135deg,#ffe79e,var(--gold) 55%,#bd7f16);box-shadow:0 8px 24px rgba(244,200,91,.18)}
.mobile-title{font-family:'Manrope';font-size:.98rem;font-weight:800;letter-spacing:.02em}.mobile-sub{font-size:.66rem;color:var(--muted);margin-top:2px}.mobile-live{padding:7px 9px;border-radius:999px;border:1px solid var(--line);font-size:.62rem;font-weight:800;color:#b9c5d4;white-space:nowrap}.mobile-live:before{content:'';display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:6px;box-shadow:0 0 0 4px rgba(34,207,160,.10)}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;border-radius:16px!important;background:linear-gradient(180deg,rgba(15,24,41,.96),rgba(8,14,25,.96))!important;box-shadow:0 14px 40px rgba(0,0,0,.18)}
[data-testid="stVerticalBlockBorderWrapper"]>div{padding:11px 12px!important}
div[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:#0a1220!important;border-color:rgba(148,163,184,.22)!important;color:var(--text)!important;border-radius:12px!important;min-height:46px!important}
.stButton>button{min-height:48px;border-radius:12px;font-weight:800;border:1px solid rgba(244,200,91,.44)!important;color:#171106!important;background:linear-gradient(135deg,#ffe28a,var(--gold) 55%,#d99d24)!important;box-shadow:0 8px 25px rgba(244,200,91,.16)}
button[kind="secondary"]{color:#e5edf7!important;background:#101c2f!important;border-color:var(--line)!important;box-shadow:none!important}
.mobile-kpis{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.mkpi{padding:11px 12px;border-radius:13px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(16,27,46,.96),rgba(8,15,27,.96));min-height:70px}.mkpi span{display:block;font-size:.61rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:800}.mkpi strong{display:block;font-family:'Manrope';font-size:1.05rem;margin-top:5px}.mkpi small{display:block;color:#718198;font-size:.62rem;margin-top:3px}.state-buy{color:var(--green)!important}.state-sell{color:var(--red)!important}.state-trap{color:var(--purple)!important}.state-stuck{color:var(--amber)!important}
.mobile-section{font-family:'Manrope';font-size:.86rem;font-weight:800;margin:14px 2px 8px}.mobile-note{font-size:.69rem;color:var(--muted);line-height:1.5}.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;background:rgba(6,10,18,.94);backdrop-filter:blur(14px)}.stTabs [data-baseweb="tab"]{height:45px;min-width:max-content;padding:0 12px;color:#93a1b5;font-size:.69rem;font-weight:800}.stTabs [aria-selected="true"]{color:var(--gold)!important}.stTabs [data-baseweb="tab-highlight"]{background:var(--gold)!important}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}.stMetric{border:1px solid var(--line);border-radius:12px;padding:9px 10px;background:#0a1322}.stMetric label{font-size:.64rem!important}.stMetric [data-testid="stMetricValue"]{font-size:1rem!important}
.js-plotly-plot .plotly .modebar{transform:scale(.9);transform-origin:right top}.mobile-callout{padding:11px 12px;border-radius:12px;border:1px solid rgba(244,200,91,.2);background:rgba(244,200,91,.06);color:#cbd5e1;font-size:.7rem;line-height:1.45}.mobile-footer{text-align:center;color:#65758b;font-size:.6rem;margin:18px 0 4px}
@media(max-width:430px){.block-container{padding-left:.52rem!important;padding-right:.52rem!important}.mobile-title{font-size:.9rem}.mobile-sub{display:none}.mobile-live{font-size:.57rem;padding:6px 7px}.mkpi{padding:10px}.stTabs [data-baseweb="tab"]{padding:0 10px}.js-plotly-plot .plotly .legend{display:none!important}}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def state_class(state: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "TRAP": "trap", "STUCK": "stuck"}.get(state, "stuck")


@st.cache_data(ttl=75, show_spinner=False)
def fetch_live_bundle(api_key: str, symbol: str, bars: int) -> MarketBundle:
    return TwelveDataClient(api_key).fetch_bundle(symbol, TIMEFRAMES, bars)


@st.cache_data(ttl=1200, show_spinner=False)
def cached_news(api_key: str, model: str) -> dict[str, Any]:
    return research_gold_news(api_key, model)


@st.cache_data(ttl=1200, show_spinner=False)
def cached_synthesis(api_key: str, model: str, report_json: str, research_text: str) -> dict[str, Any]:
    report = TechnicalReport.model_validate_json(report_json)
    result = synthesize_ai_analysis(api_key, model, report, research_text, None)
    return result.model_dump(mode="json")


def demo_bundle(bars: int) -> MarketBundle:
    frames = {tf: generate_demo_bars(tf, bars) for tf in TIMEFRAMES}
    latest = frames["M5"].iloc[-1]
    return MarketBundle(
        frames=frames,
        symbol="XAU/USD-DEMO",
        last_price=float(latest["close"]),
        data_time=pd.to_datetime(latest["time"], utc=True).isoformat(),
        source="DEMO",
        notes=["Synthetic demo data is for interface testing only."],
    )


def tradingview_widget(symbol: str, timeframe: str) -> None:
    params = {
        "symbol": symbol,
        "interval": TV_INTERVALS[timeframe],
        "hidesidetoolbar": "1",
        "symboledit": "0",
        "saveimage": "1",
        "theme": "dark",
        "style": "1",
        "timezone": "Etc/UTC",
        "withdateranges": "1",
        "locale": "en",
        "studies": "MACD@tv-basicstudies,RSI@tv-basicstudies,Volume@tv-basicstudies",
    }
    src = "https://s.tradingview.com/widgetembed/?" + urlencode(params)
    st.iframe(src, width="stretch", height=610)


# Optional private access PIN. Leave APP_PIN blank to disable.
app_pin = os.getenv("APP_PIN", "").strip()
if app_pin:
    if not st.session_state.get("mobile_unlocked"):
        st.markdown('<div class="mobile-header"><div class="mobile-brand"><div class="mobile-logo">Au</div><div><div class="mobile-title">AURUMEDGE MOBILE</div><div class="mobile-sub">Private gold intelligence terminal</div></div></div></div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("### Unlock terminal")
            pin_value = st.text_input("Access PIN", type="password")
            if st.button("UNLOCK", use_container_width=True):
                if pin_value == app_pin:
                    st.session_state["mobile_unlocked"] = True
                    st.rerun()
                else:
                    st.error("Incorrect PIN")
        st.stop()

# Streamlit Community Cloud stores secrets in st.secrets. Mirror root values into
# environment variables so the same Settings class works locally and in the cloud.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(str(_key), str(_value))
except Exception:
    pass

settings = Settings.from_env()
feed_live = bool(settings.twelve_data_api_key)
st.markdown(
    f'<div class="mobile-header"><div class="mobile-brand"><div class="mobile-logo">Au</div><div><div class="mobile-title">AURUMEDGE MOBILE</div><div class="mobile-sub">XAU/USD decision terminal · {BUILD_VERSION}</div></div></div><div class="mobile-live">{"LIVE FEED" if feed_live else "DEMO"}</div></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    c1, c2 = st.columns([1.3, .7], vertical_alignment="bottom")
    with c1:
        chart_tf = st.selectbox("Analysis timeframe", TIMEFRAMES, index=2)
    with c2:
        if st.button("REFRESH", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

load_warning = ""
try:
    if settings.twelve_data_api_key:
        with st.spinner("Updating XAU/USD…"):
            bundle = fetch_live_bundle(settings.twelve_data_api_key, settings.market_symbol, settings.bars_per_timeframe)
    else:
        bundle = demo_bundle(settings.bars_per_timeframe)
        load_warning = "No market-data key was found. Demo candles are displayed."
except Exception as exc:
    bundle = demo_bundle(settings.bars_per_timeframe)
    load_warning = f"Live feed failed; Demo mode is active: {exc}"

if load_warning:
    st.warning(load_warning)

try:
    frames = {tf: add_indicators(df) for tf, df in bundle.frames.items()}
    indicator_snapshots = [summarize_indicators(frames[tf], tf) for tf in TIMEFRAMES]
    liquidity_snapshots = [analyze_liquidity(frames[tf], tf) for tf in ["M15", "H1", "H4", "D1"]]
    report = build_technical_report(
        symbol=bundle.symbol,
        data_time=bundle.data_time,
        price=bundle.last_price,
        indicators=indicator_snapshots,
        liquidity=liquidity_snapshots,
        data_source=bundle.source,
        digits=2,
        extra_notes=bundle.notes,
    )
except Exception as exc:
    st.error(f"Technical calculation failed: {exc}")
    st.stop()

chart_liquidity = next((item for item in liquidity_snapshots if item.timeframe == chart_tf), None)
if chart_liquidity is None:
    chart_liquidity = next((item for item in liquidity_snapshots if item.timeframe == "H1"), None)

# Technical decision is primary. AI research is manual to prevent surprise API charges.
final_state = report.market_state
final_note = "Technical engine"
ai_review: AIAnalysis | None = st.session_state.get("mobile_ai_review")
if ai_review is not None:
    if report.market_state in {"STUCK", "TRAP"}:
        final_state = report.market_state
    elif ai_review.decision in {"STUCK", "TRAP"}:
        final_state = ai_review.decision
        final_note = "Manual AI risk gate"
    elif ai_review.decision in {"BUY", "SELL"} and ai_review.decision != report.market_state:
        final_state = "TRAP"
        final_note = "Technical/news conflict"
    else:
        final_note = "Technical + manual research"

state_css = state_class(final_state)
st.markdown(
    f'''<div class="mobile-kpis">
<div class="mkpi"><span>Indicative price</span><strong>{fmt(report.last_price)}</strong><small>{report.symbol}</small></div>
<div class="mkpi"><span>Decision</span><strong class="state-{state_css}">{final_state}</strong><small>{final_note}</small></div>
<div class="mkpi"><span>Confidence</span><strong>{report.confidence}%</strong><small>Technical quality</small></div>
<div class="mkpi"><span>Buy / Sell score</span><strong>{report.buy_score} / {report.sell_score}</strong><small>{report.volatility_state.upper()} volatility</small></div>
</div>''',
    unsafe_allow_html=True,
)

st.markdown(signal_panel_html(report, final_state, final_note, chart_liquidity), unsafe_allow_html=True)

signal_tab, chart_tab, levels_tab, momentum_tab, tv_tab, more_tab = st.tabs(
    ["SIGNAL", "CHART", "LEVELS", "MOMENTUM", "TRADINGVIEW", "MORE"]
)

with signal_tab:
    st.markdown('<div class="mobile-section">Multi-timeframe alignment</div>', unsafe_allow_html=True)
    st.markdown(timeframe_cards_html(indicator_snapshots), unsafe_allow_html=True)
    st.markdown('<div class="mobile-section">Why this decision</div>', unsafe_allow_html=True)
    reasons = []
    if report.active_setup and final_state in {"BUY", "SELL"}:
        reasons = report.active_setup.rationale
    elif report.trap_reason:
        reasons = [report.trap_reason]
    else:
        reasons = ["Multi-timeframe trend, momentum, liquidity and volatility gates produced the current state."]
    for reason in reasons[:8]:
        st.markdown(f'<div class="mobile-callout">• {reason}</div>', unsafe_allow_html=True)
        st.write("")
    st.markdown("**Shareable summary**")
    active = report.active_setup
    if active and final_state in {"BUY", "SELL"}:
        summary = (
            f"XAU/USD {chart_tf} — {final_state}\n"
            f"Entry: {active.entry_low:,.2f}–{active.entry_high:,.2f}\n"
            f"SL: {active.stop_loss:,.2f}\n"
            f"TP1: {active.take_profit_1:,.2f}\n"
            f"TP2: {active.take_profit_2:,.2f}\n"
            f"TP3: {active.take_profit_3:,.2f}\n"
            f"Confidence: {report.confidence}%\n"
            f"Invalidation: {active.invalidation}"
        )
    else:
        summary = f"XAU/USD {chart_tf} — {final_state}\nPrice: {report.last_price:,.2f}\nConfidence: {report.confidence}%\nNo immediate entry."
    st.code(summary, language=None)

with chart_tab:
    st.markdown('<div class="mobile-section">Professional market map</div>', unsafe_allow_html=True)
    fig = professional_market_chart(
        frames[chart_tf], report.symbol, chart_tf, chart_liquidity,
        report.active_setup if final_state in {"BUY", "SELL"} else None,
    )
    fig.update_layout(height=610, margin={"l": 24, "r": 54, "t": 50, "b": 18})
    st.plotly_chart(
        fig,
        width="stretch",
        config={"displaylogo": False, "responsive": True, "scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )
    st.caption("Pinch/zoom and drag the chart. Rotate the iPhone to landscape for the widest chart view.")

with levels_tab:
    st.markdown('<div class="mobile-section">Nearest price levels</div>', unsafe_allow_html=True)
    st.markdown(level_ladder_html(chart_liquidity, report.last_price), unsafe_allow_html=True)
    for item in liquidity_snapshots:
        with st.expander(f"{item.timeframe} support, resistance & liquidity", expanded=item.timeframe == "H1"):
            st.metric("Support", fmt(item.nearest_support))
            st.metric("Resistance", fmt(item.nearest_resistance))
            st.metric("POC", fmt(item.point_of_control))
            st.write("**Demand zones**")
            st.dataframe(pd.DataFrame(item.support_zones), hide_index=True, width="stretch")
            st.write("**Supply zones**")
            st.dataframe(pd.DataFrame(item.resistance_zones), hide_index=True, width="stretch")
            st.caption(f"Buy-side liquidity: {item.equal_highs or 'None'}")
            st.caption(f"Sell-side liquidity: {item.equal_lows or 'None'}")

with momentum_tab:
    st.markdown('<div class="mobile-section">Directional strength</div>', unsafe_allow_html=True)
    st.plotly_chart(regime_strength_chart(indicator_snapshots), width="stretch", config={"displaylogo": False})
    st.markdown(f'<div class="mobile-section">{chart_tf} MACD</div>', unsafe_allow_html=True)
    st.plotly_chart(macd_chart(frames[chart_tf]), width="stretch", config={"displaylogo": False})
    rows = []
    for item in indicator_snapshots:
        rows.append({
            "TF": item.timeframe,
            "Trend": item.trend.upper(),
            "RSI": round(item.rsi14, 1),
            "ADX": round(item.adx14, 1),
            "ATR%": round(item.atr_pct, 2),
            "MACD H": round(item.macd_hist, 2),
            "Vol ratio": round(item.volume_ratio, 2),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with tv_tab:
    st.info("TradingView is visual reference only. The signal is calculated from the independent OHLC feed.")
    tradingview_widget(settings.tradingview_symbol, chart_tf)

with more_tab:
    st.markdown('<div class="mobile-section">Optional live AI research</div>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-callout">This is OFF by default to avoid OpenAI API charges. The technical signal works without it. Run it only when you intentionally want a current macro/news check.</div>', unsafe_allow_html=True)
    can_ai = settings.openai_api_key.startswith("sk-") and bundle.source != "DEMO"
    if st.button("RUN PAID AI RESEARCH", disabled=not can_ai, use_container_width=True):
        try:
            with st.spinner("Searching current gold news and checking macro risk…"):
                research = cached_news(settings.openai_api_key, settings.openai_model)
                ai_dict = cached_synthesis(settings.openai_api_key, settings.openai_model, report.model_dump_json(), research["text"])
                st.session_state["mobile_ai_review"] = AIAnalysis.model_validate(ai_dict)
                st.session_state["mobile_research_text"] = research["text"]
                st.success("AI research completed. Refresh once to apply the risk gate to the headline decision.")
        except Exception as exc:
            st.error(f"AI research failed: {exc}")
    if not can_ai:
        st.caption("No usable OpenAI key/credit is available. Upload the chart in ChatGPT for manual review instead.")
    if st.session_state.get("mobile_research_text"):
        with st.expander("Latest research", expanded=True):
            st.markdown(st.session_state["mobile_research_text"])
    st.divider()
    st.markdown('<div class="mobile-section">Risk guide</div>', unsafe_allow_html=True)
    equity = st.number_input("Account equity", min_value=1.0, value=10000.0, step=100.0)
    risk_pct = st.number_input("Risk per setup (%)", min_value=0.1, max_value=2.0, value=float(settings.risk_percent), step=0.1)
    st.metric("Maximum planned loss", f"{equity * risk_pct / 100:,.2f}")
    st.caption("This app does not know your broker spread, CFD contract size, tick value, leverage or account currency. Calculate broker lot size separately.")
    with st.expander("Diagnostics"):
        st.json({
            "data_source": report.data_source,
            "data_time": report.data_time,
            "symbol": report.symbol,
            "build": BUILD_VERSION,
            "openai_research_default": "manual/off",
            "broker_connected": False,
            "order_execution": False,
        })

st.markdown(f'<div class="mobile-footer">AurumEdge Mobile · {BUILD_VERSION} · Indicative analysis only · No broker connection</div>', unsafe_allow_html=True)
