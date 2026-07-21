from __future__ import annotations

import json
import math
import os
import time
from html import escape
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from gold_web_terminal.adaptive_engine import AdaptiveEngine, derive_feature_votes
from gold_web_terminal.alerts import AlertConfig, AlertState, build_alert_events, dispatch_events, record_non_alert_states, send_test_alert
from gold_web_terminal.ai_engine import research_gold_news, synthesize_ai_analysis
from gold_web_terminal.config import Settings
from gold_web_terminal.demo_data import generate_demo_bars
from gold_web_terminal.indicators import add_indicators, summarize_indicators
from gold_web_terminal.fvg_strategy import detect_four_hour_fvg_signal
from gold_web_terminal.liquidity import analyze_liquidity
from gold_web_terminal.macro_mobile_v542 import fetch_macro_confirmation, refresh_macro_confirmation
from gold_web_terminal.market_data import MarketBundle, TwelveDataClient
from gold_web_terminal.mobile_svg import mobile_macd_html, mobile_market_map_html, mobile_regime_html
from gold_web_terminal.models import (
    AIAnalysis,
    MacroAssetSnapshot,
    MacroConfirmation,
    TechnicalReport,
)
from gold_web_terminal.risk_engine import RiskInputs
from gold_web_terminal.strategy import build_technical_report

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

# Streamlit Cloud secrets are mirrored into environment variables so the same
# Settings class works on Windows, Streamlit Cloud and local Wi-Fi mode.
try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(str(_key), str(_value))
except Exception:
    pass

BUILD_VERSION = "5.7.0-mobile-avwap-volume-profile"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]
TV_INTERVALS = {"M5": "5", "M15": "15", "H1": "60", "H4": "240", "D1": "D"}

st.set_page_config(
    page_title="AurumEdge Mobile Adaptive",
    page_icon="🟡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

MOBILE_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@700;800&display=swap');
:root{--bg:#060a12;--panel:#0d1625;--panel2:#101c2f;--line:rgba(148,163,184,.16);--text:#f4f7fb;--muted:#8e9bb0;--gold:#f4c85b;--green:#22cfa0;--red:#ff5d7d;--purple:#a98bff;--amber:#ffb94a;}
html,body,[class*="css"]{font-family:'Inter',sans-serif;background:var(--bg)!important;}
.stApp{background:radial-gradient(circle at 100% -10%,rgba(244,200,91,.11),transparent 27%),linear-gradient(180deg,#050811,#08101d 54%,#050810);color:var(--text);}
header[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer,[data-testid="stDecoration"],[data-testid="stStatusWidget"],[data-testid="stSidebar"]{display:none!important;}
.block-container{max-width:780px!important;padding:calc(.45rem + env(safe-area-inset-top)) .68rem calc(6rem + env(safe-area-inset-bottom))!important;}
.mobile-header{display:flex;align-items:center;justify-content:space-between;gap:10px;padding:9px 2px 13px;border-bottom:1px solid var(--line);margin-bottom:11px;}
.mobile-brand{display:flex;align-items:center;gap:10px;min-width:0}.mobile-logo{width:42px;height:42px;border-radius:14px;display:grid;place-items:center;font-family:'Manrope';font-weight:800;color:#171106;background:linear-gradient(135deg,#ffe79e,var(--gold) 55%,#bd7f16);box-shadow:0 8px 24px rgba(244,200,91,.18)}
.mobile-title{font-family:'Manrope';font-size:.98rem;font-weight:800;letter-spacing:.02em}.mobile-sub{font-size:.65rem;color:var(--muted);margin-top:2px}.mobile-live{padding:7px 9px;border-radius:999px;border:1px solid var(--line);font-size:.61rem;font-weight:800;color:#b9c5d4;white-space:nowrap}.mobile-live:before{content:'';display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:6px;box-shadow:0 0 0 4px rgba(34,207,160,.10)}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;border-radius:16px!important;background:linear-gradient(180deg,rgba(15,24,41,.96),rgba(8,14,25,.96))!important;box-shadow:0 14px 40px rgba(0,0,0,.18)}
[data-testid="stVerticalBlockBorderWrapper"]>div{padding:11px 12px!important}
div[data-baseweb="select"]>div,[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{background:#0a1220!important;border-color:rgba(148,163,184,.22)!important;color:var(--text)!important;border-radius:12px!important;min-height:46px!important}
.stButton>button{min-height:48px;border-radius:12px;font-weight:800;border:1px solid rgba(244,200,91,.44)!important;color:#171106!important;background:linear-gradient(135deg,#ffe28a,var(--gold) 55%,#d99d24)!important;box-shadow:0 8px 25px rgba(244,200,91,.16)}
button[kind="secondary"]{color:#e5edf7!important;background:#101c2f!important;border-color:var(--line)!important;box-shadow:none!important}
.mobile-kpis,.macro-grid,.risk-grid,.brain-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.mkpi,.macro-card,.risk-card,.brain-card{padding:11px 12px;border-radius:14px;border:1px solid var(--line);background:linear-gradient(180deg,rgba(16,27,46,.97),rgba(8,15,27,.97));min-height:75px}.mkpi span,.macro-card span,.risk-card span,.brain-card span{display:block;font-size:.59rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:800}.mkpi strong,.macro-card strong,.risk-card strong,.brain-card strong{display:block;font-family:'Manrope';font-size:1.03rem;margin-top:5px;line-height:1.15}.mkpi small,.macro-card small,.risk-card small,.brain-card small{display:block;color:#718198;font-size:.59rem;margin-top:4px;line-height:1.35}.state-buy{color:var(--green)!important}.state-sell{color:var(--red)!important}.state-trap{color:var(--purple)!important}.state-stuck{color:var(--amber)!important}.tone-good{color:var(--green)!important}.tone-bad{color:var(--red)!important}.tone-warn{color:var(--amber)!important}.tone-muted{color:#8e9bb0!important}
.mobile-section{font-family:'Manrope';font-size:.86rem;font-weight:800;margin:15px 2px 8px}.mobile-note{font-size:.68rem;color:var(--muted);line-height:1.5}.mobile-callout{padding:11px 12px;border-radius:12px;border:1px solid rgba(244,200,91,.2);background:rgba(244,200,91,.06);color:#cbd5e1;font-size:.69rem;line-height:1.48;margin-bottom:7px}
.signal-shell{border:1px solid var(--line);border-radius:16px;padding:14px;background:linear-gradient(145deg,rgba(17,29,49,.98),rgba(8,14,25,.98));box-shadow:0 15px 38px rgba(0,0,0,.22);margin:10px 0 12px}.signal-overline{font-size:.59rem;color:var(--muted);font-weight:800;letter-spacing:.09em;text-transform:uppercase}.signal-title{font-family:'Manrope';font-size:1.45rem;font-weight:800;margin:5px 0}.signal-copy{font-size:.7rem;color:#aab6c8;line-height:1.45}.levels-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:11px}.level-box{border:1px solid var(--line);border-radius:11px;padding:9px;background:#091321}.level-box.wide{grid-column:1/-1}.level-box span{display:block;font-size:.56rem;color:var(--muted);text-transform:uppercase;font-weight:800;letter-spacing:.07em}.level-box strong{display:block;font-family:'Manrope';font-size:.93rem;margin-top:4px}.risk-banner{margin-top:9px;padding:10px;border-radius:11px;border:1px solid var(--line);font-size:.65rem;line-height:1.45;background:#091321}
.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto;scrollbar-width:none;border-bottom:1px solid var(--line);position:sticky;top:0;z-index:10;background:rgba(6,10,18,.95);backdrop-filter:blur(14px)}.stTabs [data-baseweb="tab"]{height:45px;min-width:max-content;padding:0 11px;color:#93a1b5;font-size:.67rem;font-weight:800}.stTabs [aria-selected="true"]{color:var(--gold)!important}.stTabs [data-baseweb="tab-highlight"]{background:var(--gold)!important}
.tf-grid{display:grid;grid-template-columns:repeat(5,minmax(92px,1fr));gap:7px;overflow-x:auto;padding-bottom:4px}.tf-card{border:1px solid var(--line);border-radius:12px;padding:9px;background:#091321;min-width:92px}.tf-card .tf-head{display:flex;justify-content:space-between;gap:4px;font-size:.59rem;font-weight:800}.tf-card .tf-score{font-family:'Manrope';font-size:1rem;font-weight:800;margin-top:6px}.tf-card .tf-sub{font-size:.55rem;color:#73849a;margin-top:3px;line-height:1.35}
.ae-ladder{border:1px solid var(--line);border-radius:13px;overflow:hidden}.ae-ladder-row{display:grid;grid-template-columns:1.35fr .8fr .52fr;gap:7px;align-items:center;padding:9px 10px;border-bottom:1px solid var(--line);background:#091321}.ae-ladder-row:last-child{border-bottom:0}.ae-ladder-name{font-size:.65rem;color:#aab6c8}.ae-ladder-price{font-family:'Manrope';font-size:.72rem;font-weight:800;text-align:right}.ae-ladder-side{font-size:.52rem;font-weight:800;text-align:right}.ae-ladder-side.res{color:var(--red)}.ae-ladder-side.sup{color:var(--green)}.ae-ladder-side.neutral{color:var(--gold)}
[data-testid="stDataFrame"]{border:1px solid var(--line);border-radius:12px;overflow:hidden}.stMetric{border:1px solid var(--line);border-radius:12px;padding:9px 10px;background:#0a1322}.stMetric label{font-size:.64rem!important}.stMetric [data-testid="stMetricValue"]{font-size:1rem!important}.svg-chart-shell{width:100%;overflow:hidden;border:1px solid var(--line);border-radius:16px;background:#07101d;box-shadow:0 12px 34px rgba(0,0,0,.22)}.mobile-market-svg,.mobile-mini-svg{display:block;width:100%;height:auto;min-height:250px}.svg-chart-error{padding:18px;border-radius:14px;background:rgba(255,93,125,.10);border:1px solid rgba(255,93,125,.25);color:#ff8da3;font-weight:700}.sync-row{display:grid;grid-template-columns:1fr 1fr;gap:8px;align-items:center}.sync-badge{min-height:42px;border:1px solid var(--line);border-radius:12px;background:#091321;display:flex;align-items:center;justify-content:center;color:#c6d1df;font-size:.66rem;font-weight:800;letter-spacing:.04em}.sync-badge strong{color:var(--gold);margin-left:5px}.mobile-footer{text-align:center;color:#65758b;font-size:.59rem;margin:18px 0 4px}
@media(max-width:430px){.block-container{padding-left:.5rem!important;padding-right:.5rem!important}.mobile-title{font-size:.9rem}.mobile-sub{display:none}.mobile-live{font-size:.56rem;padding:6px 7px}.mkpi,.macro-card,.risk-card,.brain-card{padding:10px}.stTabs [data-baseweb="tab"]{padding:0 9px}.signal-title{font-size:1.3rem}}
</style>
"""
st.markdown(MOBILE_CSS, unsafe_allow_html=True)


def fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def state_class(state: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "TRAP": "trap", "STUCK": "stuck"}.get(state, "stuck")


def direction_arrow(direction: str) -> str:
    return {"UP": "↑", "DOWN": "↓", "FLAT": "→"}.get(direction, "—")


def macro_tone(direction: str, gold_asset: bool = False) -> str:
    if direction == "UNAVAILABLE":
        return "tone-muted"
    favorable = direction == ("UP" if gold_asset else "DOWN")
    adverse = direction == ("DOWN" if gold_asset else "UP")
    return "tone-good" if favorable else "tone-bad" if adverse else "tone-warn"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_live_bundle(api_key: str, symbol: str, bars: int) -> MarketBundle:
    # One synchronization loads every supported timeframe. M15 is derived from
    # M5, while H4 and D1 are derived from H1 inside the market-data client.
    return TwelveDataClient(api_key).fetch_bundle(symbol, TIMEFRAMES, bars)


def clear_full_market_cache(reason: str) -> None:
    fetch_live_bundle.clear()
    st.session_state.pop("all_timeframe_analysis", None)
    st.session_state["market_refresh_epoch"] = time.time()
    st.session_state["last_refresh_reason"] = reason


def bundle_signature(bundle: MarketBundle) -> str:
    rows: list[tuple[str, int, str, float]] = []
    for timeframe in TIMEFRAMES:
        frame = bundle.frames[timeframe]
        last = frame.iloc[-1]
        rows.append((timeframe, len(frame), pd.to_datetime(last["time"], utc=True).isoformat(), round(float(last["close"]), 6)))
    return json.dumps([bundle.source, bundle.symbol, bundle.data_time, rows], separators=(",", ":"))


@st.fragment(run_every=1)
def auto_refresh_controller(enabled: bool, live_feed: bool, interval_seconds: int) -> None:
    if not enabled or not live_feed:
        st.markdown('<div class="sync-badge">AUTO SYNC PAUSED</div>', unsafe_allow_html=True)
        return
    now = time.time()
    started = float(st.session_state.get("market_refresh_epoch", now))
    remaining = max(0, int(math.ceil(interval_seconds - (now - started))))
    st.markdown(f'<div class="sync-badge">ALL TF <strong>{remaining}s</strong></div>', unsafe_allow_html=True)
    if remaining <= 0:
        clear_full_market_cache(f"Automatic {interval_seconds}-second all-timeframe synchronization")
        st.rerun()


@st.cache_data(ttl=1200, show_spinner=False)
def cached_news(api_key: str, model: str) -> dict[str, Any]:
    return research_gold_news(api_key, model)


@st.cache_data(ttl=1200, show_spinner=False)
def cached_synthesis(api_key: str, model: str, report_json: str, research_text: str) -> dict[str, Any]:
    report = TechnicalReport.model_validate_json(report_json)
    result = synthesize_ai_analysis(api_key, model, report, research_text, None)
    return result.model_dump(mode="json")


def unavailable_macro(note: str) -> MacroConfirmation:
    return MacroConfirmation(
        dxy=MacroAssetSnapshot(symbol="DXY", label="U.S. Dollar Index", source="UNAVAILABLE"),
        us10y=MacroAssetSnapshot(symbol="US10Y", label="U.S. 10Y yield", source="UNAVAILABLE"),
        notes=[note],
    )


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
        "studies": "MACD@tv-basicstudies,RSI@tv-basicstudies,VWAP@tv-basicstudies",
    }
    st.iframe("https://s.tradingview.com/widgetembed/?" + urlencode(params), width="stretch", height=610)


def timeframe_cards(snapshots: list[Any]) -> str:
    cards: list[str] = []
    for item in snapshots:
        score = float(item.directional_score)
        state = "BUY" if score >= 14 else "SELL" if score <= -14 else "NEUTRAL"
        tone = "state-buy" if state == "BUY" else "state-sell" if state == "SELL" else "state-stuck"
        cards.append(
            f'<div class="tf-card"><div class="tf-head"><span>{escape(item.timeframe)}</span><span class="{tone}">{state}</span></div>'
            f'<div class="tf-score {tone}">{score:+.0f}</div>'
            f'<div class="tf-sub">RSI {fmt(item.rsi14,1)} · ADX {fmt(item.adx14,1)}<br>{escape(item.trend.title())} / {escape(item.momentum.title())}</div></div>'
        )
    return '<div class="tf-grid">' + "".join(cards) + "</div>"


def macro_cards(macro: MacroConfirmation) -> str:
    dxy_change = macro.dxy.change_4h if macro.dxy.change_4h is not None else macro.dxy.change_1d
    dxy_period = "4h" if macro.dxy.change_4h is not None else "1d"
    y_change = macro.us10y.change_4h if macro.us10y.change_4h is not None else macro.us10y.change_1d
    y_period = "4h" if macro.us10y.change_4h is not None else "1d"
    gate_tone = "tone-good" if macro.gate == "CONFIRM" else "tone-bad" if macro.gate == "CONFLICT" else "tone-warn"
    coverage_tone = "tone-good" if macro.coverage_score >= 100 else "tone-warn" if macro.coverage_score >= 80 else "tone-bad"
    return f'''<div class="macro-grid">
<div class="macro-card"><span>DXY</span><strong class="{macro_tone(macro.dxy.direction)}">{fmt(macro.dxy.value,3)} {direction_arrow(macro.dxy.direction)}</strong><small>{dxy_period} {fmt(dxy_change,3)} · {escape(macro.dxy.source)}<br>{escape(macro.dxy.freshness)}</small></div>
<div class="macro-card"><span>US 10Y yield</span><strong class="{macro_tone(macro.us10y.direction)}">{fmt(macro.us10y.value,3)}% {direction_arrow(macro.us10y.direction)}</strong><small>{y_period} {fmt(y_change,3)} · {escape(macro.us10y.source)}<br>{escape(macro.us10y.freshness)}</small></div>
<div class="macro-card"><span>Gold 4H flow</span><strong class="{macro_tone(macro.gold_direction, True)}">{fmt(macro.gold_change_4h,2)} {direction_arrow(macro.gold_direction)}</strong><small>Calculated from the live H1 candle series</small></div>
<div class="macro-card"><span>Decision gate</span><strong class="{gate_tone}">{escape(macro.gate)}</strong><small>{escape(macro.alignment)} · bias {escape(macro.macro_bias.replace('_',' '))}</small></div>
<div class="macro-card"><span>Data coverage</span><strong class="{coverage_tone}">{macro.coverage_score}%</strong><small>{escape(macro.data_status)} · DXY + yield + gold flow</small></div>
<div class="macro-card"><span>Macro score</span><strong>{macro.confirmation_score}/100</strong><small>Direction and agreement quality</small></div>
</div>'''


def signal_card(report: TechnicalReport, final_state: str, final_note: str) -> str:
    css_state = state_class(final_state)
    setup = report.active_setup if final_state in {"BUY", "SELL"} else None
    if final_state == "TRAP":
        title = "LIQUIDITY / MACRO TRAP"
        copy = report.trap_reason or "Technical and macro evidence conflict. Do not force an entry."
        plan = '<div class="mobile-callout">Wait for a clean reclaim, rejection, or complete macro confirmation before entering.</div>'
    elif final_state == "STUCK":
        title = "MARKET STUCK"
        copy = report.trap_reason or "Directional evidence is incomplete or the market is compressed."
        plan = '<div class="mobile-callout">No immediate trade. Refresh after a confirmed range break or after macro coverage improves.</div>'
    elif setup is None:
        title = f"{final_state} BIAS"
        copy = "Direction is visible, but no valid executable setup was produced."
        plan = '<div class="mobile-callout">No active setup.</div>'
    else:
        risk = setup.risk_plan
        risk_tone = "tone-good" if risk and risk.status == "OK" else "tone-warn" if risk and risk.status == "REDUCE_LOT" else "tone-bad"
        risk_text = "Risk plan unavailable"
        if risk:
            risk_text = (
                f"{risk.status}: requested {risk.requested_lot:.2f} lot risks about ${risk.estimated_loss_requested_lot:,.2f}; "
                f"recommended lot {risk.recommended_lot:.2f}; budget ${risk.risk_budget:,.2f}."
            )
        title = f"ENTER {final_state}" if setup.status == "ENTER" else f"{final_state} BIAS · RISK BLOCK"
        copy = f"{report.regime.replace('_',' ').title()} · {escape(final_note)} · valid until {escape(setup.valid_until)}"
        plan = f'''<div class="levels-grid">
<div class="level-box wide"><span>Entry zone</span><strong>{fmt(setup.entry_low)} – {fmt(setup.entry_high)}</strong></div>
<div class="level-box"><span>Stop loss</span><strong class="state-sell">{fmt(setup.stop_loss)}</strong></div>
<div class="level-box"><span>TP1 · {setup.risk_reward_1:.2f}R</span><strong class="state-buy">{fmt(setup.take_profit_1)}</strong></div>
<div class="level-box"><span>TP2 · {setup.risk_reward_2:.2f}R</span><strong class="state-buy">{fmt(setup.take_profit_2)}</strong></div>
<div class="level-box"><span>TP3 · {setup.risk_reward_3:.2f}R</span><strong class="state-buy">{fmt(setup.take_profit_3)}</strong></div>
</div><div class="risk-banner {risk_tone}">{escape(risk_text)}<br>{escape(setup.invalidation)}</div>'''
    return f'''<div class="signal-shell"><div class="signal-overline">Current execution state</div><div class="signal-title state-{css_state}">{escape(title)}</div><div class="signal-copy">{escape(copy)}</div>{plan}</div>'''


def special_strategy_card(signal) -> str:
    state = getattr(signal, "state", "NONE")
    side = getattr(signal, "side", "NONE")
    state_tone = "state-buy" if state == "TRIGGERED" and side == "BUY" else "state-sell" if state == "TRIGGERED" and side == "SELL" else "state-stuck"
    side_tone = "state-buy" if side == "BUY" else "state-sell" if side == "SELL" else "state-stuck"

    def fv(value):
        return "—" if value is None else f"{float(value):,.2f}"

    levels = ""
    if getattr(signal, "entry_low", None) is not None:
        levels = (
            '<div class="levels-grid">'
            f'<div class="level-box wide"><span>FVG / entry</span><strong>{fv(signal.entry_low)} – {fv(signal.entry_high)}</strong></div>'
            f'<div class="level-box"><span>Stop loss</span><strong class="state-sell">{fv(signal.stop_loss)}</strong></div>'
            f'<div class="level-box"><span>TP1</span><strong class="state-buy">{fv(signal.take_profit_1)}</strong></div>'
            f'<div class="level-box"><span>TP2</span><strong class="state-buy">{fv(signal.take_profit_2)}</strong></div>'
            f'<div class="level-box"><span>TP3</span><strong class="state-buy">{fv(signal.take_profit_3)}</strong></div>'
            '</div>'
        )
    reasons = "".join(
        f'<div class="mobile-callout">• {escape(str(item))}</div>'
        for item in getattr(signal, "rationale", [])[:5]
    )
    agreement = "YES" if getattr(signal, "aligns_with_primary", False) else "NO"
    return (
        '<div class="signal-shell">'
        '<div class="signal-overline">Specialist strategy · 4H candle + M15 FVG</div>'
        f'<div class="signal-title {side_tone}">{escape(side)} · <span class="{state_tone}">{escape(state)}</span></div>'
        f'<div class="signal-copy">Confidence {int(getattr(signal, "confidence", 0))}% · Regular engine agreement {agreement} · Macro {escape(str(getattr(signal, "macro_gate", "UNAVAILABLE")))}</div>'
        f'{levels}{reasons}</div>'
    )


# Optional private access PIN.
app_pin = os.getenv("APP_PIN", "").strip()
if app_pin and not st.session_state.get("mobile_unlocked"):
    st.markdown('<div class="mobile-header"><div class="mobile-brand"><div class="mobile-logo">Au</div><div><div class="mobile-title">AURUMEDGE MOBILE</div><div class="mobile-sub">Private adaptive gold terminal</div></div></div></div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown("### Unlock terminal")
        pin_value = st.text_input("Access PIN", type="password")
        if st.button("UNLOCK", use_container_width=True):
            if pin_value == app_pin:
                st.session_state["mobile_unlocked"] = True
                st.rerun()
            st.error("Incorrect PIN")
    st.stop()

settings = Settings.from_env()
# Backward-compatible defaults: older deployed config.py versions do not yet
# expose the v5.4 automatic-refresh fields. The entrypoint must remain usable
# while GitHub/Streamlit finishes replacing package files.
auto_refresh_default = bool(getattr(settings, "auto_refresh_enabled", True))
try:
    auto_refresh_seconds = max(60, int(getattr(settings, "auto_refresh_seconds", 1200)))
except (TypeError, ValueError):
    auto_refresh_seconds = 1200
feed_live = bool(settings.twelve_data_api_key)
st.markdown(
    f'<div class="mobile-header"><div class="mobile-brand"><div class="mobile-logo">Au</div><div><div class="mobile-title">AURUMEDGE ADAPTIVE MOBILE</div><div class="mobile-sub">XAU/USD · macro gate · adaptive brain · {BUILD_VERSION}</div></div></div><div class="mobile-live">{"LIVE FEED" if feed_live else "DEMO"}</div></div>',
    unsafe_allow_html=True,
)

with st.container(border=True):
    chart_tf = st.selectbox(
        "Chart timeframe",
        TIMEFRAMES,
        index=2,
        help="This changes only the chart. The decision always compares M5, M15, H1, H4 and D1 together.",
    )
    c1, c2 = st.columns([1, 1], vertical_alignment="center")
    with c1:
        auto_enabled = st.toggle(
            f"{max(1, auto_refresh_seconds // 60)}-minute auto sync",
            value=auto_refresh_default,
            key="mobile_auto_refresh_enabled",
            help="Refreshes all timeframes together. Switching charts uses the stored synchronized snapshot.",
        )
    with c2:
        previous_auto = st.session_state.get("_mobile_previous_auto")
        if previous_auto is None or previous_auto != auto_enabled:
            st.session_state["market_refresh_epoch"] = time.time()
        st.session_state["_mobile_previous_auto"] = auto_enabled
        auto_refresh_controller(auto_enabled, feed_live, auto_refresh_seconds)
    if st.button("↻ SYNC ALL TIMEFRAMES", use_container_width=True):
        clear_full_market_cache("Manual all-timeframe synchronization")
        st.rerun()
    if bool(getattr(settings, "free_plan_mode", True)):
        watcher_minutes = max(5, int(getattr(settings, "cloud_watcher_minutes", 5)))
        daily_limit = int(getattr(settings, "provider_daily_limit", 800))
        ui_runs = max(1, 1440 // max(1, auto_refresh_seconds // 60))
        watcher_runs = 1440 // watcher_minutes
        projected_credits = (ui_runs + watcher_runs) * 2
        st.caption(
            f"Free-plan profile: cloud watcher every {watcher_minutes} min + open-app sync every "
            f"{max(1, auto_refresh_seconds // 60)} min ≈ {projected_credits}/{daily_limit} candle credits per day. "
            "Timeframe switching uses the stored all-timeframe snapshot and consumes no extra request."
        )

with st.expander("CFD risk profile", expanded=False):
    rc1, rc2 = st.columns(2)
    account_balance = rc1.number_input("Account balance", min_value=100.0, value=float(settings.account_balance), step=500.0)
    risk_percent = rc2.number_input("Risk per trade %", min_value=0.05, max_value=5.0, value=float(settings.risk_percent), step=0.05)
    requested_lot = rc1.number_input("Intended lot", min_value=0.01, value=float(settings.requested_lot), step=float(settings.lot_step), format="%.2f")
    contract_size = rc2.number_input("Contract size oz/lot", min_value=1.0, value=float(settings.contract_size), step=1.0)
    spread_price = rc1.number_input("Spread allowance $", min_value=0.0, value=float(settings.spread_price), step=0.05)
    max_risk_dollars = rc2.number_input("Hard dollar cap (0=off)", min_value=0.0, value=float(settings.maximum_risk_dollars), step=25.0)
    st.caption("The app keeps the structural stop and reduces lot size when the requested position would risk too much.")

risk_inputs = RiskInputs(
    account_balance=account_balance,
    risk_percent=risk_percent,
    requested_lot=requested_lot,
    contract_size=contract_size,
    lot_step=settings.lot_step,
    min_lot=settings.min_lot,
    maximum_risk_dollars=max_risk_dollars,
    spread_price=spread_price,
    slippage_price=settings.slippage_price,
    minimum_stop_atr=settings.minimum_stop_atr,
    maximum_stop_atr=settings.maximum_stop_atr,
)

if not settings.twelve_data_api_key:
    bundle = demo_bundle(settings.bars_per_timeframe)
    load_warning = "No market-data key was found. Demo mode is active; do not trade synthetic prices."
else:
    try:
        with st.spinner("Updating quota-safe XAU/USD candles…"):
            bundle = fetch_live_bundle(settings.twelve_data_api_key, settings.market_symbol, settings.bars_per_timeframe)
            st.session_state["last_good_live_bundle"] = bundle
        load_warning = ""
    except Exception as exc:
        cached_bundle = st.session_state.get("last_good_live_bundle")
        if isinstance(cached_bundle, MarketBundle):
            bundle = cached_bundle
            load_warning = f"Live refresh unavailable. Showing the last successful live candles as STALE: {exc}"
        else:
            st.error(f"LIVE DATA UNAVAILABLE — NO TRADE SIGNAL: {exc}")
            st.caption("A failed live feed is never replaced by a synthetic BUY or SELL signal.")
            st.stop()

if load_warning:
    st.warning(load_warning)

try:
    signature = bundle_signature(bundle)
    analysis_cache = st.session_state.get("all_timeframe_analysis")
    if isinstance(analysis_cache, dict) and analysis_cache.get("signature") == signature:
        frames = analysis_cache["frames"]
        indicator_snapshots = analysis_cache["indicator_snapshots"]
        liquidity_snapshots = analysis_cache["liquidity_snapshots"]
    else:
        with st.spinner("Calculating M5, M15, H1, H4 and D1 together…"):
            frames = {tf: add_indicators(bundle.frames[tf]) for tf in TIMEFRAMES}
            indicator_snapshots = [summarize_indicators(frames[tf], tf) for tf in TIMEFRAMES]
            liquidity_snapshots = [analyze_liquidity(frames[tf], tf) for tf in ["M15", "H1", "H4", "D1"]]
        st.session_state["all_timeframe_analysis"] = {
            "signature": signature,
            "frames": frames,
            "indicator_snapshots": indicator_snapshots,
            "liquidity_snapshots": liquidity_snapshots,
        }
except Exception as exc:
    st.error(f"Technical calculation failed: {exc}")
    st.stop()

adaptive_path = Path(settings.adaptive_state_path)
if not adaptive_path.is_absolute():
    adaptive_path = APP_DIR / adaptive_path
adaptive = AdaptiveEngine(
    adaptive_path,
    enabled=settings.adaptive_learning,
    minimum_samples=settings.adaptive_min_samples,
    horizon_bars=settings.adaptive_horizon_bars,
    max_weight_change=settings.adaptive_max_weight_change,
)
completed_reviews = adaptive.review_pending(frames)
adaptive_summary = adaptive.summary()

decision_memory = st.session_state.setdefault(
    "decision_memory",
    {"state": None, "trap_anchor_price": None, "trap_age": 0},
)

preliminary_report = build_technical_report(
    symbol=bundle.symbol,
    data_time=bundle.data_time,
    price=bundle.last_price,
    indicators=indicator_snapshots,
    liquidity=liquidity_snapshots,
    data_source=bundle.source,
    digits=2,
    extra_notes=bundle.notes,
    adaptive_weights=adaptive.weights(),
    target_multipliers=adaptive.target_multipliers(),
    adaptive_summary=adaptive_summary,
    risk_inputs=risk_inputs,
)

if settings.macro_enabled and bundle.source != "DEMO":
    gold_h1 = frames["H1"][["time", "close"]].copy()
    now_epoch = time.time()
    macro_cache = st.session_state.get("macro_asset_cache")
    force_macro = bool(st.session_state.pop("force_macro_refresh", False))
    macro_age = (now_epoch - float(macro_cache.get("epoch", 0))) if isinstance(macro_cache, dict) else float("inf")
    macro_due = force_macro or not isinstance(macro_cache, dict) or macro_age >= settings.macro_cache_minutes * 60
    try:
        if macro_due:
            with st.spinner("Refreshing DXY and U.S. 10-year yield context…"):
                macro = fetch_macro_confirmation(
                    settings.twelve_data_api_key,
                    settings.dxy_symbol,
                    settings.us10y_symbol,
                    gold_h1,
                    preliminary_report.market_state,
                )
            if macro.coverage_score >= 60:
                st.session_state["macro_asset_cache"] = {"epoch": now_epoch, "macro": macro}
                st.session_state["last_good_macro"] = macro
        else:
            macro = refresh_macro_confirmation(macro_cache["macro"], gold_h1, preliminary_report.market_state)
        if macro.coverage_score < 60 and isinstance(st.session_state.get("last_good_macro"), MacroConfirmation):
            macro = refresh_macro_confirmation(st.session_state["last_good_macro"], gold_h1, preliminary_report.market_state)
            macro.notes.append("Current external macro refresh was incomplete; cached DXY/US10Y are STALE while Gold 4H flow is current.")
            macro.dxy.freshness = (macro.dxy.freshness + " · STALE").strip(" ·")
            macro.us10y.freshness = (macro.us10y.freshness + " · STALE").strip(" ·")
    except Exception as exc:
        previous = st.session_state.get("last_good_macro")
        if isinstance(previous, MacroConfirmation):
            macro = refresh_macro_confirmation(previous, gold_h1, preliminary_report.market_state)
            macro.notes.append(f"External macro refresh failed; cached DXY/US10Y remain STALE while Gold 4H flow was recalculated ({exc.__class__.__name__}).")
            macro.dxy.freshness = (macro.dxy.freshness + " · STALE").strip(" ·")
            macro.us10y.freshness = (macro.us10y.freshness + " · STALE").strip(" ·")
        else:
            macro = unavailable_macro(f"Macro confirmation failed: {exc}")
else:
    macro = unavailable_macro("Macro confirmation is disabled or the app is using Demo data.")

report = build_technical_report(
    symbol=bundle.symbol,
    data_time=bundle.data_time,
    price=bundle.last_price,
    indicators=indicator_snapshots,
    liquidity=liquidity_snapshots,
    data_source=bundle.source,
    digits=2,
    extra_notes=bundle.notes,
    adaptive_weights=adaptive.weights(),
    target_multipliers=adaptive.target_multipliers(),
    adaptive_summary=adaptive_summary,
    risk_inputs=risk_inputs,
    macro=macro,
    macro_required_for_entry=settings.macro_required_for_entry,
    previous_state=decision_memory.get("state"),
    trap_anchor_price=decision_memory.get("trap_anchor_price"),
    trap_age=int(decision_memory.get("trap_age", 0)),
)

signal_time = frames["M15"].iloc[-1]["time"]
feature_votes = derive_feature_votes(indicator_snapshots, liquidity_snapshots, macro, report.market_state)
report = adaptive.apply_capital_preservation(report, signal_time, feature_votes)

four_hour_fvg = detect_four_hour_fvg_signal(
    frames,
    indicators=indicator_snapshots,
    macro=macro,
    primary_state=report.market_state,
    risk_inputs=risk_inputs,
    digits=2,
) if bool(getattr(settings, "h4_fvg_strategy_enabled", True)) else None
if four_hour_fvg is not None:
    report.special_signals = [four_hour_fvg]

mobile_alert_messages: list[str] = []
if bool(getattr(settings, "local_alerts_enabled", False)):
    alert_cfg = AlertConfig.from_env()
    state_path = Path(getattr(settings, "alert_state_path", "data/alert_state.json"))
    if not state_path.is_absolute():
        state_path = APP_DIR / state_path
    alert_state = AlertState(state_path)
    sent_ids, alert_errors = dispatch_events(
        build_alert_events(report, four_hour_fvg), alert_cfg, alert_state
    )
    record_non_alert_states(report, four_hour_fvg, alert_cfg, alert_state)
    if sent_ids:
        mobile_alert_messages.append(f"Sent {len(sent_ids)} new alert(s).")
    mobile_alert_messages.extend(alert_errors)

if report.market_state == "TRAP":
    if decision_memory.get("state") != "TRAP" or decision_memory.get("trap_anchor_price") is None:
        decision_memory["trap_anchor_price"] = report.last_price
        decision_memory["trap_age"] = 1
    else:
        decision_memory["trap_age"] = int(decision_memory.get("trap_age", 0)) + 1
else:
    decision_memory["trap_anchor_price"] = None
    decision_memory["trap_age"] = 0
decision_memory["state"] = report.market_state

if report.market_state in {"BUY", "SELL"} and report.active_setup is not None:
    adaptive.register_signal(report, signal_time, feature_votes, timeframe="M15")

# Optional paid AI research remains manual. The adaptive technical/macro engine
# owns prices, entries, stops and targets.
final_state = report.market_state
final_note = "All-timeframe M5→D1 + adaptive + DXY/US10Y gate"
ai_review: AIAnalysis | None = st.session_state.get("mobile_ai_review")
if ai_review is not None:
    if report.market_state in {"STUCK", "TRAP"}:
        final_state = report.market_state
    elif ai_review.decision in {"STUCK", "TRAP"}:
        final_state = ai_review.decision
        final_note = "Manual news-risk gate"
    elif ai_review.decision in {"BUY", "SELL"} and ai_review.decision != report.market_state:
        final_state = "TRAP"
        final_note = "Technical/macro/news conflict"
    else:
        final_note = "Adaptive technical + macro + manual news"

chart_liquidity = next((item for item in liquidity_snapshots if item.timeframe == chart_tf), None)
if chart_liquidity is None:
    chart_liquidity = next((item for item in liquidity_snapshots if item.timeframe == "H1"), None)

if completed_reviews:
    st.success(f"Adaptive brain reviewed {len(completed_reviews)} completed signal(s).")

state_css = state_class(final_state)
st.markdown(
    f'''<div class="mobile-kpis">
<div class="mkpi"><span>Indicative price</span><strong>{fmt(report.last_price)}</strong><small>{escape(report.symbol)}</small></div>
<div class="mkpi"><span>Decision</span><strong class="state-{state_css}">{escape(final_state)}</strong><small>{escape(final_note)}</small></div>
<div class="mkpi"><span>Confidence</span><strong>{report.confidence}%</strong><small>Adaptive + macro quality</small></div>
<div class="mkpi"><span>Buy / Sell score</span><strong>{report.buy_score} / {report.sell_score}</strong><small>{report.volatility_state.upper()} volatility</small></div>
</div>''',
    unsafe_allow_html=True,
)

st.markdown('<div class="mobile-section">Macro confirmation</div>', unsafe_allow_html=True)
st.markdown(macro_cards(macro), unsafe_allow_html=True)
st.markdown(signal_card(report, final_state, final_note), unsafe_allow_html=True)
st.markdown('<div class="mobile-section">4H candle + M15 FVG strategy</div>', unsafe_allow_html=True)
if four_hour_fvg is not None:
    st.markdown(special_strategy_card(four_hour_fvg), unsafe_allow_html=True)
for _msg in mobile_alert_messages:
    st.caption(_msg)

tabs = st.tabs(["SIGNAL", "MACRO", "4H FVG", "ALERTS", "CHART", "LEVELS", "MOMENTUM", "BRAIN", "MORE"])
signal_tab, macro_tab, fvg_tab, alerts_tab, chart_tab, levels_tab, momentum_tab, brain_tab, more_tab = tabs

with signal_tab:
    st.markdown('<div class="mobile-section">Multi-timeframe decision stack</div>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-callout">The final decision compares D1 regime, H4 trend, H1 structure, M15 confirmation and M5 timing. The selected chart does not change the decision.</div>', unsafe_allow_html=True)
    st.markdown(timeframe_cards(indicator_snapshots), unsafe_allow_html=True)
    st.markdown('<div class="mobile-section">Why this decision</div>', unsafe_allow_html=True)
    reasons: list[str] = []
    if report.active_setup and final_state in {"BUY", "SELL"}:
        reasons.extend(report.active_setup.rationale)
        if report.macro:
            reasons.extend(report.macro.reasons[:3])
    elif report.trap_reason:
        reasons.append(report.trap_reason)
        reasons.extend(report.macro.conflicts[:3] if report.macro else [])
    else:
        reasons.append("Trend, momentum, liquidity, macro coverage and risk gates produced the current state.")
    for reason in reasons[:9]:
        st.markdown(f'<div class="mobile-callout">• {escape(str(reason))}</div>', unsafe_allow_html=True)

    setup = report.active_setup
    if setup and final_state in {"BUY", "SELL"}:
        rp = setup.risk_plan
        risk_line = (
            f"Risk: {rp.status}; recommended lot {rp.recommended_lot:.2f}"
            if rp else "Risk: UNAVAILABLE"
        )
        summary = (
            f"XAU/USD {chart_tf} — {final_state}\n"
            f"Entry: {setup.entry_low:,.2f}–{setup.entry_high:,.2f}\n"
            f"SL: {setup.stop_loss:,.2f}\n"
            f"TP1: {setup.take_profit_1:,.2f}\n"
            f"TP2: {setup.take_profit_2:,.2f}\n"
            f"TP3: {setup.take_profit_3:,.2f}\n"
            f"Confidence: {report.confidence}%\n"
            f"Macro gate: {macro.gate} ({macro.coverage_score}% coverage)\n"
            f"{risk_line}\n"
            f"Invalidation: {setup.invalidation}"
        )
    else:
        summary = (
            f"XAU/USD {chart_tf} — {final_state}\nPrice: {report.last_price:,.2f}\n"
            f"Confidence: {report.confidence}%\nMacro gate: {macro.gate} ({macro.coverage_score}% coverage)\nNo immediate entry."
        )
    st.markdown("**Shareable summary**")
    st.code(summary, language=None)

with macro_tab:
    if st.button("REFRESH MACRO", use_container_width=True, help="Forces a fresh DXY and yield lookup. Gold 4H is refreshed on every all-timeframe sync."):
        st.session_state["force_macro_refresh"] = True
        st.rerun()
    st.markdown(macro_cards(macro), unsafe_allow_html=True)
    st.markdown('<div class="mobile-section">Confirmations</div>', unsafe_allow_html=True)
    for item in macro.reasons or ["No confirming macro factor is currently available."]:
        st.markdown(f'<div class="mobile-callout">• {escape(item)}</div>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-section">Conflicts and diagnostics</div>', unsafe_allow_html=True)
    for item in (macro.conflicts + macro.notes) or ["No macro conflict or source warning is currently reported."]:
        st.markdown(f'<div class="mobile-callout">• {escape(item)}</div>', unsafe_allow_html=True)
    with st.expander("Macro source details"):
        st.json({
            "coverage_score": macro.coverage_score,
            "data_status": macro.data_status,
            "alignment": macro.alignment,
            "gate": macro.gate,
            "dxy_source": macro.dxy.source,
            "dxy_time": macro.dxy.data_time,
            "dxy_freshness": macro.dxy.freshness,
            "us10y_source": macro.us10y.source,
            "us10y_time": macro.us10y.data_time,
            "us10y_freshness": macro.us10y.freshness,
            "gold_change_4h": macro.gold_change_4h,
        })

with fvg_tab:
    st.markdown('<div class="mobile-section">4H displacement and fair-value-gap continuation</div>', unsafe_allow_html=True)
    if four_hour_fvg is not None:
        st.markdown(special_strategy_card(four_hour_fvg), unsafe_allow_html=True)
        for item in four_hour_fvg.warnings:
            st.markdown(f'<div class="mobile-callout">• {escape(str(item))}</div>', unsafe_allow_html=True)
    st.caption("WATCH = parent candle qualifies. ARMED = price is inside the FVG. TRIGGERED = an M15 confirmation candle has closed in the parent direction.")

with alerts_tab:
    st.markdown('<div class="mobile-section">Signal notifications</div>', unsafe_allow_html=True)
    cfg = AlertConfig.from_env()
    telegram_status = "READY" if cfg.telegram_enabled else "NOT SET"
    email_status = "READY" if cfg.email_enabled else "NOT SET"
    forming_status = "ON" if cfg.forming_alerts else "OFF"
    st.markdown(
        '<div class="brain-grid">'
        f'<div class="brain-card"><span>Telegram push</span><strong>{telegram_status}</strong><small>Best option for instant iPhone notification</small></div>'
        f'<div class="brain-card"><span>Email alerts</span><strong>{email_status}</strong><small>SMTP delivery</small></div>'
        f'<div class="brain-card"><span>Minimum confidence</span><strong>{cfg.minimum_confidence}%</strong><small>Lower-quality signals are not sent</small></div>'
        f'<div class="brain-card"><span>Forming alerts</span><strong>{forming_status}</strong><small>ARMED FVG alerts; TRIGGERED is always eligible</small></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    if st.button("SEND TEST NOTIFICATION", key="mobile_test_notification", use_container_width=True):
        delivered, errors = send_test_alert(cfg)
        if delivered:
            st.success("Test notification sent.")
        else:
            st.error("No channel configured or delivery failed: " + "; ".join(errors))
    st.markdown('<div class="mobile-callout">For alerts while this app is closed, install the included GitHub Actions workflow. Streamlit auto-refresh only runs while a session is open.</div>', unsafe_allow_html=True)

with chart_tab:
    st.markdown('<div class="mobile-section">Professional market map</div>', unsafe_allow_html=True)
    st.markdown(
        mobile_market_map_html(
            frames[chart_tf], report.symbol, chart_tf, chart_liquidity,
            report.active_setup if final_state in {"BUY", "SELL"} else None,
        ),
        unsafe_allow_html=True,
    )
    st.caption("Safari-safe chart with market structure, anchored VWAP, volume profile, secondary EMA context, support/resistance, liquidity and active setup levels.")

with levels_tab:
    st.markdown('<div class="mobile-section">Nearest price levels</div>', unsafe_allow_html=True)
    if chart_liquidity is None:
        st.info("Liquidity map unavailable for this timeframe.")
    else:
        rows = [
            ("Nearest resistance", chart_liquidity.nearest_resistance, "res"),
            ("Value area high", chart_liquidity.value_area_high, "res"),
            ("Previous-day high", chart_liquidity.previous_day_high, "res"),
            ("Current market", report.last_price, "neutral"),
            ("Volume POC", chart_liquidity.point_of_control, "neutral"),
            ("Previous-day low", chart_liquidity.previous_day_low, "sup"),
            ("Value area low", chart_liquidity.value_area_low, "sup"),
            ("Nearest support", chart_liquidity.nearest_support, "sup"),
        ]
        valid = sorted([(n, v, s) for n, v, s in rows if v is not None], key=lambda x: float(x[1]), reverse=True)
        body = "".join(f'<div class="ae-ladder-row"><div class="ae-ladder-name">{escape(n)}</div><div class="ae-ladder-price">{fmt(float(v))}</div><div class="ae-ladder-side {s}">{"MARKET" if n=="Current market" else s.upper()}</div></div>' for n,v,s in valid)
        st.markdown(f'<div class="ae-ladder">{body}</div>', unsafe_allow_html=True)
    for item in liquidity_snapshots:
        with st.expander(f"{item.timeframe} support, resistance & liquidity", expanded=item.timeframe == "H1"):
            c1, c2 = st.columns(2)
            c1.metric("Support", fmt(item.nearest_support))
            c2.metric("Resistance", fmt(item.nearest_resistance))
            st.metric("POC", fmt(item.point_of_control))
            st.write("**Demand zones**")
            st.dataframe(pd.DataFrame(item.support_zones), hide_index=True, width="stretch")
            st.write("**Supply zones**")
            st.dataframe(pd.DataFrame(item.resistance_zones), hide_index=True, width="stretch")

with momentum_tab:
    st.markdown('<div class="mobile-section">Directional strength</div>', unsafe_allow_html=True)
    st.markdown(mobile_regime_html(indicator_snapshots), unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-section">{chart_tf} MACD</div>', unsafe_allow_html=True)
    st.markdown(mobile_macd_html(frames[chart_tf], chart_tf), unsafe_allow_html=True)
    rows = []
    for item in indicator_snapshots:
        rows.append({
            "TF": item.timeframe,
            "Trend": item.trend.upper(),
            "RSI": round(float(item.rsi14 or 0), 1),
            "ADX": round(float(item.adx14 or 0), 1),
            "ATR%": round(float(item.atr_pct or 0), 2),
            "MACD H": round(float(item.macd_hist or 0), 2),
            "Structure": item.market_structure,
            "AVWAP": round(float(item.avwap_active or 0), 2),
            "Profile": item.profile_state,
            "POC": round(float(item.profile_poc or 0), 2),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

with brain_tab:
    summary = adaptive.summary()
    st.markdown(f'''<div class="brain-grid">
<div class="brain-card"><span>Reviewed signals</span><strong>{summary.reviewed_signals}</strong><small>Completed evidence only</small></div>
<div class="brain-card"><span>Win rate</span><strong>{summary.win_rate:.1f}%</strong><small>{summary.wins} wins · {summary.losses} losses · {summary.timeouts} timeouts</small></div>
<div class="brain-card"><span>Adaptive targets</span><strong>{summary.target_r_multipliers['tp1']:.2f}R / {summary.target_r_multipliers['tp2']:.2f}R / {summary.target_r_multipliers['tp3']:.2f}R</strong><small>Capital-preservation defaults, then clean pre-exit movement</small></div>
<div class="brain-card"><span>Learning mode</span><strong>{'ACTIVE' if summary.enabled else 'OFF'}</strong><small>Confidence cap {adaptive.confidence_cap()}% · full learning near {settings.adaptive_min_samples} samples</small></div>
</div>''', unsafe_allow_html=True)
    st.markdown(f'<div class="mobile-callout"><strong>Latest review</strong><br>{escape(summary.last_review)}</div>', unsafe_allow_html=True)
    weight_rows = [{"Feature": k.replace("_", " ").title(), "Weight": round(v, 3), "Samples": summary.indicator_samples.get(k, 0)} for k, v in summary.indicator_weights.items()]
    st.dataframe(pd.DataFrame(weight_rows), hide_index=True, width="stretch")
    st.caption("Streamlit Cloud storage can reset after redeployment. Download an adaptive-state backup periodically.")
    try:
        state_bytes = adaptive_path.read_bytes() if adaptive_path.exists() else json.dumps(adaptive.state, indent=2).encode("utf-8")
        st.download_button("DOWNLOAD BRAIN BACKUP", data=state_bytes, file_name="adaptive_state.json", mime="application/json", use_container_width=True)
    except Exception as exc:
        st.warning(f"Could not prepare adaptive backup: {exc}")
    restore = st.file_uploader("Restore adaptive-state JSON", type=["json"])
    if restore is not None and st.button("RESTORE BRAIN STATE", use_container_width=True):
        try:
            parsed = json.loads(restore.getvalue().decode("utf-8"))
            if not isinstance(parsed, dict) or "features" not in parsed:
                raise ValueError("Not a valid AurumEdge adaptive state")
            adaptive_path.parent.mkdir(parents=True, exist_ok=True)
            adaptive_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            st.success("Adaptive state restored. Refreshing…")
            st.rerun()
        except Exception as exc:
            st.error(f"Restore failed: {exc}")

with more_tab:
    st.markdown('<div class="mobile-section">Optional paid AI research</div>', unsafe_allow_html=True)
    st.markdown('<div class="mobile-callout">OpenAI web research is manual and off by default. DXY, US10Y, Gold 4H, adaptive learning and technical analysis work without it.</div>', unsafe_allow_html=True)
    can_ai = settings.openai_api_key.startswith("sk-") and bundle.source != "DEMO"
    if st.button("RUN PAID AI RESEARCH", disabled=not can_ai, use_container_width=True):
        try:
            with st.spinner("Searching current gold news and checking event risk…"):
                research = cached_news(settings.openai_api_key, settings.openai_model)
                ai_dict = cached_synthesis(settings.openai_api_key, settings.openai_model, report.model_dump_json(), research["text"])
                st.session_state["mobile_ai_review"] = AIAnalysis.model_validate(ai_dict)
                st.session_state["mobile_research_text"] = research["text"]
                st.success("AI research completed. Refresh once to apply the manual news gate.")
        except Exception as exc:
            st.error(f"AI research failed: {exc}")
    if st.session_state.get("mobile_research_text"):
        with st.expander("Latest research", expanded=True):
            st.markdown(st.session_state["mobile_research_text"])
    st.divider()
    st.markdown('<div class="mobile-section">TradingView reference</div>', unsafe_allow_html=True)
    with st.expander("Open TradingView chart"):
        tradingview_widget(settings.tradingview_symbol, chart_tf)
    with st.expander("Diagnostics"):
        setup = report.active_setup
        rp = setup.risk_plan if setup else None
        st.json({
            "data_source": report.data_source,
            "data_time": report.data_time,
            "symbol": report.symbol,
            "build": BUILD_VERSION,
            "auto_refresh_enabled": auto_enabled,
            "auto_refresh_seconds": auto_refresh_seconds,
            "all_timeframes_compared": TIMEFRAMES,
            "last_refresh_reason": st.session_state.get("last_refresh_reason", "Initial synchronization"),
            "macro_gate": macro.gate,
            "macro_coverage": macro.coverage_score,
            "dxy_source": macro.dxy.source,
            "us10y_source": macro.us10y.source,
            "gold_4h_move": macro.gold_change_4h,
            "adaptive_reviewed_signals": adaptive.summary().reviewed_signals,
            "h4_fvg_state": four_hour_fvg.state if four_hour_fvg else "DISABLED",
            "h4_fvg_side": four_hour_fvg.side if four_hour_fvg else "NONE",
            "risk_status": rp.status if rp else "NO_SETUP",
            "recommended_lot": rp.recommended_lot if rp else None,
            "broker_connected": False,
            "order_execution": False,
        })

st.markdown(f'<div class="mobile-footer">AurumEdge Adaptive Mobile · {BUILD_VERSION} · All-timeframe sync · No broker connection · No automatic execution</div>', unsafe_allow_html=True)
