from __future__ import annotations

"""AurumEdge Mobile v5.8.1 — clean three-pillar terminal."""

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv

from gold_web_terminal.adaptive_engine import AdaptiveEngine, derive_feature_votes
from gold_web_terminal.config import Settings
from gold_web_terminal.indicators import add_indicators, summarize_indicators
from gold_web_terminal.liquidity import analyze_liquidity
from gold_web_terminal.market_data import TwelveDataClient
from gold_web_terminal.risk_engine import RiskInputs
from gold_web_terminal.strategy import build_technical_report

APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")

try:
    for _key, _value in st.secrets.items():
        if isinstance(_value, (str, int, float, bool)):
            os.environ.setdefault(str(_key), str(_value))
except Exception:
    pass

BUILD_VERSION = "5.8.1-mobile-entry-lifecycle"
TIMEFRAMES = ["M5", "M15", "H1", "H4", "D1"]

st.set_page_config(
    page_title="AurumEdge Mobile",
    page_icon="🟡",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items=None,
)

st.markdown(
    """
<style>
:root{--bg:#050914;--panel:#0c1626;--line:rgba(148,163,184,.18);--text:#f5f7fb;--muted:#8c9bb0;--gold:#f4c85b;--green:#21d39b;--red:#ff6480;--amber:#ffbb55}
html,body,[class*="css"]{font-family:Inter,system-ui,sans-serif;background:var(--bg)!important}
.stApp{background:radial-gradient(circle at 100% -10%,rgba(244,200,91,.12),transparent 27%),linear-gradient(180deg,#050914,#08111f 60%,#050914);color:var(--text)}
header[data-testid="stHeader"],#MainMenu,footer,[data-testid="stToolbar"],[data-testid="stDecoration"],[data-testid="stStatusWidget"],section[data-testid="stSidebar"]{display:none!important}
.block-container{max-width:760px;padding:.55rem .65rem 5rem!important}
.ae-head{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:8px 2px 13px;border-bottom:1px solid var(--line)}
.ae-brand{font-size:1rem;font-weight:900;letter-spacing:.04em}.ae-sub{font-size:.62rem;color:var(--muted);margin-top:3px}.ae-build{font-size:.58rem;border:1px solid var(--line);border-radius:999px;padding:7px 9px;color:#cad5e3;white-space:nowrap}
.ae-status{border:1px solid var(--line);border-radius:18px;padding:16px;margin:12px 0;background:linear-gradient(145deg,rgba(16,29,49,.98),rgba(7,13,24,.98));box-shadow:0 16px 36px rgba(0,0,0,.25)}
.ae-label{font-size:.58rem;letter-spacing:.1em;color:var(--muted);font-weight:800;text-transform:uppercase}.ae-title{font-size:1.45rem;font-weight:900;margin:5px 0}.buy{color:var(--green)}.sell{color:var(--red)}.wait{color:var(--amber)}
.ae-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:10px 0}.ae-box{border:1px solid var(--line);border-radius:13px;background:#091321;padding:11px}.ae-box span{display:block;font-size:.56rem;color:var(--muted);text-transform:uppercase;letter-spacing:.08em;font-weight:800}.ae-box strong{display:block;font-size:1rem;margin-top:5px}.ae-box small{display:block;font-size:.59rem;color:#718198;margin-top:4px;line-height:1.35}
.ae-pillars{display:grid;grid-template-columns:repeat(3,1fr);gap:7px;margin:9px 0}.ae-pillar{border:1px solid var(--line);border-radius:12px;padding:10px;background:#091321;text-align:center}.ae-pillar span{font-size:.55rem;color:var(--muted);font-weight:800}.ae-pillar strong{display:block;margin-top:5px;font-size:.82rem}
[data-testid="stVerticalBlockBorderWrapper"]{border:1px solid var(--line)!important;border-radius:15px!important;background:rgba(9,19,33,.9)!important}.stButton>button{min-height:46px;border-radius:12px;font-weight:800;background:linear-gradient(135deg,#ffe38f,var(--gold));color:#171106;border:0}.stTabs [data-baseweb="tab-list"]{gap:4px;overflow-x:auto}.stTabs [data-baseweb="tab"]{min-width:max-content;font-size:.66rem;font-weight:800}.stTabs [aria-selected="true"]{color:var(--gold)!important}
.ae-foot{text-align:center;color:#65758b;font-size:.57rem;margin-top:18px}
@media(max-width:430px){.ae-title{font-size:1.25rem}.ae-build{font-size:.52rem}.ae-box{padding:9px}.ae-pillars{gap:5px}.ae-pillar{padding:8px 4px}}
</style>
""",
    unsafe_allow_html=True,
)


def _setting(settings: Any, name: str, default: Any) -> Any:
    value = getattr(settings, name, default)
    return default if value is None else value


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _float(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(_get(obj, name, default))
    except (TypeError, ValueError):
        return default


def _risk_inputs(settings: Settings) -> RiskInputs:
    values = {
        "account_balance": _setting(settings, "account_balance", 10000.0),
        "risk_percent": _setting(settings, "risk_percent", 1.0),
        "requested_lot": _setting(settings, "requested_lot", 0.10),
        "contract_size": _setting(settings, "contract_size", 100.0),
        "lot_step": _setting(settings, "lot_step", 0.01),
        "min_lot": _setting(settings, "min_lot", 0.01),
        "maximum_risk_dollars": _setting(settings, "maximum_risk_dollars", 0.0),
        "spread_price": _setting(settings, "spread_price", 0.0),
        "slippage_price": _setting(settings, "slippage_price", 0.0),
        "minimum_stop_atr": _setting(settings, "minimum_stop_atr", 0.55),
        "maximum_stop_atr": _setting(settings, "maximum_stop_atr", 1.60),
    }
    try:
        return RiskInputs(**values)
    except TypeError:
        fields = getattr(RiskInputs, "model_fields", None) or getattr(RiskInputs, "__fields__", {})
        return RiskInputs(**{key: value for key, value in values.items() if key in fields})


@st.cache_data(ttl=25, show_spinner=False)
def _fetch_bundle(api_key: str, symbol: str, bars: int) -> Any:
    return TwelveDataClient(api_key).fetch_bundle(symbol, TIMEFRAMES, bars)


@st.cache_data(ttl=300, show_spinner=False)
def _macro_context(api_key: str, dxy: str, us10y: str, gold_h1_json: str, state: str) -> Any | None:
    gold_h1 = pd.read_json(gold_h1_json, orient="split")
    try:
        from gold_web_terminal.macro_mobile_v542 import fetch_macro_confirmation

        return fetch_macro_confirmation(api_key, dxy, us10y, gold_h1, state)
    except Exception:
        try:
            from gold_web_terminal.macro_data import fetch_macro_confirmation

            return fetch_macro_confirmation(api_key, dxy, us10y, gold_h1, state)
        except Exception:
            return None


def _macro_direction(macro: Any | None, asset: str) -> str:
    if macro is None:
        return "UNAVAILABLE"
    item = _get(macro, asset)
    return str(_get(item, "direction", "UNAVAILABLE"))


def _arrow(direction: str) -> str:
    return {"UP": "↑", "DOWN": "↓", "FLAT": "→"}.get(direction.upper(), "—")


def _setup(report: Any) -> Any | None:
    return _get(report, "active_setup")


def _setup_value(report: Any, name: str, default: Any = None) -> Any:
    setup = _setup(report)
    return _get(setup, name, default) if setup is not None else default


def _pillar_html(votes: dict[str, int]) -> str:
    labels = {
        "market_structure": "Market Structure",
        "anchored_vwap": "Anchored VWAP",
        "volume_profile": "Volume Profile",
    }
    cards = []
    for key, label in labels.items():
        vote = int(votes.get(key, 0))
        state = "BUY" if vote > 0 else "SELL" if vote < 0 else "NEUTRAL"
        css = "buy" if vote > 0 else "sell" if vote < 0 else "wait"
        cards.append(f'<div class="ae-pillar"><span>{label}</span><strong class="{css}">{state}</strong></div>')
    return '<div class="ae-pillars">' + "".join(cards) + "</div>"


def _chart_frame(frame: pd.DataFrame, snapshot: Any) -> pd.DataFrame:
    result = frame[["time", "close"]].tail(180).copy()
    result["time"] = pd.to_datetime(result["time"], utc=True, errors="coerce")
    result = result.dropna(subset=["time"]).set_index("time").rename(columns={"close": "Price"})
    for column in ("anchored_vwap", "active_anchored_vwap", "active_avwap", "avwap", "vwap"):
        if column in frame.columns:
            series = frame[["time", column]].tail(180).copy()
            series["time"] = pd.to_datetime(series["time"], utc=True, errors="coerce")
            result["Anchored VWAP"] = series.set_index("time")[column]
            break
    aliases = {
        "POC": ("profile_poc", "volume_profile_poc", "poc"),
        "VAH": ("profile_vah", "volume_profile_vah", "vah", "value_area_high"),
        "VAL": ("profile_val", "volume_profile_val", "val", "value_area_low"),
    }
    for label, names in aliases.items():
        value = next((_get(snapshot, name) for name in names if _get(snapshot, name) is not None), None)
        if value is not None:
            result[label] = float(value)
    return result


app_pin = os.getenv("APP_PIN", "").strip()
if app_pin and not st.session_state.get("mobile_unlocked"):
    st.markdown('<div class="ae-head"><div><div class="ae-brand">AURUMEDGE MOBILE</div><div class="ae-sub">Private gold CFD decision terminal</div></div></div>', unsafe_allow_html=True)
    pin = st.text_input("Access PIN", type="password")
    if st.button("UNLOCK", use_container_width=True):
        if pin == app_pin:
            st.session_state["mobile_unlocked"] = True
            st.rerun()
        st.error("Incorrect PIN")
    st.stop()

settings = Settings.from_env()
api_key = str(_setting(settings, "twelve_data_api_key", "")).strip()
if not api_key:
    st.error("TWELVE_DATA_API_KEY is missing from Streamlit Secrets.")
    st.stop()

symbol = str(_setting(settings, "market_symbol", "XAU/USD"))
bars = int(_setting(settings, "bars_per_timeframe", 500))
base_refresh = max(30, int(_setting(settings, "auto_refresh_seconds", 300)))

st.markdown(
    f'<div class="ae-head"><div><div class="ae-brand">AURUMEDGE MOBILE</div><div class="ae-sub">XAU/USD · structure · anchored VWAP · volume profile</div></div><div class="ae-build">{BUILD_VERSION}</div></div>',
    unsafe_allow_html=True,
)

try:
    with st.spinner("Synchronizing M5 to D1…"):
        bundle = _fetch_bundle(api_key, symbol, bars)
except Exception as exc:
    st.error(f"Market-data synchronization failed: {exc}")
    st.stop()

frames = {tf: add_indicators(bundle.frames[tf]) for tf in TIMEFRAMES}
indicators = [summarize_indicators(frames[tf], tf) for tf in TIMEFRAMES]
liquidity = [analyze_liquidity(frames[tf], tf) for tf in ["M15", "H1", "H4", "D1"]]
adaptive_path = APP_DIR / str(_setting(settings, "adaptive_state_path", "data/adaptive_state.json"))
adaptive = AdaptiveEngine(
    adaptive_path,
    enabled=bool(_setting(settings, "adaptive_learning", True)),
    minimum_samples=int(_setting(settings, "adaptive_min_samples", 20)),
    horizon_bars=int(_setting(settings, "adaptive_horizon_bars", 12)),
    max_weight_change=float(_setting(settings, "adaptive_max_weight_change", 0.05)),
)
risk = _risk_inputs(settings)
preliminary = build_technical_report(
    symbol=bundle.symbol,
    data_time=bundle.data_time,
    price=bundle.last_price,
    indicators=indicators,
    liquidity=liquidity,
    data_source=bundle.source,
    digits=2,
    adaptive_weights=adaptive.weights(),
    target_multipliers=adaptive.target_multipliers(),
    adaptive_summary=adaptive.summary(),
    risk_inputs=risk,
    macro_required_for_entry=False,
)
gold_h1_json = frames["H1"][["time", "close"]].to_json(orient="split", date_format="iso")
macro = _macro_context(
    api_key,
    str(_setting(settings, "dxy_symbol", "DXY")),
    str(_setting(settings, "us10y_symbol", "US10Y")),
    gold_h1_json,
    str(_get(preliminary, "market_state", "STUCK")),
)
report = build_technical_report(
    symbol=bundle.symbol,
    data_time=bundle.data_time,
    price=bundle.last_price,
    indicators=indicators,
    liquidity=liquidity,
    data_source=bundle.source,
    digits=2,
    adaptive_weights=adaptive.weights(),
    target_multipliers=adaptive.target_multipliers(),
    adaptive_summary=adaptive.summary(),
    risk_inputs=risk,
    macro=macro,
    macro_required_for_entry=False,
)
signal_time = frames["M15"].iloc[-1]["time"]
votes = dict(_get(report, "pillar_votes", {}) or {})
if not votes:
    votes = derive_feature_votes(indicators, liquidity, macro, str(_get(report, "market_state", "STUCK")))
report = adaptive.apply_capital_preservation(report, signal_time, votes)

state = str(_get(report, "market_state", "STUCK"))
entry_live = bool(_get(report, "entry_live", False) or _setup_value(report, "entry_live", False))
near_entry = bool(_get(report, "near_entry", False) or _setup_value(report, "near_entry", False))
near_exit = adaptive.pending_near_exit(float(_get(report, "last_price", bundle.last_price)))
refresh_seconds = 30 if entry_live or near_exit else 45 if near_entry else 120 if state in {"BUY", "SELL"} else base_refresh
components.html(
    f"<script>setTimeout(function(){{window.parent.location.reload();}}, {refresh_seconds * 1000});</script>",
    height=0,
)

execution_label = str(_get(report, "execution_label", "NO TRADE"))
css = "buy" if state == "BUY" else "sell" if state == "SELL" else "wait"
st.markdown(
    f'<div class="ae-status"><div class="ae-label">Current execution state</div><div class="ae-title {css}">{execution_label}</div><div style="font-size:.67rem;color:#9cabbf">Price {float(_get(report, "last_price", 0)):.2f} · Confidence {int(_get(report, "confidence", 0))}% · next refresh {refresh_seconds}s</div>{_pillar_html(votes)}</div>',
    unsafe_allow_html=True,
)

setup = _setup(report)
if setup is not None:
    risk_plan = _get(setup, "risk_plan")
    st.markdown(
        f'''<div class="ae-grid">
<div class="ae-box"><span>Entry zone</span><strong>{_float(setup,"entry_low"):.2f} – {_float(setup,"entry_high"):.2f}</strong><small>{str(_get(setup,"setup_type","THREE PILLAR"))}</small></div>
<div class="ae-box"><span>Stop loss</span><strong class="sell">{_float(setup,"stop_loss"):.2f}</strong><small>Local M15 structure / AVWAP / profile invalidation</small></div>
<div class="ae-box"><span>TP1 · monitored close</span><strong class="buy">{_float(setup,"take_profit_1"):.2f}</strong><small>{_float(setup,"risk_reward_1"):.2f}R</small></div>
<div class="ae-box"><span>TP2 / TP3</span><strong>{_float(setup,"take_profit_2"):.2f} / {_float(setup,"take_profit_3"):.2f}</strong><small>Management targets</small></div>
<div class="ae-box"><span>Risk gate</span><strong>{str(_get(risk_plan,"status","—"))}</strong><small>Recommended lot {_float(risk_plan,"recommended_lot"):.2f}</small></div>
<div class="ae-box"><span>Cloud monitor</span><strong>{'FAST' if refresh_seconds <= 45 else 'NORMAL'}</strong><small>Near entry/exit burst is handled by GitHub Actions</small></div>
</div>''',
        unsafe_allow_html=True,
    )

with st.container(border=True):
    c1, c2 = st.columns(2)
    with c1:
        if st.button("SYNC NOW", use_container_width=True):
            _fetch_bundle.clear()
            st.rerun()
    with c2:
        st.metric("Data time", str(bundle.data_time).replace("T", " ")[:19])

signal_tab, chart_tab, context_tab, brain_tab = st.tabs(["SIGNAL", "CHART", "CONTEXT", "BRAIN"])

with signal_tab:
    st.subheader("Three-pillar decision")
    rows = []
    tf_votes = _get(report, "pillar_timeframe_votes", {}) or {}
    for tf in ("M15", "H1"):
        row = {"Timeframe": tf}
        for key, label in (("market_structure", "Structure"), ("anchored_vwap", "AVWAP"), ("volume_profile", "Profile")):
            value = int((tf_votes.get(tf) or {}).get(key, 0))
            row[label] = "BUY" if value > 0 else "SELL" if value < 0 else "NEUTRAL"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.caption("EMA, MACD, RSI, ADX/DMI, ordinary volume, FVG, liquidity labels, DXY and US10Y do not vote on direction.")

with chart_tab:
    chart_tf = st.selectbox("Chart timeframe", ["M5", "M15", "H1"], index=1)
    snapshot = next((item for item in indicators if str(_get(item, "timeframe", "")) == chart_tf), None)
    st.line_chart(_chart_frame(frames[chart_tf], snapshot), use_container_width=True)
    st.caption("Clean display: price, active anchored VWAP, POC, VAH and VAL when available.")

with context_tab:
    dxy = _macro_direction(macro, "dxy")
    us10y = _macro_direction(macro, "us10y")
    st.markdown(
        f'''<div class="ae-grid">
<div class="ae-box"><span>DXY direction</span><strong>{_arrow(dxy)} {dxy}</strong><small>Display only — never blocks a signal</small></div>
<div class="ae-box"><span>US 10Y direction</span><strong>{_arrow(us10y)} {us10y}</strong><small>Display only — never blocks a signal</small></div>
</div>''',
        unsafe_allow_html=True,
    )
    st.info("The terminal has no broker connection and does not execute orders automatically.")

with brain_tab:
    summary = adaptive.summary()
    st.markdown(
        f'''<div class="ae-grid">
<div class="ae-box"><span>Reviewed trades</span><strong>{summary.reviewed_signals}</strong><small>Learning starts with the first completed delivered trade</small></div>
<div class="ae-box"><span>Pending trades</span><strong>{summary.pending_signals}</strong><small>TP1 / SL / timeout monitored</small></div>
<div class="ae-box"><span>Wins / losses</span><strong>{summary.wins} / {summary.losses}</strong><small>Timeouts {summary.timeouts}</small></div>
<div class="ae-box"><span>Last review</span><strong style="font-size:.72rem">{summary.last_review or '—'}</strong><small>Bounded changes prevent one trade dominating</small></div>
</div>''',
        unsafe_allow_html=True,
    )
    state_bytes = json.dumps(adaptive.state, indent=2).encode("utf-8")
    st.download_button("DOWNLOAD BRAIN BACKUP", state_bytes, "adaptive_state.json", "application/json", use_container_width=True)
    restore = st.file_uploader("Restore adaptive_state.json", type=["json"])
    if restore is not None and st.button("RESTORE BRAIN STATE", use_container_width=True):
        try:
            parsed = json.loads(restore.getvalue().decode("utf-8"))
            if not isinstance(parsed, dict) or "features" not in parsed:
                raise ValueError("Invalid AurumEdge adaptive state")
            adaptive_path.parent.mkdir(parents=True, exist_ok=True)
            adaptive_path.write_text(json.dumps(parsed, indent=2), encoding="utf-8")
            st.success("Brain state restored.")
            st.rerun()
        except Exception as exc:
            st.error(f"Restore failed: {exc}")

with st.expander("Diagnostics"):
    st.json(
        {
            "build": BUILD_VERSION,
            "source": bundle.source,
            "state": state,
            "entry_live": entry_live,
            "near_entry": near_entry,
            "near_exit": near_exit,
            "refresh_seconds": refresh_seconds,
            "pillar_votes": votes,
            "DXY": _macro_direction(macro, "dxy"),
            "US10Y": _macro_direction(macro, "us10y"),
            "macro_blocks_signal": False,
            "broker_connected": False,
            "automatic_execution": False,
        }
    )

st.markdown(f'<div class="ae-foot">AurumEdge Mobile · {BUILD_VERSION} · Decision support only</div>', unsafe_allow_html=True)
