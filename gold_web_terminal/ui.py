from __future__ import annotations

from html import escape
from typing import Iterable

from .models import IndicatorSnapshot, LiquiditySnapshot, TechnicalReport


GLOBAL_CSS = r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Manrope:wght@600;700;800&display=swap');

:root {
  --bg-0:#060913;
  --bg-1:#090f1c;
  --panel:#0d1524;
  --panel-2:#111c2f;
  --panel-3:#0a1220;
  --line:rgba(148,163,184,.14);
  --line-strong:rgba(148,163,184,.24);
  --text:#f4f7fb;
  --muted:#8e9bb0;
  --gold:#f4c85b;
  --gold-2:#d9a72e;
  --cyan:#5dd6f4;
  --green:#22cfa0;
  --red:#ff5d7d;
  --purple:#a98bff;
  --amber:#ffb94a;
}

html, body, [class*="css"] { font-family:'Inter',sans-serif; }
.stApp {
  background:
    radial-gradient(circle at 88% -10%, rgba(244,200,91,.09), transparent 27%),
    radial-gradient(circle at 0% 20%, rgba(93,214,244,.06), transparent 24%),
    linear-gradient(180deg,var(--bg-0),var(--bg-1) 46%,#070b14);
  color:var(--text);
}
header[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer,
[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display:none !important; }
[data-testid="stSidebar"] { display:none !important; }
[data-testid="stAppViewContainer"] > .main { overflow:visible; }
.block-container { max-width:1780px; padding:0.7rem 1.25rem 2rem; }

/* Top brand bar */
.ae-topbar {
  display:flex; align-items:center; justify-content:space-between; gap:20px;
  min-height:64px; padding:10px 2px 14px; border-bottom:1px solid var(--line);
  margin-bottom:12px;
}
.ae-brand { display:flex; align-items:center; gap:12px; min-width:0; }
.ae-logo {
  width:38px; height:38px; border-radius:12px; display:grid; place-items:center;
  font-family:'Manrope'; font-weight:800; color:#15100a;
  background:linear-gradient(135deg,#ffe79e,var(--gold) 55%,#bd7f16);
  box-shadow:0 8px 24px rgba(244,200,91,.22),inset 0 1px rgba(255,255,255,.45);
}
.ae-brand-title { font-family:'Manrope'; font-weight:800; font-size:1.02rem; letter-spacing:.03em; }
.ae-brand-sub { color:var(--muted); font-size:.73rem; margin-top:2px; }
.ae-top-meta { display:flex; align-items:center; justify-content:flex-end; gap:8px; flex-wrap:wrap; }
.ae-badge {
  display:inline-flex; align-items:center; gap:7px; padding:7px 10px; border-radius:999px;
  border:1px solid var(--line); background:rgba(13,21,36,.72); color:#b8c3d2;
  font-size:.71rem; font-weight:650; white-space:nowrap;
}
.ae-dot { width:7px; height:7px; border-radius:50%; display:inline-block; }
.ae-dot.live { background:var(--green); box-shadow:0 0 0 4px rgba(34,207,160,.10); }
.ae-dot.demo { background:var(--amber); box-shadow:0 0 0 4px rgba(255,185,74,.10); }

/* Native controls */
.ae-control-label { color:#8e9bb0; font-size:.68rem; letter-spacing:.08em; text-transform:uppercase; font-weight:750; margin-bottom:5px; }
[data-testid="stVerticalBlockBorderWrapper"] {
  border:1px solid var(--line) !important; border-radius:16px !important;
  background:linear-gradient(180deg,rgba(15,24,41,.92),rgba(10,17,30,.92)) !important;
  box-shadow:0 12px 38px rgba(0,0,0,.16);
}
[data-testid="stVerticalBlockBorderWrapper"] > div { padding:12px 14px !important; }
div[data-baseweb="select"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
  background:#0a1220 !important; border-color:var(--line-strong) !important; color:var(--text) !important;
  border-radius:10px !important;
}
.stButton > button {
  min-height:42px; border-radius:10px; font-weight:800; letter-spacing:.01em;
  border:1px solid rgba(244,200,91,.45) !important;
  color:#171106 !important; background:linear-gradient(135deg,#ffe28a,var(--gold) 54%,#d99d24) !important;
  box-shadow:0 8px 26px rgba(244,200,91,.18);
}
.stButton > button:hover { transform:translateY(-1px); border-color:#ffe9a7 !important; }
button[kind="secondary"] { color:#d9e1eb !important; background:#111c2f !important; border-color:var(--line-strong) !important; box-shadow:none; }

/* KPI strip */
.ae-kpi-grid { display:grid; grid-template-columns:1.25fr repeat(5,1fr); gap:9px; margin:11px 0 12px; }
.ae-kpi {
  position:relative; overflow:hidden; min-height:76px; padding:12px 14px;
  border:1px solid var(--line); border-radius:14px;
  background:linear-gradient(180deg,rgba(16,27,46,.94),rgba(10,17,30,.94));
  box-shadow:0 12px 34px rgba(0,0,0,.12);
}
.ae-kpi:before { content:""; position:absolute; left:0; top:0; bottom:0; width:2px; background:rgba(244,200,91,.72); }
.ae-kpi-label { color:var(--muted); font-size:.65rem; font-weight:760; letter-spacing:.08em; text-transform:uppercase; }
.ae-kpi-value { margin-top:5px; font-family:'Manrope'; font-size:1.13rem; line-height:1.1; font-weight:800; color:var(--text); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ae-kpi-sub { margin-top:4px; color:#6f7f96; font-size:.66rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.ae-state-buy .ae-kpi-value { color:var(--green); }
.ae-state-sell .ae-kpi-value { color:var(--red); }
.ae-state-trap .ae-kpi-value { color:var(--purple); }
.ae-state-stuck .ae-kpi-value { color:var(--amber); }

/* Section shells */
.ae-panel-head { display:flex; align-items:flex-start; justify-content:space-between; gap:16px; margin-bottom:8px; }
.ae-panel-title { font-family:'Manrope'; font-size:.92rem; font-weight:800; }
.ae-panel-sub { color:var(--muted); font-size:.69rem; margin-top:3px; }
.ae-mini-badges { display:flex; gap:6px; flex-wrap:wrap; justify-content:flex-end; }
.ae-mini-badge { padding:5px 8px; border-radius:7px; border:1px solid var(--line); background:#0a1220; color:#aeb9c8; font-size:.65rem; font-weight:700; }

/* Decision card */
.ae-signal {
  position:relative; overflow:hidden; min-height:400px; padding:18px;
  border-radius:18px; border:1px solid var(--line-strong);
  background:linear-gradient(160deg,rgba(17,28,47,.98),rgba(8,15,27,.98));
  box-shadow:0 20px 55px rgba(0,0,0,.22);
}
.ae-signal:after { content:""; position:absolute; width:180px; height:180px; right:-75px; top:-85px; border-radius:50%; filter:blur(2px); opacity:.16; background:var(--tone); }
.ae-signal.buy { --tone:var(--green); border-top:3px solid var(--green); }
.ae-signal.sell { --tone:var(--red); border-top:3px solid var(--red); }
.ae-signal.trap { --tone:var(--purple); border-top:3px solid var(--purple); }
.ae-signal.stuck { --tone:var(--amber); border-top:3px solid var(--amber); }
.ae-signal-overline { color:var(--muted); font-size:.64rem; letter-spacing:.13em; text-transform:uppercase; font-weight:800; }
.ae-signal-title { font-family:'Manrope'; font-size:1.65rem; line-height:1.05; margin:7px 0 5px; font-weight:800; color:var(--tone); }
.ae-signal-copy { color:#aeb9c8; font-size:.76rem; line-height:1.5; }
.ae-confidence-row { display:flex; align-items:center; justify-content:space-between; margin:14px 0 7px; }
.ae-confidence-row span { color:var(--muted); font-size:.68rem; }
.ae-confidence-row strong { font-family:'Manrope'; font-size:.8rem; }
.ae-track { height:7px; border-radius:999px; background:#070d17; border:1px solid rgba(148,163,184,.09); overflow:hidden; }
.ae-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--tone),color-mix(in srgb,var(--tone) 55%,white)); }
.ae-level-grid { display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-top:15px; }
.ae-level { min-height:62px; padding:10px 11px; border:1px solid var(--line); background:rgba(7,13,23,.72); border-radius:10px; }
.ae-level.wide { grid-column:1/-1; }
.ae-level-label { color:#73849a; font-size:.59rem; letter-spacing:.09em; text-transform:uppercase; font-weight:800; }
.ae-level-value { margin-top:5px; font-family:'Manrope'; font-size:.91rem; font-weight:800; color:#f3f6fb; }
.ae-level.entry .ae-level-value { color:var(--gold); }
.ae-level.stop .ae-level-value { color:var(--red); }
.ae-level.target .ae-level-value { color:var(--green); }
.ae-note { margin-top:12px; padding:10px 11px; border-radius:10px; border:1px solid var(--line); background:rgba(10,18,32,.8); color:#92a1b6; font-size:.68rem; line-height:1.45; }

/* Market pulse */
.ae-pulse { margin-top:12px; padding:13px 14px; border:1px solid var(--line); border-radius:14px; background:rgba(10,18,32,.82); }
.ae-pulse-title { font-family:'Manrope'; font-size:.75rem; font-weight:800; margin-bottom:9px; }
.ae-pulse-row { display:grid; grid-template-columns:58px 1fr 36px; gap:8px; align-items:center; margin:7px 0; }
.ae-pulse-row span { color:#8e9bb0; font-size:.63rem; }
.ae-pulse-row strong { text-align:right; font-size:.63rem; }
.ae-pulse-track { height:6px; background:#070d17; border-radius:999px; overflow:hidden; }
.ae-pulse-fill { height:100%; border-radius:999px; }

/* Reason chips and price ladder */
.ae-chip-wrap { display:flex; gap:7px; flex-wrap:wrap; }
.ae-chip { display:inline-flex; align-items:center; gap:6px; padding:7px 9px; border-radius:9px; border:1px solid var(--line); background:#0b1423; color:#b6c0ce; font-size:.67rem; line-height:1.25; }
.ae-chip:before { content:""; width:5px; height:5px; border-radius:50%; background:var(--gold); flex:0 0 auto; }
.ae-ladder { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.ae-ladder-row { display:grid; grid-template-columns:1.3fr .8fr .55fr; gap:8px; padding:9px 11px; border-bottom:1px solid rgba(148,163,184,.09); align-items:center; }
.ae-ladder-row:last-child { border-bottom:0; }
.ae-ladder-name { color:#aeb9c8; font-size:.68rem; font-weight:650; }
.ae-ladder-price { font-family:'Manrope'; text-align:right; font-size:.72rem; font-weight:800; }
.ae-ladder-side { text-align:right; font-size:.58rem; letter-spacing:.08em; font-weight:850; }
.ae-ladder-side.res { color:var(--red); }
.ae-ladder-side.sup { color:var(--green); }
.ae-ladder-side.neutral { color:var(--gold); }

/* Tabs / dataframes / charts */
.stTabs [data-baseweb="tab-list"] { gap:5px; border-bottom:1px solid var(--line); }
.stTabs [data-baseweb="tab"] { height:41px; padding:0 13px; border-radius:9px 9px 0 0; color:#8795a9; font-size:.72rem; font-weight:750; background:transparent; border:0; }
.stTabs [aria-selected="true"] { color:var(--gold) !important; background:rgba(244,200,91,.06) !important; border-bottom:2px solid var(--gold) !important; }
[data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
.js-plotly-plot { border-radius:14px; overflow:hidden; }
[data-testid="stExpander"] { border:1px solid var(--line) !important; border-radius:12px !important; background:rgba(10,18,32,.62); }
hr { border-color:var(--line) !important; }

.ae-section-title { font-family:'Manrope'; font-size:.92rem; font-weight:800; margin:5px 0 10px; }
.ae-footer { margin-top:20px; padding-top:14px; border-top:1px solid var(--line); display:flex; justify-content:space-between; gap:12px; color:#68788f; font-size:.62rem; }

@media(max-width:1150px) {
  .ae-kpi-grid { grid-template-columns:repeat(3,1fr); }
  .ae-top-meta .ae-badge:nth-child(2) { display:none; }
}
@media(max-width:760px) {
  .block-container { padding:.45rem .65rem 1.4rem; }
  .ae-topbar { align-items:flex-start; }
  .ae-top-meta { display:none; }
  .ae-kpi-grid { grid-template-columns:1fr 1fr; }
  .ae-kpi { min-height:69px; }
  .ae-level-grid { grid-template-columns:1fr; }
  .ae-level.wide { grid-column:auto; }
  .ae-footer { flex-direction:column; }
}
</style>
"""


def _fmt(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _state_class(state: str) -> str:
    return {"BUY": "buy", "SELL": "sell", "TRAP": "trap", "STUCK": "stuck"}.get(state, "stuck")


def _state_kpi_class(state: str) -> str:
    return f"ae-state-{_state_class(state)}"


def brand_bar_html(data_source: str, symbol: str, data_time: str, build: str) -> str:
    is_live = data_source == "TWELVE_DATA"
    dot = "live" if is_live else "demo"
    feed = "LIVE MARKET FEED" if is_live else data_source.replace("_", " ")
    return f"""
<div class="ae-topbar">
  <div class="ae-brand">
    <div class="ae-logo">Au</div>
    <div>
      <div class="ae-brand-title">AURUMEDGE <span style="color:var(--gold)">PRO</span></div>
      <div class="ae-brand-sub">Gold market intelligence · XAU/USD decision terminal</div>
    </div>
  </div>
  <div class="ae-top-meta">
    <span class="ae-badge"><span class="ae-dot {dot}"></span>{escape(feed)}</span>
    <span class="ae-badge">{escape(symbol)}</span>
    <span class="ae-badge">CANDLE {escape(data_time[:19].replace('T',' '))} UTC</span>
    <span class="ae-badge">BUILD {escape(build)}</span>
  </div>
</div>
"""


def kpi_strip_html(report: TechnicalReport, final_state: str, final_note: str) -> str:
    bias = "BUYERS CONTROL" if report.buy_score > report.sell_score else "SELLERS CONTROL"
    values = [
        ("Indicative price", _fmt(report.last_price), report.symbol, ""),
        ("Market decision", final_state, final_note, _state_kpi_class(final_state)),
        ("Confidence", f"{report.confidence}%", "Signal quality", ""),
        ("Trend strength", f"{report.trend_strength}/100", bias, ""),
        ("Buy / Sell", f"{report.buy_score} / {report.sell_score}", "Directional score", ""),
        ("Volatility", report.volatility_state.upper(), report.regime.replace("_", " ").title(), ""),
    ]
    cards = "".join(
        f"""
<div class="ae-kpi {css}">
  <div class="ae-kpi-label">{escape(label)}</div>
  <div class="ae-kpi-value">{escape(value)}</div>
  <div class="ae-kpi-sub">{escape(sub)}</div>
</div>
"""
        for label, value, sub, css in values
    )
    return f'<div class="ae-kpi-grid">{cards}</div>'


def signal_panel_html(
    report: TechnicalReport,
    final_state: str,
    final_note: str,
    liquidity: LiquiditySnapshot | None,
) -> str:
    tone = _state_class(final_state)
    setup = report.active_setup if final_state in {"BUY", "SELL"} else None
    if final_state == "TRAP":
        title = "LIQUIDITY TRAP"
        copy = report.trap_reason or "A false break or two-sided liquidity sweep makes direction unreliable."
        plan = f"""
<div class="ae-level-grid">
  <div class="ae-level"><div class="ae-level-label">Action</div><div class="ae-level-value">NO ENTRY</div></div>
  <div class="ae-level"><div class="ae-level-label">Recheck</div><div class="ae-level-value">NEXT CANDLE</div></div>
  <div class="ae-level wide entry"><div class="ae-level-label">Price location</div><div class="ae-level-value">{_fmt(report.last_price)}</div></div>
</div>
<div class="ae-note">Wait for price to reclaim a broken zone or close cleanly beyond the nearest support/resistance. A trap classification protects against entering directly into stop-hunting.</div>
"""
    elif final_state == "STUCK":
        title = "MARKET STUCK"
        copy = report.trap_reason or "Trend strength is weak and price is compressing inside a range."
        plan = f"""
<div class="ae-level-grid">
  <div class="ae-level"><div class="ae-level-label">Action</div><div class="ae-level-value">NO ENTRY</div></div>
  <div class="ae-level"><div class="ae-level-label">State</div><div class="ae-level-value">RANGE / CHOP</div></div>
  <div class="ae-level wide entry"><div class="ae-level-label">Current price</div><div class="ae-level-value">{_fmt(report.last_price)}</div></div>
</div>
<div class="ae-note">A trade is intentionally blocked while ADX, choppiness and compression show poor directional follow-through. Refresh after a clean range break.</div>
"""
    else:
        risk_blocked = bool(setup is not None and setup.status == "NO_TRADE")
        title = f"{final_state} BIAS · RISK GATE" if risk_blocked else f"ENTER {final_state}"
        copy = f"{report.regime.replace('_',' ').title()} · {'direction valid but position risk blocked' if risk_blocked else 'active market zone'} · valid until {escape(setup.valid_until if setup else '—')}"
        if setup is None:
            plan = '<div class="ae-note">No active setup was generated.</div>'
        else:
            risk_html = ""
            if setup.risk_plan is not None:
                rp = setup.risk_plan
                risk_color = "var(--green)" if rp.status == "OK" else "var(--amber)" if rp.status == "REDUCE_LOT" else "var(--red)"
                risk_html = f"""<div class="ae-note" style="border-color:{risk_color}"><strong style="color:{risk_color}">RISK GATE · {escape(rp.status)}</strong><br>Requested lot {rp.requested_lot:.2f} · recommended {rp.recommended_lot:.2f} · estimated requested-lot loss ${rp.estimated_loss_requested_lot:,.2f} · budget ${rp.risk_budget:,.2f}<br>{escape(rp.message)}</div>"""
            plan = f"""
<div class="ae-level-grid">
  <div class="ae-level wide entry"><div class="ae-level-label">Entry zone</div><div class="ae-level-value">{_fmt(setup.entry_low)} – {_fmt(setup.entry_high)}</div></div>
  <div class="ae-level stop"><div class="ae-level-label">Stop loss</div><div class="ae-level-value">{_fmt(setup.stop_loss)}</div></div>
  <div class="ae-level target"><div class="ae-level-label">Take profit 1 · {setup.risk_reward_1}R</div><div class="ae-level-value">{_fmt(setup.take_profit_1)}</div></div>
  <div class="ae-level target"><div class="ae-level-label">Take profit 2 · {setup.risk_reward_2}R</div><div class="ae-level-value">{_fmt(setup.take_profit_2)}</div></div>
  <div class="ae-level target"><div class="ae-level-label">Take profit 3 · {setup.risk_reward_3}R</div><div class="ae-level-value">{_fmt(setup.take_profit_3)}</div></div>
</div>
<div class="ae-note">{escape(setup.invalidation)}<br>{escape(setup.stop_basis)}<br>{escape(setup.target_basis)}</div>
{risk_html}
"""

    buy_width = max(2, min(100, report.buy_score))
    sell_width = max(2, min(100, report.sell_score))
    trend_width = max(2, min(100, report.trend_strength))
    nearest_support = _fmt(liquidity.nearest_support) if liquidity else "—"
    nearest_resistance = _fmt(liquidity.nearest_resistance) if liquidity else "—"
    return f"""
<div class="ae-signal {tone}">
  <div class="ae-signal-overline">Current execution state</div>
  <div class="ae-signal-title">{escape(title)}</div>
  <div class="ae-signal-copy">{escape(copy)}</div>
  <div class="ae-confidence-row"><span>MODEL CONFIDENCE</span><strong>{report.confidence}%</strong></div>
  <div class="ae-track"><div class="ae-fill" style="width:{report.confidence}%"></div></div>
  {plan}
</div>
<div class="ae-pulse">
  <div class="ae-pulse-title">Market pulse</div>
  <div class="ae-pulse-row"><span>BUY</span><div class="ae-pulse-track"><div class="ae-pulse-fill" style="width:{buy_width}%;background:var(--green)"></div></div><strong>{report.buy_score}</strong></div>
  <div class="ae-pulse-row"><span>SELL</span><div class="ae-pulse-track"><div class="ae-pulse-fill" style="width:{sell_width}%;background:var(--red)"></div></div><strong>{report.sell_score}</strong></div>
  <div class="ae-pulse-row"><span>TREND</span><div class="ae-pulse-track"><div class="ae-pulse-fill" style="width:{trend_width}%;background:var(--gold)"></div></div><strong>{report.trend_strength}</strong></div>
  <div class="ae-note" style="margin-top:10px">Support <strong style="color:var(--green)">{nearest_support}</strong> &nbsp; · &nbsp; Resistance <strong style="color:var(--red)">{nearest_resistance}</strong><br><span style="color:#65758b">Decision gate: {escape(final_note)}</span></div>
</div>
"""


def chips_html(items: Iterable[str], empty_text: str = "No additional reasons available.") -> str:
    clean = [escape(str(item)) for item in items if str(item).strip()]
    if not clean:
        clean = [escape(empty_text)]
    return '<div class="ae-chip-wrap">' + "".join(f'<span class="ae-chip">{item}</span>' for item in clean) + "</div>"


def level_ladder_html(liquidity: LiquiditySnapshot | None, price: float) -> str:
    if liquidity is None:
        return '<div class="ae-note">Liquidity map is unavailable for this timeframe.</div>'
    rows: list[tuple[str, float | None, str]] = [
        ("Nearest resistance", liquidity.nearest_resistance, "res"),
        ("Value area high", liquidity.value_area_high, "res"),
        ("Previous-day high", liquidity.previous_day_high, "res"),
        ("Current market", price, "neutral"),
        ("Volume POC", liquidity.point_of_control, "neutral"),
        ("Previous-day low", liquidity.previous_day_low, "sup"),
        ("Value area low", liquidity.value_area_low, "sup"),
        ("Nearest support", liquidity.nearest_support, "sup"),
    ]
    valid = [(name, value, side) for name, value, side in rows if value is not None]
    valid.sort(key=lambda row: float(row[1]), reverse=True)
    body = "".join(
        f"""
<div class="ae-ladder-row">
  <div class="ae-ladder-name">{escape(name)}</div>
  <div class="ae-ladder-price">{_fmt(float(value))}</div>
  <div class="ae-ladder-side {side}">{'MARKET' if side == 'neutral' and name == 'Current market' else side.upper()}</div>
</div>
"""
        for name, value, side in valid
    )
    return f'<div class="ae-ladder">{body}</div>'


def timeframe_cards_html(snapshots: list[IndicatorSnapshot]) -> str:
    cards = []
    for item in snapshots:
        score = float(item.directional_score)
        state = "BUY" if score >= 14 else "SELL" if score <= -14 else "NEUTRAL"
        color = "var(--green)" if state == "BUY" else "var(--red)" if state == "SELL" else "var(--amber)"
        cards.append(
            f"""
<div style="border:1px solid var(--line);border-radius:12px;padding:11px;background:#0a1220;min-height:102px">
  <div style="display:flex;justify-content:space-between;gap:8px"><strong style="font-family:Manrope;font-size:.77rem">{escape(item.timeframe)}</strong><span style="font-size:.6rem;font-weight:850;color:{color}">{state}</span></div>
  <div style="font-family:Manrope;font-size:1.12rem;font-weight:800;margin-top:8px;color:{color}">{score:+.0f}</div>
  <div style="font-size:.6rem;color:#73849a;margin-top:4px">RSI {_fmt(item.rsi14,1)} · ADX {_fmt(item.adx14,1)}</div>
  <div style="font-size:.6rem;color:#73849a;margin-top:2px">{escape(item.trend.title())} trend · {escape(item.momentum.title())} momentum</div>
</div>
"""
        )
    return '<div style="display:grid;grid-template-columns:repeat(5,minmax(110px,1fr));gap:8px">' + "".join(cards) + "</div>"



def macro_confirmation_html(report: TechnicalReport) -> str:
    macro = report.macro
    if macro is None:
        return '<div class="ae-note">Macro confirmation is unavailable. Directional execution is blocked while the macro gate is required.</div>'

    def arrow(direction: str) -> str:
        return {"UP": "↑", "DOWN": "↓", "FLAT": "→"}.get(direction, "—")

    def tone(direction: str, gold_positive: bool = False) -> str:
        if direction == "UNAVAILABLE":
            return "#8e9bb0"
        favorable = direction == ("UP" if gold_positive else "DOWN")
        adverse = direction == ("DOWN" if gold_positive else "UP")
        return "var(--green)" if favorable else "var(--red)" if adverse else "var(--amber)"

    dxy_value = _fmt(macro.dxy.value, 3)
    y_value = _fmt(macro.us10y.value, 3)
    gold_move = _fmt(macro.gold_change_4h, 2)
    dxy_change = macro.dxy.change_4h if macro.dxy.change_4h is not None else macro.dxy.change_1d
    dxy_period = "4h" if macro.dxy.change_4h is not None else "1d"
    yield_change = macro.us10y.change_4h if macro.us10y.change_4h is not None else macro.us10y.change_1d
    yield_period = "4h" if macro.us10y.change_4h is not None else "1d"
    gate_color = "var(--green)" if macro.gate == "CONFIRM" else "var(--red)" if macro.gate == "CONFLICT" else "var(--amber)"
    coverage_color = "var(--green)" if macro.coverage_score >= 100 else "var(--amber)" if macro.coverage_score >= 80 else "var(--red)"
    return f"""
<div class="ae-kpi-grid" style="grid-template-columns:repeat(6,1fr)">
  <div class="ae-kpi"><div class="ae-kpi-label">DXY</div><div class="ae-kpi-value" style="color:{tone(macro.dxy.direction)}">{dxy_value} {arrow(macro.dxy.direction)}</div><div class="ae-kpi-sub">{dxy_period} {dxy_change if dxy_change is not None else '—'} · {escape(macro.dxy.source)}</div><div class="ae-kpi-sub">{escape(macro.dxy.freshness)}</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">US 10Y YIELD</div><div class="ae-kpi-value" style="color:{tone(macro.us10y.direction)}">{y_value}% {arrow(macro.us10y.direction)}</div><div class="ae-kpi-sub">{yield_period} {yield_change if yield_change is not None else '—'} · {escape(macro.us10y.source)}</div><div class="ae-kpi-sub">{escape(macro.us10y.freshness)}</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">GOLD 4H FLOW</div><div class="ae-kpi-value" style="color:{tone(macro.gold_direction, True)}">{gold_move} {arrow(macro.gold_direction)}</div><div class="ae-kpi-sub">Calculated from live H1 candles</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">MACRO ALIGNMENT</div><div class="ae-kpi-value">{escape(macro.alignment)}</div><div class="ae-kpi-sub">Bias {escape(macro.macro_bias.replace('_', ' '))} · score {macro.confirmation_score}/100</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">DATA COVERAGE</div><div class="ae-kpi-value" style="color:{coverage_color}">{macro.coverage_score}%</div><div class="ae-kpi-sub">{escape(macro.data_status)} · DXY + yield + gold flow</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">DECISION GATE</div><div class="ae-kpi-value" style="color:{gate_color}">{escape(macro.gate)}</div><div class="ae-kpi-sub">Directional entry requires complete confirmation</div></div>
</div>
"""


def adaptive_learning_html(report: TechnicalReport) -> str:
    adaptive = report.adaptive
    if adaptive is None:
        return '<div class="ae-note">Adaptive-learning state is unavailable.</div>'
    weights = sorted(adaptive.indicator_weights.items(), key=lambda item: item[1], reverse=True)
    chips = ''.join(f'<span class="ae-chip">{escape(name.replace("_"," ").upper())} {value:.2f}× · n={adaptive.indicator_samples.get(name,0)}</span>' for name, value in weights)
    targets = adaptive.target_r_multipliers
    return f"""
<div class="ae-kpi-grid" style="grid-template-columns:repeat(4,1fr)">
  <div class="ae-kpi"><div class="ae-kpi-label">Reviewed signals</div><div class="ae-kpi-value">{adaptive.reviewed_signals}</div><div class="ae-kpi-sub">Completed evidence only</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">Win rate</div><div class="ae-kpi-value">{adaptive.win_rate:.1f}%</div><div class="ae-kpi-sub">{adaptive.wins} wins · {adaptive.losses} losses · {adaptive.timeouts} timeouts</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">Adaptive targets</div><div class="ae-kpi-value">{targets['tp1']:.2f}R / {targets['tp2']:.2f}R / {targets['tp3']:.2f}R</div><div class="ae-kpi-sub">Based on historical MFE after enough samples</div></div>
  <div class="ae-kpi"><div class="ae-kpi-label">Learning mode</div><div class="ae-kpi-value">{'ACTIVE' if adaptive.enabled else 'OFF'}</div><div class="ae-kpi-sub">Bounded and evidence-weighted</div></div>
</div>
<div class="ae-note"><strong>Latest review</strong><br>{escape(adaptive.last_review)}</div>
<div class="ae-chip-wrap" style="margin-top:10px">{chips}</div>
"""

def footer_html(build: str) -> str:
    return f"""
<div class="ae-footer">
  <span>AurumEdge Pro is an analytical decision-support terminal. It does not connect to a broker or place orders.</span>
  <span>Build {escape(build)} · OpenAI research · Independent XAU/USD data</span>
</div>
"""
