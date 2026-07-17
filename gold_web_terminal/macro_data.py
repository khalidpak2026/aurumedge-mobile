from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import StringIO
from typing import Any
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests

from .models import MacroAssetSnapshot, MacroConfirmation


DXY_COMPONENTS: dict[str, float] = {
    "EUR/USD": -0.576,
    "USD/JPY": 0.136,
    "GBP/USD": -0.119,
    "USD/CAD": 0.091,
    "USD/SEK": 0.042,
    "USD/CHF": 0.036,
}
DXY_CONSTANT = 50.14348112
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AurumEdge/7.3"


@dataclass(slots=True)
class MacroSeries:
    symbol: str
    label: str
    data: pd.DataFrame
    source: str
    freshness: str


def _normalize_series(times: list[Any] | pd.Series | pd.Index, values: list[Any] | pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "time": pd.to_datetime(times, utc=True, errors="coerce"),
            "close": pd.to_numeric(pd.Series(values), errors="coerce"),
        }
    ).dropna()
    return frame.sort_values("time").drop_duplicates("time").reset_index(drop=True)


def _safe_message(payload: Any, fallback: str) -> str:
    if isinstance(payload, dict):
        value = payload.get("message") or payload.get("code") or fallback
    else:
        value = fallback
    return str(value).replace("\n", " ")[:240]


def _twelve_series(api_key: str, symbol: str, interval: str = "1h", outputsize: int = 120) -> pd.DataFrame:
    if not api_key:
        raise ValueError("No Twelve Data key")
    try:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol,
                "interval": interval,
                "outputsize": outputsize,
                "timezone": "UTC",
                "format": "JSON",
                "apikey": api_key,
            },
            timeout=20,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Twelve Data connection failed: {exc.__class__.__name__}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code == 429:
        raise RuntimeError("Twelve Data macro quota reached (HTTP 429)")
    if response.status_code >= 400:
        raise RuntimeError(f"Twelve Data macro request failed: {_safe_message(payload, f'HTTP {response.status_code}')}")
    if not isinstance(payload, dict) or payload.get("status") == "error" or "values" not in payload:
        raise RuntimeError(_safe_message(payload, "Twelve Data macro symbol unavailable"))
    values = payload["values"]
    return _normalize_series([row.get("datetime") for row in values], [row.get("close") for row in values])


def _twelve_batch_series(api_key: str, symbols: list[str], interval: str = "1h", outputsize: int = 120) -> dict[str, pd.DataFrame]:
    """Fetch several forex pairs in one HTTP request.

    Twelve Data still charges one API credit per symbol, but this exact six-pair
    DXY basket plus the two candle-source calls stays within an eight-credit
    free-plan minute when the app is loaded once.
    """
    if not api_key:
        raise ValueError("No Twelve Data key")
    symbol_param = ",".join(symbols)
    try:
        response = requests.get(
            "https://api.twelvedata.com/time_series",
            params={
                "symbol": symbol_param,
                "interval": interval,
                "outputsize": outputsize,
                "timezone": "UTC",
                "format": "JSON",
                "apikey": api_key,
            },
            timeout=25,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"Twelve Data DXY-basket connection failed: {exc.__class__.__name__}") from exc
    try:
        payload = response.json()
    except ValueError:
        payload = {}
    if response.status_code == 429:
        raise RuntimeError("Twelve Data macro quota reached while loading the DXY basket (HTTP 429)")
    if response.status_code >= 400:
        raise RuntimeError(f"DXY basket request failed: {_safe_message(payload, f'HTTP {response.status_code}')}")

    # A multi-symbol response is normally keyed by symbol. Some deployments may
    # return a single standard response when only one symbol was accepted.
    if isinstance(payload, dict) and "values" in payload and len(symbols) == 1:
        payload = {symbols[0]: payload}
    if not isinstance(payload, dict):
        raise RuntimeError("Unexpected DXY basket response")

    result: dict[str, pd.DataFrame] = {}
    errors: list[str] = []
    for symbol in symbols:
        node = payload.get(symbol)
        if node is None:
            # Be tolerant of provider-normalized keys and case differences.
            node = next((value for key, value in payload.items() if str(key).upper() == symbol.upper()), None)
        if not isinstance(node, dict) or node.get("status") == "error" or "values" not in node:
            errors.append(f"{symbol}: {_safe_message(node, 'unavailable')}")
            continue
        values = node["values"]
        frame = _normalize_series([row.get("datetime") for row in values], [row.get("close") for row in values])
        if len(frame) >= 5:
            result[symbol] = frame
        else:
            errors.append(f"{symbol}: insufficient rows")
    if len(result) != len(symbols):
        raise RuntimeError("DXY component data incomplete: " + "; ".join(errors[:3]))
    return result


def _synthetic_dxy(api_key: str) -> pd.DataFrame:
    components = _twelve_batch_series(api_key, list(DXY_COMPONENTS), "1h", 140)
    columns: list[pd.Series] = []
    for symbol, frame in components.items():
        series = frame.set_index("time")["close"].astype(float).rename(symbol)
        columns.append(series)
    aligned = pd.concat(columns, axis=1).sort_index().ffill(limit=2).dropna()
    if len(aligned) < 5:
        raise RuntimeError("DXY components could not be aligned")
    dxy = pd.Series(DXY_CONSTANT, index=aligned.index, dtype=float)
    for symbol, exponent in DXY_COMPONENTS.items():
        dxy = dxy * aligned[symbol].pow(exponent)
    return _normalize_series(dxy.index, dxy.values)


def _yahoo_series(symbol: str, interval: str = "1h", range_: str = "5d") -> pd.DataFrame:
    """Read a delayed public chart series with host fallback.

    Yahoo is used only as a best-effort public intraday source. Failure is
    expected on some networks, so callers always provide official daily
    fallbacks. No exception ever includes a credential or full request URL.
    """
    encoded = requests.utils.quote(symbol, safe="")
    errors: list[str] = []
    for host in ("query1.finance.yahoo.com", "query2.finance.yahoo.com"):
        try:
            response = requests.get(
                f"https://{host}/v8/finance/chart/{encoded}",
                params={"interval": interval, "range": range_, "includePrePost": "true", "events": "div,splits"},
                headers={"User-Agent": USER_AGENT, "Accept": "application/json,text/plain,*/*"},
                timeout=15,
            )
            if response.status_code == 429:
                errors.append(f"{host}: rate limited")
                continue
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result")
            if not result:
                errors.append(f"{host}: no chart result")
                continue
            node = result[0]
            timestamps = node.get("timestamp") or []
            quote = (node.get("indicators", {}).get("quote") or [{}])[0]
            closes = quote.get("close") or []
            frame = _normalize_series(pd.to_datetime(timestamps, unit="s", utc=True), closes)
            if len(frame) >= 2:
                return frame
            errors.append(f"{host}: insufficient observations")
        except Exception as exc:
            errors.append(f"{host}: {exc.__class__.__name__}")
    raise RuntimeError("Yahoo public chart unavailable (" + "; ".join(errors[:2]) + ")")


def _stooq_series(symbol: str = "dx.f") -> pd.DataFrame:
    """Daily public market series from Stooq, used as a DXY fallback."""
    response = requests.get(
        "https://stooq.com/q/d/l/",
        params={"s": symbol, "i": "d"},
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,*/*"},
        timeout=18,
    )
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    columns = {str(c).strip().lower(): c for c in raw.columns}
    if "date" not in columns or "close" not in columns:
        raise RuntimeError("Stooq DXY response had no Date/Close columns")
    frame = _normalize_series(raw[columns["date"]], raw[columns["close"]])
    if len(frame) < 2:
        raise RuntimeError("Stooq DXY response had insufficient rows")
    return frame.tail(90).reset_index(drop=True)


def _ecb_dxy_series() -> pd.DataFrame:
    """Construct a daily ICE-method DXY from official ECB reference rates.

    ECB rates are published as currency units per euro. Cross rates are
    transformed into the six component pairs used by the DXY formula. This is
    a daily fallback and is labelled as such in the UI.
    """
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=45)
    response = requests.get(
        "https://data-api.ecb.europa.eu/service/data/EXR/D..EUR.SP00.A",
        params={"startPeriod": start.isoformat(), "endPeriod": end.isoformat(), "format": "csvdata"},
        headers={"User-Agent": USER_AGENT, "Accept": "text/csv,application/vnd.sdmx.data+csv;version=2.0.0"},
        timeout=25,
    )
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    upper = {str(c).strip().upper(): c for c in raw.columns}
    time_col = upper.get("TIME_PERIOD") or upper.get("TIME PERIOD")
    value_col = upper.get("OBS_VALUE") or upper.get("OBS VALUE")
    currency_col = upper.get("CURRENCY")
    if not time_col or not value_col or not currency_col:
        raise RuntimeError("ECB exchange-rate response lacked required columns")
    raw = raw[[time_col, currency_col, value_col]].copy()
    raw[currency_col] = raw[currency_col].astype(str).str.upper().str.strip()
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw[time_col] = pd.to_datetime(raw[time_col], utc=True, errors="coerce")
    wanted = {"USD", "JPY", "GBP", "CAD", "SEK", "CHF"}
    raw = raw[raw[currency_col].isin(wanted)].dropna()
    pivot = raw.pivot_table(index=time_col, columns=currency_col, values=value_col, aggfunc="last").sort_index().dropna()
    if len(pivot) < 2 or not wanted.issubset(set(pivot.columns)):
        raise RuntimeError("ECB reference rates did not contain all DXY components")
    usd = pivot["USD"]  # USD per EUR
    components = pd.DataFrame(index=pivot.index)
    components["EUR/USD"] = usd
    components["USD/JPY"] = pivot["JPY"] / usd
    components["GBP/USD"] = usd / pivot["GBP"]
    components["USD/CAD"] = pivot["CAD"] / usd
    components["USD/SEK"] = pivot["SEK"] / usd
    components["USD/CHF"] = pivot["CHF"] / usd
    dxy = pd.Series(DXY_CONSTANT, index=components.index, dtype=float)
    for pair, exponent in DXY_COMPONENTS.items():
        dxy = dxy * components[pair].pow(exponent)
    return _normalize_series(dxy.index, dxy.values)


def decode_gold_h1_json(payload: str) -> pd.DataFrame:
    """Decode cached H1 JSON compatibly with pandas 2.x and 3.x."""
    frame = pd.read_json(StringIO(payload), orient="split")
    if "time" not in frame or "close" not in frame:
        raise ValueError("Cached gold H1 payload is missing time/close columns")
    frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["time", "close"]).sort_values("time").reset_index(drop=True)
    if len(frame) < 5:
        raise ValueError("Cached gold H1 payload has insufficient observations")
    return frame


def _fred_series(series_id: str) -> pd.DataFrame:
    response = requests.get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": series_id},
        headers={"User-Agent": USER_AGENT},
        timeout=18,
    )
    response.raise_for_status()
    raw = pd.read_csv(StringIO(response.text))
    raw.columns = [str(c).strip() for c in raw.columns]
    if len(raw.columns) < 2:
        raise RuntimeError(f"FRED {series_id} response had no data column")
    date_col = raw.columns[0]
    value_col = raw.columns[1]
    raw[value_col] = pd.to_numeric(raw[value_col], errors="coerce")
    raw = raw.dropna(subset=[value_col]).tail(90)
    return _normalize_series(raw[date_col], raw[value_col])


def _treasury_dgs10() -> pd.DataFrame:
    """Official daily 10-year par yield from the U.S. Treasury XML feed."""
    current_year = datetime.now(timezone.utc).year
    rows: list[tuple[Any, Any]] = []
    for year in (current_year, current_year - 1):
        response = requests.get(
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/pages/xml",
            params={"data": "daily_treasury_yield_curve", "field_tdr_date_value": str(year)},
            headers={"User-Agent": USER_AGENT},
            timeout=20,
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        for properties in root.iter():
            if not str(properties.tag).endswith("properties"):
                continue
            date_value = None
            yield_value = None
            for child in list(properties):
                local = str(child.tag).split("}")[-1]
                if local == "NEW_DATE":
                    date_value = child.text
                elif local == "BC_10YEAR":
                    yield_value = child.text
            if date_value and yield_value not in (None, ""):
                rows.append((date_value, yield_value))
        if len(rows) >= 10:
            break
    if not rows:
        raise RuntimeError("U.S. Treasury XML contained no 10-year observations")
    return _normalize_series([row[0] for row in rows], [row[1] for row in rows])


def _change_near_hours(frame: pd.DataFrame, hours: int) -> float | None:
    if frame.empty or len(frame) < 2:
        return None
    ordered = frame.sort_values("time").reset_index(drop=True)
    latest_time = pd.to_datetime(ordered["time"].iloc[-1], utc=True)
    target = latest_time - pd.Timedelta(hours=hours)
    prior = ordered[ordered["time"] <= target]
    if prior.empty:
        prior = ordered.iloc[:-1]
    if prior.empty:
        return None
    return float(ordered["close"].iloc[-1] - prior["close"].iloc[-1])


def _changes(frame: pd.DataFrame, daily_series: bool = False) -> tuple[float | None, float | None, float | None, float | None]:
    if frame.empty:
        return None, None, None, None
    frame = frame.sort_values("time").reset_index(drop=True)
    values = frame["close"].astype(float).to_numpy()
    last = values[-1]
    if daily_series:
        one = None
        four = None
        day = last - values[-2] if len(values) >= 2 else None
        base = values[-6] if len(values) >= 6 else values[0]
        pct = ((last - base) / base * 100) if len(values) >= 2 and base else None
        return one, four, day, pct
    one = _change_near_hours(frame, 1)
    four = _change_near_hours(frame, 4)
    day = _change_near_hours(frame, 24)
    base_row = frame[frame["time"] <= pd.to_datetime(frame["time"].iloc[-1], utc=True) - pd.Timedelta(hours=24)]
    base = float(base_row["close"].iloc[-1]) if not base_row.empty else float(values[0])
    pct = (day / base * 100) if day is not None and base else None
    return one, four, day, pct


def _snapshot(series: MacroSeries, is_yield: bool = False, daily_series: bool = False) -> MacroAssetSnapshot:
    frame = series.data.copy()
    if is_yield and not frame.empty and float(frame["close"].iloc[-1]) > 20:
        frame["close"] = frame["close"] / 10.0
    one, four, day, pct = _changes(frame, daily_series=daily_series)
    value = float(frame["close"].iloc[-1]) if not frame.empty else None
    if is_yield:
        threshold = 0.015
        reference = day if daily_series else four if four is not None else day
    else:
        threshold = max(0.015, abs(value or 0) * 0.00030)
        reference = four if four is not None else day
    direction = "UNAVAILABLE"
    if reference is not None:
        direction = "UP" if reference > threshold else "DOWN" if reference < -threshold else "FLAT"
    data_time = pd.to_datetime(frame["time"].iloc[-1], utc=True).isoformat() if not frame.empty else ""
    return MacroAssetSnapshot(
        symbol=series.symbol,
        label=series.label,
        value=round(value, 4) if value is not None else None,
        change_1h=round(one, 4) if one is not None else None,
        change_4h=round(four, 4) if four is not None else None,
        change_1d=round(day, 4) if day is not None else None,
        change_pct_1d=round(pct, 4) if pct is not None else None,
        direction=direction,  # type: ignore[arg-type]
        source=series.source,
        data_time=data_time,
        freshness=series.freshness,
    )


def _gold_change(frame: pd.DataFrame, hours: int) -> float | None:
    if frame.empty:
        return None
    work = frame[["time", "close"]].copy()
    work["time"] = pd.to_datetime(work["time"], utc=True, errors="coerce")
    work["close"] = pd.to_numeric(work["close"], errors="coerce")
    work = work.dropna().sort_values("time")
    return _change_near_hours(work, hours)


def fetch_macro_confirmation(
    api_key: str,
    dxy_symbol: str,
    us10y_symbol: str,
    gold_h1: pd.DataFrame,
    technical_decision: str,
) -> MacroConfirmation:
    notes: list[str] = []

    # DXY hierarchy deliberately avoids consuming the constrained Twelve Data
    # quota used by gold candles. Public intraday DXY is attempted first, then
    # two independent daily fallbacks, including an exact six-component
    # calculation from official ECB reference rates.
    dxy_daily = False
    try:
        dxy_frame = _yahoo_series("DX-Y.NYB")
        dxy_series = MacroSeries("DX-Y.NYB", "U.S. Dollar Index", dxy_frame, "YAHOO_DXY", "Intraday; may be delayed")
    except Exception as yahoo_error:
        try:
            dxy_frame = _stooq_series("dx.f")
            dxy_series = MacroSeries("DX.F", "U.S. Dollar Index futures", dxy_frame, "STOOQ_DXY_DAILY", "Daily public close")
            dxy_daily = True
            notes.append(f"Intraday DXY was unavailable; Stooq daily DXY is used: {yahoo_error}")
        except Exception as stooq_error:
            try:
                dxy_frame = _ecb_dxy_series()
                dxy_series = MacroSeries(
                    "DXY-ECB",
                    "ICE-method DXY synthetic",
                    dxy_frame,
                    "ECB_DXY_DAILY",
                    "Official ECB daily FX reference rates; exact six-component formula",
                )
                dxy_daily = True
                notes.append("Intraday DXY feeds were unavailable, so an exact daily DXY is calculated from official ECB reference rates.")
            except Exception as ecb_error:
                try:
                    dxy_frame = _fred_series("DTWEXBGS")
                    dxy_series = MacroSeries(
                        "DTWEXBGS",
                        "Broad U.S. Dollar Index fallback",
                        dxy_frame,
                        "FRED_BROAD_USD_DAILY",
                        "Official daily broad-dollar index; directional fallback, not ICE DXY",
                    )
                    dxy_daily = True
                    notes.append("All DXY feeds failed; the official Federal Reserve broad-dollar daily index is used as a directional fallback.")
                except Exception as fred_error:
                    dxy_series = MacroSeries(dxy_symbol, "U.S. Dollar Index", pd.DataFrame(columns=["time", "close"]), "UNAVAILABLE", "")
                    notes.append(
                        "DXY unavailable after independent sources: "
                        f"yahoo={yahoo_error.__class__.__name__}; stooq={stooq_error.__class__.__name__}; "
                        f"ecb={ecb_error.__class__.__name__}; fred={fred_error.__class__.__name__}"
                    )

    # Yield hierarchy: intraday market quote when reachable; otherwise official
    # U.S. Treasury daily curve, then FRED DGS10.
    yield_daily = False
    try:
        yield_frame = _yahoo_series("^TNX")
        yield_series = MacroSeries("^TNX", "U.S. 10Y yield", yield_frame, "YAHOO_TNX_FALLBACK", "Intraday; may be delayed")
    except Exception as intraday_error:
        try:
            yield_frame = _treasury_dgs10()
            yield_series = MacroSeries("UST10Y", "U.S. 10Y yield", yield_frame, "US_TREASURY_DAILY", "Official daily par yield close")
            yield_daily = True
            notes.append(f"Intraday US10Y was unavailable, so the official U.S. Treasury daily 10-year rate is used: {intraday_error}")
        except Exception as treasury_error:
            try:
                yield_frame = _fred_series("DGS10")
                yield_series = MacroSeries("DGS10", "U.S. 10Y yield", yield_frame, "FRED_DGS10_DAILY", "Official daily close; not intraday")
                yield_daily = True
                notes.append("US10Y uses the official FRED DGS10 daily series because intraday and Treasury XML feeds were unavailable.")
            except Exception as fred_error:
                yield_series = MacroSeries(us10y_symbol, "U.S. 10Y yield", pd.DataFrame(columns=["time", "close"]), "UNAVAILABLE", "")
                notes.append(f"US10Y unavailable: intraday={intraday_error}; treasury={treasury_error}; fred={fred_error}")

    dxy = _snapshot(dxy_series, False, daily_series=dxy_daily)
    us10y = _snapshot(yield_series, True, daily_series=yield_daily)

    gold_change_1h = _gold_change(gold_h1, 1)
    gold_change_4h = _gold_change(gold_h1, 4)
    gold_direction = "UNAVAILABLE"
    if gold_change_4h is not None:
        recent = gold_h1["close"].astype(float)
        scale = max(float(recent.tail(24).std(ddof=0) or 0), float(recent.iloc[-1]) * 0.00015, 0.25)
        gold_direction = "UP" if gold_change_4h > scale * 0.25 else "DOWN" if gold_change_4h < -scale * 0.25 else "FLAT"

    # Intraday macro inputs earn full weight. Official/daily fallbacks still
    # permit a decision but are intentionally capped, so confidence cannot look
    # identical to a fully live macro feed.
    dxy_points = 40 if dxy.direction != "UNAVAILABLE" and not dxy_daily else 30 if dxy.direction != "UNAVAILABLE" else 0
    yield_points = 40 if us10y.direction != "UNAVAILABLE" and not yield_daily else 30 if us10y.direction != "UNAVAILABLE" else 0
    gold_points = 20 if gold_change_4h is not None else 0
    available = dxy_points + yield_points + gold_points
    data_status = "COMPLETE" if available >= 90 else "PARTIAL" if available >= 60 else "INSUFFICIENT"

    score = 50
    reasons: list[str] = []
    conflicts: list[str] = []
    if dxy.direction == "DOWN":
        score += 20
        reasons.append("DXY is down, removing a major headwind from dollar-priced gold.")
    elif dxy.direction == "UP":
        score -= 20
        conflicts.append("DXY is up, creating a major headwind for gold.")
    if us10y.direction == "DOWN":
        score += 20
        reasons.append("The U.S. 10-year yield is down, reducing opportunity-cost pressure on gold.")
    elif us10y.direction == "UP":
        score -= 20
        conflicts.append("The U.S. 10-year yield is up, increasing opportunity-cost pressure on gold.")
    if gold_direction == "UP":
        score += 8
        reasons.append("Gold's own four-hour move is positive, confirming current upside flow.")
    elif gold_direction == "DOWN":
        score -= 8
        conflicts.append("Gold's own four-hour move is negative, confirming current downside flow.")
    score = int(max(0, min(100, score)))

    if data_status == "INSUFFICIENT":
        bias = "UNAVAILABLE"
    elif dxy.direction == "DOWN" and us10y.direction == "DOWN":
        bias = "BULLISH_GOLD"
    elif dxy.direction == "UP" and us10y.direction == "UP":
        bias = "BEARISH_GOLD"
    elif score >= 64:
        bias = "BULLISH_GOLD"
    elif score <= 36:
        bias = "BEARISH_GOLD"
    else:
        bias = "MIXED"

    # Strict execution gate: DXY/yield context and gold's own four-hour flow
    # must point in the same direction before the macro layer returns CONFIRM.
    # This prevents an apparently supportive dollar/yield pair from confirming
    # a trade while gold itself is already moving the other way.
    bullish_pair = dxy.direction == "DOWN" and us10y.direction == "DOWN"
    bearish_pair = dxy.direction == "UP" and us10y.direction == "UP"
    if data_status == "INSUFFICIENT":
        gate = "UNAVAILABLE"
    elif technical_decision == "BUY":
        if bearish_pair or (gold_direction == "DOWN" and bias != "BULLISH_GOLD"):
            gate = "CONFLICT"
        elif bullish_pair and gold_direction == "UP":
            gate = "CONFIRM"
        elif bias == "BULLISH_GOLD" and gold_direction == "UP" and available >= 80:
            gate = "CONFIRM"
        else:
            gate = "NEUTRAL"
    elif technical_decision == "SELL":
        if bullish_pair or (gold_direction == "UP" and bias != "BEARISH_GOLD"):
            gate = "CONFLICT"
        elif bearish_pair and gold_direction == "DOWN":
            gate = "CONFIRM"
        elif bias == "BEARISH_GOLD" and gold_direction == "DOWN" and available >= 80:
            gate = "CONFIRM"
        else:
            gate = "NEUTRAL"
    else:
        gate = "NEUTRAL"

    if gate == "CONFIRM":
        reasons.append(f"DXY/yield/gold-flow conditions confirm the technical {technical_decision} direction.")
    elif gate == "CONFLICT":
        conflicts.append(f"DXY/yield/gold-flow conditions conflict with the technical {technical_decision} direction.")
    elif gate == "UNAVAILABLE":
        conflicts.append("Macro coverage is insufficient for a high-conviction entry decision.")

    if dxy.direction == us10y.direction and dxy.direction in {"UP", "DOWN"}:
        alignment = f"DXY and US10Y both {dxy.direction}"
    elif dxy.direction == "UNAVAILABLE" or us10y.direction == "UNAVAILABLE":
        alignment = "Macro pair incomplete"
    else:
        alignment = "DXY and US10Y mixed"

    return MacroConfirmation(
        dxy=dxy,
        us10y=us10y,
        gold_change_1h=round(gold_change_1h, 4) if gold_change_1h is not None else None,
        gold_change_4h=round(gold_change_4h, 4) if gold_change_4h is not None else None,
        gold_direction=gold_direction,  # type: ignore[arg-type]
        macro_bias=bias,  # type: ignore[arg-type]
        confirmation_score=score,
        coverage_score=available,
        data_status=data_status,  # type: ignore[arg-type]
        alignment=alignment,
        gate=gate,  # type: ignore[arg-type]
        reasons=reasons,
        conflicts=conflicts,
        news_risk="UNAVAILABLE",
        notes=notes,
    )
