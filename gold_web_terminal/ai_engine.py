from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from typing import Any

from .models import AIAnalysis, ChartVisionAnalysis, TechnicalReport

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore


def _client(api_key: str):
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is missing.")
    return OpenAI(api_key=api_key, base_url="https://api.openai.com/v1")


def _extract_sources(response: Any) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for content in getattr(item, "content", []) or []:
            for annotation in getattr(content, "annotations", []) or []:
                if getattr(annotation, "type", None) != "url_citation":
                    continue
                url = getattr(annotation, "url", "")
                title = getattr(annotation, "title", "") or url
                if url and url not in seen:
                    seen.add(url)
                    sources.append({"title": title, "url": url})
    return sources


def research_gold_news(api_key: str, model: str, city: str = "Dubai", country: str = "AE") -> dict[str, Any]:
    client = _client(api_key)
    now = datetime.now(timezone.utc).isoformat()
    prompt = f"""
You are the real-time macro and news researcher for a professional XAU/USD dashboard.
Current UTC time: {now}.

Search the web for confirmed gold-relevant developments from the last 24 hours and scheduled risks during the next 48 hours. Prioritize official or highly reputable sources and cover:
- US dollar and broad dollar direction
- US Treasury nominal and real-yield expectations
- Federal Reserve decisions, speeches and rate expectations
- CPI, PCE, employment, GDP, retail sales and other high-impact US data
- geopolitical and safe-haven developments
- central-bank buying, ETF flows and physical-demand news when genuinely current
- exact upcoming high-impact event times when available

Return a compact memo with: confirmed facts, bullish gold drivers, bearish gold drivers, event-risk windows, and an overall macro risk rating. Never invent a live gold price or technical level. The separate deterministic engine supplies price, entry, SL and TP.
""".strip()
    response = client.responses.create(
        model=model,
        reasoning={"effort": "medium"},
        tools=[
            {
                "type": "web_search",
                "user_location": {
                    "type": "approximate",
                    "country": country,
                    "city": city,
                    "region": city,
                },
                "search_context_size": "medium",
            }
        ],
        input=prompt,
    )
    return {
        "text": response.output_text,
        "sources": _extract_sources(response),
        "response_id": getattr(response, "id", None),
        "created_at": now,
    }


def analyze_chart_screenshot(
    api_key: str,
    model: str,
    image_bytes: bytes,
    mime_type: str,
    technical: TechnicalReport,
) -> ChartVisionAnalysis:
    client = _client(api_key)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    report_json = json.dumps(technical.model_dump(mode="json"), ensure_ascii=False)
    prompt = f"""
Inspect this TradingView screenshot as a secondary visual check for XAU/USD.
The deterministic report below is authoritative for numerical levels:
{report_json}

Read only visible information. Identify market structure, support/resistance, liquidity sweeps, visible EMA/volume/momentum evidence and whether the image confirms or conflicts with the report. Do not create replacement entry, SL or TP numbers from pixels. Lower confidence when the scale or timeframe is unreadable.
""".strip()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:{mime_type};base64,{encoded}", "detail": "high"},
                ],
            }
        ],
        text_format=ChartVisionAnalysis,
    )
    if response.output_parsed is None:
        raise RuntimeError("The model did not return structured screenshot analysis.")
    return response.output_parsed


def synthesize_ai_analysis(
    api_key: str,
    model: str,
    technical: TechnicalReport,
    research_text: str,
    chart_vision: ChartVisionAnalysis | None = None,
) -> AIAnalysis:
    client = _client(api_key)
    technical_json = json.dumps(technical.model_dump(mode="json"), ensure_ascii=False, indent=2)
    visual_json = json.dumps(chart_vision.model_dump(mode="json"), ensure_ascii=False, indent=2) if chart_vision else "No screenshot supplied."
    system = """
You are the independent risk-review layer in a professional GOLD/XAUUSD dashboard.
The deterministic OHLC engine owns all numerical entry, stop and target levels. Never change or invent them.
The final decision must be BUY, SELL, STUCK or TRAP; never output WAIT.
- BUY/SELL means the directional technical setup remains usable.
- STUCK means range/compression makes an entry unattractive.
- TRAP means a false break, liquidity sweep, major conflict or immediate event risk makes the apparent direction unreliable.
Do not promise profit or describe any setup as perfect.
""".strip()
    user = f"""
TECHNICAL REPORT:
{technical_json}

CURRENT WEB RESEARCH:
{research_text}

OPTIONAL CHART SCREENSHOT REVIEW:
{visual_json}

Review alignment among EMA structure and slope, Supertrend, MACD, RSI, ATR, ADX/DMI, VWAP, volume/activity, Choppiness, Donchian breakout, support/resistance, volume profile and liquidity. Confirm the supplied BUY/SELL decision or classify the market as STUCK/TRAP when the evidence warrants it. Keep the output concise and actionable without changing any numerical level.
""".strip()
    response = client.responses.parse(
        model=model,
        reasoning={"effort": "medium"},
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        text_format=AIAnalysis,
    )
    if response.output_parsed is None:
        raise RuntimeError("The model did not return a structured AI analysis.")
    return response.output_parsed
