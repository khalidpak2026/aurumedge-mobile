from __future__ import annotations

import os
from dataclasses import dataclass


def _as_bool(value: str, default: bool = True) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class Settings:
    openai_api_key: str
    openai_model: str
    twelve_data_api_key: str
    market_symbol: str
    tradingview_symbol: str
    bars_per_timeframe: int
    risk_percent: float
    journal_path: str
    auto_research: bool
    research_cache_minutes: int

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.6-luna"),
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
            market_symbol=os.getenv("MARKET_SYMBOL", "XAU/USD"),
            tradingview_symbol=os.getenv("TRADINGVIEW_SYMBOL", "OANDA:XAUUSD"),
            bars_per_timeframe=max(250, int(os.getenv("BARS_PER_TIMEFRAME", "500"))),
            risk_percent=float(os.getenv("RISK_PERCENT", "1.0")),
            journal_path=os.getenv("JOURNAL_PATH", "data/trade_journal.csv"),
            auto_research=_as_bool(os.getenv("AUTO_RESEARCH", "true"), True),
            research_cache_minutes=max(5, int(os.getenv("RESEARCH_CACHE_MINUTES", "20"))),
        )
