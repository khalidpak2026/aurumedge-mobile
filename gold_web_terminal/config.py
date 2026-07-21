from __future__ import annotations

import os
import math
from dataclasses import dataclass


def _as_bool(value: str, default: bool = True) -> bool:
    if value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _as_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _free_plan_safe_interval(daily_limit: int, watcher_minutes: int) -> int:
    """Return a conservative UI refresh interval when the 5-minute cloud watcher is active.

    The candle loader uses two provider credits per synchronized cycle (M5 and
    H1; M15/H4/D1 are derived locally). Ten percent of the daily allowance is
    reserved for manual checks and retries.
    """
    limit = max(100, int(daily_limit))
    watcher = max(5, int(watcher_minutes))
    watcher_runs = math.ceil(1440 / watcher)
    watcher_credits = watcher_runs * 2
    reserve = max(40, int(limit * 0.10))
    available_ui_credits = max(2, limit - watcher_credits - reserve)
    ui_runs = max(1, available_ui_credits // 2)
    seconds = math.ceil((86400 / ui_runs) / 60) * 60
    return max(60, int(seconds))


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

    # Practical CFD risk model. These are defaults; the Windows UI can override them.
    account_balance: float
    requested_lot: float
    contract_size: float
    lot_step: float
    min_lot: float
    maximum_risk_dollars: float
    spread_price: float
    slippage_price: float
    minimum_stop_atr: float
    maximum_stop_atr: float
    target_profile: str

    # Controlled adaptive-learning state.
    adaptive_learning: bool
    adaptive_state_path: str
    adaptive_min_samples: int
    adaptive_horizon_bars: int
    adaptive_max_weight_change: float

    # Macro confirmation inputs.
    macro_enabled: bool
    macro_required_for_entry: bool
    dxy_symbol: str
    us10y_symbol: str
    macro_cache_minutes: int

    # Automatic full-market synchronization.
    auto_refresh_enabled: bool
    auto_refresh_seconds: int
    free_plan_mode: bool
    provider_daily_limit: int
    cloud_watcher_minutes: int

    # Specialist strategy and optional alert delivery.
    h4_fvg_strategy_enabled: bool
    local_alerts_enabled: bool
    alert_state_path: str

    @classmethod
    def from_env(cls) -> "Settings":
        free_plan_mode = _as_bool(os.getenv("FREE_PLAN_MODE", "true"), True)
        provider_daily_limit = max(100, _as_int("PROVIDER_DAILY_LIMIT", 800))
        cloud_watcher_minutes = max(5, _as_int("CLOUD_WATCHER_MINUTES", 5))
        requested_refresh = max(60, _as_int("AUTO_REFRESH_SECONDS", 1200))
        if free_plan_mode:
            requested_refresh = max(
                requested_refresh,
                _free_plan_safe_interval(provider_daily_limit, cloud_watcher_minutes),
            )
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.5"),
            twelve_data_api_key=os.getenv("TWELVE_DATA_API_KEY", ""),
            market_symbol=os.getenv("MARKET_SYMBOL", "XAU/USD"),
            tradingview_symbol=os.getenv("TRADINGVIEW_SYMBOL", "OANDA:XAUUSD"),
            bars_per_timeframe=max(250, _as_int("BARS_PER_TIMEFRAME", 500)),
            risk_percent=max(0.05, _as_float("RISK_PERCENT", 1.0)),
            journal_path=os.getenv("JOURNAL_PATH", "data/trade_journal.csv"),
            auto_research=_as_bool(os.getenv("AUTO_RESEARCH", "false"), False),
            research_cache_minutes=max(5, _as_int("RESEARCH_CACHE_MINUTES", 20)),
            account_balance=max(1.0, _as_float("ACCOUNT_BALANCE", 10000.0)),
            requested_lot=max(0.0, _as_float("REQUESTED_LOT", 0.10)),
            contract_size=max(0.01, _as_float("XAU_CONTRACT_SIZE", 100.0)),
            lot_step=max(0.001, _as_float("LOT_STEP", 0.01)),
            min_lot=max(0.001, _as_float("MIN_LOT", 0.01)),
            maximum_risk_dollars=max(0.0, _as_float("MAXIMUM_RISK_DOLLARS", 0.0)),
            spread_price=max(0.0, _as_float("SPREAD_PRICE", 0.50)),
            slippage_price=max(0.0, _as_float("SLIPPAGE_PRICE", 0.20)),
            minimum_stop_atr=max(0.25, _as_float("MINIMUM_STOP_ATR", 0.55)),
            maximum_stop_atr=max(0.65, _as_float("MAXIMUM_STOP_ATR", 1.55)),
            target_profile=os.getenv("TARGET_PROFILE", "balanced").strip().lower(),
            adaptive_learning=_as_bool(os.getenv("ADAPTIVE_LEARNING", "true"), True),
            adaptive_state_path=os.getenv("ADAPTIVE_STATE_PATH", "data/adaptive_state.json"),
            adaptive_min_samples=max(8, _as_int("ADAPTIVE_MIN_SAMPLES", 20)),
            adaptive_horizon_bars=max(4, _as_int("ADAPTIVE_HORIZON_BARS", 12)),
            adaptive_max_weight_change=max(0.01, min(0.20, _as_float("ADAPTIVE_MAX_WEIGHT_CHANGE", 0.05))),
            macro_enabled=_as_bool(os.getenv("MACRO_ENABLED", "true"), True),
            macro_required_for_entry=_as_bool(os.getenv("MACRO_REQUIRED_FOR_ENTRY", "true"), True),
            dxy_symbol=os.getenv("DXY_SYMBOL", "DXY"),
            us10y_symbol=os.getenv("US10Y_SYMBOL", "US10Y"),
            macro_cache_minutes=max(2, _as_int("MACRO_CACHE_MINUTES", 10)),
            auto_refresh_enabled=_as_bool(os.getenv("AUTO_REFRESH_ENABLED", "true"), True),
            auto_refresh_seconds=requested_refresh,
            free_plan_mode=free_plan_mode,
            provider_daily_limit=provider_daily_limit,
            cloud_watcher_minutes=cloud_watcher_minutes,
            h4_fvg_strategy_enabled=_as_bool(os.getenv("H4_FVG_STRATEGY_ENABLED", "true"), True),
            local_alerts_enabled=_as_bool(os.getenv("LOCAL_ALERTS_ENABLED", "false"), False),
            alert_state_path=os.getenv("ALERT_STATE_PATH", "data/alert_state.json"),
        )
