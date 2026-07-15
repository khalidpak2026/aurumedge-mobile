from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


Direction = Literal["bullish", "bearish", "neutral"]
MarketDecision = Literal["BUY", "SELL", "STUCK", "TRAP"]


class IndicatorSnapshot(BaseModel):
    timeframe: str
    timestamp: str
    close: float
    ema9: float | None = None
    ema20: float | None = None
    ema50: float | None = None
    ema200: float | None = None
    ema20_slope_atr: float | None = None
    ema50_slope_atr: float | None = None
    rsi14: float | None = None
    atr14: float | None = None
    atr_pct: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_hist: float | None = None
    macd_hist_slope: float | None = None
    adx14: float | None = None
    plus_di: float | None = None
    minus_di: float | None = None
    stoch_k: float | None = None
    stoch_d: float | None = None
    bb_upper: float | None = None
    bb_mid: float | None = None
    bb_lower: float | None = None
    bb_width_pct: float | None = None
    vwap: float | None = None
    obv: float | None = None
    obv_slope: float | None = None
    volume_zscore: float | None = None
    volume_ratio: float | None = None
    volume_delta_proxy: float | None = None
    choppiness14: float | None = None
    supertrend: float | None = None
    supertrend_direction: Direction = "neutral"
    donchian_high: float | None = None
    donchian_low: float | None = None
    breakout_up: bool = False
    breakout_down: bool = False
    compression: bool = False
    trend: Direction = "neutral"
    momentum: Direction = "neutral"
    directional_score: float = Field(default=0.0, ge=-100, le=100)


class LiquiditySnapshot(BaseModel):
    timeframe: str
    previous_day_high: float | None = None
    previous_day_low: float | None = None
    current_day_high: float | None = None
    current_day_low: float | None = None
    recent_swing_highs: list[float] = Field(default_factory=list)
    recent_swing_lows: list[float] = Field(default_factory=list)
    equal_highs: list[float] = Field(default_factory=list)
    equal_lows: list[float] = Field(default_factory=list)
    support_zones: list[dict] = Field(default_factory=list)
    resistance_zones: list[dict] = Field(default_factory=list)
    bullish_fvgs: list[dict] = Field(default_factory=list)
    bearish_fvgs: list[dict] = Field(default_factory=list)
    sweep_above: float | None = None
    sweep_below: float | None = None
    trap_type: Literal["bull_trap", "bear_trap", "none"] = "none"
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    point_of_control: float | None = None
    value_area_high: float | None = None
    value_area_low: float | None = None


class TradeSetup(BaseModel):
    side: Literal["BUY", "SELL"]
    status: Literal["ENTER", "NO_TRADE"]
    confidence: int = Field(ge=0, le=100)
    entry_low: float
    entry_high: float
    entry_type: str
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: float
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: float
    valid_until: str
    invalidation: str
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class TechnicalReport(BaseModel):
    symbol: str
    data_time: str
    last_price: float
    data_source: Literal["TWELVE_DATA", "CSV_UPLOAD", "DEMO"]
    regime: Literal[
        "bullish_trend",
        "bearish_trend",
        "breakout_up",
        "breakout_down",
        "stuck_range",
        "liquidity_trap",
        "volatile_bullish",
        "volatile_bearish",
    ]
    market_state: MarketDecision
    recommendation: MarketDecision
    signal_label: str
    confidence: int = Field(ge=0, le=100)
    buy_score: int = Field(ge=0, le=100)
    sell_score: int = Field(ge=0, le=100)
    trend_strength: int = Field(ge=0, le=100)
    volatility_state: Literal["low", "normal", "high", "extreme"]
    trap_reason: str = ""
    indicators: list[IndicatorSnapshot]
    liquidity: list[LiquiditySnapshot]
    active_setup: TradeSetup | None = None
    buy_setup: TradeSetup
    sell_setup: TradeSetup
    data_quality_notes: list[str] = Field(default_factory=list)


class ChartVisionAnalysis(BaseModel):
    visual_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED", "UNREADABLE"]
    confidence: int = Field(ge=0, le=100)
    visible_timeframe: str
    visible_structure: list[str]
    indicator_observations: list[str]
    confirmations: list[str]
    conflicts: list[str]
    limitations: list[str]


class AIAnalysis(BaseModel):
    market_bias: Literal["BULLISH", "BEARISH", "NEUTRAL", "MIXED"]
    confidence: int = Field(ge=0, le=100)
    decision: MarketDecision
    macro_risk: Literal["LOW", "MEDIUM", "HIGH", "EXTREME"]
    executive_summary: str
    technical_read: list[str]
    macro_news_read: list[str]
    chart_screenshot_read: list[str]
    confirmations: list[str]
    conflicts: list[str]
    key_invalidation: list[str]
    event_risks: list[str]
    data_quality_notes: list[str]
