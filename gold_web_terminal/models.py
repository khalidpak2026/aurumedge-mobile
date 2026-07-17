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
    impulse_1_atr: float | None = None
    impulse_3_atr: float | None = None
    close_location: float | None = None
    structure_break_up: bool = False
    structure_break_down: bool = False


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
    sweep_above_age: int | None = None
    sweep_below_age: int | None = None
    trap_type: Literal["bull_trap", "bear_trap", "none"] = "none"
    nearest_support: float | None = None
    nearest_resistance: float | None = None
    point_of_control: float | None = None
    value_area_high: float | None = None
    value_area_low: float | None = None


class MacroAssetSnapshot(BaseModel):
    symbol: str
    label: str
    value: float | None = None
    change_1h: float | None = None
    change_4h: float | None = None
    change_1d: float | None = None
    change_pct_1d: float | None = None
    direction: Literal["UP", "DOWN", "FLAT", "UNAVAILABLE"] = "UNAVAILABLE"
    source: str = ""
    data_time: str = ""
    freshness: str = ""


class MacroConfirmation(BaseModel):
    dxy: MacroAssetSnapshot
    us10y: MacroAssetSnapshot
    gold_change_1h: float | None = None
    gold_change_4h: float | None = None
    gold_direction: Literal["UP", "DOWN", "FLAT", "UNAVAILABLE"] = "UNAVAILABLE"
    macro_bias: Literal["BULLISH_GOLD", "BEARISH_GOLD", "MIXED", "UNAVAILABLE"] = "UNAVAILABLE"
    confirmation_score: int = Field(default=50, ge=0, le=100)
    coverage_score: int = Field(default=0, ge=0, le=100)
    data_status: Literal["COMPLETE", "PARTIAL", "INSUFFICIENT"] = "INSUFFICIENT"
    alignment: str = "Macro pair unavailable"
    gate: Literal["CONFIRM", "NEUTRAL", "CONFLICT", "UNAVAILABLE"] = "UNAVAILABLE"
    reasons: list[str] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    news_risk: Literal["LOW", "MEDIUM", "HIGH", "EXTREME", "UNAVAILABLE"] = "UNAVAILABLE"
    notes: list[str] = Field(default_factory=list)


class PositionRiskPlan(BaseModel):
    account_balance: float
    risk_percent: float
    risk_budget: float
    requested_lot: float
    recommended_lot: float
    maximum_safe_lot: float
    contract_size: float
    lot_step: float
    stop_distance: float
    estimated_loss_requested_lot: float
    estimated_loss_recommended_lot: float
    status: Literal["OK", "REDUCE_LOT", "NO_TRADE"]
    message: str


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
    stop_basis: str = ""
    target_basis: str = ""
    management_plan: list[str] = Field(default_factory=list)
    risk_plan: PositionRiskPlan | None = None
    rationale: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class AdaptiveLearningSummary(BaseModel):
    enabled: bool = True
    reviewed_signals: int = 0
    wins: int = 0
    losses: int = 0
    timeouts: int = 0
    win_rate: float = 0.0
    indicator_weights: dict[str, float] = Field(default_factory=dict)
    indicator_samples: dict[str, int] = Field(default_factory=dict)
    target_r_multipliers: dict[str, float] = Field(default_factory=lambda: {"tp1": 0.8, "tp2": 1.3, "tp3": 1.8})
    last_review: str = "No completed signals have been reviewed yet."
    safeguards: list[str] = Field(default_factory=list)


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
    macro: MacroConfirmation | None = None
    adaptive: AdaptiveLearningSummary | None = None
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
