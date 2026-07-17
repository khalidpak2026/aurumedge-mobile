# AurumEdge Adaptive Mobile v5.4

A mobile-first Streamlit XAU/USD decision terminal using the same all-timeframe, adaptive, macro-confirmed and risk-controlled engine as AurumEdge Adaptive Windows v7.5.

## All-timeframe decision process

The selected timeframe changes only the chart shown on screen. The final market decision always compares all five timeframes:

- **D1:** dominant regime and large support/resistance
- **H4:** major trend and supply/demand
- **H1:** active directional structure
- **M15:** breakout, pullback, liquidity and momentum confirmation
- **M5:** entry timing

A noisy M5 or M15 move cannot independently reverse the H1/H4/D1 conclusion. Lower-timeframe opposition is labelled as a pullback or mixed market unless structure confirms a real reversal.

## 90-second synchronization

- Auto-sync is enabled by default every 90 seconds.
- One synchronization updates all timeframes together.
- M5 and H1 are downloaded from the market-data provider.
- M15 is derived locally from M5.
- H4 and D1 are derived locally from H1.
- Indicator and liquidity calculations for all timeframes are stored in memory.
- Switching the visible chart between M5, M15, H1, H4 and D1 is therefore much faster and does not spend an additional API request.
- A visible countdown and manual **SYNC ALL TIMEFRAMES** button are included.

## Market-state corrections

The corrected state machine distinguishes:

- BUY trend or confirmed bullish reversal
- SELL trend or confirmed bearish continuation
- Pullback against the higher-timeframe trend
- STUCK/mixed range
- A short-lived unresolved liquidity TRAP

TRAP expires after the relevant candles or immediately when price displacement and structure confirm a direction. Macro conflict produces STUCK rather than keeping the terminal trapped indefinitely.

## Macro confirmation

The main page shows:

- DXY value and direction
- U.S. 10-year yield value and direction
- Gold four-hour flow calculated from current H1 candles
- Macro alignment, data coverage and decision gate

DXY and US10Y are cached for about 10 minutes. Gold four-hour flow and the final gate are recalculated every 90-second gold sync.

## Adaptive brain and realistic risk

- Completed signals are evaluated after their outcome window.
- Indicator reliability changes are bounded and require evidence.
- Target distances adapt gradually to historical favourable movement.
- Stops remain beyond real structure and ATR noise.
- If the requested lot risks too much, the app recommends a smaller lot or blocks the trade instead of placing an unrealistically close stop.

## Streamlit Secrets

Your existing `TWELVE_DATA_API_KEY` and other Secrets remain valid. Optional controls:

```toml
AUTO_REFRESH_ENABLED = "true"
AUTO_REFRESH_SECONDS = "90"
MACRO_CACHE_MINUTES = "10"
```

OpenAI research remains optional and off by default.

## Adaptive-state persistence

Streamlit Cloud storage may reset after redeployment. Use the **BRAIN** tab to download and restore `adaptive_state.json`.

## Safety

This is an analysis-only application. It is not connected to a broker and cannot execute orders. No classifier can guarantee profitable trades. Verify the broker quote, spread and GOLD contract size before using any level or position-size estimate.
