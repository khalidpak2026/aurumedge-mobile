# AurumEdge Adaptive Mobile v5.7

Mobile XAU/USD decision-support terminal using:

- Market structure: HH/HL, LH/LL, BOS and CHOCH
- Automatically anchored VWAP
- Volume Profile: POC, VAH, VAL, high-volume nodes and value acceptance
- MACD, RSI and ADX/DMI momentum confirmation
- DXY, US10Y and Gold 4H macro gate
- Capital-preservation adaptive learning
- Separate 4H Candle + M15 FVG specialist strategy
- Telegram/email alerts from an always-on GitHub Actions watcher

## Indicator hierarchy

Market structure, anchored VWAP and volume profile are the primary directional layer. EMA is retained only as secondary context. Raw time-volume is not used as a standalone BUY/SELL vote; XAU/USD tick/activity volume is converted into a price-distribution profile.

## Background operation

The Streamlit/iPhone interface updates only while it is open. The included GitHub Actions watcher performs the independent background checks and sends alerts even when the app is closed or the laptop is off.

The free-plan profile uses:

- Background watcher every 5 minutes
- Open-app synchronization every 20 minutes
- Two market-data credits per synchronized cycle
- M15, H4 and D1 generated locally from M5 and H1
- Approximately 720 credits/day if both watcher and open app run continuously
- Approximately 80 credits/day reserved for retries and manual checks

## Safety

This is analytical decision support, not a broker connection. It cannot guarantee profitable signals and does not execute trades.
