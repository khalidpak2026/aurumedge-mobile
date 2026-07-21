# AurumEdge background alerts

## What runs when the app is closed

The iPhone/Streamlit page cannot remain as a permanent background process. The included GitHub Actions workflow is the always-on checker. It downloads the synchronized M5 and H1 candles, derives M15/H4/D1 locally, evaluates the regular engine and the 4H-FVG strategy, and sends a deduplicated Telegram or email alert.

## Timing

- Main weekday sessions: every five minutes
- Sunday reopen window: every five minutes from 22:00 through 23:59 UTC
- GitHub schedule timing is approximate, not broker-grade low latency

## Recommended: Telegram

Add these repository Actions secrets:

- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Message the bot once before testing. Use the workflow's **Run workflow → Send a test notification** option.

## Email alternative

SMTP is supported. For Gmail, use an app password rather than your normal Google password.

## Alert events

- Primary engine starts a new executable BUY or SELL
- 4H-FVG strategy becomes TRIGGERED
- Optional FVG ARMED alert when `ALERT_FORMING_ENABLED=true`
- Duplicate notifications for the same continuous state are suppressed

The watcher uses the same market-structure, anchored-VWAP, volume-profile, macro and adaptive decision modules as the mobile app.
