# GitHub update guide — Mobile v5.7

## 1. Back up the adaptive brain

In the current mobile app open **BRAIN** and download `adaptive_state.json`. Keep the existing GitHub `data/adaptive_state.json`; do not replace it with the package copy unless restoring a backup deliberately.

## 2. Replace repository-root files

At the root of `khalidpak2026/aurumedge-mobile`, replace:

- `mobile_app.py`
- `signal_watcher.py`
- `requirements.txt`
- `requirements-alerts.txt`
- `VERSION.txt`
- `README.md`
- `README_ALERTS.md`
- `.env.example`
- `.gitignore`
- `GITHUB_UPDATE_GUIDE_V5_7.md`

Do not upload `.env`, `secrets.toml`, `__pycache__`, `.pyc`, `.pytest_cache`, or the ZIP file.

## 3. Replace the Python package

Open the GitHub folder `gold_web_terminal` and upload every `.py` file from the package's own `gold_web_terminal` folder.

Confirm these new/updated files are inside that folder:

- `market_context.py`
- `indicators.py`
- `strategy.py`
- `adaptive_engine.py`
- `liquidity.py`
- `mobile_svg.py`
- `models.py`
- `config.py`

Do not place those files at the repository root.

## 4. Replace the workflow

Upload the package file to this exact path:

`.github/workflows/aurumedge_signal_watch.yml`

The workflow checks every five minutes Monday-Friday UTC and during the first two hours of the Sunday reopen window. Scheduled delivery is approximate and can be delayed by GitHub.

## 5. Keep the existing secrets

No new key is required. Keep these GitHub Actions secrets:

- `TWELVE_DATA_API_KEY`
- `TELEGRAM_BOT_TOKEN`
- `TELEGRAM_CHAT_ID`

Email secrets remain optional.

Keep the existing Streamlit secret. Add or update these only when missing:

```toml
AUTO_REFRESH_ENABLED = "true"
AUTO_REFRESH_SECONDS = "1200"
FREE_PLAN_MODE = "true"
PROVIDER_DAILY_LIMIT = "800"
CLOUD_WATCHER_MINUTES = "5"
```

When free-plan mode is enabled, the app automatically prevents a shorter UI interval from exhausting the daily allowance.

## 6. Commit and redeploy

Suggested commit message:

`Add market structure AVWAP and volume profile v5.7`

Streamlit normally redeploys automatically. If the old build remains visible, use **Streamlit Cloud → My apps → Reboot app**.

## 7. Verify

The app header must show:

`5.7.0-mobile-avwap-volume-profile`

The chart and indicator tables should show:

- Market structure
- Anchored VWAP
- Profile POC, VAH and VAL
- Profile acceptance

The alert workflow should remain enabled in the GitHub **Actions** tab.
