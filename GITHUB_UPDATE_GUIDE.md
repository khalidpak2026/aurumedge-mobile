# Update the existing AurumEdge iPhone app to v5.4

Repository: `khalidpak2026/aurumedge-mobile`

This update adds the same all-timeframe synchronization and corrected market-state engine as Windows v7.5.

## Before uploading

Open the current mobile app, go to **BRAIN**, and download `adaptive_state.json` if it has accumulated useful trade reviews. Streamlit Cloud storage may reset during deployment.

## 1. Upload repository-root files

On the GitHub repository home page choose:

`Add file` → `Upload files`

Upload these files from the extracted v5.4 folder:

- `mobile_app.py`
- `requirements.txt`
- `VERSION.txt`
- `.gitignore`
- `README.md`
- `GITHUB_UPDATE_GUIDE.md`

You may also upload `.streamlit/config.toml`. Do not upload a real `.env`, `secrets.toml`, `.venv`, `__pycache__`, or `.pyc` file.

## 2. Replace the Python package inside the correct folder

Open the existing GitHub folder:

`gold_web_terminal`

Choose:

`Add file` → `Upload files`

Upload every `.py` file from the update package's own `gold_web_terminal` folder. These files must remain inside `gold_web_terminal`, especially:

- `gold_web_terminal/strategy.py`
- `gold_web_terminal/indicators.py`
- `gold_web_terminal/liquidity.py`
- `gold_web_terminal/models.py`
- `gold_web_terminal/macro_data.py`
- `gold_web_terminal/market_data.py`
- `gold_web_terminal/config.py`
- `gold_web_terminal/mobile_svg.py`
- `gold_web_terminal/adaptive_engine.py`
- `gold_web_terminal/risk_engine.py`

## 3. Preserve or initialize adaptive state

At the repository root, keep:

`data/adaptive_state.json`

To preserve your learned state, upload the backup downloaded from the BRAIN tab instead of replacing it with the blank initial file.

## 4. Commit

Use this commit message:

`Add 90-second all-timeframe mobile sync`

Streamlit should redeploy automatically.

## 5. Secrets

Your existing Streamlit Secrets continue to work. No API key needs to be entered again.

Optional settings:

```toml
AUTO_REFRESH_ENABLED = "true"
AUTO_REFRESH_SECONDS = "90"
MACRO_CACHE_MINUTES = "10"
```

Gold M5 and H1 candles refresh every 90 seconds. M15, H4 and D1 are rebuilt locally. DXY and US10Y remain cached for about 10 minutes, while Gold 4H flow is recalculated on every cycle.

## 6. Confirm the deployment

The header must display:

`5.4.0-mobile-all-timeframe-sync`

The control area must include:

- Chart timeframe selector
- 90-second auto-sync switch
- Countdown showing `ALL TF · ...s`
- `SYNC ALL TIMEFRAMES` button

The SIGNAL tab must say that the final decision compares:

- D1 regime
- H4 trend
- H1 structure
- M15 confirmation
- M5 timing

Changing the visible chart timeframe must not make a new provider request or change the decision by itself.

## 7. If Streamlit does not update

In Streamlit Community Cloud:

`My apps` → app menu → `Reboot app`

Then reopen the existing iPhone app URL and refresh Safari once.
