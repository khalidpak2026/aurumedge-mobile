# Quick GitHub update checklist

## Upload 1 — repository home page

At `khalidpak2026/aurumedge-mobile`, choose **Add file → Upload files** and upload:

- `mobile_app.py`
- `requirements.txt`
- `VERSION.txt`
- `README.md`
- `GITHUB_UPDATE_GUIDE.md`
- `QUICK_UPDATE_CHECKLIST.md`
- `.gitignore`

Do not upload `.env` or a real `secrets.toml`.

## Upload 2 — inside `gold_web_terminal`

Open the GitHub folder `gold_web_terminal`, choose **Add file → Upload files**, and upload every `.py` file from the package's `gold_web_terminal` folder.

Important: `strategy.py`, `macro_data.py`, `market_data.py`, `config.py`, `models.py`, `indicators.py`, `liquidity.py` and `mobile_svg.py` must appear inside `gold_web_terminal/`.

## Adaptive brain

Keep your existing `data/adaptive_state.json` if it contains learned results. Upload the blank file from this package only for a fresh reset.

## Commit

Use: `Add 90-second all-timeframe mobile sync`

## Confirm

After Streamlit redeploys, the header must show:

`5.4.0-mobile-all-timeframe-sync`

The page must show a 90-second switch, countdown, and **SYNC ALL TIMEFRAMES** button.
