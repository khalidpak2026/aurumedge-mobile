# Update the existing iPhone Streamlit app

Repository: `khalidpak2026/aurumedge-mobile`

## 1. Back up the current adaptive state

The previous mobile version did not have the full adaptive brain, so there may be nothing to preserve. For later updates, use the BRAIN tab and download `adaptive_state.json` first.

## 2. Replace repository-root files

At the GitHub repository home page choose:

`Add file` → `Upload files`

Upload these files from the extracted v5.3 package:

- `mobile_app.py`
- `requirements.txt`
- `VERSION.txt`
- `.gitignore`
- `README.md`
- `GITHUB_UPDATE_GUIDE.md`

Do not upload `.env`, `.venv`, `__pycache__`, or `.pyc` files.

## 3. Replace the Python package files in the correct folder

Open the existing GitHub folder:

`gold_web_terminal`

Choose:

`Add file` → `Upload files`

Upload every `.py` file from the extracted package's own `gold_web_terminal` folder. Confirm that GitHub shows files such as:

- `gold_web_terminal/adaptive_engine.py`
- `gold_web_terminal/macro_data.py`
- `gold_web_terminal/risk_engine.py`
- `gold_web_terminal/mobile_svg.py`
- `gold_web_terminal/strategy.py`
- `gold_web_terminal/models.py`

These files must be inside `gold_web_terminal`, not at the repository root.

## 4. Upload the initial adaptive data folder

At the repository root upload the `data` folder containing:

`data/adaptive_state.json`

This is the initial state. Future accumulated learning should be backed up from the mobile app before a redeployment.

## 5. Commit

Use the commit message:

`Add adaptive macro and risk engine to mobile app`

Streamlit should redeploy automatically.

## 6. Reboot only if needed

In Streamlit Community Cloud:

`My apps` → app menu → `Reboot app`

Then open the existing app URL. Streamlit Secrets do not need to be re-entered.

## 7. Confirm the update

The header must show:

`5.3.0-mobile-adaptive-macro`

The main page must show:

- DXY
- US 10Y yield
- Gold 4H flow
- Data coverage
- Decision gate
- CFD risk profile

The tabs must include:

- SIGNAL
- MACRO
- CHART
- LEVELS
- MOMENTUM
- BRAIN
- MORE
