# AurumEdge Mobile v5

A mobile-first XAU/USD technical terminal for iPhone and other phones.

## Recommended use: cloud web app installed on iPhone

The phone should open a secure cloud URL. Your laptop does not need to remain on. The app runs server-side, so Twelve Data and optional OpenAI keys are not placed in Safari or JavaScript.

### Deploy on Streamlit Community Cloud

1. Create a private GitHub repository.
2. Upload this project. Do **not** upload `.env` or `.streamlit/secrets.toml`.
3. Sign in to Streamlit Community Cloud and create an app from the repository.
4. Set the entrypoint to `mobile_app.py`.
5. Open **App settings > Secrets** and paste values using the syntax shown in `.streamlit/secrets.toml.example`.
6. Deploy and open the resulting `https://...streamlit.app` address in Safari.
7. In Safari, tap **Share > Add to Home Screen**, enable **Open as Web App**, and tap **Add**.

Set an `APP_PIN` in cloud secrets if you want a second privacy gate. Keep the Streamlit app private whenever the hosting plan/workspace permits it.

## Local test on the same Wi-Fi

1. Extract this folder beside the old v4 project.
2. Run `run_mobile_terminal.bat`.
3. The launcher reuses the old `.env` automatically.
4. It prints a phone URL such as `http://192.168.1.20:8515`.
5. Connect the iPhone and laptop to the same Wi-Fi and open that address in Safari.
6. Allow Python through Windows Firewall if Windows asks.

Local Wi-Fi mode stops when the laptop sleeps, closes, changes network, or the terminal window closes. Use cloud deployment for access from anywhere.

## Mobile design

- Signal card and actionable levels appear first.
- Touch-friendly timeframe and refresh controls.
- Tabs for signal, chart, levels, momentum and TradingView.
- Pinch/zoom Plotly chart; rotate to landscape for full chart inspection.
- Manual paid OpenAI research is OFF by default.
- No broker connection and no order execution.

## Security

- Never put real API keys into GitHub files.
- Revoke any key that was pasted into a chat, screenshot, public repository or shared file.
- Store cloud keys only in the hosting provider's secrets manager.
