from __future__ import annotations

import sys

from dotenv import load_dotenv

from gold_web_terminal.config import Settings
from gold_web_terminal.market_data import TwelveDataClient


def main() -> int:
    load_dotenv()
    settings = Settings.from_env()
    print("Gold AI Professional Web Terminal diagnostics")
    print(f"Python: {sys.version.split()[0]}")
    print(f"OpenAI key configured: {bool(settings.openai_api_key)}")
    print(f"Twelve Data key configured: {bool(settings.twelve_data_api_key)}")
    print(f"Automatic research: {settings.auto_research}")
    print("Broker connection: disabled / not present")
    print("Order execution: disabled / not present")
    if settings.twelve_data_api_key:
        try:
            frame, has_volume = TwelveDataClient(settings.twelve_data_api_key).fetch(settings.market_symbol, "H1", 250)
            print(f"Market data: OK ({len(frame)} H1 bars, latest close {frame.iloc[-1]['close']})")
            print(f"Provider volume present: {has_volume}")
        except Exception as exc:
            print(f"Market data: FAILED - {exc}")
            return 2
    else:
        print("Market data: skipped because TWELVE_DATA_API_KEY is empty")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
