"""Alpaca paper-trading REST client."""
import os

import requests

BASE_URL = "https://paper-api.alpaca.markets"
_KEY = os.environ.get("ALPACA_API_KEY")
_SECRET = os.environ.get("ALPACA_SECRET")
ENABLED = bool(os.environ.get("ALPACA_PAPER_ENABLED", "").lower() == "true" and _KEY and _SECRET)
_HEADERS = {"APCA-API-KEY-ID": _KEY, "APCA-API-SECRET-KEY": _SECRET}


def api(method: str, path: str, **kwargs):
    """Returns parsed JSON, or None on 404 (e.g. no open position)."""
    r = requests.request(method, f"{BASE_URL}{path}", headers=_HEADERS, timeout=15, **kwargs)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.json() if r.content else {}


def account() -> dict:
    return api("GET", "/v2/account")


def position(symbol: str):
    return api("GET", f"/v2/positions/{symbol}")


def clock() -> dict:
    return api("GET", "/v2/clock")


def has_open_order(symbol: str) -> bool:
    return bool(api("GET", "/v2/orders", params={"status": "open", "symbols": symbol}))


def market_order(symbol: str, side: str, qty: int = None, notional: float = None) -> dict:
    order = {"symbol": symbol, "side": side, "type": "market", "time_in_force": "day"}
    order.update({"qty": str(qty)} if qty is not None else {"notional": str(notional)})
    return api("POST", "/v2/orders", json=order)


def close_position(symbol: str):
    """Market-sells the whole position. Returns None if there was nothing to close."""
    return api("DELETE", f"/v2/positions/{symbol}")
