"""
Alpaca paper trading integration.
Reads signal, decides whether to enter/exit SPY, places fractional orders.
"""
import os
import json
import requests

ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY")
ALPACA_SECRET = os.environ.get("ALPACA_SECRET")
ALPACA_ENABLED = os.environ.get("ALPACA_PAPER_ENABLED", "false").lower() == "true"
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"

# How much of available cash to deploy on each BUY signal (0.0–1.0)
POSITION_SIZE_PCT = 0.95


def _headers():
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET,
        "Content-Type": "application/json",
    }


def is_enabled() -> bool:
    return ALPACA_ENABLED and ALPACA_API_KEY and ALPACA_SECRET


def get_account() -> dict | None:
    if not is_enabled():
        return None
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/account", headers=_headers(), timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Alpaca] ❌ get_account failed: {e}")
        return None


def get_position(symbol: str = "SPY") -> dict | None:
    """Returns position dict if held, None if flat."""
    if not is_enabled():
        return None
    try:
        r = requests.get(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=_headers(), timeout=10)
        if r.status_code == 404:
            return None  # no position
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[Alpaca] ❌ get_position failed: {e}")
        return None


def submit_buy(symbol: str = "SPY") -> dict | None:
    """Buy SPY with ~95% of available buying power, fractional shares."""
    if not is_enabled():
        print("[Alpaca] ⚠️ Disabled, skipping buy.")
        return None
    account = get_account()
    if not account:
        return None
    buying_power = float(account.get("buying_power", 0))
    notional = round(buying_power * POSITION_SIZE_PCT, 2)
    if notional < 1:
        print(f"[Alpaca] ⚠️ Not enough buying power: ${buying_power}")
        return None
    try:
        order = {
            "symbol": symbol,
            "notional": notional,  # fractional, dollar-denominated
            "side": "buy",
            "type": "market",
            "time_in_force": "day",
        }
        r = requests.post(f"{ALPACA_BASE_URL}/v2/orders", headers=_headers(), json=order, timeout=15)
        r.raise_for_status()
        result = r.json()
        print(f"[Alpaca] ✅ BUY submitted: ${notional} of {symbol} (order_id={result.get('id')})")
        return result
    except Exception as e:
        print(f"[Alpaca] ❌ submit_buy failed: {e}")
        return None


def submit_sell_all(symbol: str = "SPY") -> dict | None:
    """Close entire position in symbol."""
    if not is_enabled():
        print("[Alpaca] ⚠️ Disabled, skipping sell.")
        return None
    try:
        r = requests.delete(f"{ALPACA_BASE_URL}/v2/positions/{symbol}", headers=_headers(), timeout=15)
        if r.status_code == 404:
            print(f"[Alpaca] ⚠️ No position in {symbol} to close.")
            return None
        r.raise_for_status()
        result = r.json()
        print(f"[Alpaca] ✅ SELL submitted: closing {symbol} position")
        return result
    except Exception as e:
        print(f"[Alpaca] ❌ submit_sell_all failed: {e}")
        return None


def execute_signal(signal: int, symbol: str = "SPY") -> str:
    """
    Reconcile current position with model signal.
    signal: 1 = should be long, 0 = should be flat.
    Returns a status string for logging/Telegram.
    """
    if not is_enabled():
        return "Paper trading disabled."

    position = get_position(symbol)
    currently_long = position is not None

    if signal == 1 and not currently_long:
        result = submit_buy(symbol)
        return f"Paper BUY {symbol} submitted." if result else "Paper BUY failed."
    elif signal == 0 and currently_long:
        result = submit_sell_all(symbol)
        return f"Paper SELL {symbol} submitted." if result else "Paper SELL failed."
    elif signal == 1 and currently_long:
        return f"✅ Already long {symbol}, no action."
    else:
        return f"✅ Already flat, no action."


def get_status_message() -> str:
    """Build a /paper status message for Telegram."""
    if not is_enabled():
        return "Paper trading is disabled."
    account = get_account()
    if not account:
        return "❌ Could not fetch Alpaca account."
    equity = float(account.get("equity", 0))
    cash = float(account.get("cash", 0))
    position = get_position("SPY")
    lines = [
        "📊 *Alpaca Paper Account*",
        f"",
        f"Equity: `${equity:,.2f}`",
        f"Cash: `${cash:,.2f}`",
    ]
    if position:
        qty = float(position.get("qty", 0))
        market_value = float(position.get("market_value", 0))
        unrealized_pl = float(position.get("unrealized_pl", 0))
        unrealized_plpc = float(position.get("unrealized_plpc", 0)) * 100
        lines.extend([
            f"",
            f"📌 *Position: SPY*",
            f"Qty: `{qty:.4f}`",
            f"Value: `${market_value:,.2f}`",
            f"P&L: `${unrealized_pl:+,.2f}` (`{unrealized_plpc:+.2f}%`)",
        ])
    else:
        lines.append(f"\nPosition: *FLAT*")
    return "\n".join(lines)
