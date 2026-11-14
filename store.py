"""Key/value state in the Supabase `bot_state` table. Values are stored as JSON text."""
import json
import os

import requests

_URL = f"{os.environ.get('SUPABASE_URL')}/rest/v1/bot_state"
_KEY = os.environ.get("SUPABASE_KEY")
_HEADERS = {
    "apikey": _KEY,
    "Authorization": f"Bearer {_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",
}


def get(key: str, default=None):
    # Raises on HTTP errors so callers never overwrite real data after a failed read.
    r = requests.get(_URL, params={"key": f"eq.{key}", "select": "value"}, headers=_HEADERS, timeout=10)
    r.raise_for_status()
    rows = r.json()
    return json.loads(rows[0]["value"]) if rows else default


def set(key: str, value) -> None:
    r = requests.post(_URL, params={"on_conflict": "key"}, headers=_HEADERS,
                      json={"key": key, "value": json.dumps(value)}, timeout=10)
    if not r.ok:
        print(f"[Supabase] ❌ write {key} failed: {r.status_code} {r.text}")
    r.raise_for_status()
