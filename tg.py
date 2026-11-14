"""Telegram: sending, subscriber list, and the long-polling update loop."""
import os
import time

import requests

import store

TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
DEFAULT_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
ADMIN_CHAT_ID = "224111652"
_API = f"https://api.telegram.org/bot{TOKEN}"


def send(text: str, chat_id=None) -> bool:
    chat_id = str(chat_id) if chat_id is not None else DEFAULT_CHAT_ID
    if not TOKEN or not chat_id:
        print(f"[Telegram] ⚠️ Not configured — message not sent:\n{text}")
        return False
    try:
        requests.post(f"{_API}/sendMessage", timeout=10,
                      json={"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}).raise_for_status()
        return True
    except Exception as e:
        print(f"[Telegram] ❌ send to {chat_id} failed: {e}")
        return False


def call(method: str, **payload) -> dict:
    r = requests.post(f"{_API}/{method}", json=payload, timeout=10)
    r.raise_for_status()
    return r.json()


def send_admin(text: str) -> bool:
    return send(text, ADMIN_CHAT_ID)


def subscribers() -> list:
    return store.get("subscribers", [])


def add_subscriber(chat_id: str):
    subs = subscribers()
    if chat_id not in subs:
        store.set("subscribers", sorted(subs + [chat_id]))
        print(f"[Telegram] Added subscriber {chat_id}")


def remove_subscriber(chat_id: str):
    subs = subscribers()
    if chat_id in subs:
        store.set("subscribers", [c for c in subs if c != chat_id])
        print(f"[Telegram] Removed subscriber {chat_id}")


def broadcast(text: str) -> bool:
    subs = subscribers()
    if not subs:
        return send(text)
    sent = False
    for chat_id in subs:
        sent = send(text, chat_id) or sent
        time.sleep(0.05)  # stays under Telegram's ~30 msgs/sec limit
    return sent


def poll_forever(handle_update):
    """Long-poll getUpdates. The offset is saved before handling, so an update that
    crashes the bot is never replayed in a restart loop."""
    try:
        offset = store.get("update_offset")
    except Exception as e:
        print(f"[Telegram] ⚠️ could not load offset, Telegram will resume from its own: {e}")
        offset = None
    while True:
        try:
            r = requests.get(f"{_API}/getUpdates", params={"offset": offset, "timeout": 50}, timeout=60)
            r.raise_for_status()
            for update in r.json()["result"]:
                offset = update["update_id"] + 1
                try:
                    store.set("update_offset", offset)
                    handle_update(update)
                except Exception as e:
                    print(f"[Telegram] ❌ handler failed for update {update['update_id']}: {e}")
        except Exception as e:
            print(f"[Telegram] ❌ poll failed: {e}")
            time.sleep(5)
