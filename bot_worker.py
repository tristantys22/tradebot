"""Entry point: Telegram commands on the main thread, scheduled jobs on a background thread."""
import threading
import time
from datetime import datetime

import schedule

import nuai_mean_revert as nuai
import spy_model
import tg

SPY_RUN_UTC = "13:00"  # 21:00 SGT


def _safe(name, fn):
    def run():
        try:
            fn()
        except Exception as e:
            print(f"[{name}] ❌ failed: {e}")
    return run


def spy_job():
    if datetime.utcnow().weekday() < 5:
        spy_model.run_daily(force_notify=True)


def _scheduler():
    schedule.every().day.at(SPY_RUN_UTC).do(_safe("SPY", spy_job))
    schedule.every().minute.do(_safe("NUAI", nuai.run_cycle))
    while True:
        schedule.run_pending()
        time.sleep(1)


HELP = ("📋 *Available Commands:*\n"
        "/start — Subscribe to daily alerts\n"
        "/stop — Unsubscribe from alerts\n"
        "/history — View the last 5 predictions\n"
        "/accuracy — See the bot's prediction accuracy\n")


def _broadcast(chat_id, arg):
    if not arg:
        return tg.send("Usage: /broadcast your message here", chat_id)
    tg.send("📢 Broadcasting message...", chat_id)
    tg.broadcast(f"📢 *Broadcast*\n\n{arg}")


def _force(chat_id, _):
    tg.send("⚡ Forcing prediction...", chat_id)
    spy_model.run_daily(force_notify=True)


PUBLIC = {
    "/start": lambda c, _: (tg.add_subscriber(c), tg.send(f"✅ *Subscribed to Tristan's SPY ML Bot!*\n\n{HELP}", c)),
    "/stop": lambda c, _: (tg.remove_subscriber(c), tg.send("You have been unsubscribed from me. I hate you.", c)),
    "/history": lambda c, _: tg.send(spy_model.history_message(), c),
    "/accuracy": lambda c, _: tg.send(spy_model.accuracy_message(), c),
}
ADMIN = {
    "/force": _force,
    "/paper": lambda c, _: tg.send(spy_model.paper_status_message(), c),
    "/nuai": lambda c, _: tg.send(nuai.status_message(), c),
    "/broadcast": _broadcast,
}


def handle_update(update: dict):
    member = update.get("my_chat_member")
    if member:
        chat, status = member["chat"], member["new_chat_member"]["status"]
        group_id = str(chat["id"])
        if status == "member":
            tg.add_subscriber(group_id)
            tg.send(f"👋 Thanks for adding me to *{chat.get('title', 'this group')}*!\n\n"
                    f"I'll send daily SPY signals here automatically.\n\n{HELP}", group_id)
        elif status in ("kicked", "left"):
            tg.remove_subscriber(group_id)

    msg = update.get("message") or {}
    text = (msg.get("text") or "").strip()
    if not text.startswith("/") or "chat" not in msg:
        return
    chat_id = str(msg["chat"]["id"])
    command, _, arg = text.partition(" ")
    command = command.split("@")[0].lower()  # "/start@MyBot" in groups
    print(f"[Telegram] {command} from {chat_id}")

    if command in PUBLIC:
        PUBLIC[command](chat_id, arg.strip())
    elif command in ADMIN:
        if chat_id != tg.ADMIN_CHAT_ID:
            tg.send(f"❌ You are not authorized. Your chat_id is {chat_id}", chat_id)
        else:
            ADMIN[command](chat_id, arg.strip())


if __name__ == "__main__":
    print("Bot worker started")
    threading.Thread(target=_scheduler, daemon=True).start()
    tg.poll_forever(handle_update)
