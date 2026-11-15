"""Entry point: Telegram commands on the main thread, scheduled jobs on a background thread."""
import threading
import time
from datetime import datetime

import schedule

import nuai_mean_revert as nuai
import spy_strategy as spy
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
        spy.run_daily(force_notify=True)


def _scheduler():
    schedule.every().day.at(SPY_RUN_UTC).do(_safe("SPY", spy_job))
    schedule.every().minute.do(_safe("NUAI", nuai.run_cycle))
    while True:
        schedule.run_pending()
        time.sleep(1)


ABOUT = ("🤖 *Tristan's SPY Bot*\n\n"
         "Every weekday at *9pm SGT* I send how much of a portfolio should be in SPY (the S&P 500 ETF), "
         "from 0% to 100%. Half follows the long-term trend (sized down when markets are volatile), "
         "half buys short dips during uptrends.\n\n"
         "I only understand the commands below — normal chat messages won't get a reply.")


def _help_text(chat_id) -> str:
    lines = [ABOUT, "", "📋 *Commands*"] + [f"/{name} — {desc}" for name, (desc, _) in PUBLIC.items()]
    if chat_id == tg.ADMIN_CHAT_ID:
        lines += ["", "🔐 *Admin*"] + [f"/{name} — {desc}" for name, (desc, _) in ADMIN.items()]
    return "\n".join(lines)


def _start(chat_id, _):
    tg.add_subscriber(chat_id)
    tg.send("✅ *Subscribed!* You'll get the SPY signal every weekday at 9pm SGT.\n\n" + _help_text(chat_id), chat_id)

def _broadcast(chat_id, arg):
    if not arg:
        return tg.send("Usage: /broadcast your message here", chat_id)
    tg.send("📢 Broadcasting message...", chat_id)
    tg.broadcast(f"📢 *Broadcast*\n\n{arg}")


def _force(chat_id, _):
    tg.send("⚡ Running the SPY strategy now...", chat_id)
    spy.run_daily(force_notify=True)


PUBLIC = {
    "start": ("Subscribe to the daily signal", _start),
    "signal": ("Latest SPY signal", lambda c, _: tg.send(spy.latest_message(), c)),
    "history": ("Last 5 signals", lambda c, _: tg.send(spy.history_message(), c)),
    "accuracy": ("How often the bot was right", lambda c, _: tg.send(spy.accuracy_message(), c)),
    "help": ("What this bot does + commands", lambda c, _: tg.send(_help_text(c), c)),
    "stop": ("Unsubscribe", lambda c, _: (tg.remove_subscriber(c), tg.send(
        "👋 Unsubscribed — you won't get daily signals anymore. Send /start to come back.", c))),
}
ADMIN = {
    "paper": ("Alpaca paper account + SPY position", lambda c, _: tg.send(spy.paper_status_message(), c)),
    "nuai": ("NUAI mean-reversion status", lambda c, _: tg.send(nuai.status_message(), c)),
    "force": ("Run the SPY strategy now and broadcast", _force),
    "broadcast": ("Message every subscriber", _broadcast),
}


def register_commands():
    """Fills Telegram's '/' menu and the intro text people see before pressing Start."""
    tg.call("setMyCommands", commands=[{"command": n, "description": d} for n, (d, _) in PUBLIC.items()])
    tg.call("setMyCommands", scope={"type": "chat", "chat_id": int(tg.ADMIN_CHAT_ID)},
            commands=[{"command": n, "description": d} for n, (d, _) in (PUBLIC | ADMIN).items()])
    tg.call("setMyShortDescription", short_description="Daily SPY position signal (trend + dip-buying). Send /start.")
    tg.call("setMyDescription", description=(
        "Every weekday at 9pm SGT this bot sends how much of a portfolio should be in SPY (S&P 500 ETF), from 0% to "
        "100% — half trend-following with a volatility target, half RSI(2) dip-buying.\n\n"
        "Press Start to subscribe. The bot only answers commands "
        "(/signal, /history, /accuracy, /help, /stop), not normal chat messages."))


def handle_update(update: dict):
    member = update.get("my_chat_member")
    if member:
        chat, status = member["chat"], member["new_chat_member"]["status"]
        group_id = str(chat["id"])
        if status == "member":
            tg.add_subscriber(group_id)
            tg.send(f"👋 Thanks for adding me to *{chat.get('title', 'this group')}*!\n\n"
                    f"I'll post the SPY signal here every weekday at 9pm SGT.\n\n{_help_text(group_id)}", group_id)
        elif status in ("kicked", "left"):
            tg.remove_subscriber(group_id)

    msg = update.get("message") or {}
    if "chat" not in msg:
        return
    chat_id, private = str(msg["chat"]["id"]), msg["chat"].get("type") == "private"
    text = (msg.get("text") or "").strip()
    if not text.startswith("/"):
        if private:  # in groups, stay quiet on normal chatter
            tg.send("🤔 I only respond to commands.\n\n" + _help_text(chat_id), chat_id)
        return
    command, _, arg = text.partition(" ")
    command = command[1:].split("@")[0].lower()  # "/start@MyBot" in groups
    print(f"[Telegram] /{command} from {chat_id}")

    if command in PUBLIC:
        PUBLIC[command][1](chat_id, arg.strip())
    elif command in ADMIN:
        if chat_id != tg.ADMIN_CHAT_ID:
            tg.send("🔐 That command is admin-only. Send /help to see what you can use.", chat_id)
        else:
            ADMIN[command][1](chat_id, arg.strip())
    elif private:
        tg.send(f"❓ I don't know /{command}.\n\n" + _help_text(chat_id), chat_id)


if __name__ == "__main__":
    print("Bot worker started")
    try:
        register_commands()
    except Exception as e:
        print(f"[Telegram] ⚠️ could not register command menu: {e}")
    threading.Thread(target=_scheduler, daemon=True).start()
    tg.poll_forever(handle_update)
