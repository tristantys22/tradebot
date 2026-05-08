import time
import threading
import schedule
from datetime import datetime
from ml_trading_bot import sync_subscribers, run_pipeline

print("Bot worker started...")

def scheduler_job():
    now = datetime.now()
    if now.weekday() >= 5:
        print(f"[Scheduler] Weekend ({now.strftime('%A')}) — skipping.")
        return
    print(f"[Scheduler] Running pipeline at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        run_pipeline(backtest_mode=False, force_notify=True)
    except Exception as e:
        print(f"[Scheduler] ❌ Pipeline failed: {e}")

# Schedule 9pm daily
schedule.every().day.at("21:00").do(scheduler_job)

def scheduler_loop():
    while True:
        schedule.run_pending()
        time.sleep(30)

# Run scheduler in background thread
threading.Thread(target=scheduler_loop, daemon=True).start()
print("[Scheduler] Scheduled to run daily at 21:00 (weekdays only)")

# Main loop — Telegram sync
while True:
    try:
        sync_subscribers()
    except Exception as e:
        print(f"[Worker Error] {e}")
    time.sleep(5)
