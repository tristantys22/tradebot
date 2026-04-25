"""
scheduler.py — runs the signal bot every weekday at a set time.

Options:
  A) Run this script directly (keeps running in background)
  B) Use cron instead (see README) — more reliable for 24/7

Usage:
    python scheduler.py
    python scheduler.py --time 21:30   # custom time (24h, your local time)
"""

import argparse
import time
import schedule
from datetime import datetime
from ml_trading_bot import run_pipeline

DEFAULT_RUN_TIME = "21:30"  # 9:30 PM SGT = after US market close + model update


def job():
    now = datetime.now()
    # Skip weekends
    if now.weekday() >= 5:
        print(f"[Scheduler] Weekend ({now.strftime('%A')}) — skipping.")
        return
    print(f"[Scheduler] Running pipeline at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    try:
        # force_notify=True ensures the broadcast always fires at 9pm
        # even if the signal hasn't changed from yesterday
        run_pipeline(backtest_mode=False, force_notify=True)
    except Exception as e:
        print(f"[Scheduler] ❌ Pipeline failed: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--time", default=DEFAULT_RUN_TIME,
                        help=f"Time to run daily (HH:MM, 24h). Default: {DEFAULT_RUN_TIME}")
    args = parser.parse_args()

    print(f"[Scheduler] Scheduled to run daily at {args.time} (weekdays only)")
    print(f"[Scheduler] Keep this terminal open, or use cron for reliability.")
    print(f"[Scheduler] Press Ctrl+C to stop.\n")

    schedule.every().day.at(args.time).do(job)

    while True:
        schedule.run_pending()
        time.sleep(30)
