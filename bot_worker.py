import time
from ml_trading_bot import sync_subscribers

print("🚀 Bot worker started...")

while True:
    try:
        sync_subscribers()
    except Exception as e:
        print(f"[Worker Error] {e}")

    time.sleep(5)  # check Telegram every 5 seconds
