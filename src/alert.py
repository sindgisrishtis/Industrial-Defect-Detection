import os
import cv2
import time
from datetime import datetime

ALERT_THRESHOLD = 0.80
ALERT_COOLDOWN = 2  # seconds between alerts

last_alert_time = 0


def trigger_alert(class_name, confidence, frame):
    global last_alert_time

    current_time = time.time()

    if confidence >= ALERT_THRESHOLD and (current_time - last_alert_time > ALERT_COOLDOWN):

        print(f"\n🚨 DEFECT DETECTED: {class_name} ({confidence:.2f})")

        os.makedirs("results/defects", exist_ok=True)

        filename = f"results/defects/{class_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)

        print(f"Saved defect frame: {filename}")

        last_alert_time = current_time
