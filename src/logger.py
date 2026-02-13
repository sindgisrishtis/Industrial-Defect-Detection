import csv
import os
from datetime import datetime


def log_event(class_name, confidence):
    os.makedirs("results", exist_ok=True)

    file_path = "results/detection_log.csv"
    file_exists = os.path.isfile(file_path)

    with open(file_path, "a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["timestamp", "class", "confidence"])

        writer.writerow([datetime.now(), class_name, confidence])
