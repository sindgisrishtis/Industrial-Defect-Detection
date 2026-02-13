import cv2
import argparse
import time
import os

from inference import load_model, predict_image
from alert import trigger_alert
from logger import log_event

# -----------------------------
# Configuration
# -----------------------------
CONF_THRESHOLD = 0.70  # Below this → Normal Surface
HEADLESS = os.environ.get("HEADLESS", "0") == "1"


def main(video_path=None):
    model = load_model()

    # -----------------------------
    # Select Video Source
    # -----------------------------
    if video_path:
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open video source")
        return

    print("🏭 Industrial Defect Monitoring Started")
    if not HEADLESS:
        print("Press 'q' to exit\n")

    prev_time = 0

    # -----------------------------
    # Main Loop
    # -----------------------------
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # -----------------------------
        # FPS Calculation
        # -----------------------------
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        # -----------------------------
        # Prediction
        # -----------------------------
        class_name, confidence = predict_image(model, frame)

        # -----------------------------
        # Normal Surface Logic
        # -----------------------------
        if confidence < CONF_THRESHOLD:
            display_text = "Normal Surface"
            color = (0, 255, 0)  # Green

        else:
            display_text = f"{class_name}: {confidence:.2f}"

            # Confidence Color Coding
            if confidence > 0.90:
                color = (0, 255, 0)      # High confidence → Green
            elif confidence > 0.80:
                color = (0, 255, 255)    # Medium → Yellow
            else:
                color = (0, 0, 255)      # Low → Red

            # Trigger alert + log only for defects
            trigger_alert(class_name, confidence, frame)
            log_event(class_name, confidence)

        # -----------------------------
        # Overlay Display Text
        # -----------------------------
        cv2.putText(frame,
                    display_text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        cv2.putText(frame,
                    f"FPS: {int(fps)}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2)

        # -----------------------------
        # GUI Display (Only if not headless)
        # -----------------------------
        if not HEADLESS:
            cv2.imshow("Industrial Defect Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()

    if not HEADLESS:
        try:
            cv2.destroyAllWindows()
        except:
            pass

    print("✅ Monitoring Finished.")


# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Path to video file")
    args = parser.parse_args()

    main(video_path=args.video)
