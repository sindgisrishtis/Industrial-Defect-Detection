import cv2
import os
import random

# =========================
# CONFIG
# =========================
DATA_DIR = "data/NEU-DET/validation/images"
OUTPUT_VIDEO = "results/demo_inspection_video.mp4"
IMG_SIZE = 512
FPS = 2   # images per second
IMAGES_PER_CLASS = 20  # how many per defect type

# =========================
# Collect Images
# =========================
image_paths = []

for defect_class in os.listdir(DATA_DIR):
    class_folder = os.path.join(DATA_DIR, defect_class)
    if os.path.isdir(class_folder):
        images = os.listdir(class_folder)
        selected = random.sample(images, min(IMAGES_PER_CLASS, len(images)))
        for img in selected:
            image_paths.append(os.path.join(class_folder, img))

random.shuffle(image_paths)

print(f"Total images selected: {len(image_paths)}")

# =========================
# Create Video Writer
# =========================
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (IMG_SIZE, IMG_SIZE))

# =========================
# Write Images to Video
# =========================
for path in image_paths:
    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    video_writer.write(img)

video_writer.release()

print(f"\n✅ Demo inspection video saved at: {OUTPUT_VIDEO}")
