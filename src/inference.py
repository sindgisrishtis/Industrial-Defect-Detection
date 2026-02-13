import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

from model import DefectCNN

IMG_SIZE = 128
MODEL_PATH = "results/best_model.pth"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Class names (must match training order)
CLASS_NAMES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches"
]

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def load_model():
    model = DefectCNN(num_classes=len(CLASS_NAMES)).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))

    model.eval()
    return model


def predict_image(model, image_np):
    image = Image.fromarray(image_np)
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    class_name = CLASS_NAMES[predicted.item()]
    confidence = confidence.item()

    return class_name, confidence
