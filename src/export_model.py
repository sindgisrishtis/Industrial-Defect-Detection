import torch
from model import DefectCNN

MODEL_PATH = "results/best_model.pth"
EXPORT_PATH = "results/model_scripted.pt"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CLASS_COUNT = 6

model = DefectCNN(num_classes=CLASS_COUNT).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Convert to TorchScript
scripted_model = torch.jit.script(model)
scripted_model.save(EXPORT_PATH)

print(f"✅ TorchScript model saved at: {EXPORT_PATH}")
