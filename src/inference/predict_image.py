import os
import sys

import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# Allow importing ASLCNN from src/models
# ============================================================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models"
        )
    )
)

from cnn_model import ASLCNN


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "models/asl_cnn_best.pth"

IMAGE_PATH = (
    "data/raw/asl_alphabet_train/"
    "asl_alphabet_train/A/A1.jpg"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


# Create model
model = ASLCNN(
    num_classes=29
)


# Load trained weights
model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# Get class mapping
# ============================================================

class_to_index = checkpoint[
    "class_to_index"
]

index_to_class = {
    index: class_name
    for class_name, index
    in class_to_index.items()
}


# ============================================================
# Image preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (64, 64)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# Load image
# ============================================================

image = Image.open(
    IMAGE_PATH
).convert("RGB")


# Transform image
image_tensor = transform(
    image
).unsqueeze(0).to(device)


# ============================================================
# Prediction
# ============================================================

with torch.no_grad():

    outputs = model(
        image_tensor
    )

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, predicted_index = (
        torch.max(
            probabilities,
            dim=1
        )
    )


predicted_class = index_to_class[
    predicted_index.item()
]


# ============================================================
# Display result
# ============================================================

print("\n" + "=" * 50)
print("ASL IMAGE PREDICTION")
print("=" * 50)

print(
    "Image:",
    IMAGE_PATH
)

print(
    "Prediction:",
    predicted_class
)

print(
    f"Confidence: "
    f"{confidence.item() * 100:.2f}%"
)

print("=" * 50)