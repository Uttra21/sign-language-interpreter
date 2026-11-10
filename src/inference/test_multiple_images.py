import os
import sys

import torch
from PIL import Image
from torchvision import transforms


# ============================================================
# Import CNN
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

DATASET_ROOT = (
    "data/raw/asl_alphabet_train/"
    "asl_alphabet_train"
)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = ASLCNN(
    num_classes=29
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# Class mapping
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
# Image transformation
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
# Classes to test
# ============================================================

classes_to_test = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I"
]


# ============================================================
# Test predictions
# ============================================================

print("\n" + "=" * 65)
print("MULTI-CLASS ASL MODEL TEST")
print("=" * 65)

correct = 0
total = 0


for class_name in classes_to_test:

    class_folder = os.path.join(
        DATASET_ROOT,
        class_name
    )

    # Pick first JPG
    image_files = [
        file
        for file in os.listdir(class_folder)
        if file.lower().endswith(".jpg")
    ]

    image_file = sorted(
        image_files
    )[0]

    image_path = os.path.join(
        class_folder,
        image_file
    )


    # Load image
    image = Image.open(
        image_path
    ).convert("RGB")


    # Transform
    image_tensor = transform(
        image
    ).unsqueeze(0).to(device)


    # Predict
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


    prediction = index_to_class[
        predicted_index.item()
    ]

    confidence_value = (
        confidence.item() * 100
    )


    # Check
    is_correct = (
        prediction == class_name
    )

    if is_correct:
        correct += 1

    total += 1


    print(
        f"{class_name:>2}  →  "
        f"{prediction:<8} "
        f"{confidence_value:6.2f}% "
        f"{'✓' if is_correct else '✗'}"
    )


# ============================================================
# Summary
# ============================================================

accuracy = (
    100 * correct / total
)

print("=" * 65)

print(
    f"Correct: {correct}/{total}"
)

print(
    f"Sample accuracy: {accuracy:.2f}%"
)

print("=" * 65)