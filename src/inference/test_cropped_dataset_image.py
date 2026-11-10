import os
import sys

import cv2
import torch
import mediapipe as mp

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
# Paths
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
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# Load model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = ASLCNN(num_classes=29)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()


# ============================================================
# Class mapping
# ============================================================

class_to_index = checkpoint["class_to_index"]

index_to_class = {
    index: name
    for name, index in class_to_index.items()
}


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


# ============================================================
# MediaPipe
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)


# ============================================================
# Load dataset image
# ============================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(IMAGE_PATH)


rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# Detect hand
# ============================================================

results = hands.process(rgb)


if not results.multi_hand_landmarks:

    print("MediaPipe could not detect the hand.")
    sys.exit()


hand = results.multi_hand_landmarks[0]

height, width, _ = image.shape


# ============================================================
# Calculate bounding box
# ============================================================

xs = [
    int(landmark.x * width)
    for landmark in hand.landmark
]

ys = [
    int(landmark.y * height)
    for landmark in hand.landmark
]


padding = 30

x_min = max(
    min(xs) - padding,
    0
)

x_max = min(
    max(xs) + padding,
    width
)

y_min = max(
    min(ys) - padding,
    0
)

y_max = min(
    max(ys) + padding,
    height
)


# ============================================================
# Crop
# ============================================================

crop = image[
    y_min:y_max,
    x_min:x_max
]


# Save/show crop
cv2.imwrite(
    "results/debug_A_mediapipe_crop.jpg",
    crop
)

cv2.imshow(
    "MediaPipe Cropped A",
    crop
)

cv2.waitKey(2000)
cv2.destroyAllWindows()


# ============================================================
# Convert to PIL
# ============================================================

crop_rgb = cv2.cvtColor(
    crop,
    cv2.COLOR_BGR2RGB
)

pil_image = Image.fromarray(
    crop_rgb
)


# ============================================================
# CNN prediction
# ============================================================

tensor = transform(
    pil_image
).unsqueeze(0).to(device)


with torch.no_grad():

    outputs = model(tensor)

    probabilities = torch.softmax(
        outputs,
        dim=1
    )

    confidence, prediction = torch.max(
        probabilities,
        dim=1
    )


predicted_class = index_to_class[
    prediction.item()
]


print("\n" + "=" * 55)
print("CROPPED DATASET IMAGE TEST")
print("=" * 55)

print("Actual class: A")

print(
    f"Prediction: {predicted_class}"
)

print(
    f"Confidence: "
    f"{confidence.item() * 100:.2f}%"
)

print("=" * 55)