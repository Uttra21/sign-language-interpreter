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
# Configuration
# ============================================================

MODEL_PATH = "models/asl_cnn_best.pth"

IMAGE_SIZE = 64

CONFIDENCE_THRESHOLD = 0.70


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
# Load trained CNN
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
# Image preprocessing
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

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

mp_drawing = mp.solutions.drawing_utils


hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# OpenCV webcam
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")

    sys.exit()


print("\nWebcam started.")
print("Show your hand to the camera.")
print("Press Q to quit.")


# ============================================================
# Main loop
# ============================================================

# ============================================================
# Main webcam loop
# ============================================================

while True:

    ret, raw_frame = cap.read()

    if not ret:
        print("ERROR: Could not read webcam frame.")
        break

    # --------------------------------------------------------
    # Mirror the original webcam image
    # --------------------------------------------------------

    raw_frame = cv2.flip(raw_frame, 1)

    # --------------------------------------------------------
    # IMPORTANT:
    # raw_frame = CLEAN IMAGE
    # display_frame = image used for drawing
    # --------------------------------------------------------

    clean_frame = raw_frame.copy()
    display_frame = raw_frame.copy()

    # --------------------------------------------------------
    # MediaPipe processing
    # --------------------------------------------------------

    rgb_frame = cv2.cvtColor(
        clean_frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb_frame)

    prediction_text = "No hand detected"

    # --------------------------------------------------------
    # Hand detected
    # --------------------------------------------------------

    if results.multi_hand_landmarks:

        hand_landmarks = results.multi_hand_landmarks[0]

        height, width, _ = clean_frame.shape

        # ----------------------------------------------------
        # Draw landmarks ONLY on display_frame
        # ----------------------------------------------------

        mp_drawing.draw_landmarks(
            display_frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        # ----------------------------------------------------
        # Get landmark coordinates
        # ----------------------------------------------------

        x_coordinates = [
            int(landmark.x * width)
            for landmark in hand_landmarks.landmark
        ]

        y_coordinates = [
            int(landmark.y * height)
            for landmark in hand_landmarks.landmark
        ]

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x_min_raw = min(x_coordinates)
        x_max_raw = max(x_coordinates)

        y_min_raw = min(y_coordinates)
        y_max_raw = max(y_coordinates)

        hand_width = x_max_raw - x_min_raw
        hand_height = y_max_raw - y_min_raw

        box_size = max(
            hand_width,
            hand_height
        )

        padding = int(
            box_size * 0.25
        )

        box_size = box_size + (
            2 * padding
        )

        center_x = (
            x_min_raw + x_max_raw
        ) // 2

        center_y = (
            y_min_raw + y_max_raw
        ) // 2

        x_min = center_x - box_size // 2
        x_max = center_x + box_size // 2

        y_min = center_y - box_size // 2
        y_max = center_y + box_size // 2

        # ----------------------------------------------------
        # Keep bounding box inside image
        # ----------------------------------------------------

        x_min = max(x_min, 0)
        y_min = max(y_min, 0)

        x_max = min(x_max, width)
        y_max = min(y_max, height)

        # ----------------------------------------------------
        # Draw bounding box ONLY on display frame
        # ----------------------------------------------------

        cv2.rectangle(
            display_frame,
            (x_min, y_min),
            (x_max, y_max),
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # CRITICAL:
        # Crop from CLEAN frame, NOT display_frame
        # ----------------------------------------------------

        hand_crop = clean_frame[
            y_min:y_max,
            x_min:x_max
        ]

        # ----------------------------------------------------
        # Debug window
        #
        # This MUST contain ONLY the hand.
        # No red dots.
        # No white lines.
        # ----------------------------------------------------

        if (
            hand_crop.size != 0
            and hand_crop.shape[0] > 20
            and hand_crop.shape[1] > 20
        ):

            cv2.imshow(
                "CNN Input",
                hand_crop
            )

            # ------------------------------------------------
            # Convert BGR → RGB
            # ------------------------------------------------

            hand_rgb = cv2.cvtColor(
                hand_crop,
                cv2.COLOR_BGR2RGB
            )

            pil_image = Image.fromarray(
                hand_rgb
            )

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            image_tensor = transform(
                pil_image
            ).unsqueeze(0).to(device)

            # ------------------------------------------------
            # CNN prediction
            # ------------------------------------------------

            with torch.no_grad():

                outputs = model(
                    image_tensor
                )

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                confidence, predicted_index = torch.max(
                    probabilities,
                    dim=1
                )

            predicted_class = index_to_class[
                predicted_index.item()
            ]

            confidence_value = confidence.item()

            # ------------------------------------------------
            # Confidence filter
            # ------------------------------------------------

            if confidence_value >= CONFIDENCE_THRESHOLD:

                prediction_text = (
                    f"{predicted_class} "
                    f"({confidence_value * 100:.1f}%)"
                )

            else:

                prediction_text = "Uncertain"

    # --------------------------------------------------------
    # Prediction display
    # --------------------------------------------------------

    cv2.rectangle(
        display_frame,
        (10, 10),
        (400, 70),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display_frame,
        f"Prediction: {prediction_text}",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Show annotated webcam
    # --------------------------------------------------------

    cv2.imshow(
        "ASL Sign Language Interpreter",
        display_frame
    )

    # --------------------------------------------------------
    # Quit
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
# ============================================================
# Cleanup
# ============================================================

cap.release()

cv2.destroyAllWindows()

hands.close()

print("Webcam interpreter stopped.")