
import cv2
import numpy as np
import torch
import mediapipe as mp
import os
import sys

# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../..")
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "asl_landmark_best.pth"
)

sys.path.append(
    os.path.join(PROJECT_ROOT, "src", "models")
)

from landmark_model import ASLLandmarkNet


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("ASL LANDMARK WEBCAM INTERPRETER")
print("=" * 60)

print("Device:", device)


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

class_to_index = checkpoint["class_to_index"]

index_to_class = {
    int(k): v
    for k, v in checkpoint["index_to_class"].items()
}

model = ASLLandmarkNet(
    num_classes=len(class_to_index)
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)
model.eval()

print("Model loaded successfully!")
print("Number of classes:", len(class_to_index))


# ============================================================
# MEDIAPIPE INITIALIZATION
# ============================================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# LANDMARK PREPROCESSING
# MUST MATCH extract_landmarks.py
# ============================================================

def extract_webcam_features(hand_landmarks):

    landmarks = []

    for landmark in hand_landmarks.landmark:

        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    # Convert into 21 x 3 array
    coordinates = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(21, 3)

    # Normalize relative to wrist
    wrist = coordinates[0].copy()

    coordinates = coordinates - wrist

    # Scale normalization
    distances = np.linalg.norm(
        coordinates,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:
        coordinates = coordinates / max_distance

    # Return 63 features
    return coordinates.flatten()


# ============================================================
# WEBCAM INITIALIZATION
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    hands.close()

    raise RuntimeError(
        "Could not open webcam. Check camera permissions."
    )

print("\nWebcam started.")
print("Show your ASL hand signs.")
print("Press Q to quit.")


# ============================================================
# MAIN WEBCAM LOOP
# ============================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break

        # IMPORTANT:
        # No horizontal flipping during this test.

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(rgb_frame)

        # Reset display for every frame
        prediction_text = "Nothing"
        confidence_text = ""

        # ----------------------------------------------------
        # HAND DETECTION
        # ----------------------------------------------------

        if results.multi_hand_landmarks:

            hand_landmarks = (
                results.multi_hand_landmarks[0]
            )

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 63 features
            features = extract_webcam_features(
                hand_landmarks
            )
            print("\nFirst 15 webcam features:")
            print(np.round(features[:15], 4))

            print("\nFeature shape:")
            print(features.shape)

            print("\nFeature range:")
            print("Minimum:", np.min(features))
            print("Maximum:", np.max(features))

            # Convert features to tensor
            input_tensor = torch.tensor(
                features,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # ------------------------------------------------
            # MODEL INFERENCE
            # ------------------------------------------------

            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

                top_probs, top_indices = torch.topk(
                    probabilities,
                    k=min(3, len(class_to_index)),
                    dim=1
                )

            # Best prediction
            confidence = top_probs[0][0].item()

            prediction = top_indices[0][0].item()

            predicted_label = index_to_class[
                prediction
            ]

            confidence_text = (
                f"{confidence * 100:.2f}%"
            )

            # Top 3 predictions
            top3 = [
                (
                    index_to_class[idx.item()],
                    round(prob.item() * 100, 2)
                )
                for idx, prob in zip(
                    top_indices[0],
                    top_probs[0]
                )
            ]

            # Print top predictions periodically
            print(
                "Top predictions:",
                top3
            )

            # Confidence filtering
            if confidence >= 0.70:

                prediction_text = predicted_label

            else:

                prediction_text = "Uncertain"

        # ----------------------------------------------------
        # DISPLAY PREDICTION
        # ----------------------------------------------------

        cv2.rectangle(
            frame,
            (0, 0),
            (500, 95),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Prediction: {prediction_text}",
            (15, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        if confidence_text:

            cv2.putText(
                frame,
                f"Confidence: {confidence_text}",
                (15, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2
            )

        cv2.putText(
            frame,
            "Press Q to quit",
            (15, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "ASL Landmark Interpreter",
            frame
        )

        # ----------------------------------------------------
        # QUIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

finally:

    cap.release()

    cv2.destroyAllWindows()

    hands.close()

    print("\nWebcam closed.")
