import cv2
import mediapipe as mp


# -----------------------------
# Initialize MediaPipe Hands
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# -----------------------------
# Start Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Show your hand to the camera.")
print("Press Q to quit.")


# -----------------------------
# Main Loop
# -----------------------------
while True:

    success, frame = cap.read()

    if not success:
        print("ERROR: Could not read webcam frame.")
        break

    # Mirror the webcam
    frame = cv2.flip(frame, 1)

    # OpenCV uses BGR
    # MediaPipe expects RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Detect hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks
    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # Display webcam
    cv2.imshow(
        "Sign Language - Hand Detection",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()
hands.close()

print("Webcam stopped.")