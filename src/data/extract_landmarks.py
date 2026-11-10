import os
import csv
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp


# ============================================================
# PROJECT PATHS
# ============================================================

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"
TEST_CSV = "data/processed/test.csv"

OUTPUT_DIR = "data/landmarks"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# DATASET ROOT
# ============================================================

DATASET_ROOT = "data/raw"


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
# LANDMARK FEATURE HEADER
# ============================================================

header = ["label"]

for i in range(21):
    header.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# ============================================================
# FIND IMAGE PATH
# ============================================================

def find_image_path(path):

    path = str(path).replace("\\", "/")

    # Already valid
    if os.path.exists(path):
        return path

    # Try from project root
    candidate = os.path.join(
        ".",
        path
    )

    if os.path.exists(candidate):
        return candidate

    # Try from dataset root
    candidate = os.path.join(
        DATASET_ROOT,
        path
    )

    if os.path.exists(candidate):
        return candidate

    # Try just the filename
    filename = os.path.basename(path)

    for root, dirs, files in os.walk(DATASET_ROOT):

        if filename in files:

            return os.path.join(
                root,
                filename
            )

    return None


# ============================================================
# EXTRACT LANDMARKS
# ============================================================

def extract_landmarks(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    landmarks = []

    for landmark in hand.landmark:

        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])


    # --------------------------------------------------------
    # Convert to 21 x 3
    # --------------------------------------------------------

    coordinates = np.array(
        landmarks,
        dtype=np.float32
    ).reshape(21, 3)


    # --------------------------------------------------------
    # Normalize relative to wrist
    # --------------------------------------------------------

    wrist = coordinates[0].copy()

    coordinates = coordinates - wrist


    # --------------------------------------------------------
    # Scale normalization
    # --------------------------------------------------------

    distances = np.linalg.norm(
        coordinates,
        axis=1
    )

    max_distance = np.max(distances)

    if max_distance > 0:

        coordinates = (
            coordinates / max_distance
        )


    return coordinates.flatten().tolist()


# ============================================================
# PROCESS ONE SPLIT
# ============================================================

def process_split(
    csv_path,
    output_name
):

    print("\n" + "=" * 65)

    print(
        f"PROCESSING: {output_name.upper()}"
    )

    print("=" * 65)


    df = pd.read_csv(csv_path)

    print(
        f"CSV rows: {len(df):,}"
    )

    print(
        "Columns:",
        list(df.columns)
    )


    # --------------------------------------------------------
    # Find label column
    # --------------------------------------------------------

    possible_label_columns = [
        "label",
        "class",
        "category",
        "target"
    ]

    label_column = None

    for column in possible_label_columns:

        if column in df.columns:

            label_column = column
            break

    if label_column is None:

        raise ValueError(
            "Could not find label column. "
            f"Available columns: {list(df.columns)}"
        )


    # --------------------------------------------------------
    # Find image path column
    # --------------------------------------------------------

    possible_path_columns = [
        "filepath",
        "file_path",
        "path",
        "image_path",
        "image",
        "filename",
        "file"
    ]

    path_column = None

    for column in possible_path_columns:

        if column in df.columns:

            path_column = column
            break

    if path_column is None:

        raise ValueError(
            "Could not find image path column. "
            f"Available columns: {list(df.columns)}"
        )


    print(
        f"Label column: {label_column}"
    )

    print(
        f"Image column: {path_column}"
    )


    output_path = os.path.join(
        OUTPUT_DIR,
        output_name
    )


    successful = 0
    failed = 0
    missing = 0


    with open(
        output_path,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow(header)


        for index, row in df.iterrows():

            image_path = find_image_path(
                row[path_column]
            )

            if image_path is None:

                missing += 1

                continue


            features = extract_landmarks(
                image_path
            )

            if features is None:

                failed += 1

                continue


            label = row[label_column]


            writer.writerow(
                [label] + features
            )

            successful += 1


            if successful % 1000 == 0:

                print(
                    f"Processed: "
                    f"{successful:,}"
                )


    print("\nRESULT")

    print(
        f"Successful: {successful:,}"
    )

    print(
        f"Hand not detected: {failed:,}"
    )

    print(
        f"Missing images: {missing:,}"
    )

    print(
        f"Saved to: {output_path}"
    )

    return successful


# ============================================================
# MAIN
# ============================================================

print("\n" + "=" * 65)

print(
    "ASL LANDMARK DATASET EXTRACTION"
)

print("=" * 65)


train_count = process_split(
    TRAIN_CSV,
    "train_landmarks.csv"
)


val_count = process_split(
    VAL_CSV,
    "validation_landmarks.csv"
)


test_count = process_split(
    TEST_CSV,
    "test_landmarks.csv"
)


# ============================================================
# CLEANUP
# ============================================================

hands.close()


print("\n" + "=" * 65)

print(
    "LANDMARK EXTRACTION COMPLETE"
)

print("=" * 65)

print(
    f"Training samples:   {train_count:,}"
)

print(
    f"Validation samples: {val_count:,}"
)

print(
    f"Testing samples:    {test_count:,}"
)

print("\nFiles created:")

print(
    "data/landmarks/train_landmarks.csv"
)

print(
    "data/landmarks/validation_landmarks.csv"
)

print(
    "data/landmarks/test_landmarks.csv"
)

print("=" * 65)