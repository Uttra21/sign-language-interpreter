import numpy as np
import pandas as pd
import torch

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from landmark_model import ASLLandmarkNet


# ============================================================
# PATHS
# ============================================================

TEST_FILE = (
    "data/landmarks/test_landmarks.csv"
)

MODEL_PATH = (
    "models/asl_landmark_best.pth"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("=" * 65)
print("ASL LANDMARK MODEL TEST")
print("=" * 65)

print(
    "Device:",
    device
)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)


class_to_index = (
    checkpoint["class_to_index"]
)

index_to_class = {
    int(index): label
    for index, label
    in checkpoint["index_to_class"].items()
}


feature_columns = (
    checkpoint["feature_columns"]
)


model = ASLLandmarkNet(
    num_classes=len(class_to_index)
)


model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(device)

model.eval()


# ============================================================
# LOAD TEST DATA
# ============================================================

df = pd.read_csv(
    TEST_FILE
)


# Remove "nothing"
df = df[
    df["label"] != "nothing"
].copy()


X = df[
    feature_columns
].values.astype(
    np.float32
)


y = df[
    "label"
].map(
    class_to_index
).values.astype(
    np.int64
)


# ============================================================
# PREDICTION
# ============================================================

X_tensor = torch.tensor(
    X,
    dtype=torch.float32
).to(device)


with torch.no_grad():

    outputs = model(
        X_tensor
    )

    predictions = torch.argmax(
        outputs,
        dim=1
    )


predictions = (
    predictions
    .cpu()
    .numpy()
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y,
    predictions
)


print("\n" + "=" * 65)
print("TEST RESULTS")
print("=" * 65)

print(
    f"Test samples: {len(y):,}"
)

print(
    f"Test accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

target_names = [
    index_to_class[i]
    for i in range(
        len(index_to_class)
    )
]


print("\n" + "=" * 65)
print("CLASSIFICATION REPORT")
print("=" * 65)


print(
    classification_report(
        y,
        predictions,
        labels=list(
            range(
                len(index_to_class)
            )
        ),
        target_names=target_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y,
    predictions
)


print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

np.savetxt(
    "results/landmark_confusion_matrix.csv",
    cm,
    delimiter=",",
    fmt="%d"
)


print(
    "\nConfusion matrix saved to:"
)

print(
    "results/landmark_confusion_matrix.csv"
)


print("\n" + "=" * 65)
print("TEST COMPLETE")
print("=" * 65)