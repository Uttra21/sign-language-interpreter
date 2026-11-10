import os

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    Dataset,
    DataLoader
)

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

from landmark_model import ASLLandmarkNet


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = (
    "data/landmarks/train_landmarks.csv"
)

VAL_FILE = (
    "data/landmarks/validation_landmarks.csv"
)

MODEL_DIR = "models"

MODEL_PATH = (
    "models/asl_landmark_best.pth"
)

BATCH_SIZE = 128

EPOCHS = 30

LEARNING_RATE = 0.001


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("=" * 65)
print("ASL LANDMARK MODEL TRAINING")
print("=" * 65)

print(
    "Device:",
    device
)


if device.type == "cuda":

    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ============================================================
# LOAD DATA
# ============================================================

train_df = pd.read_csv(
    TRAIN_FILE
)

val_df = pd.read_csv(
    VAL_FILE
)


# ============================================================
# REMOVE 'nothing'
# ============================================================

train_df = train_df[
    train_df["label"] != "nothing"
].copy()

val_df = val_df[
    val_df["label"] != "nothing"
].copy()


print(
    "\nTraining samples:",
    len(train_df)
)

print(
    "Validation samples:",
    len(val_df)
)


# ============================================================
# CLASS MAPPING
# ============================================================

classes = sorted(
    train_df["label"].unique()
)

class_to_index = {
    label: index
    for index, label in enumerate(classes)
}

index_to_class = {
    index: label
    for label, index in class_to_index.items()
}


print("\nClasses:")

for index, label in index_to_class.items():

    print(
        f"{index:2d} -> {label}"
    )


print(
    "\nNumber of classes:",
    len(classes)
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    column
    for column in train_df.columns
    if column != "label"
]


X_train = train_df[
    feature_columns
].values.astype(
    np.float32
)

y_train = train_df[
    "label"
].map(
    class_to_index
).values.astype(
    np.int64
)


X_val = val_df[
    feature_columns
].values.astype(
    np.float32
)

y_val = val_df[
    "label"
].map(
    class_to_index
).values.astype(
    np.int64
)


# ============================================================
# DATASET
# ============================================================

class LandmarkDataset(Dataset):

    def __init__(
        self,
        X,
        y
    ):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )


    def __len__(self):

        return len(self.X)


    def __getitem__(
        self,
        index
    ):

        return (
            self.X[index],
            self.y[index]
        )


train_dataset = LandmarkDataset(
    X_train,
    y_train
)

val_dataset = LandmarkDataset(
    X_val,
    y_val
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

class_counts = np.bincount(
    y_train,
    minlength=len(classes)
)

total = class_counts.sum()

weights = total / (
    len(classes) * class_counts
)

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)


# ============================================================
# MODEL
# ============================================================

model = ASLLandmarkNet(
    num_classes=len(classes)
)

model = model.to(device)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=weights
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

best_val_accuracy = 0.0


os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    train_predictions = []

    train_targets = []


    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(device)

        y_batch = y_batch.to(device)


        optimizer.zero_grad()


        outputs = model(
            X_batch
        )


        loss = criterion(
            outputs,
            y_batch
        )


        loss.backward()

        optimizer.step()


        running_loss += (
            loss.item()
            * X_batch.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        train_predictions.extend(
            predictions.cpu().numpy()
        )

        train_targets.extend(
            y_batch.cpu().numpy()
        )


    train_loss = (
        running_loss
        / len(train_dataset)
    )


    train_accuracy = accuracy_score(
        train_targets,
        train_predictions
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss = 0.0

    val_predictions = []

    val_targets = []


    with torch.no_grad():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(device)

            y_batch = y_batch.to(device)


            outputs = model(
                X_batch
            )


            loss = criterion(
                outputs,
                y_batch
            )


            val_loss += (
                loss.item()
                * X_batch.size(0)
            )


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )

            val_targets.extend(
                y_batch.cpu().numpy()
            )


    val_loss = (
        val_loss
        / len(val_dataset)
    )


    val_accuracy = accuracy_score(
        val_targets,
        val_predictions
    )


    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss: {val_loss:.4f}"
    )

    print(
        f"Val Accuracy: "
        f"{val_accuracy * 100:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "class_to_index":
                    class_to_index,

                "index_to_class":
                    index_to_class,

                "feature_columns":
                    feature_columns
            },
            MODEL_PATH
        )

        print(
            "✓ BEST MODEL SAVED"
        )


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 65)

print(
    "LANDMARK TRAINING COMPLETE"
)

print("=" * 65)

print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print(
    f"Model saved to: {MODEL_PATH}"
)

print("=" * 65)