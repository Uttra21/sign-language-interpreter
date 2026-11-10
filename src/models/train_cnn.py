import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Allow imports from src/data and src/models
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "data")
    )
)

from asl_dataset import ASLDataset

from cnn_model import ASLCNN


# ============================================================
# Configuration
# ============================================================

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/validation.csv"

BATCH_SIZE = 32

# Small sanity-check run
MAX_TRAIN_SAMPLES = 2_900
MAX_VAL_SAMPLES = 580

EPOCHS = 2

LEARNING_RATE = 0.001

MODEL_PATH = "models/cnn_sanity_check.pth"


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

print("=" * 60)
print("CNN TRAINING")
print("=" * 60)

print(f"Device: {device}")


# ============================================================
# Load datasets
# ============================================================

train_dataset = ASLDataset(TRAIN_CSV)

val_dataset = ASLDataset(VAL_CSV)


# ============================================================
# Use a small subset for the first test
# ============================================================

train_indices = list(
    range(
        min(MAX_TRAIN_SAMPLES, len(train_dataset))
    )
)

val_indices = list(
    range(
        min(MAX_VAL_SAMPLES, len(val_dataset))
    )
)

train_dataset = Subset(
    train_dataset,
    train_indices
)

val_dataset = Subset(
    val_dataset,
    val_indices
)


print(f"Training samples:   {len(train_dataset):,}")
print(f"Validation samples: {len(val_dataset):,}")


# ============================================================
# DataLoaders
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# Model
# ============================================================

model = ASLCNN(
    num_classes=29
).to(device)


# ============================================================
# Loss + Optimizer
# ============================================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# Training
# ============================================================

for epoch in range(EPOCHS):

    # -------------------------
    # Training
    # -------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * images.size(0)
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        100 * correct / total
    )


    # -------------------------
    # Validation
    # -------------------------

    model.eval()

    val_loss_total = 0.0

    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * images.size(0)
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()


    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        100 * val_correct / val_total
    )


    # -------------------------
    # Print results
    # -------------------------

    print(
        f"\nEpoch [{epoch + 1}/{EPOCHS}]"
    )

    print(
        f"Training Loss: {train_loss:.4f}"
    )

    print(
        f"Training Accuracy: "
        f"{train_accuracy:.2f}%"
    )

    print(
        f"Validation Loss: {val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.2f}%"
    )


# ============================================================
# Save model
# ============================================================

os.makedirs(
    os.path.dirname(MODEL_PATH),
    exist_ok=True
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    f"Model saved to: {MODEL_PATH}"
)