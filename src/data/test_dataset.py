import torch

from torch.utils.data import DataLoader

from asl_dataset import ASLDataset


# ---------------------------------------
# Create dataset
# ---------------------------------------

train_dataset = ASLDataset(
    "data/processed/train.csv"
)


# ---------------------------------------
# Create DataLoader
# ---------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)


# ---------------------------------------
# Print dataset information
# ---------------------------------------

print("=" * 60)
print("PYTORCH DATASET TEST")
print("=" * 60)

print(
    f"Number of training images: "
    f"{len(train_dataset):,}"
)


# ---------------------------------------
# Get one batch
# ---------------------------------------

images, labels = next(iter(train_loader))


print(f"\nImage batch shape: {images.shape}")
print(f"Label batch shape: {labels.shape}")

print(f"\nImage data type: {images.dtype}")
print(f"Label data type: {labels.dtype}")

print(
    f"\nMinimum pixel value: "
    f"{images.min().item():.3f}"
)

print(
    f"Maximum pixel value: "
    f"{images.max().item():.3f}"
)

print("\nFirst 10 labels:")
print(labels[:10])

print("\nDataset test successful!")