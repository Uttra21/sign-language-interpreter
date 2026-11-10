import os
from collections import Counter

from PIL import Image


# ---------------------------------------
# Dataset path
# ---------------------------------------

DATASET_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"


# ---------------------------------------
# Check classes
# ---------------------------------------

classes = sorted(os.listdir(DATASET_PATH))

print("=" * 60)
print("ASL DATASET EXPLORATION")
print("=" * 60)

print(f"\nNumber of classes: {len(classes)}")

print("\nClasses:")
print(classes)


# ---------------------------------------
# Count images per class
# ---------------------------------------

image_counts = {}

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    if not os.path.isdir(class_path):
        continue

    images = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    image_counts[class_name] = len(images)


print("\n" + "=" * 60)
print("IMAGES PER CLASS")
print("=" * 60)

for class_name, count in image_counts.items():

    print(
        f"{class_name:10s} : {count:,}"
    )


# ---------------------------------------
# Dataset statistics
# ---------------------------------------

total_images = sum(image_counts.values())

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Total images: {total_images:,}")
print(f"Number of classes: {len(image_counts)}")

if image_counts:

    print(
        f"Minimum images in a class: "
        f"{min(image_counts.values()):,}"
    )

    print(
        f"Maximum images in a class: "
        f"{max(image_counts.values()):,}"
    )


# ---------------------------------------
# Inspect first image
# ---------------------------------------

first_class = classes[0]

class_path = os.path.join(
    DATASET_PATH,
    first_class
)

image_files = [
    file
    for file in os.listdir(class_path)
    if file.lower().endswith(
        (".jpg", ".jpeg", ".png")
    )
]

if image_files:

    first_image = os.path.join(
        class_path,
        image_files[0]
    )

    image = Image.open(first_image)

    print("\n" + "=" * 60)
    print("SAMPLE IMAGE INFORMATION")
    print("=" * 60)

    print(f"Class: {first_class}")
    print(f"File: {image_files[0]}")
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")