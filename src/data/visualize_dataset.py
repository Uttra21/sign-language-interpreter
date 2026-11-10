import os

import matplotlib.pyplot as plt
from PIL import Image


# ---------------------------------------
# Dataset path
# ---------------------------------------

DATASET_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"

OUTPUT_PATH = "results/dataset_samples.png"


# ---------------------------------------
# Get classes
# ---------------------------------------

classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, folder)
        )
    ]
)


# ---------------------------------------
# Create figure
# ---------------------------------------

fig, axes = plt.subplots(
    5,
    6,
    figsize=(15, 13)
)

axes = axes.flatten()


# ---------------------------------------
# Display one image per class
# ---------------------------------------

for index, class_name in enumerate(classes):

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    image_files = [
        file
        for file in os.listdir(class_path)
        if file.lower().endswith(
            (".jpg", ".jpeg", ".png")
        )
    ]

    if not image_files:
        continue

    image_path = os.path.join(
        class_path,
        image_files[0]
    )

    image = Image.open(image_path)

    axes[index].imshow(image)
    axes[index].set_title(
        class_name,
        fontsize=12,
        fontweight="bold"
    )

    axes[index].axis("off")


# ---------------------------------------
# Hide unused subplot
# ---------------------------------------

for index in range(len(classes), len(axes)):
    axes[index].axis("off")


# ---------------------------------------
# Figure title
# ---------------------------------------

fig.suptitle(
    "ASL Alphabet Dataset - Sample Images",
    fontsize=18,
    fontweight="bold"
)

plt.tight_layout()


# ---------------------------------------
# Save result
# ---------------------------------------

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(
    f"Sample grid saved to: {OUTPUT_PATH}"
)