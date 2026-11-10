import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ==========================================
# Paths
# ==========================================

DATASET_PATH = "data/raw/asl_alphabet_train/asl_alphabet_train"

OUTPUT_DIR = "data/processed"


# ==========================================
# Create output directory
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# Class mapping
# ==========================================

classes = sorted(
    [
        folder
        for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(
            os.path.join(DATASET_PATH, folder)
        )
    ]
)

class_to_index = {
    class_name: index
    for index, class_name in enumerate(classes)
}


print("=" * 60)
print("CLASS MAPPING")
print("=" * 60)

for class_name, index in class_to_index.items():
    print(f"{index:2d} -> {class_name}")


# ==========================================
# Collect image paths
# ==========================================

records = []

for class_name in classes:

    class_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    for filename in os.listdir(class_path):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png")
        ):

            image_path = os.path.join(
                class_path,
                filename
            )

            records.append(
                {
                    "image_path": image_path,
                    "label": class_name,
                    "label_id": class_to_index[class_name]
                }
            )


# ==========================================
# Create dataframe
# ==========================================

df = pd.DataFrame(records)

print("\n" + "=" * 60)
print("DATASET")
print("=" * 60)

print(f"Total images: {len(df):,}")


# ==========================================
# Train/Test split
# ==========================================

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)


# ==========================================
# Validation/Test split
# ==========================================

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# ==========================================
# Save CSV files
# ==========================================

train_df.to_csv(
    os.path.join(OUTPUT_DIR, "train.csv"),
    index=False
)

validation_df.to_csv(
    os.path.join(OUTPUT_DIR, "validation.csv"),
    index=False
)

test_df.to_csv(
    os.path.join(OUTPUT_DIR, "test.csv"),
    index=False
)


# ==========================================
# Print results
# ==========================================

print("\n" + "=" * 60)
print("DATA SPLIT")
print("=" * 60)

print(f"Training images:   {len(train_df):,}")
print(f"Validation images: {len(validation_df):,}")
print(f"Testing images:    {len(test_df):,}")

print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)

print("\nTraining:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation:")
print(validation_df["label"].value_counts().sort_index())

print("\nTesting:")
print(test_df["label"].value_counts().sort_index())

print("\nCSV files created successfully!")