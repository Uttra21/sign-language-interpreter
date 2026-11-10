import pandas as pd
import os


# ============================================================
# Paths
# ============================================================

files = {
    "Training": "data/landmarks/train_landmarks.csv",
    "Validation": "data/landmarks/validation_landmarks.csv",
    "Testing": "data/landmarks/test_landmarks.csv"
}


# ============================================================
# Check each split
# ============================================================

for split_name, file_path in files.items():

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} LANDMARK DATASET")
    print("=" * 65)

    if not os.path.exists(file_path):

        print(
            f"File not found: {file_path}"
        )

        continue


    df = pd.read_csv(file_path)


    print(
        f"Total samples: {len(df):,}"
    )

    print(
        f"Total columns: {len(df.columns)}"
    )


    # --------------------------------------------------------
    # Missing values
    # --------------------------------------------------------

    missing = df.isnull().sum().sum()

    print(
        f"Missing values: {missing}"
    )


    # --------------------------------------------------------
    # Feature count
    # --------------------------------------------------------

    feature_count = len(df.columns) - 1

    print(
        f"Landmark features: {feature_count}"
    )


    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print("\nClass distribution:")

    counts = (
        df["label"]
        .value_counts()
        .sort_index()
    )

    print(counts)


    # --------------------------------------------------------
    # Min / max samples
    # --------------------------------------------------------

    print(
        f"\nMinimum class samples: "
        f"{counts.min():,}"
    )

    print(
        f"Maximum class samples: "
        f"{counts.max():,}"
    )


print("\n" + "=" * 65)
print("CHECK COMPLETE")
print("=" * 65)