import os
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------

FEATURES_DIR = r"C:\Users\omar1\Downloads\Footskillz\features"
INPUT_FILE = "features_all_windows.csv"
OUTPUT_FILE = "features_balanced_windows.csv"
RANDOM_SEED = 42   # for reproducible sampling & shuffling


# -----------------------------
# MAIN
# -----------------------------

def main():
    features_root = Path(FEATURES_DIR)
    in_path = features_root / INPUT_FILE

    if not in_path.exists():
        raise FileNotFoundError(f"Could not find input features file at: {in_path}")

    # Load all window features
    df = pd.read_csv(in_path)

    if "skill" not in df.columns:
        raise ValueError("Input CSV must have a 'skill' column for class labels.")

    # Count windows per skill
    counts = df["skill"].value_counts()
    print("Original window counts per skill:")
    print(counts)
    print()

    # Determine target count = minimum across skills (under-sampling)
    min_count = counts.min()
    print(f"Target windows per skill (under-sample to smallest class): {min_count}\n")

    balanced_parts = []

    # Under-sample each skill
    for skill, count in counts.items():
        df_skill = df[df["skill"] == skill]

        # sample without replacement
        df_sampled = df_skill.sample(
            n=min_count,
            replace=False,
            random_state=RANDOM_SEED
        )

        balanced_parts.append(df_sampled)

    # Concatenate and shuffle all sampled windows
    balanced_df = pd.concat(balanced_parts, axis=0)
    balanced_df = balanced_df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # Sanity check counts
    new_counts = balanced_df["skill"].value_counts()
    print("Balanced window counts per skill:")
    print(new_counts)
    print()

    # Save to output CSV
    out_path = features_root / OUTPUT_FILE
    balanced_df.to_csv(out_path, index=False)
    print(f"Balanced dataset saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
