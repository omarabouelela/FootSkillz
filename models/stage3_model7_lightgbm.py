"""
Stage 3 - Model 7: LightGBM (LGBMClassifier)
-------------------------------------------
- Loads features_balanced_windows.csv
- Encodes labels
- Train/test split (stratified)
- Pipeline: SimpleImputer -> StandardScaler -> LGBMClassifier
- 5-fold cross-validation (accuracy)
- Reports training time, test metrics, inference time, learning rate.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from lightgbm import LGBMClassifier

# -----------------------------
# CONFIG
# -----------------------------

FEATURES_DIR = "features"
INPUT_FILE = r"C:\Users\omar1\Downloads\Footskillz\features\features_balanced_windows.csv"
RANDOM_SEED = 42

# batch size used ONLY for measuring inference time
INFERENCE_BATCH_SIZE = 64

# LightGBM hyperparameters (tune later if you want)
LGBM_N_ESTIMATORS = 300
LGBM_LEARNING_RATE = 0.05
LGBM_NUM_LEAVES = 31
LGBM_MAX_DEPTH = -1          # -1 means no explicit limit
LGBM_SUBSAMPLE = 0.9
LGBM_COLSAMPLE_BYTREE = 0.9


# -----------------------------
# MAIN
# -----------------------------

def main():
    features_path = Path(FEATURES_DIR) / INPUT_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Could not find {features_path}")

    # 1) Load data
    df = pd.read_csv(features_path)

    if "skill" not in df.columns:
        raise ValueError("Expected a 'skill' column as label in the features CSV.")

    print(f"Loaded dataset from {features_path}")
    print(f"Shape (rows, cols): {df.shape}\n")

    # 2) Separate features and labels
    y = df["skill"].astype(str)

    # Non-feature columns (IDs / metadata)
    drop_cols = [
        "skill",
        "recording",
        "window_id",
        "start_time",
        "end_time",
        "n_samples",
        "fs",
        "pause_time",
    ]

    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].to_numpy(dtype=float)

    print(f"Number of feature columns: {X.shape[1]}")
    print(f"Feature columns example: {feature_cols[:10]}\n")

    # 3) Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = list(le.classes_)
    n_classes = len(class_names)

    print("Classes and encoded labels:")
    for idx, name in enumerate(class_names):
        print(f"  {idx}: {name}")
    print(f"\nNumber of classes: {n_classes}\n")

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y_enc,
    )

    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}\n")

    # 5) Build pipeline: Imputer -> Scaler -> LGBMClassifier
    lgbm_clf = LGBMClassifier(
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=LGBM_LEARNING_RATE,
        num_leaves=LGBM_NUM_LEAVES,
        max_depth=LGBM_MAX_DEPTH,
        subsample=LGBM_SUBSAMPLE,
        colsample_bytree=LGBM_COLSAMPLE_BYTREE,
        objective="multiclass",
        num_class=n_classes,
        random_state=RANDOM_SEED,
        n_jobs=1,
        verbosity=-1
    )

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", lgbm_clf),
        ]
    )

    # 6) Cross-validation on full dataset with pipeline
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        pipeline,
        X,
        y_enc,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print("5-fold cross-validation accuracy scores:")
    print(cv_scores)
    print(f"Mean CV accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}\n")

    # 7) Fit pipeline on training split
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    train_time = time.perf_counter() - t0

    # 8) Evaluate on test set
    y_pred = pipeline.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            digits=4,
        )
    )

    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))
    print()

    # 9) Inference time measurement (batch prediction)
    n_test = X_test.shape[0]
    batches_idx = [
        slice(i, i + INFERENCE_BATCH_SIZE)
        for i in range(0, n_test, INFERENCE_BATCH_SIZE)
    ]

    t0 = time.perf_counter()
    for sl in batches_idx:
        _ = pipeline.predict(X_test[sl])
    infer_time = time.perf_counter() - t0

    total_predictions = n_test
    time_per_sample = infer_time / total_predictions if total_predictions > 0 else float(
        "nan"
    )

    print(f"Training time: {train_time:.6f} seconds")
    print(
        f"Inference time (all {n_test} test samples, batch size={INFERENCE_BATCH_SIZE}): "
        f"{infer_time:.6f} seconds"
    )
    print(f"Average inference time per sample: {time_per_sample * 1000:.6f} ms\n")

    # 10) Learning rate / batch size note
    print("Model details:")
    print("  Model type: LGBMClassifier (lightgbm.LGBMClassifier)")
    print(f"  learning_rate: {LGBM_LEARNING_RATE}")
    print(f"  n_estimators: {LGBM_N_ESTIMATORS}")
    print(f"  num_leaves: {LGBM_NUM_LEAVES}")
    print(f"  max_depth: {LGBM_MAX_DEPTH}")
    print(f"  subsample: {LGBM_SUBSAMPLE}")
    print(f"  colsample_bytree: {LGBM_COLSAMPLE_BYTREE}")
    print("  Gradient boosting model; learning_rate controls step size of each boosting iteration.")
    print(f"  Inference batch size used for timing: {INFERENCE_BATCH_SIZE}")


if __name__ == "__main__":
    main()
