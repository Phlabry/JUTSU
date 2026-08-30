import math
import os
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..")))

import cv2 as cv
import joblib
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.ensemble import (HistGradientBoostingClassifier, VotingClassifier)
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from gesture.custom_base import _landmarks_to_array, _features_from_array

_ROOT       = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATASET_DIR = os.path.join(_ROOT, "datasets", "hollow_purple")
MODEL_OUT   = os.path.join(_ROOT, "models", "hollow_purple", "model.pkl")
HAND_MODEL  = os.path.join(_ROOT, "hand_landmarker.task")
CLASSES     = ["charge", "release"]
EXTS        = {".jpg", ".jpeg", ".png"}

_aug_rng = np.random.default_rng(42)


def _augment(pts: np.ndarray):
    # Augmentation applied to raw (21,3) arrays before feature extraction so
    # derived features (angles, distances) stay consistent with the geometry.
    yield pts
    m = pts.copy();  m[:, 0] *= -1
    yield m
    for deg in (-12, -6, 6, 12):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        yield pts @ R.T
    for sigma in (0.025, 0.055):
        yield pts + _aug_rng.normal(0, sigma, pts.shape).astype(np.float32)
    for factor in (0.85, 1.15):
        yield pts * factor


def _extract_dataset():
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
    )
    detector = vision.HandLandmarker.create_from_options(options)

    pts_list, labels = [], []

    for label in CLASSES:
        folder = os.path.join(DATASET_DIR, label)
        files  = sorted(f for f in os.listdir(folder)
                        if os.path.splitext(f)[1].lower() in EXTS)
        found  = 0
        for fname in files:
            img = cv.imread(os.path.join(folder, fname))
            if img is None:
                continue
            rgb    = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_img)
            if result.hand_landmarks:
                pts_list.append(_landmarks_to_array(result.hand_landmarks[0]))
                labels.append(label)
                found += 1
        print(f"  {label:>10}: {found}/{len(files)} usable images")

    detector.close()
    return pts_list, labels


def _build_augmented(pts_list, labels):
    X, y = [], []
    for pts, label in zip(pts_list, labels):
        for aug_pts in _augment(pts):
            X.append(_features_from_array(aug_pts))
            y.append(label)
    return np.array(X, dtype=np.float32), np.array(y)


def _build_model():
    svm = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    SVC(
            kernel="rbf", probability=True,
            C=10, gamma="scale",
            class_weight="balanced",
        )),
    ])

    hgb = HistGradientBoostingClassifier(
        max_iter=150,           # capped for fast inference (~1ms vs 7ms at 500 trees)
        learning_rate=0.08,
        max_depth=5,
        min_samples_leaf=15,
        l2_regularization=0.5,
        max_features=0.8,
        early_stopping=True,
        validation_fraction=0.08,
        n_iter_no_change=20,
        random_state=42,
        class_weight="balanced",
    )

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            learning_rate_init=3e-4,
            max_iter=800,
            early_stopping=True,
            validation_fraction=0.08,
            n_iter_no_change=20,
            random_state=42,
        )),
    ])

    # weights 1:2:1 — HGB empirically best on tabular pose data
    return VotingClassifier(
        estimators=[("svm", svm), ("hgb", hgb), ("mlp", mlp)],
        voting="soft",
        weights=[1, 2, 1],
        n_jobs=1,   # each sub-model already uses all cores where possible
    )


def main():
    print("=" * 58)
    print("  Hollow Purple — gesture classifier training  v2")
    print("=" * 58)

    print("\n[1/4] Extracting landmarks from dataset images...")
    pts_list, labels = _extract_dataset()
    print(f"        Raw samples: {len(pts_list)}")
    if not pts_list:
        print("ERROR: no landmarks extracted — check dataset paths.")
        sys.exit(1)

    print("\n[2/4] Augmenting + extracting 83-D features...")
    X, y = _build_augmented(pts_list, labels)
    print(f"        Total samples after 10x augmentation: {len(X)}")
    print(f"        Feature dimensionality: {X.shape[1]}")

    le    = LabelEncoder()
    y_enc = le.fit_transform(y)
    print(f"        Classes: {list(le.classes_)}")
    counts = {c: int((y == c).sum()) for c in le.classes_}
    print(f"        Class distribution: {counts}")

    print("\n[3/4] Training SVM + HistGradientBoosting + MLP ensemble...")
    model = _build_model()
    model.fit(X, y_enc)

    print("\n[4/4] Evaluating with 5-fold stratified cross-validation...")
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y_enc, cv=cv_strategy, n_jobs=-1)

    acc = float((y_pred == y_enc).mean()) * 100
    print(f"\n  5-fold CV accuracy: {acc:.2f}%")
    print("\n  Per-class report (cross-validated):")
    print(classification_report(y_enc, y_pred, target_names=le.classes_, digits=3))

    print("  Accuracy on original samples only (no augmentation):")
    # originals come first in _build_augmented — slice without augmented copies
    X_clean = np.array([_features_from_array(pts) for pts in pts_list], dtype=np.float32)
    y_clean = le.transform(labels)
    y_clean_pred = model.predict(X_clean)
    acc_clean = float((y_clean_pred == y_clean).mean()) * 100
    print(f"    Training-set accuracy (upper bound): {acc_clean:.2f}%")

    print("\n  Saving model...")
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump({"model": model, "labels": le.classes_}, MODEL_OUT)
    print(f"  Saved -> {MODEL_OUT}")
    print("\nDone. Run main.py to use the trained gestures.")


if __name__ == "__main__":
    main()
