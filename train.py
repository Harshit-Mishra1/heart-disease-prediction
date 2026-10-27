"""Train and evaluate baseline classifiers without preprocessing leakage."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import dump
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "heart.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
RANDOM_STATE = 42
TARGET = "target"


def build_pipeline(model, numeric_features: list[str]) -> Pipeline:
    """Keep imputation/scaling inside the pipeline to avoid leakage."""
    prep = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]), numeric_features),
        ],
        remainder="drop",
    )
    return Pipeline([("preprocess", prep), ("classifier", model)])


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Dataset must contain a '{TARGET}' column.")
    if df[TARGET].nunique() != 2:
        raise ValueError("Expected a binary target with exactly two classes.")

    X = df.drop(columns=TARGET)
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight=None),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "RBF SVM": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted = {}
    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} features")
    print(f"Train rows: {len(X_train)} | Test rows: {len(X_test)}\n")

    for name, estimator in candidates.items():
        pipeline = build_pipeline(estimator, list(X.columns))
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
        results.append({
            "model": name,
            "cv_roc_auc_mean": float(scores.mean()),
            "cv_roc_auc_std": float(scores.std()),
        })
        print(f"{name:22s} CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    comparison = pd.DataFrame(results).sort_values("cv_roc_auc_mean", ascending=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = fitted[best_name]
    pred = best_model.predict(X_test)
    proba = best_model.predict_proba(X_test)[:, 1]
    metrics = {
        "selected_model_by_cv_roc_auc": best_name,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
        "cv_results": results,
        "note": "Educational dataset only; metrics are not evidence of clinical validity.",
    }
    comparison.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORT_DIR / "classification_report.txt").write_text(
        classification_report(y_test, pred, zero_division=0), encoding="utf-8"
    )
    dump(best_model, MODEL_DIR / "heart_disease_pipeline.joblib")

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Predicted 0", "Predicted 1"],
                yticklabels=["Actual 0", "Actual 1"])
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "confusion_matrix.png", dpi=160)
    plt.close()

    print("\nHeld-out test metrics (small educational dataset):")
    for key in ["test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]:
        print(f"{key.replace('_', ' ').title():18s}: {metrics[key]:.3f}")
    print(f"\nSaved model: {MODEL_DIR / 'heart_disease_pipeline.joblib'}")
    print(f"Saved reports: {REPORT_DIR}")
    print("Reminder: do not use this model for medical decisions.")


if __name__ == "__main__":
    main()
