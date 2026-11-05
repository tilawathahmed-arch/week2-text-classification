#!/usr/bin/env python3
"""Train and evaluate baseline text classifiers on Cornell sentence polarity v1.0.

Expected input files (one snippet per line):
  * rt-polarity.pos and rt-polarity.neg (official archive), or
  * rt-polarity-pos.txt and rt-polarity-neg.txt (public mirror).

Example:
  python train_classifier.py --download --data_dir data --out_dir outputs

Vectorizers are inside sklearn Pipelines so vocabulary and IDF statistics are
learned only from each training fold during cross-validation.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

CORNELL_ARCHIVE = (
    "https://www.cs.cornell.edu/people/pabo/movie-review-data/"
    "rt-polaritydata.tar.gz"
)
MIRROR_BASE = (
    "https://raw.githubusercontent.com/Elijas/sentence-polarity-dataset-v1.0/"
    "master/"
)
MIRROR_FILES = {"positive": "rt-polarity.pos", "negative": "rt-polarity.neg"}
TOKEN_PATTERN = r"(?u)\b\w+(?:'\w+)?\b"
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")
TAG_RE = re.compile(r"<[^>]+>")


def locate_file(data_dir: Path, label: str) -> Path | None:
    names = {
        "positive": ["rt-polarity.pos", "rt-polarity-pos.txt", "rt-polarity.pos.txt"],
        "negative": ["rt-polarity.neg", "rt-polarity-neg.txt", "rt-polarity.neg.txt"],
    }[label]
    folders = [data_dir]
    if data_dir.exists():
        folders.extend(p for p in data_dir.rglob("*") if p.is_dir())
    for folder in folders:
        for name in names:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def safe_extract(tar: tarfile.TarFile, destination: Path) -> None:
    """Extract regular files/directories only and block archive path traversal."""
    base = destination.resolve()
    for member in tar.getmembers():
        if member.issym() or member.islnk() or not (member.isfile() or member.isdir()):
            continue
        target = (destination / member.name).resolve()
        if target == base or base in target.parents:
            tar.extract(member, destination)


def download_dataset(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "rt-polaritydata.tar.gz"
    try:
        print(f"Trying official Cornell archive: {CORNELL_ARCHIVE}")
        urllib.request.urlretrieve(CORNELL_ARCHIVE, archive)
        with tarfile.open(archive, "r:gz") as tar:
            safe_extract(tar, data_dir)
        if locate_file(data_dir, "positive") and locate_file(data_dir, "negative"):
            print("Official Cornell archive downloaded and extracted.")
            return
        raise FileNotFoundError("Expected class files were not found in the archive.")
    except Exception as official_error:
        print(f"Official archive retrieval failed: {official_error}")
        print("Trying public plain-text mirror...")
        for label, filename in MIRROR_FILES.items():
            destination = data_dir / filename
            urllib.request.urlretrieve(MIRROR_BASE + filename, destination)
            print(f"Downloaded {label}: {destination}")


def load_dataset(data_dir: Path) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for label in ("negative", "positive"):
        path = locate_file(data_dir, label)
        if path is None:
            raise FileNotFoundError(
                f"Could not find the {label} file in {data_dir}. Use --download or "
                "place the Cornell source text files in the data directory."
            )
        lines = path.read_text(encoding="latin-1", errors="replace").splitlines()
        for index, line in enumerate(lines, start=1):
            text = line.strip()
            if text:
                records.append({
                    "snippet_id": f"{label[:3]}_{index:05d}",
                    "label": label,
                    "text": text,
                    "source_file": path.name,
                })
    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No non-empty snippets were loaded.")
    if data["label"].nunique() != 2:
        raise ValueError("Both positive and negative class files are required.")
    return data


def normalize_text(text: str) -> str:
    """Conservative normalization; keep negation, contractions, and punctuation."""
    value = (text or "").lower()
    value = URL_RE.sub(" URLTOKEN ", value)
    value = EMAIL_RE.sub(" EMAILTOKEN ", value)
    value = TAG_RE.sub(" ", value)
    value = re.sub(r"[^a-z0-9'!?.,;:\s]", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def build_candidates() -> dict[str, object]:
    count_vectorizer = CountVectorizer(
        ngram_range=(1, 2), min_df=2, max_df=0.95,
        token_pattern=TOKEN_PATTERN, lowercase=True,
    )
    tfidf_params = dict(
        ngram_range=(1, 2), min_df=2, max_df=0.95,
        sublinear_tf=True, strip_accents="unicode",
        token_pattern=TOKEN_PATTERN, lowercase=True,
    )
    return {
        "Dummy most-frequent baseline": DummyClassifier(strategy="most_frequent"),
        "CountVectorizer + MultinomialNB": Pipeline([
            ("vectorizer", count_vectorizer),
            ("classifier", MultinomialNB(alpha=1.0)),
        ]),
        "TF-IDF + MultinomialNB": Pipeline([
            ("vectorizer", TfidfVectorizer(**tfidf_params)),
            ("classifier", MultinomialNB(alpha=0.5)),
        ]),
        "TF-IDF + LogisticRegression": Pipeline([
            ("vectorizer", TfidfVectorizer(**tfidf_params)),
            ("classifier", LogisticRegression(
                C=1.0, max_iter=2000, solver="liblinear", random_state=42
            )),
        ]),
        "TF-IDF + LinearSVC": Pipeline([
            ("vectorizer", TfidfVectorizer(**tfidf_params)),
            ("classifier", LinearSVC(C=1.0, random_state=42)),
        ]),
    }


def get_decision_scores(model: object, texts: pd.Series) -> np.ndarray | None:
    """Return ranking scores for either a Pipeline or a plain estimator."""
    classifier = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(texts)).ravel()
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(texts)
        classes = list(classifier.classes_)
        return probabilities[:, classes.index("positive")]
    return None


def save_top_features(model: Pipeline, output_path: Path) -> None:
    vectorizer = model.named_steps["vectorizer"]
    classifier = model.named_steps["classifier"]
    terms = vectorizer.get_feature_names_out()
    weights: np.ndarray | None = None
    if hasattr(classifier, "coef_"):
        weights = np.asarray(classifier.coef_).ravel()
    elif hasattr(classifier, "feature_log_prob_"):
        classes = list(classifier.classes_)
        pos_idx, neg_idx = classes.index("positive"), classes.index("negative")
        weights = classifier.feature_log_prob_[pos_idx] - classifier.feature_log_prob_[neg_idx]
    if weights is None or len(weights) != len(terms):
        return
    order = np.argsort(weights)
    rows = [
        {"class_association": "negative", "term": terms[i], "weight": float(weights[i])}
        for i in order[:20]
    ]
    rows += [
        {"class_association": "positive", "term": terms[i], "weight": float(weights[i])}
        for i in order[-20:][::-1]
    ]
    pd.DataFrame(rows).to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline Cornell sentiment classifiers")
    parser.add_argument("--data_dir", default="data", help="Folder containing source files")
    parser.add_argument("--out_dir", default="outputs", help="Output folder")
    parser.add_argument("--download", action="store_true", help="Download the public corpus")
    parser.add_argument("--test_size", type=float, default=0.20, help="Final holdout fraction")
    parser.add_argument("--folds", type=int, default=5, help="Stratified CV folds on training partition")
    parser.add_argument("--random_state", type=int, default=42, help="Reproducible split/CV seed")
    args = parser.parse_args()

    if not 0.1 <= args.test_size <= 0.4:
        parser.error("--test_size must be between 0.10 and 0.40")
    if args.folds < 2:
        parser.error("--folds must be at least 2")

    data_dir, out_dir = Path(args.data_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.download and (not locate_file(data_dir, "positive") or not locate_file(data_dir, "negative")):
        download_dataset(data_dir)

    data = load_dataset(data_dir)
    data["normalized_text"] = data["text"].map(normalize_text)
    data = data.loc[data["normalized_text"].str.len() > 0].reset_index(drop=True)
    if data["label"].value_counts().min() < args.folds:
        raise ValueError("Each class must contain at least as many rows as CV folds.")

    X, y = data["normalized_text"], data["label"]
    X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
        X, y, data["snippet_id"], test_size=args.test_size,
        random_state=args.random_state, stratify=y,
    )

    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.random_state)
    scoring = {
        "accuracy": "accuracy",
        "precision_macro": make_scorer(precision_score, average="macro", zero_division=0),
        "recall_macro": make_scorer(recall_score, average="macro", zero_division=0),
        "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    }
    candidates = build_candidates()
    comparison_rows = []
    for name, estimator in candidates.items():
        print(f"Cross-validating: {name}")
        scores = cross_validate(
            estimator, X_train, y_train, cv=cv, scoring=scoring,
            n_jobs=-1, return_train_score=False, error_score="raise",
        )
        row = {"model": name}
        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values, ddof=1))
        comparison_rows.append(row)

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["f1_macro_mean", "accuracy_mean"], ascending=False
    ).reset_index(drop=True)
    comparison.to_csv(out_dir / "cross_validation_comparison.csv", index=False)
    best_name = str(comparison.iloc[0]["model"])
    best_model = candidates[best_name]
    print(f"Selected by mean CV macro-F1: {best_name}")
    best_model.fit(X_train, y_train)
    predictions = best_model.predict(X_test)

    labels = ["negative", "positive"]
    cm = confusion_matrix(y_test, predictions, labels=labels)
    metrics = {
        "selected_model": best_name,
        "selection_rule": "Highest mean stratified CV macro-F1 on training partition; accuracy breaks ties",
        "test_documents": int(len(y_test)),
        "test_class_counts": {str(k): int(v) for k, v in y_test.value_counts().items()},
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_test, predictions, labels=labels, output_dict=True, zero_division=0
        ),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
        "run_metadata": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "pandas_version": pd.__version__,
            "numpy_version": np.__version__,
            "random_state": args.random_state,
            "test_size": args.test_size,
            "cv_folds": args.folds,
        },
    }
    scores = get_decision_scores(best_model, X_test)
    if scores is not None:
        positive_binary = (y_test == "positive").astype(int)
        metrics["roc_auc"] = float(roc_auc_score(positive_binary, scores))

    (out_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    test_texts = data.set_index("snippet_id").loc[id_test.to_numpy(), "text"].to_numpy()
    pred_frame = pd.DataFrame({
        "snippet_id": id_test.to_numpy(),
        "text": test_texts,
        "true_label": y_test.to_numpy(),
        "predicted_label": predictions,
    })
    pred_frame["error_type"] = np.where(
        pred_frame["true_label"] == pred_frame["predicted_label"], "correct",
        np.where(pred_frame["predicted_label"] == "positive", "false_positive", "false_negative")
    )
    pred_frame.to_csv(out_dir / "test_predictions.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Negative", "Positive"])
    ax.set_yticks([0, 1], labels=["Negative", "Positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"Confusion matrix: {best_name}")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center", fontsize=13)
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    plot_df = comparison.sort_values("f1_macro_mean")
    bars = ax.barh(plot_df["model"], plot_df["f1_macro_mean"], color="#2C8C84")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean cross-validation macro-F1")
    ax.set_title("Model selection: stratified cross-validation")
    ax.grid(axis="x", alpha=.25)
    for bar, value in zip(bars, plot_df["f1_macro_mean"]):
        ax.text(value + .008, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center")
    fig.tight_layout()
    fig.savefig(out_dir / "cv_model_comparison.png", dpi=180)
    plt.close(fig)

    if isinstance(best_model, Pipeline):
        save_top_features(best_model, out_dir / "top_model_features.csv")
    else:
        pd.DataFrame(columns=["class_association", "term", "weight"]).to_csv(
            out_dir / "top_model_features.csv", index=False
        )
    joblib.dump(best_model, out_dir / "best_model.joblib")
    data[["snippet_id", "label", "text", "normalized_text"]].to_csv(
        out_dir / "prepared_dataset.csv", index=False
    )

    print("\\nFinal held-out test metrics")
    for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc"):
        if key in metrics:
            print(f"{key}: {metrics[key]:.4f}")
    print(f"Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
