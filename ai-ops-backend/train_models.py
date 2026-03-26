"""
train_models.py
================================================================================
Trains TWO ML models used by the AI Support system:

  MODEL 1 — ticket_model.pkl
    Input  : ticket title + description (raw text)
    Output : predicted category  (database / application / infrastructure /
                                   network / security)
             predicted priority  (P1 / P2 / P3 / P4)
    Algorithm : embedding system + MultiOutput RandomForestClassifier

  MODEL 2 — anomaly_model.pkl
    Input  : system metrics snapshot
             (response_time_ms, error_rate, cpu_usage_pct,
              memory_usage_pct, request_count)
    Output : anomaly_score (float, lower = more anomalous)
             is_anomaly    (bool)
    Algorithm : IsolationForest (unsupervised)

HOW TO RUN:
  python scripts/generate_sample_data.py   # generate data first
  python train_models.py                   # then train
================================================================================
"""

import json
import os
import pickle
import re
import string
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
os.makedirs("models", exist_ok=True)


# ==============================================================================
# SHARED UTILITY — TEXT PREPROCESSING
# ==============================================================================
# IMPORTANT: This function is used in BOTH training and inference (nlp_service.py)
# They MUST be identical. If they differ, the model gets different input
# at inference time vs training time -> predictions degrade.
# ==============================================================================

def preprocess_text(text: str) -> str:
    """
    Cleans raw ticket text before TF-IDF vectorization.

    Steps:
      1. Lowercase           -> "Database" and "database" are the same word
      2. Remove URLs         -> http links are noise
      3. Remove IPs          -> 192.168.1.1 adds no classification signal
      4. Remove ticket IDs   -> TKT-A1B2C3D4 are random, not meaningful
      5. Numbers -> NUM token -> "timeout after 30 retries" and
                                 "timeout after 500 retries" mean the same thing
      6. Remove punctuation  -> keeps only words
      7. Normalize spaces    -> clean final text
    """
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " ", text)
    text = re.sub(r"tkt-[a-z0-9]+", " ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# MODEL 1 — TICKET CLASSIFICATION (NLP)
# ==============================================================================

def train_ticket_model():
    """
    WHAT THIS MODEL DOES:
      Given a new ticket's title + description, predict:
        1. category : database | application | infrastructure | network | security
        2. priority : P1 | P2 | P3 | P4

    PIPELINE:
      Raw text
        -> preprocess_text()       (clean + normalize)
        -> TfidfVectorizer         (convert words to numeric feature vector)
        -> MultiOutputClassifier   (predict category AND priority together)
           wrapping RandomForestClassifier

    WHY TF-IDF + RANDOM FOREST (not BERT)?
      - Inference speed  : TF-IDF+RF takes ~2ms vs ~200ms for BERT
      - No GPU needed    : Can run on any server
      - Easy to retrain  : Retraining takes seconds, not hours
      - Good accuracy    : For IT support vocabulary, TF-IDF captures the
                           important keywords (connection, timeout, memory,
                           certificate, CVE etc.) very well

    MULTIOUTPUT:
      We predict category AND priority from the SAME text in one shot.
      MultiOutputClassifier trains one separate RandomForest for each output
      but they share the same TF-IDF feature vector.
    """
    print("\n" + "=" * 60)
    print("  TRAINING MODEL 1: Ticket Classifier (NLP)")
    print("=" * 60)

    # ── Load tickets.csv ──────────────────────────────────────────
    print("\n  Loading data/tickets.csv ...")
    df = pd.read_csv("data/tickets.csv")
    print(f"  Total rows loaded : {len(df)}")
    print(f"  Categories        : {df['category'].value_counts().to_dict()}")
    print(f"  Priorities        : {df['priority'].value_counts().to_dict()}")

    # ── Build input text ──────────────────────────────────────────
    # Title is repeated 3x to give it more weight.
    # Why? The title is written by the user as a summary — it contains
    # the strongest classification signal. "Database connection timeout"
    # instantly tells us category=database. The description has more
    # noise (service names, numbers, etc.)
    df["text"] = (
        df["title"].apply(preprocess_text) + " " +
        df["title"].apply(preprocess_text) + " " +
        df["title"].apply(preprocess_text) + " " +
        df["description"].apply(preprocess_text)
    )

    X = df["text"].values

    # ── Encode labels ─────────────────────────────────────────────
    # LabelEncoder converts string labels to integers:
    #   "application" -> 0, "database" -> 1, "infrastructure" -> 2 etc.
    # We save the encoders in the model bundle so we can decode
    # predictions back to strings during inference.
    le_category = LabelEncoder()
    le_priority = LabelEncoder()

    y_cat = le_category.fit_transform(df["category"].values)
    y_pri = le_priority.fit_transform(df["priority"].values)

    # Stack into 2D array: shape (n_samples, 2)
    # Column 0 = category label, Column 1 = priority label
    Y = np.column_stack([y_cat, y_pri])

    print(f"\n  Category classes  : {list(le_category.classes_)}")
    print(f"  Priority classes  : {list(le_priority.classes_)}")

    # ── Train / Test split ────────────────────────────────────────
    # stratify=y_cat ensures each split has proportional category distribution
    # (prevents all "security" tickets ending up in test set by chance)
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=y_cat
    )
    print(f"\n  Train size        : {len(X_train)}")
    print(f"  Test size         : {len(X_test)}")

    # ── Build Pipeline ────────────────────────────────────────────
    # Pipeline chains steps so that fit() and predict() apply all steps
    # in order. This prevents data leakage during cross-validation.
    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                # Use unigrams + bigrams.
                # "connection timeout" as a bigram is more informative
                # than "connection" and "timeout" separately.

                max_features=8000,
                # Keep only the 8000 most important terms.
                # Prevents overfitting on rare words.

                min_df=2,
                # Ignore terms appearing in fewer than 2 documents.
                # Removes unique noise terms.

                sublinear_tf=True,
                # Apply log(1+tf) instead of raw tf.
                # Prevents very frequent words from dominating.

                analyzer="word",
            )
        ),
        (
            "clf",
            MultiOutputClassifier(
                RandomForestClassifier(
                    n_estimators=200,
                    # 200 decision trees. More trees = more stable predictions.
                    # After ~200, adding more gives diminishing returns.

                    max_depth=None,
                    # Allow trees to grow fully.
                    # TF-IDF features are sparse, deep trees work well.

                    min_samples_leaf=2,
                    # Each leaf needs at least 2 samples.
                    # Light regularization to prevent overfitting.

                    class_weight="balanced",
                    # Automatically adjusts weights inversely proportional
                    # to class frequency. Handles class imbalance.
                    # (e.g., if "security" has fewer tickets than "application")

                    random_state=42,
                    n_jobs=-1,
                    # Use all CPU cores for parallel training.
                )
            )
        ),
    ])

    # ── Train ─────────────────────────────────────────────────────
    print("\n  Training pipeline (TF-IDF + RandomForest)...")
    pipeline.fit(X_train, Y_train)
    print("  Training complete.")

    # ── Evaluate ──────────────────────────────────────────────────
    Y_pred = pipeline.predict(X_test)

    print("\n  ── CATEGORY RESULTS ──────────────────────────────────")
    print(classification_report(
        Y_test[:, 0], Y_pred[:, 0],
        target_names=le_category.classes_
    ))

    print("  ── PRIORITY RESULTS ──────────────────────────────────")
    print(classification_report(
        Y_test[:, 1], Y_pred[:, 1],
        target_names=le_priority.classes_
    ))

    cat_f1 = f1_score(Y_test[:, 0], Y_pred[:, 0], average="weighted")
    pri_f1 = f1_score(Y_test[:, 1], Y_pred[:, 1], average="weighted")
    print(f"  Category F1 (weighted) : {cat_f1:.4f}")
    print(f"  Priority  F1 (weighted): {pri_f1:.4f}")

    # ── Cross-validation (category only) ─────────────────────────
    # 5-fold CV gives a more reliable accuracy estimate than a single split.
    print("\n  Running 5-fold cross-validation on category...")
    tfidf_cv = TfidfVectorizer(ngram_range=(1, 2), max_features=8000,
                               min_df=2, sublinear_tf=True)
    X_vec = tfidf_cv.fit_transform(X)
    cv_scores = cross_val_score(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_vec, y_cat, cv=5, scoring="f1_weighted"
    )
    print(f"  CV F1 (5-fold): {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

    # ── Save model bundle ─────────────────────────────────────────
    # We save everything needed for inference in one pickle file:
    #   - pipeline        : the trained TF-IDF + classifier
    #   - label_encoders  : to decode int predictions back to strings
    #   - feature_names   : for display purposes
    #   - metrics         : so we can show model accuracy in the dashboard
    model_bundle = {
        "pipeline":       pipeline,
        "label_encoders": {
            "category": le_category,
            "priority": le_priority,
        },
        "feature_names": {
            "categories": list(le_category.classes_),
            "priorities":  list(le_priority.classes_),
        },
        "metrics": {
            "category_f1": float(cat_f1),
            "priority_f1": float(pri_f1),
            "cv_f1_mean":  float(cv_scores.mean()),
        },
        "version": "1.0.0",
    }

    with open("models/ticket_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    print("\n  Saved -> models/ticket_model.pkl")
    return model_bundle


# ==============================================================================
# MODEL 2 — ANOMALY DETECTION (Unsupervised)
# ==============================================================================

def train_anomaly_model():
    """
    WHAT THIS MODEL DOES:
      Given a system metrics snapshot, detect if the system is behaving
      abnormally (anomaly) or normally.

    INPUT FEATURES (from logs.json metrics):
      1. response_time_ms   — high latency = possible issue
      2. error_rate         — high errors = definite issue
      3. cpu_usage_pct      — high CPU = resource problem
      4. memory_usage_pct   — high memory = leak or resource problem
      5. request_count      — spike or drop = traffic anomaly

    ENGINEERED FEATURES (derived from above):
      6. error_x_latency    = error_rate * response_time_ms
                              High ONLY when BOTH error and latency are high.
                              Single strongest signal for cascading failures.
      7. resource_pressure  = (cpu + memory) / 2
                              Composite resource health indicator.

    ALGORITHM — WHY ISOLATION FOREST?
      Isolation Forest is an unsupervised anomaly detection algorithm.

      Core idea: anomalies are EASY TO ISOLATE.
        - Normal points are dense — need many cuts to isolate one point
        - Anomaly points are sparse — isolated with very few cuts

      How it works:
        1. Build many random decision trees (n_estimators=200)
        2. At each node, randomly pick a feature and a random split value
        3. Recursively partition until each point is isolated
        4. Anomaly score = average path length to isolation
           (short path = easy to isolate = anomaly)

      Why not supervised (LogisticRegression, SVM)?
        - In production, we don't always have labeled anomaly data
        - System behavior changes over time (new services, new patterns)
        - Isolation Forest adapts to the current baseline automatically

    contamination = 0.10:
      We tell the model we expect ~10% of data to be anomalous.
      This sets the internal decision threshold.
      Must match the actual anomaly rate in training data.
    """
    print("\n" + "=" * 60)
    print("  TRAINING MODEL 2: Anomaly Detector (IsolationForest)")
    print("=" * 60)

    # ── Load logs.json ────────────────────────────────────────────
    print("\n  Loading data/logs.json ...")
    with open("data/logs.json") as f:
        logs = json.load(f)
    print(f"  Total log entries : {len(logs)}")

    # ── Extract metrics into DataFrame ────────────────────────────
    records = []
    for log in logs:
        m = log["metrics"]
        records.append({
            "response_time_ms":  m["response_time_ms"],
            "error_rate":        m["error_rate"],
            "cpu_usage_pct":     m["cpu_usage_pct"],
            "memory_usage_pct":  m["memory_usage_pct"],
            "request_count":     m["request_count"],
            "is_anomaly":        log.get("is_anomaly", False),
        })

    df = pd.DataFrame(records)
    anomaly_count = df["is_anomaly"].sum()
    print(f"  Anomalies in data : {anomaly_count} / {len(df)} ({anomaly_count/len(df):.1%})")

    # ── Feature Engineering ───────────────────────────────────────
    df["error_x_latency"]   = df["error_rate"] * df["response_time_ms"]
    df["resource_pressure"] = (df["cpu_usage_pct"] + df["memory_usage_pct"]) / 2

    feature_cols = [
        "response_time_ms",
        "error_rate",
        "cpu_usage_pct",
        "memory_usage_pct",
        "request_count",
        "error_x_latency",    # engineered
        "resource_pressure",  # engineered
    ]

    X      = df[feature_cols].values
    y_true = df["is_anomaly"].values

    # ── Scale features ────────────────────────────────────────────
    # StandardScaler converts each feature to mean=0, std=1.
    # Why? IsolationForest randomly picks features and split points.
    # Without scaling, response_time_ms (0-3000) dominates over
    # error_rate (0-1) just because of magnitude difference.
    # After scaling, all features contribute equally.
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"\n  Feature stats after scaling:")
    for i, col in enumerate(feature_cols):
        print(f"    {col:25s}: mean={X_scaled[:, i].mean():.3f}  std={X_scaled[:, i].std():.3f}")

    # ── Train IsolationForest ─────────────────────────────────────
    print("\n  Training IsolationForest...")
    iso_forest = IsolationForest(
        n_estimators=200,
        # Number of isolation trees.
        # More trees = more stable anomaly scores.

        contamination=0.10,
        # Expected fraction of anomalies in training data.
        # Must match actual anomaly rate (we generated 10%).

        max_samples="auto",
        # Subsample size per tree. "auto" = min(256, n_samples).
        # Subsampling makes it fast and prevents overfitting.

        random_state=42,
        n_jobs=-1,
    )
    iso_forest.fit(X_scaled)

    # ── Score all samples ─────────────────────────────────────────
    # score_samples() returns anomaly score for each point.
    # LOWER score = MORE anomalous (shorter path to isolation).
    # Typical range: -0.7 (very anomalous) to 0.0 (very normal)
    raw_preds     = iso_forest.predict(X_scaled)   # -1=anomaly, 1=normal
    scores        = iso_forest.score_samples(X_scaled)
    y_pred        = (raw_preds == -1).astype(int)

    print("\n  ── ANOMALY DETECTION RESULTS ─────────────────────────")
    print(classification_report(
        y_true.astype(int), y_pred,
        target_names=["Normal", "Anomaly"]
    ))

    # Score distribution
    normal_scores  = scores[~y_true]
    anomaly_scores = scores[y_true]
    print(f"  Normal score  : min={normal_scores.min():.3f}  max={normal_scores.max():.3f}  mean={normal_scores.mean():.3f}")
    print(f"  Anomaly score : min={anomaly_scores.min():.3f}  max={anomaly_scores.max():.3f}  mean={anomaly_scores.mean():.3f}")

    # ── Find optimal threshold ────────────────────────────────────
    # IsolationForest has a default threshold, but we can find a better
    # one by searching for the value that maximizes F1 on training data.
    print("\n  Finding optimal score threshold...")
    thresholds = np.linspace(scores.min(), scores.max(), 300)
    best_f1, best_threshold = 0.0, -0.1

    for t in thresholds:
        preds = (scores < t).astype(int)
        f1    = f1_score(y_true.astype(int), preds, zero_division=0)
        if f1 > best_f1:
            best_f1       = f1
            best_threshold = t

    print(f"  Optimal threshold : {best_threshold:.4f}")
    print(f"  Optimal F1        : {best_f1:.4f}")

    # ── Save model bundle ─────────────────────────────────────────
    model_bundle = {
        "model":        iso_forest,
        "scaler":       scaler,
        "feature_cols": feature_cols,
        "threshold":    float(best_threshold),
        "metrics": {
            "optimal_f1":    float(best_f1),
            "contamination": 0.10,
        },
        "score_stats": {
            "normal_mean":  float(normal_scores.mean()),
            "anomaly_mean": float(anomaly_scores.mean()),
        },
        "version": "1.0.0",
    }

    with open("models/anomaly_model.pkl", "wb") as f:
        pickle.dump(model_bundle, f)

    print("\n  Saved -> models/anomaly_model.pkl")
    return model_bundle


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    print("\n" + "#" * 60)
    print("  AI SUPPORT OPS — MODEL TRAINING PIPELINE")
    print("#" * 60)

    # Check data exists
    if not os.path.exists("data/tickets.csv"):
        print("\n  ERROR: data/tickets.csv not found.")
        print("  Run first: python scripts/generate_sample_data.py")
        exit(1)

    if not os.path.exists("data/logs.json"):
        print("\n  ERROR: data/logs.json not found.")
        print("  Run first: python scripts/generate_sample_data.py")
        exit(1)

    # Train both models
    ticket_bundle  = train_ticket_model()
    anomaly_bundle = train_anomaly_model()

    # Final summary
    print("\n" + "#" * 60)
    print("  TRAINING COMPLETE")
    print(f"  ticket_model.pkl  -> Category F1: {ticket_bundle['metrics']['category_f1']:.4f}")
    print(f"  ticket_model.pkl  -> Priority  F1: {ticket_bundle['metrics']['priority_f1']:.4f}")
    print(f"  anomaly_model.pkl -> Anomaly   F1: {anomaly_bundle['metrics']['optimal_f1']:.4f}")
    print("\n  Next step -> python api.py")
    print("#" * 60 + "\n")
