"""
app/services/nlp_service.py
================================================================================
PURPOSE:
  Loads the trained ticket_model.pkl and classifies incoming tickets.

  Given a ticket title + description, returns:
    - predicted_category  : database | application | infrastructure |
                            network | security
    - predicted_priority  : P1 | P2 | P3 | P4
    - confidence scores   : per prediction (0.0 -> 1.0)
    - full probabilities  : distribution across all classes

HOW THE ML PIPELINE WORKS:
  Raw text
    -> preprocess_text()         clean + normalize (MUST match training)
    -> TfidfVectorizer           words -> sparse numeric vector
    -> MultiOutputClassifier     predict [category, priority] together
       wrapping RandomForest
    -> predict_proba()           returns probability array per class
    -> decode with LabelEncoder  int index -> string label

SINGLETON PATTERN:
  Model is loaded ONCE at startup via get_nlp_service().
  All requests share the same loaded model object.
  Why? Loading a .pkl file takes ~0.5s. Loading on every
  request would make the API 0.5s slower per ticket.

CRITICAL — PREPROCESSING MUST MATCH TRAINING:
  preprocess_text() here must be IDENTICAL to the one in train_models.py.
  If they differ, the model receives different input at inference time
  vs training time and predictions degrade silently.
================================================================================
"""

import pickle
import re
import string
from pathlib import Path
from typing import Dict, List, Optional

from app.config import get_settings
from app.schemas import TicketClassification

settings = get_settings()


# ==============================================================================
# SECTION 1: TEXT PREPROCESSING
# ==============================================================================
# MUST be identical to preprocess_text() in train_models.py
# Any change here requires retraining the model.
# ==============================================================================

def preprocess_text(text: str) -> str:
    """
    Cleans raw ticket text before TF-IDF vectorization.

    Steps and WHY each step matters:

    1. Lowercase
       "Database" and "database" are the same word.
       Without this, TF-IDF treats them as different features.

    2. Remove URLs
       "See http://jira.company.com/TKT-123" adds no classification signal.
       URLs are unique per ticket -> terrible for generalization.

    3. Remove IP addresses
       "192.168.1.45 is unreachable" — the IP is noise.
       The word "unreachable" is the signal.

    4. Remove ticket IDs
       "Related to TKT-A1B2C3D4" — ticket IDs are random.
       Keeping them would cause the model to memorize IDs, not learn patterns.

    5. Replace numbers with NUM token
       "timeout after 30 retries" and "timeout after 500 retries"
       mean the same thing for classification purposes.
       Replacing both with NUM lets the model generalize across quantities.

    6. Remove punctuation
       "connection-timeout" and "connection timeout" should be equivalent.

    7. Normalize whitespace
       Clean up any extra spaces left by previous steps.
    """
    # Step 1: lowercase
    text = text.lower()

    # Step 2: remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Step 3: remove IP addresses (e.g., 192.168.1.1)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " ", text)

    # Step 4: remove ticket IDs (e.g., TKT-A1B2C3D4)
    text = re.sub(r"tkt-[a-z0-9]+", " ", text)

    # Step 5: replace all numbers with NUM token
    text = re.sub(r"\d+", " NUM ", text)

    # Step 6: remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Step 7: normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def build_input_text(title: str, description: str) -> str:
    """
    Combines title + description into a single string for the model.

    WHY TITLE IS REPEATED 3x:
      The title is a concise human summary of the issue.
      "Database connection timeout" immediately signals category=database.
      The description contains more noise (service names, numbers, IPs).
      Repeating the title 3x gives it 3x more weight in the TF-IDF vector
      compared to body words — same effect as a feature weight multiplier.

      This must match what was done in train_models.py during training.
    """
    clean_title = preprocess_text(title)
    clean_desc  = preprocess_text(description)

    # Title repeated 3x + description
    return f"{clean_title} {clean_title} {clean_title} {clean_desc}"


# ==============================================================================
# SECTION 2: NLP SERVICE CLASS
# ==============================================================================

class NLPService:
    """
    Singleton service that loads ticket_model.pkl once and classifies tickets.

    Attributes:
      _bundle : the full model bundle loaded from .pkl:
        {
          "pipeline"       : TF-IDF + MultiOutputClassifier (the trained model)
          "label_encoders" : {"category": LabelEncoder, "priority": LabelEncoder}
          "feature_names"  : {"categories": [...], "priorities": [...]}
          "metrics"        : {"category_f1": 1.0, "priority_f1": 1.0}
          "version"        : "1.0.0"
        }
    """

    def __init__(self):
        self._bundle = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────────────

    def _load_model(self):
        """
        Loads ticket_model.pkl from disk into memory.
        Called once at startup.

        If the file doesn't exist (not trained yet), logs a warning
        and sets _bundle = None. The service still works but returns
        default predictions with 0.0 confidence.
        """
        model_path = Path(settings.ticket_model_path)

        if not model_path.exists():
            print(
                f"  [NLPService] WARNING: Model not found at {model_path}\n"
                f"  Run: python train_models.py"
            )
            return

        try:
            with open(model_path, "rb") as f:
                self._bundle = pickle.load(f)

            print(
                f"  [NLPService] Loaded ticket_model.pkl"
                f"  v{self._bundle['version']}"
                f"  | Category F1={self._bundle['metrics']['category_f1']:.3f}"
                f"  | Priority F1={self._bundle['metrics']['priority_f1']:.3f}"
            )
        except Exception as e:
            print(f"  [NLPService] ERROR loading model: {e}")
            self._bundle = None

    def is_ready(self) -> bool:
        """Returns True if model is loaded and ready for inference."""
        return self._bundle is not None

    def get_model_info(self) -> dict:
        """Returns model metadata for health checks and dashboard."""
        if not self.is_ready():
            return {"status": "not_loaded", "version": None, "metrics": {}}
        return {
            "status":   "ready",
            "version":  self._bundle["version"],
            "metrics":  self._bundle["metrics"],
            "categories": self._bundle["feature_names"]["categories"],
            "priorities": self._bundle["feature_names"]["priorities"],
        }

    # ── Core Classification ───────────────────────────────────────

    def classify(self, title: str, description: str) -> TicketClassification:
        """
        Main classification method. Called by the ticket route.

        Args:
            title       : Ticket title (e.g., "DB connection timeout on payment-svc")
            description : Full ticket description

        Returns:
            TicketClassification with:
              - predicted_category + category_confidence
              - predicted_priority  + priority_confidence
              - full probability distributions for both

        Flow:
          title + description
              -> build_input_text()      preprocess + weight title 3x
              -> pipeline.predict_proba()
                 proba[0] -> category probabilities  shape: (1, n_categories)
                 proba[1] -> priority probabilities  shape: (1, n_priorities)
              -> argmax()                best class index
              -> label_encoder.classes_  decode index -> string
        """
        if not self.is_ready():
            print("  [NLPService] Model not loaded — returning fallback classification")
            return self._fallback_classification()

        # ── Step 1: Build input text ──────────────────────────────
        combined = build_input_text(title, description)

        # ── Step 2: Get pipeline components ──────────────────────
        pipeline   = self._bundle["pipeline"]
        le_category = self._bundle["label_encoders"]["category"]
        le_priority = self._bundle["label_encoders"]["priority"]

        # ── Step 3: Predict probabilities ────────────────────────
        # predict_proba() returns a list because we have MultiOutput:
        #   proba[0] shape: (1, n_categories) — one row, one col per category
        #   proba[1] shape: (1, n_priorities) — one row, one col per priority
        proba     = pipeline.predict_proba([combined])
        cat_proba = proba[0][0]   # e.g., [0.05, 0.82, 0.08, 0.03, 0.02]
        pri_proba = proba[1][0]   # e.g., [0.74, 0.20, 0.06]

        # ── Step 4: Best prediction = highest probability class ───
        cat_idx = int(cat_proba.argmax())
        pri_idx = int(pri_proba.argmax())

        predicted_category  = le_category.classes_[cat_idx]
        predicted_priority  = le_priority.classes_[pri_idx]
        category_confidence = round(float(cat_proba[cat_idx]), 4)
        priority_confidence = round(float(pri_proba[pri_idx]), 4)

        # ── Step 5: Build full probability dicts ─────────────────
        # These are returned for transparency / explainability.
        # The Swagger UI shows exactly WHY the model chose "database".
        category_probabilities = {
            le_category.classes_[i]: round(float(p), 4)
            for i, p in enumerate(cat_proba)
        }
        priority_probabilities = {
            le_priority.classes_[i]: round(float(p), 4)
            for i, p in enumerate(pri_proba)
        }

        print(
            f"  [NLPService] Classified: "
            f"category={predicted_category} ({category_confidence:.0%}) | "
            f"priority={predicted_priority} ({priority_confidence:.0%})"
        )

        return TicketClassification(
            predicted_category     = predicted_category,
            predicted_priority     = predicted_priority,
            category_confidence    = category_confidence,
            priority_confidence    = priority_confidence,
            category_probabilities = category_probabilities,
            priority_probabilities = priority_probabilities,
        )

    # ── Keyword Extraction ────────────────────────────────────────

    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extracts the most important terms from ticket text.

        Uses the TF-IDF weights from the trained vectorizer.
        Terms with highest TF-IDF score are the most important.

        Used by:
          - similarity_service: enrich similarity search queries
          - dashboard: show trending topics across recent tickets

        Args:
            text  : Raw ticket text (title + description)
            top_n : How many top keywords to return

        Returns:
            List of top keywords sorted by importance
        """
        if not self.is_ready():
            return []

        try:
            tfidf       = self._bundle["pipeline"].named_steps["tfidf"]
            clean_text  = preprocess_text(text)
            vec         = tfidf.transform([clean_text])
            feature_names = tfidf.get_feature_names_out()
            scores      = vec.toarray()[0]

            # Get indices of top N scores (descending)
            top_indices = scores.argsort()[-top_n:][::-1]
            return [
                feature_names[i]
                for i in top_indices
                if scores[i] > 0
            ]
        except Exception as e:
            print(f"  [NLPService] Keyword extraction failed: {e}")
            return []

    # ── Model Reload (for continuous learning) ────────────────────

    def reload_model(self):
        """
        Hot-reloads the model from disk WITHOUT restarting the server.
        Called automatically after model retraining completes.

        Flow in learning_service.py:
          retraining finishes
              -> new ticket_model.pkl written to disk
              -> get_nlp_service().reload_model() called
              -> next request uses the new model immediately
        """
        print("  [NLPService] Reloading model from disk...")
        self._bundle = None
        self._load_model()
        print("  [NLPService] Reload complete.")

    # ── Fallback ──────────────────────────────────────────────────

    def _fallback_classification(self) -> TicketClassification:
        """
        Returns safe defaults when model is not loaded.

        confidence = 0.0 means governance_service will always
        route to human_review (safer than auto-resolving with no model).
        """
        return TicketClassification(
            predicted_category     = "application",
            predicted_priority     = "P3",
            category_confidence    = 0.0,
            priority_confidence    = 0.0,
            category_probabilities = {},
            priority_probabilities = {},
        )


# ==============================================================================
# SECTION 3: SINGLETON ACCESSOR
# ==============================================================================
# _nlp_service is a module-level variable.
# get_nlp_service() creates it once, then returns the same
# instance on every subsequent call.
#
# Used in routes like:
#   nlp_svc = get_nlp_service()
#   classification = nlp_svc.classify(title, description)
# ==============================================================================

_nlp_service: Optional[NLPService] = None


def get_nlp_service() -> NLPService:
    """
    Returns the singleton NLPService instance.
    Creates it on first call (lazy initialization).

    Usage:
        from app.services.nlp_service import get_nlp_service
        nlp = get_nlp_service()
        result = nlp.classify(title, description)
    """
    global _nlp_service
    if _nlp_service is None:
        _nlp_service = NLPService()
    return _nlp_service