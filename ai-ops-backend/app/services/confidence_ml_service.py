import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


class ConfidenceMLService:

    def __init__(self):
        print("🤖 ML Confidence Model Initialized")

        self.model = LogisticRegression()
        self.scaler = StandardScaler()
        self.is_trained = False

        # Temporary training data
        self._train_initial_model()

    # ─────────────────────────────────────────────
    # INITIAL TRAINING (BOOTSTRAP)
    # ─────────────────────────────────────────────
    def _train_initial_model(self):

        # Synthetic training data (you can replace later)
        X = [
            # [cls_conf, sim_score, num_sim, avg_time, category_match]

            [0.9, 0.85, 5, 30, 1],
            [0.8, 0.75, 4, 40, 1],
            [0.7, 0.65, 3, 60, 1],

            [0.6, 0.50, 2, 90, 0],
            [0.5, 0.45, 1, 120, 0],

            [0.4, 0.30, 1, 200, 0],
            [0.3, 0.25, 0, 300, 0],
        ]

        y = [
            1, 1, 1,   # good predictions
            0, 0,      # medium
            0, 0       # bad
        ]

        X = np.array(X)
        y = np.array(y)

        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.is_trained = True

        print("✅ Confidence ML model trained")

    # ─────────────────────────────────────────────
    # FEATURE EXTRACTION
    # ─────────────────────────────────────────────
    def _extract_features(self, classification, similar_tickets):

        cls_conf = (
            classification.category_confidence +
            classification.priority_confidence
        ) / 2

        if not similar_tickets:
            return [cls_conf, 0, 0, 300, 0]

        best_score = similar_tickets[0]["similarity_score"]
        num_sim = len(similar_tickets)

        avg_time = np.mean([
            t.get("resolution_time_minutes", 120)
            for t in similar_tickets
        ])

        category_match = int(
            any(t["category"] == classification.predicted_category for t in similar_tickets)
        )

        return [
            cls_conf,
            best_score,
            num_sim,
            avg_time,
            category_match
        ]

    # ─────────────────────────────────────────────
    # COMPUTE CONFIDENCE
    # ─────────────────────────────────────────────
    def compute(self, classification, similar_tickets):

        features = self._extract_features(classification, similar_tickets)

        X = np.array([features])
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0][1]

        return round(float(prob), 4)