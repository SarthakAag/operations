"""
app/services/confidence_service.py
================================================================================
PURPOSE:
  Aggregates signals from NLP, similarity, and history into ONE
  overall confidence score per ticket.

  This single score drives the governance decision:
    confidence >= 0.80  ->  auto_resolve  (AI handles it)
    confidence <  0.80  ->  human_review  (route to team)

WHY A WEIGHTED COMBINATION?
  No single signal is reliable alone:

  - Classification confidence ALONE can be misleading.
    A model can be 90% confident about a wrong category.
    (Overconfident models are a known ML problem.)

  - Similarity score ALONE doesn't account for whether
    the similar ticket's fix actually worked reliably.

  - Historical resolution time ALONE is just a proxy —
    some complex issues are still fast to resolve.

  Together, three independent signals that ALL agree = trustworthy.
  If they disagree, the score drops → human review triggered.

WEIGHT RATIONALE:
  Similarity (40%) — highest weight because:
    A 0.90 similarity score means we found a nearly identical past ticket.
    If we know exactly what fixed it before, we can fix it again.
    This is the most direct evidence available.

  Classification (35%) — second because:
    If the model is highly confident about category, the fix space narrows.
    Wrong category = wrong fix = failed auto-resolution.
    Category confidence (60%) weighted more than priority (40%) because
    category determines which team and which fix applies.

  Historical rate (25%) — lowest because:
    Resolution time is an indirect proxy for difficulty.
    A ticket that took 5 minutes to fix has high repeatability.
    A ticket that took 3 days is likely complex and unpredictable.
    But this signal can be noisy — not all fast fixes are simple.

CONFIDENCE SCORE SCALE:
  0.90 - 1.00  : Very high — identical past issue, confident fix
  0.80 - 0.90  : High      — auto-resolve threshold
  0.60 - 0.80  : Medium    — recommend fix but require human review
  0.40 - 0.60  : Low       — weak match, human review required
  0.00 - 0.40  : Very low  — no useful history, human must decide
================================================================================
"""

from typing import List, Optional

from app.config import get_settings
from app.schemas import SimilarTicket, TicketClassification

settings = get_settings()

# ==============================================================================
# SECTION 1: CONFIDENCE WEIGHTS
# ==============================================================================
# Must sum to 1.0
# Change these to tune the balance between signals.
# ==============================================================================

W_CLASSIFICATION = 0.35   # NLP model confidence
W_SIMILARITY     = 0.40   # best similar ticket score
W_HISTORY        = 0.25   # historical resolution time proxy

# Within classification signal:
# Category confidence is weighted more than priority confidence.
# Reason: wrong category = completely wrong fix applied.
# Wrong priority = wrong urgency, but fix might still be correct.
W_CATEGORY_IN_CLASS = 0.60
W_PRIORITY_IN_CLASS = 0.40

# Category penalty threshold:
# If model is < 40% confident about category, apply a penalty.
# Prevents the similarity signal from carrying the whole score
# when the model basically guessed the category.
CATEGORY_UNCERTAIN_THRESHOLD = 0.40
CATEGORY_UNCERTAIN_PENALTY   = 0.80    # multiply score by 0.80 (20% penalty)

# Minimum history tickets needed to use the history signal reliably.
# Below this, we fall back to a conservative default.
MIN_HISTORY_TICKETS = 3

# Resolution time normalization constant.
# avg_time / HISTORY_NORM_CONSTANT gives a value in [0, 1].
# Chosen so that 0 min -> 1.0, 2000 min (33h) -> 0.0.
HISTORY_NORM_CONSTANT = 2000

# Minimum history signal (floor) even for very long resolutions.
HISTORY_SIGNAL_FLOOR = 0.30

# Confidence floor — never go below 5%.
# Even with no history, something is better than nothing.
CONFIDENCE_FLOOR = 0.05


# ==============================================================================
# SECTION 2: CONFIDENCE SERVICE CLASS
# ==============================================================================

class ConfidenceService:
    """
    Stateless service — no model loaded, pure computation.
    No singleton needed: instantiate fresh each time or keep one instance.

    All methods are deterministic given the same inputs.
    """

    # ── Core: Compute Overall Confidence ─────────────────────────

    def compute(
        self,
        classification:  TicketClassification,
        similar_tickets: List[SimilarTicket],
    ) -> float:
        """
        Computes the overall confidence score for a ticket.

        Args:
            classification  : output of nlp_service.classify()
            similar_tickets : output of similarity_service.find_similar()

        Returns:
            float in [0.05, 1.0] — overall confidence

        Steps:
          1. Compute classification_signal from category + priority confidence
          2. Compute similarity_signal from top similar tickets
          3. Compute history_signal from resolution time proxy
          4. Combine with weights
          5. Apply category uncertainty penalty if needed
          6. Clamp to [CONFIDENCE_FLOOR, 1.0]
        """

        # ── Signal 1: Classification ──────────────────────────────
        # Weighted blend of category and priority confidence.
        # Both must be high for the classification to be reliable.
        cat_conf = classification.category_confidence
        pri_conf = classification.priority_confidence

        classification_signal = (
            W_CATEGORY_IN_CLASS * cat_conf +
            W_PRIORITY_IN_CLASS * pri_conf
        )
        # Example:
        #   cat_conf=0.82, pri_conf=0.74
        #   signal = 0.60*0.82 + 0.40*0.74 = 0.492 + 0.296 = 0.788

        # ── Signal 2: Similarity ──────────────────────────────────
        # Blend of best match score and average of top-3.
        # Why blend instead of just best?
        #   Single best score can be noisy (one lucky match).
        #   Average of top-3 is more stable and representative.
        #   We weight best (70%) more than average (30%) because
        #   the single best match IS the most direct evidence.
        if similar_tickets:
            best_score  = similar_tickets[0].similarity_score
            top3_scores = [t.similarity_score for t in similar_tickets[:3]]
            avg_top3    = sum(top3_scores) / len(top3_scores)

            similarity_signal = 0.70 * best_score + 0.30 * avg_top3
            # Example:
            #   best=0.87, top3=[0.87, 0.72, 0.65], avg=0.747
            #   signal = 0.70*0.87 + 0.30*0.747 = 0.609 + 0.224 = 0.833
        else:
            # No similar tickets found at all.
            # We have zero historical evidence for this fix.
            similarity_signal = 0.0

        # ── Signal 3: Historical Resolution Rate ──────────────────
        # Proxy: faster past resolution = simpler problem = higher confidence.
        #
        # Normalization:
        #   0 minutes   → 1.0  (instant fix = trivial problem)
        #   2000 minutes → 0.0  (33 hours = very complex problem)
        #   Clamped to [HISTORY_SIGNAL_FLOOR, 1.0]
        #
        # We only use this signal if we have enough history tickets.
        # With fewer than 3 tickets, the average is unreliable.
        if len(similar_tickets) >= MIN_HISTORY_TICKETS:
            valid_times = [
                t.resolution_time_minutes
                for t in similar_tickets[:5]
                if t.resolution_time_minutes > 0
            ]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                history_signal = max(
                    HISTORY_SIGNAL_FLOOR,
                    min(1.0, 1.0 - avg_time / HISTORY_NORM_CONSTANT)
                )
                # Example:
                #   avg_time=200min → signal = 1.0 - 200/2000 = 0.90
                #   avg_time=600min → signal = 1.0 - 600/2000 = 0.70
                #   avg_time=1800min→ signal = max(0.30, 0.10) = 0.30
            else:
                history_signal = 0.50   # no time data, use neutral default
        else:
            # Not enough history → conservative default
            # Don't punish too much — just be cautious
            history_signal = 0.40

        # ── Weighted Combination ──────────────────────────────────
        overall = (
            W_CLASSIFICATION * classification_signal +
            W_SIMILARITY     * similarity_signal     +
            W_HISTORY        * history_signal
        )
        # Example:
        #   class=0.788, sim=0.833, hist=0.90
        #   overall = 0.35*0.788 + 0.40*0.833 + 0.25*0.90
        #           = 0.276 + 0.333 + 0.225
        #           = 0.834  →  AUTO RESOLVE

        # ── Category Uncertainty Penalty ─────────────────────────
        # If the model is very unsure about category, penalize.
        # Why? Wrong category → wrong team → wrong fix applied.
        # Even a high similarity score can't save a wrong-category fix.
        if cat_conf < CATEGORY_UNCERTAIN_THRESHOLD:
            overall *= CATEGORY_UNCERTAIN_PENALTY
            print(
                f"  [ConfidenceService] Category uncertainty penalty applied: "
                f"cat_conf={cat_conf:.0%} < {CATEGORY_UNCERTAIN_THRESHOLD:.0%} threshold"
            )

        # ── Clamp to valid range ──────────────────────────────────
        final = round(max(CONFIDENCE_FLOOR, min(1.0, overall)), 4)

        print(
            f"  [ConfidenceService] Signals: "
            f"classification={classification_signal:.3f} (w={W_CLASSIFICATION}) | "
            f"similarity={similarity_signal:.3f} (w={W_SIMILARITY}) | "
            f"history={history_signal:.3f} (w={W_HISTORY}) | "
            f"OVERALL={final:.3f}"
        )

        return final

    # ── Build Recommended Fix Text ────────────────────────────────

    def build_recommended_fix(
        self,
        similar_tickets: List[SimilarTicket],
        confidence:      float,
    ) -> Optional[str]:
        """
        Builds the recommended fix text shown to the engineer.

        Three cases based on confidence level:

        HIGH (>= 0.70):
          "Based on 4 similar past incidents (87% match):
           Increased connection pool size from 50 to 200..."
          → Direct, confident recommendation

        MEDIUM (0.40 - 0.70):
          "Suggested fix (58% confidence) — review before applying:
           Primary: Increased connection pool size...
           Alternative: Restarted DB service and..."
          → Hedged recommendation with alternatives shown

        LOW (< 0.40 or no similar tickets):
          Returns None
          → Triggers human_review in governance regardless
          → No bad advice is better than wrong advice

        Args:
            similar_tickets : output of similarity_service.find_similar()
            confidence      : overall confidence score

        Returns:
            Formatted recommendation string, or None if confidence too low
        """
        # No similar tickets or confidence too low → no recommendation
        if not similar_tickets or confidence < 0.25:
            print(
                f"  [ConfidenceService] No fix recommendation: "
                f"confidence={confidence:.0%} or no similar tickets"
            )
            return None

        best = similar_tickets[0]

        if confidence >= 0.70:
            # ── High confidence: direct recommendation ────────────
            fix = (
                f"Based on {len(similar_tickets)} similar past incident(s) "
                f"(best match: {best.similarity_score:.0%} similarity):\n\n"
                f"{best.resolution}"
            )

        else:
            # ── Medium confidence: hedged with alternatives ────────
            fix_parts = [
                f"Suggested fix (confidence: {confidence:.0%}) "
                f"— review carefully before applying:\n\n"
                f"Primary approach:\n{best.resolution}"
            ]

            # Add alternative if available
            if len(similar_tickets) > 1:
                alt = similar_tickets[1]
                fix_parts.append(
                    f"\n\nAlternative approach "
                    f"({alt.similarity_score:.0%} match):\n{alt.resolution}"
                )

            fix = "\n".join(fix_parts)

        print(
            f"  [ConfidenceService] Fix recommendation built: "
            f"confidence={confidence:.0%} | "
            f"best_match={best.similarity_score:.0%} | "
            f"length={len(fix)} chars"
        )
        return fix

    # ── Breakdown for Explainability ──────────────────────────────

    def get_confidence_breakdown(
        self,
        classification:  TicketClassification,
        similar_tickets: List[SimilarTicket],
    ) -> dict:
        """
        Returns the full confidence breakdown for transparency.
        Shown in the audit trail and dashboard explainability view.

        Returns:
            {
              "overall":                 0.834,
              "classification_signal":   0.788,
              "similarity_signal":       0.833,
              "history_signal":          0.900,
              "category_confidence":     0.820,
              "priority_confidence":     0.740,
              "best_similarity_score":   0.870,
              "similar_tickets_count":   4,
              "avg_resolution_time_min": 200,
              "penalty_applied":         False,
              "weights": { "classification": 0.35, "similarity": 0.40, "history": 0.25 }
            }
        """
        cat_conf = classification.category_confidence
        pri_conf = classification.priority_confidence

        classification_signal = (
            W_CATEGORY_IN_CLASS * cat_conf +
            W_PRIORITY_IN_CLASS * pri_conf
        )

        if similar_tickets:
            best_score = similar_tickets[0].similarity_score
            top3       = [t.similarity_score for t in similar_tickets[:3]]
            sim_signal = 0.70 * best_score + 0.30 * (sum(top3) / len(top3))
        else:
            best_score = 0.0
            sim_signal = 0.0

        valid_times = [
            t.resolution_time_minutes
            for t in similar_tickets[:5]
            if t.resolution_time_minutes > 0
        ] if len(similar_tickets) >= MIN_HISTORY_TICKETS else []

        avg_time  = sum(valid_times) / len(valid_times) if valid_times else None
        hist_signal = (
            max(HISTORY_SIGNAL_FLOOR, min(1.0, 1.0 - avg_time / HISTORY_NORM_CONSTANT))
            if avg_time is not None else 0.40
        )

        overall = (
            W_CLASSIFICATION * classification_signal +
            W_SIMILARITY     * sim_signal +
            W_HISTORY        * hist_signal
        )
        penalty_applied = cat_conf < CATEGORY_UNCERTAIN_THRESHOLD
        if penalty_applied:
            overall *= CATEGORY_UNCERTAIN_PENALTY

        overall = round(max(CONFIDENCE_FLOOR, min(1.0, overall)), 4)

        return {
            "overall":                  overall,
            "classification_signal":    round(classification_signal, 4),
            "similarity_signal":        round(sim_signal, 4),
            "history_signal":           round(hist_signal, 4),
            "category_confidence":      round(cat_conf, 4),
            "priority_confidence":      round(pri_conf, 4),
            "best_similarity_score":    round(best_score, 4),
            "similar_tickets_count":    len(similar_tickets),
            "avg_resolution_time_min":  round(avg_time, 1) if avg_time else None,
            "penalty_applied":          penalty_applied,
            "weights": {
                "classification": W_CLASSIFICATION,
                "similarity":     W_SIMILARITY,
                "history":        W_HISTORY,
            },
        }


# ==============================================================================
# SECTION 3: SINGLETON ACCESSOR
# ==============================================================================

_confidence_service: Optional[ConfidenceService] = None


def get_confidence_service() -> ConfidenceService:
    """
    Returns singleton ConfidenceService instance.

    Usage:
        from app.services.confidence_service import get_confidence_service
        conf_svc   = get_confidence_service()
        confidence = conf_svc.compute(classification, similar_tickets)
        fix        = conf_svc.build_recommended_fix(similar_tickets, confidence)
    """
    global _confidence_service
    if _confidence_service is None:
        _confidence_service = ConfidenceService()
    return _confidence_service