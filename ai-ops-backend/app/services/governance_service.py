"""
app/services/governance_service.py
================================================================================
PURPOSE:
  Makes the final routing decision for every ticket:
    AUTO_RESOLVE  → AI applies fix, notifies user, closes ticket
    HUMAN_REVIEW  → Routes to correct team with AI recommendations attached

  Also computes:
    risk_score    → how dangerous is applying an auto-fix? (0.0 → 1.0)
    assigned_to   → which team or system handles this ticket
    sla_deadline  → when must it be resolved to avoid SLA breach

HOW RISK SCORE IS COMPUTED:
  Risk = weighted combination of 3 components:

  ┌──────────────────────────────────┬────────┬──────────────────────────────┐
  │ Component                        │ Weight │ Logic                        │
  ├──────────────────────────────────┼────────┼──────────────────────────────┤
  │ Priority risk                    │  40%   │ P1=1.0, P2=0.65, P3=0.35    │
  │ Category risk                    │  35%   │ security=1.0, app=0.40       │
  │ Uncertainty risk (1-confidence)  │  25%   │ low confidence = high risk   │
  └──────────────────────────────────┴────────┴──────────────────────────────┘

  Boost: No similar tickets found → risk *= 1.25
  (No precedent = unpredictable outcome = more dangerous to auto-fix)

ROUTING DECISION RULES:
  A ticket MUST go to HUMAN_REVIEW if ANY condition is true.
  ALL conditions must be false for AUTO_RESOLVE.

  This is intentionally conservative (fail-safe):
    False negative (auto-resolve something risky) = bad outcome
    False positive (human-review something safe)  = just slower

TEAM ROUTING:
  auto_resolve  → "system-bot"    (AI handles it)
  P1 tickets    → "team-oncall-p1" (dedicated P1 on-call team)
  others        → category-specific team from config

SLA DEADLINE:
  Computed from ticket creation time + priority SLA window.
  Stored in ticket record. Monitoring job checks this periodically
  and sets sla_breached=True if ticket is still open past deadline.
================================================================================
"""

from datetime import datetime, timedelta
from typing import List, Optional, Tuple

from app.config import get_settings
from app.schemas import GovernanceDecision, SimilarTicket, TicketClassification

settings = get_settings()


# ==============================================================================
# SECTION 1: RISK BOOST CONSTANT
# ==============================================================================

# When no similar tickets are found (no historical precedent),
# we don't know if the auto-fix will work → boost risk by 25%
NO_PRECEDENT_RISK_BOOST = 1.25

# Team assigned to all P1 critical incidents
P1_ONCALL_TEAM = "team-oncall-p1"


# ==============================================================================
# SECTION 2: RISK SCORE COMPUTATION
# ==============================================================================

def compute_risk_score(
    category:        str,
    priority:        str,
    confidence:      float,
    similar_tickets: List[SimilarTicket],
) -> float:
    """
    Computes a risk score in [0.0, 1.0] for this ticket's proposed fix.

    A high risk score means applying an auto-fix is DANGEROUS:
      - Could make things worse (P1 during active outage)
      - Could cause data loss (database schema changes)
      - Could introduce security holes (security-category fixes)
      - AI is uncertain about what the fix should be (low confidence)

    Formula:
      risk = 0.40 × priority_risk
           + 0.35 × category_risk
           + 0.25 × uncertainty_risk

    Where:
      priority_risk   = from settings.priority_risk dict
      category_risk   = from settings.category_risk dict
      uncertainty_risk = 1.0 - confidence
        (inverse of confidence: 0% confidence = 100% uncertainty = max risk)

    Post-processing:
      If no similar tickets found → risk *= 1.25 (no precedent boost)
      Clamped to [0.0, 1.0]

    Args:
        category        : predicted ticket category
        priority        : predicted ticket priority
        confidence      : overall confidence score (from confidence_service)
        similar_tickets : similar past tickets (empty = no precedent)

    Returns:
        risk score float in [0.0, 1.0]
    """
    priority_risk     = settings.priority_risk.get(priority, 0.35)
    category_risk     = settings.category_risk.get(category, 0.50)
    uncertainty_risk  = 1.0 - confidence    # low confidence = high uncertainty

    risk = (
        0.40 * priority_risk    +
        0.35 * category_risk    +
        0.25 * uncertainty_risk
    )

    # No precedent boost: if AI has never seen this type of issue,
    # auto-fixing is a gamble — make it more likely to go to human
    if not similar_tickets:
        risk = risk * NO_PRECEDENT_RISK_BOOST

    return round(min(1.0, risk), 4)


# ==============================================================================
# SECTION 3: ROUTING RULES
# ==============================================================================

def evaluate_routing_rules(
    category:    str,
    priority:    str,
    confidence:  float,
    risk_score:  float,
    is_vip:      bool,
    similar_tickets: List[SimilarTicket],
) -> List[str]:
    """
    Checks all routing rules and returns the list of triggered reasons.

    If the returned list is EMPTY  → AUTO_RESOLVE
    If the returned list is NON-EMPTY → HUMAN_REVIEW (reasons explain why)

    Rules are checked independently — any single rule can trigger human review.
    This is the "human-in-the-loop governance" layer from the architecture.

    Args:
        category    : predicted ticket category
        priority    : predicted ticket priority
        confidence  : overall confidence score
        risk_score  : computed risk score
        is_vip      : is the reporter a VIP?

    Returns:
        List[str] of triggered reasons (empty = auto_resolve)
    """
    reasons = []

    # ── Rule 1: VIP reporter ──────────────────────────────────────
    # Enterprise clients and executives always get human attention.
    # An auto-fix failure on a VIP ticket = major relationship damage.
    if is_vip:
        reasons.append(
            "VIP reporter — manual review required per escalation policy"
        )

    # ── Rule 2: P1 critical priority ─────────────────────────────
    # P1 = active outage or critical business impact.
    # Auto-fixing during an active outage can make things worse.
    # Always needs a human engineer to verify before applying.
# ── Rule 2: P1 critical priority (SMART HANDLING) 🔥
# Allow auto-resolve ONLY if extremely high confidence + strong precedent
    if priority == "P1":
      best_sim = similar_tickets[0].similarity_score if similar_tickets else 0

    if confidence < 0.85 or best_sim < 0.9:
        reasons.append(
            "P1 priority — requires human oversight unless very high confidence with strong precedent"
        )

    # ── Rule 3: Security category ────────────────────────────────
    # Security fixes must ALWAYS be reviewed by a human.
    # Auto-applying security remediations can:
    #   - Expose new attack surfaces
    #   - Break authentication systems
    #   - Violate compliance requirements
    # This rule cannot be overridden by confidence or risk thresholds.
    if category == "security":
        reasons.append(
            "Security category — all security fixes require human approval"
        )

    # ── Rule 4: High risk score ───────────────────────────────────
    # Even if confidence is high, a high-risk action needs human sign-off.
    # Example: 85% confident about a database schema change is still risky.
    if risk_score >= settings.high_risk_threshold:
        reasons.append(
            f"High risk score ({risk_score:.2f} >= "
            f"threshold {settings.high_risk_threshold:.2f})"
        )

    # ── Rule 5: Low confidence ────────────────────────────────────
    # The AI isn't sure enough about the fix.
    # Below 80%, the probability of a wrong auto-fix is too high.
    if confidence < settings.auto_resolve_confidence:
        reasons.append(
            f"Insufficient confidence ({confidence:.0%} < "
            f"threshold {settings.auto_resolve_confidence:.0%})"
        )

    return reasons


# ==============================================================================
# SECTION 4: TEAM ASSIGNMENT
# ==============================================================================

def assign_team(
    routing_decision: str,
    category:         str,
    priority:         str,
) -> str:
    """
    Assigns the ticket to the correct team or system.

    Assignment logic:
      auto_resolve  → "system-bot"      (AI handles end-to-end)
      P1 ticket     → "team-oncall-p1"  (dedicated P1 on-call team, always)
      others        → category team     (team-db, team-app, etc.)
      fallback      → "team-l2-support" (general L2 if category unknown)

    Note: P1 team overrides category team because P1 needs the fastest
    available engineer, not necessarily the category specialist.
    """
    if routing_decision == "auto_resolve":
        return "system-bot"

    if priority == "P1":
        return P1_ONCALL_TEAM

    return settings.category_team.get(category, "team-l2-support")


# ==============================================================================
# SECTION 5: SLA DEADLINE
# ==============================================================================

def compute_sla_deadline(priority: str) -> str:
    """
    Computes the SLA deadline ISO string based on priority.

    SLA windows (from settings):
      P1 → 60 min
      P2 → 240 min  (4 hours)
      P3 → 1440 min (24 hours)
      P4 → 4320 min (72 hours)

    Returns:
      ISO format datetime string of when ticket must be resolved.
      Stored in TicketDB.sla_deadline for monitoring.
    """
    sla_minutes = settings.sla_map.get(priority, 1440)
    deadline    = datetime.utcnow() + timedelta(minutes=sla_minutes)
    return deadline.isoformat()


# ==============================================================================
# SECTION 6: GOVERNANCE SERVICE CLASS
# ==============================================================================

class GovernanceService:
    """
    Stateless service — pure decision logic, no state to manage.

    All inputs come from previous pipeline stages.
    All outputs feed into the ticket route handler.
    """

    def decide(
        self,
        classification:  TicketClassification,
        confidence:      float,
        is_vip:          bool,
        similar_tickets: List[SimilarTicket],
    ) -> GovernanceDecision:
        """
        Makes the final routing + risk decision for a ticket.

        This is the last stage of the AI pipeline before the
        ticket is saved to DB and the response is returned.

        Args:
            classification  : NLP output (category, priority, confidence)
            confidence      : overall confidence score from confidence_service
            is_vip          : is the reporter a VIP?
            similar_tickets : similar past tickets from similarity_service

        Returns:
            GovernanceDecision with:
              routing_decision  : "auto_resolve" | "human_review"
              risk_score        : float 0.0-1.0
              risk_reasons      : list of triggered rules (empty if auto)
              assigned_to       : team name or "system-bot"
              requires_approval : bool
              sla_deadline      : ISO datetime string

        Full flow:
          1. Extract category + priority from classification
          2. Compute risk score
          3. Evaluate all 5 routing rules
          4. Determine routing decision (any rule → human_review)
          5. Assign team
          6. Compute SLA deadline
          7. Return GovernanceDecision
        """
        category = classification.predicted_category
        priority = classification.predicted_priority

        # ── Step 1: Compute risk score ────────────────────────────
        risk_score = compute_risk_score(
            category        = category,
            priority        = priority,
            confidence      = confidence,
            similar_tickets = similar_tickets,
        )

        # ── Step 2: Evaluate routing rules ────────────────────────
        reasons = evaluate_routing_rules(
            category   = category,
            priority   = priority,
            confidence = confidence,
            risk_score = risk_score,
            is_vip     = is_vip,
            similar_tickets = similar_tickets,
        )

        # ── Step 3: Final routing decision ────────────────────────
        # ANY triggered rule → human_review
        # ALL rules clear   → auto_resolve
        routing_decision  = "human_review" if reasons else "auto_resolve"
        requires_approval = bool(reasons)

        # ── Step 4: Assign team ───────────────────────────────────
        assigned_to = assign_team(routing_decision, category, priority)

        # ── Step 5: SLA deadline ──────────────────────────────────
        sla_deadline = compute_sla_deadline(priority)

        # ── Step 6: Log the decision ──────────────────────────────
        print(
            f"  [GovernanceService] "
            f"category={category} | priority={priority} | "
            f"confidence={confidence:.0%} | risk={risk_score:.2f} | "
            f"decision={routing_decision} | assigned={assigned_to}"
        )
        if reasons:
            for r in reasons:
                print(f"    reason: {r}")

        return GovernanceDecision(
            routing_decision  = routing_decision,
            risk_score        = risk_score,
            risk_reasons      = reasons,
            assigned_to       = assigned_to,
            requires_approval = requires_approval,
            sla_deadline      = sla_deadline,
        )

    def explain(
        self,
        classification:  TicketClassification,
        confidence:      float,
        is_vip:          bool,
        similar_tickets: List[SimilarTicket],
    ) -> dict:
        """
        Returns a full human-readable explanation of the governance decision.
        Used for the audit trail and dashboard explainability view.

        Returns:
            {
              "decision":          "human_review",
              "risk_score":        0.73,
              "risk_breakdown": {
                "priority_component":  0.400,
                "category_component":  0.263,
                "uncertainty_component": 0.068,
                "no_precedent_boost":  False,
                "raw_before_boost":    0.730,
              },
              "rules_checked": [
                {"rule": "VIP reporter",    "triggered": False},
                {"rule": "P1 priority",     "triggered": True,  "reason": "..."},
                {"rule": "Security",        "triggered": False},
                {"rule": "High risk",       "triggered": True,  "reason": "..."},
                {"rule": "Low confidence",  "triggered": False},
              ],
              "sla_minutes":     60,
              "assigned_to":     "team-oncall-p1",
            }
        """
        category   = classification.predicted_category
        priority   = classification.predicted_priority

        # Risk breakdown
        pri_comp   = 0.40 * settings.priority_risk.get(priority, 0.35)
        cat_comp   = 0.35 * settings.category_risk.get(category, 0.50)
        unc_comp   = 0.25 * (1.0 - confidence)
        raw_risk   = pri_comp + cat_comp + unc_comp
        boosted    = not bool(similar_tickets)
        risk_score = min(1.0, raw_risk * (NO_PRECEDENT_RISK_BOOST if boosted else 1.0))

        reasons = evaluate_routing_rules(
            category=category, priority=priority,
            confidence=confidence, risk_score=risk_score, is_vip=is_vip
        )

        routing  = "human_review" if reasons else "auto_resolve"
        reasons_set = set(reasons)

        # Build per-rule check list
        rule_defs = [
            ("VIP reporter",   is_vip,              "VIP reporter — manual review required per escalation policy"),
            ("P1 priority",    priority == "P1",    "P1 priority — critical incident requires human oversight"),
            ("Security",       category == "security", "Security category — all security fixes require human approval"),
            ("High risk",      risk_score >= settings.high_risk_threshold,
                               f"High risk score ({risk_score:.2f} >= {settings.high_risk_threshold:.2f})"),
            ("Low confidence", confidence < settings.auto_resolve_confidence,
                               f"Insufficient confidence ({confidence:.0%} < {settings.auto_resolve_confidence:.0%})"),
        ]

        rules_checked = []
        for rule_name, triggered, reason_text in rule_defs:
            entry = {"rule": rule_name, "triggered": triggered}
            if triggered:
                entry["reason"] = reason_text
            rules_checked.append(entry)

        return {
            "decision":    routing,
            "risk_score":  round(risk_score, 4),
            "risk_breakdown": {
                "priority_component":   round(pri_comp, 4),
                "category_component":   round(cat_comp, 4),
                "uncertainty_component":round(unc_comp, 4),
                "no_precedent_boost":   boosted,
                "raw_before_boost":     round(raw_risk, 4),
            },
            "rules_checked": rules_checked,
            "sla_minutes":   settings.sla_map.get(priority, 1440),
            "assigned_to":   assign_team(routing, category, priority),
        }


# ==============================================================================
# SECTION 7: SINGLETON ACCESSOR
# ==============================================================================

_governance_service: Optional[GovernanceService] = None


def get_governance_service() -> GovernanceService:
    """
    Returns the singleton GovernanceService instance.

    Usage:
        from app.services.governance_service import get_governance_service
        gov_svc  = get_governance_service()
        decision = gov_svc.decide(classification, confidence, is_vip, similar)
    """
    global _governance_service
    if _governance_service is None:
        _governance_service = GovernanceService()
    return _governance_service