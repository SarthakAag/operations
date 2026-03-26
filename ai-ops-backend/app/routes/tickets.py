"""
app/routes/tickets.py
================================================================================
PURPOSE:
  All ticket-related API endpoints.
  Wires together all 6 services into clean route handlers.

ENDPOINTS:
  POST   /api/v1/tickets/                  Submit new ticket → full AI pipeline
  GET    /api/v1/tickets/                  List all tickets  (paginated + filtered)
  GET    /api/v1/tickets/{ticket_id}       Get one ticket by ID
  PUT    /api/v1/tickets/{ticket_id}       Update ticket status / assignment
  POST   /api/v1/tickets/{ticket_id}/resolve   Resolve ticket → triggers learning
  GET    /api/v1/tickets/{ticket_id}/audit     Full AI decision audit trail

PIPELINE FLOW (POST /tickets/):
  TicketCreate (request)
      ↓
  nlp_service.classify()              Step A: what category + priority?
      ↓
  similarity_service.find_similar()   Step B: what fixed this before?
      ↓
  confidence_service.compute()        Step C: how confident are we?
  confidence_service.build_fix()      Step C: build recommended fix text
      ↓
  governance_service.decide()         Step D: auto_resolve or human_review?
      ↓
  AuditLogDB.save()                   Step E: record full decision trail
  TicketDB.save()                     Step F: persist ticket
      ↓
  TicketResponse (response)
================================================================================
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AuditLogDB, TicketDB, get_db
from app.schemas import (
    SuccessResponse,
    TicketCreate,
    TicketListItem,
    TicketResolve,
    TicketResolveResponse,
    TicketResponse,
)
from app.services.confidence_service import get_confidence_service
from app.services.governance_service import get_governance_service
from app.services.learning_service import record_resolution
from app.services.nlp_service import get_nlp_service
from app.services.similarity_service import get_similarity_service

# ── Router ────────────────────────────────────────────────────────────────────
# prefix   : all routes start with /api/v1/tickets
# tags     : groups endpoints under "Tickets" in Swagger UI
router = APIRouter(
    prefix="/api/v1/tickets",
    tags=["Tickets"],
)


# ==============================================================================
# ENDPOINT 1: POST /api/v1/tickets/
# ==============================================================================

@router.post(
    "/",
    response_model=TicketResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit Ticket",
    description="""
Submit a new support ticket through the complete AI pipeline.

**Pipeline stages:**
1. **NLP Classification** — predicts category + priority from text
2. **Similarity Search**  — finds top-5 matching past incidents
3. **Confidence Scoring** — combines 3 signals into one score
4. **Governance**         — decides auto_resolve vs human_review
5. **Audit Logging**      — records full decision trail
6. **DB Persist**         — saves ticket with all AI metadata
    """,
)
async def submit_ticket(
    ticket: TicketCreate,
    db:     AsyncSession = Depends(get_db),
):
    """
    Full AI pipeline for a new ticket.

    This is the main endpoint that ties all 6 services together.
    Each step feeds its output into the next step.
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    print(f"\n  [TicketsRoute] Processing {ticket_id} — '{ticket.title[:50]}...'")

    # ── STEP A: NLP Classification ────────────────────────────────
    # Input : title + description (raw text)
    # Output: predicted_category, predicted_priority, confidence scores
    nlp_svc        = get_nlp_service()
    classification = nlp_svc.classify(ticket.title, ticket.description)

    # ── STEP B: Similarity Search ─────────────────────────────────
    # Input : title + description + predicted_category
    # Output: List[SimilarTicket] — top-5 most similar past tickets
    from app.services.similarity_service import SimilarityService

    sim_svc = SimilarityService()
    
    similar_tickets = sim_svc.find_similar(
        title              = ticket.title,
        description        = ticket.description,
        predicted_category = classification.predicted_category,
        top_k              = 5,
    )

    # ── STEP C: Confidence Scoring ────────────────────────────────
    # Input : classification + similar_tickets
    # Output: overall_confidence (float 0-1) + recommended_fix (str|None)

    from app.services.confidence_ml_service import ConfidenceMLService

    conf_svc = ConfidenceMLService()

    confidence = conf_svc.compute(
        classification=classification,
        similar_tickets=similar_tickets
    )

    # Build recommended fix
    recommended_fix = None
    if similar_tickets:
        recommended_fix = (
            f"Suggested fix (confidence {int(confidence * 100)}%) — review before applying:\n\n"
            f"Primary: {similar_tickets[0]['resolution']}"
        )

    # Simple breakdown (for audit log)
    breakdown = {
        "model": "ml_confidence",
        "confidence": confidence
    }

    # ── STEP D: Governance Decision ───────────────────────────────
    # Input : classification + confidence + is_vip + similar_tickets
    # Output: routing_decision, risk_score, assigned_to, sla_deadline
    gov_svc    = get_governance_service()
    governance = gov_svc.decide(
        classification  = classification,
        confidence      = confidence,
        is_vip          = ticket.is_vip or False,
        similar_tickets = similar_tickets,
    )

    # ── STEP E: Determine ticket status ───────────────────────────
    if governance.routing_decision == "auto_resolve":
        ticket_status  = "auto_resolved"
        resolved_at    = datetime.utcnow()
        resolved_by    = "system-bot"
        actual_fix     = recommended_fix
    else:
        ticket_status  = "open"
        resolved_at    = None
        resolved_by    = None
        actual_fix     = None

    # ── STEP F: Save ticket to DB ─────────────────────────────────
    sla_deadline = datetime.fromisoformat(governance.sla_deadline)

    db_ticket = TicketDB(
        ticket_id              = ticket_id,
        title                  = ticket.title,
        description            = ticket.description,
        reporter               = ticket.reporter,
        source                 = ticket.source or "api",
        is_vip                 = ticket.is_vip or False,

        # NLP outputs
        predicted_category     = classification.predicted_category,
        predicted_priority     = classification.predicted_priority,
        category_confidence    = classification.category_confidence,
        priority_confidence    = classification.priority_confidence,
        category_probabilities = classification.category_probabilities,
        priority_probabilities = classification.priority_probabilities,

        # Similarity outputs
        similar_tickets        = [s.__dict__ for s in similar_tickets],

        # Confidence outputs
        confidence_score       = confidence,
        recommended_fix        = recommended_fix,

        # Governance outputs
        routing_decision       = governance.routing_decision,
        risk_score             = governance.risk_score,
        risk_reasons           = governance.risk_reasons,
        assigned_to            = governance.assigned_to,
        requires_approval      = governance.requires_approval,
        sla_deadline           = sla_deadline,

        # Status
        status                 = ticket_status,
        resolved_at            = resolved_at,
        resolved_by            = resolved_by,
        actual_resolution      = actual_fix,
    )
    db.add(db_ticket)

    # ── STEP G: Write audit log ───────────────────────────────────
    # Records the complete AI decision trail for compliance.
    # decision_path stores the full reasoning so we can answer:
    # "Why did the AI route this ticket to team-oncall-p1?"
    audit = AuditLogDB(
        ticket_id        = ticket_id,
        event_type       = "ticket_submitted",
        model_version    = "1.0.0",
        confidence_score = confidence,
        risk_score       = governance.risk_score,
        decision         = f"{governance.routing_decision} → {governance.assigned_to}",
        decision_path    = {
            "classification": {
                "category":             classification.predicted_category,
                "category_confidence":  classification.category_confidence,
                "priority":             classification.predicted_priority,
                "priority_confidence":  classification.priority_confidence,
            },
            "similarity": {
                "tickets_found":  len(similar_tickets),
                "best_score":     similar_tickets[0].similarity_score if similar_tickets else 0,
            },
            "confidence": breakdown,
            "governance": {
                "routing":   governance.routing_decision,
                "risk":      governance.risk_score,
                "reasons":   governance.risk_reasons,
                "assigned":  governance.assigned_to,
            },
        },
        performed_by = "ai-system",
    )
    db.add(audit)
    await db.flush()

    print(
        f"  [TicketsRoute] {ticket_id} done: "
        f"category={classification.predicted_category} | "
        f"confidence={confidence:.0%} | "
        f"decision={governance.routing_decision} | "
        f"assigned={governance.assigned_to}"
    )

    # ── Build response ────────────────────────────────────────────
    return TicketResponse(
        ticket_id        = ticket_id,
        title            = ticket.title,
        description      = ticket.description,
        reporter         = ticket.reporter,
        source           = ticket.source or "api",
        is_vip           = ticket.is_vip or False,
        classification   = classification,
        similar_tickets  = similar_tickets,
        confidence_score = confidence,
        recommended_fix  = recommended_fix,
        governance       = governance,
        status           = ticket_status,
        created_at       = db_ticket.created_at,
    )


# ==============================================================================
# ENDPOINT 2: GET /api/v1/tickets/
# ==============================================================================

@router.get(
    "/",
    summary="List Tickets",
    description="List all tickets with optional filters. Returns paginated results.",
)
async def list_tickets(
    status:   Optional[str] = Query(None, description="Filter by status: open|auto_resolved|resolved|closed"),
    priority: Optional[str] = Query(None, description="Filter by priority: P1|P2|P3|P4"),
    category: Optional[str] = Query(None, description="Filter by category: database|application|infrastructure|network|security"),
    routing:  Optional[str] = Query(None, description="Filter by routing: auto_resolve|human_review"),
    limit:    int           = Query(50, ge=1, le=200, description="Max results per page"),
    offset:   int           = Query(0,  ge=0,         description="Number of records to skip"),
    db:       AsyncSession  = Depends(get_db),
):
    """
    Returns a paginated list of tickets.

    Supports filtering by:
      - status   : open | auto_resolved | resolved | closed
      - priority : P1 | P2 | P3 | P4
      - category : database | application | infrastructure | network | security
      - routing  : auto_resolve | human_review
    """
    query = (
        select(TicketDB)
        .order_by(TicketDB.created_at.desc())
    )

    if status:
        query = query.where(TicketDB.status == status)
    if priority:
        query = query.where(TicketDB.predicted_priority == priority)
    if category:
        query = query.where(TicketDB.predicted_category == category)
    if routing:
        query = query.where(TicketDB.routing_decision == routing)

    # Get total count for pagination metadata
    from sqlalchemy import func, select as sa_select
    count_query = sa_select(func.count(TicketDB.id))
    if status:
        count_query = count_query.where(TicketDB.status == status)
    if priority:
        count_query = count_query.where(TicketDB.predicted_priority == priority)
    if category:
        count_query = count_query.where(TicketDB.predicted_category == category)
    if routing:
        count_query = count_query.where(TicketDB.routing_decision == routing)

    total  = await db.scalar(count_query) or 0
    result = await db.execute(query.limit(limit).offset(offset))
    tickets = result.scalars().all()

    return {
        "total":   total,
        "limit":   limit,
        "offset":  offset,
        "tickets": [_ticket_to_list_item(t) for t in tickets],
    }


# ==============================================================================
# ENDPOINT 3: GET /api/v1/tickets/{ticket_id}
# ==============================================================================

@router.get(
    "/{ticket_id}",
    response_model=TicketResponse,
    summary="Get Ticket",
    description="Get full ticket details including all AI predictions.",
)
async def get_ticket(
    ticket_id: str,
    db:        AsyncSession = Depends(get_db),
):
    """
    Returns the full ticket record including:
      - Original ticket fields
      - NLP classification + probabilities
      - Similar past tickets
      - Confidence score + breakdown
      - Governance decision + risk reasons
      - Resolution status
    """
    ticket = await _get_ticket_or_404(ticket_id, db)
    return _db_ticket_to_response(ticket)


# ==============================================================================
# ENDPOINT 4: PUT /api/v1/tickets/{ticket_id}
# ==============================================================================

@router.put(
    "/{ticket_id}",
    response_model=SuccessResponse,
    summary="Update Ticket",
    description="Update ticket status, assignment or add notes.",
)
async def update_ticket(
    ticket_id:  str,
    new_status: Optional[str] = Query(None, description="New status: in_progress|escalated"),
    assigned_to: Optional[str] = Query(None, description="Reassign to another team"),
    db:         AsyncSession   = Depends(get_db),
):
    """
    Updates a ticket's status or assignment.
    Used when an engineer picks up a ticket or escalates it.
    """
    ticket = await _get_ticket_or_404(ticket_id, db)

    if ticket.status in ("resolved", "closed", "auto_resolved"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot update ticket with status '{ticket.status}'"
        )

    update_values = {"updated_at": datetime.utcnow()}

    if new_status:
        valid = {"in_progress", "escalated", "open"}
        if new_status not in valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status '{new_status}'. Valid: {valid}"
            )
        update_values["status"] = new_status

    if assigned_to:
        update_values["assigned_to"] = assigned_to

    await db.execute(
        update(TicketDB)
        .where(TicketDB.ticket_id == ticket_id)
        .values(**update_values)
    )

    # Audit log for manual update
    db.add(AuditLogDB(
        ticket_id    = ticket_id,
        event_type   = "ticket_updated",
        model_version = "1.0.0",
        confidence_score = ticket.confidence_score,
        risk_score   = ticket.risk_score,
        decision     = f"Updated: {update_values}",
        decision_path = update_values,
        performed_by = assigned_to or "system",
    ))

    return SuccessResponse(
        message=f"Ticket {ticket_id} updated",
        data=update_values
    )


# ==============================================================================
# ENDPOINT 5: POST /api/v1/tickets/{ticket_id}/resolve
# ==============================================================================

@router.post(
    "/{ticket_id}/resolve",
    response_model=TicketResolveResponse,
    summary="Resolve Ticket",
    description="""
Mark a ticket as resolved with the actual fix applied.

This triggers the **Continuous Learning Pipeline**:
1. Resolution saved to DB
2. Appended to tickets.csv
3. Similarity index updated immediately
4. Counter incremented (every 50 → models retrained)
    """,
)
async def resolve_ticket(
    ticket_id: str,
    body:      TicketResolve,
    db:        AsyncSession = Depends(get_db),
):
    """
    Resolves a ticket and feeds the resolution into the learning pipeline.

    ai_recommendation_was_correct:
      True  → AI suggested the right fix (positive training signal)
      False → AI was wrong (teaches model to correct this case)
    """
    ticket = await _get_ticket_or_404(ticket_id, db)

    # ── Validation ────────────────────────────────────────────────
    if ticket.status in ("resolved", "closed", "auto_resolved"):
        raise HTTPException(
            status_code=400,
            detail=f"Ticket {ticket_id} is already {ticket.status}"
        )

    # ── Compute resolution time ───────────────────────────────────
    created_at        = ticket.created_at
    resolution_time   = int(
        (datetime.utcnow() - created_at).total_seconds() / 60
    )

    # ── Update ticket in DB ───────────────────────────────────────
    await db.execute(
        update(TicketDB)
        .where(TicketDB.ticket_id == ticket_id)
        .values(
            status                    = "resolved",
            actual_resolution         = body.resolution_text,
            resolved_by               = body.resolved_by,
            resolved_at               = datetime.utcnow(),
            resolution_time_minutes   = resolution_time,
            ai_recommendation_correct = body.ai_recommendation_was_correct,
            updated_at                = datetime.utcnow(),
        )
    )

    # ── Audit log ─────────────────────────────────────────────────
    db.add(AuditLogDB(
        ticket_id    = ticket_id,
        event_type   = "ticket_resolved",
        model_version = "1.0.0",
        confidence_score = ticket.confidence_score,
        risk_score   = ticket.risk_score,
        decision     = f"Resolved by {body.resolved_by} in {resolution_time}min",
        decision_path = {
            "resolution_text":              body.resolution_text[:300],
            "resolved_by":                  body.resolved_by,
            "resolution_time_minutes":      resolution_time,
            "ai_recommendation_was_correct": body.ai_recommendation_was_correct,
        },
        performed_by = body.resolved_by,
    ))
    await db.flush()

    # ── Trigger Continuous Learning Pipeline ──────────────────────
    # This is the key step: every resolution feeds back into the system.
    # record_resolution() handles:
    #   1. Save to ResolutionDB
    #   2. Append to tickets.csv
    #   3. Update similarity index immediately
    #   4. Increment counter → trigger retrain at threshold
    learning_result = await record_resolution(
        ticket_id                     = ticket_id,
        title                         = ticket.title,
        description                   = ticket.description,
        category                      = ticket.predicted_category or "application",
        priority                      = ticket.predicted_priority or "P3",
        resolution_text               = body.resolution_text,
        resolved_by                   = body.resolved_by,
        resolution_time_minutes       = resolution_time,
        ai_recommendation_was_correct = body.ai_recommendation_was_correct,
        db                            = db,
    )

    print(
        f"  [TicketsRoute] {ticket_id} resolved by {body.resolved_by} "
        f"in {resolution_time}min | "
        f"AI correct: {body.ai_recommendation_was_correct} | "
        f"counter: {learning_result['counter']}/{learning_result['threshold']}"
    )

    return TicketResolveResponse(
        success                   = True,
        ticket_id                 = ticket_id,
        resolution_time_minutes   = resolution_time,
        ai_recommendation_correct = body.ai_recommendation_was_correct,
        message                   = (
            f"Ticket resolved. "
            f"{learning_result['counter']}/{learning_result['threshold']} "
            f"resolutions collected for retraining."
            + (" 🔄 Model retraining triggered!" if learning_result["retrain_triggered"] else "")
        ),
    )


# ==============================================================================
# ENDPOINT 6: GET /api/v1/tickets/{ticket_id}/audit
# ==============================================================================

@router.get(
    "/{ticket_id}/audit",
    summary="Get Audit Trail",
    description="Get the full AI decision audit trail for a ticket.",
)
async def get_audit_trail(
    ticket_id: str,
    db:        AsyncSession = Depends(get_db),
):
    """
    Returns the complete AI decision trail for a ticket.

    Shows every event from submission to resolution:
      - ticket_submitted  : classification + governance decision
      - ticket_updated    : manual status changes
      - ticket_resolved   : final resolution + learning trigger

    Used for:
      - Compliance audits
      - Debugging AI decisions
      - Performance reviews
    """
    # Verify ticket exists
    await _get_ticket_or_404(ticket_id, db)

    result = await db.execute(
        select(AuditLogDB)
        .where(AuditLogDB.ticket_id == ticket_id)
        .order_by(AuditLogDB.created_at.asc())
    )
    logs = result.scalars().all()

    if not logs:
        raise HTTPException(
            status_code=404,
            detail=f"No audit logs found for {ticket_id}"
        )

    return {
        "ticket_id":   ticket_id,
        "total_events": len(logs),
        "audit_trail": [
            {
                "event_type":       log.event_type,
                "decision":         log.decision,
                "confidence_score": log.confidence_score,
                "risk_score":       log.risk_score,
                "decision_path":    log.decision_path,
                "performed_by":     log.performed_by,
                "model_version":    log.model_version,
                "timestamp":        log.created_at.isoformat(),
            }
            for log in logs
        ],
    }


# ==============================================================================
# HELPERS
# ==============================================================================

async def _get_ticket_or_404(
    ticket_id: str,
    db:        AsyncSession,
) -> TicketDB:
    """
    Fetches a ticket by ticket_id or raises 404.
    Reused across GET, PUT, POST /resolve, GET /audit.
    """
    result = await db.execute(
        select(TicketDB).where(TicketDB.ticket_id == ticket_id)
    )
    ticket = result.scalar_one_or_none()

    if not ticket:
        raise HTTPException(
            status_code=404,
            detail=f"Ticket '{ticket_id}' not found"
        )
    return ticket


def _ticket_to_list_item(ticket: TicketDB) -> dict:
    """
    Converts a TicketDB row to a compact list item dict.
    Used by GET /tickets/ list endpoint.
    """
    return {
        "ticket_id":          ticket.ticket_id,
        "title":              ticket.title,
        "predicted_category": ticket.predicted_category,
        "predicted_priority": ticket.predicted_priority,
        "confidence_score":   ticket.confidence_score,
        "routing_decision":   ticket.routing_decision,
        "assigned_to":        ticket.assigned_to,
        "status":             ticket.status,
        "sla_breached":       ticket.sla_breached or False,
        "is_vip":             ticket.is_vip or False,
        "created_at":         ticket.created_at.isoformat() if ticket.created_at else None,
    }


def _db_ticket_to_response(ticket: TicketDB) -> TicketResponse:
    """
    Converts a TicketDB ORM object to a TicketResponse schema.
    Used by GET /tickets/{ticket_id}.
    Reconstructs nested schemas from stored JSON fields.
    """
    from app.schemas import TicketClassification, GovernanceDecision, SimilarTicket

    classification = TicketClassification(
        predicted_category     = ticket.predicted_category or "application",
        predicted_priority     = ticket.predicted_priority or "P3",
        category_confidence    = ticket.category_confidence or 0.0,
        priority_confidence    = ticket.priority_confidence or 0.0,
        category_probabilities = ticket.category_probabilities or {},
        priority_probabilities = ticket.priority_probabilities or {},
    )

    similar_tickets = []
    if ticket.similar_tickets:
        for s in ticket.similar_tickets:
            try:
                similar_tickets.append(SimilarTicket(
                    ticket_id               = s.get("ticket_id", ""),
                    title                   = s.get("title", ""),
                    similarity_score        = s.get("similarity_score", 0.0),
                    category                = s.get("category", ""),
                    priority                = s.get("priority", ""),
                    resolution              = s.get("resolution", ""),
                    resolution_time_minutes = s.get("resolution_time_minutes", 0),
                ))
            except Exception:
                pass

    governance = GovernanceDecision(
        routing_decision  = ticket.routing_decision or "human_review",
        risk_score        = ticket.risk_score or 0.0,
        risk_reasons      = ticket.risk_reasons or [],
        assigned_to       = ticket.assigned_to or "team-l2-support",
        requires_approval = ticket.requires_approval if ticket.requires_approval is not None else True,
        sla_deadline      = ticket.sla_deadline.isoformat() if ticket.sla_deadline else datetime.utcnow().isoformat(),
    )

    return TicketResponse(
        ticket_id        = ticket.ticket_id,
        title            = ticket.title,
        description      = ticket.description,
        reporter         = ticket.reporter,
        source           = ticket.source or "api",
        is_vip           = ticket.is_vip or False,
        classification   = classification,
        similar_tickets  = similar_tickets,
        confidence_score = ticket.confidence_score or 0.0,
        recommended_fix  = ticket.recommended_fix,
        governance       = governance,
        status           = ticket.status or "open",
        created_at       = ticket.created_at,
    )