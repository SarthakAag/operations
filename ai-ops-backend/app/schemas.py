"""
app/schemas.py
================================================================================
PURPOSE:
  Pydantic models for request validation and response serialization.

  Every API endpoint uses these schemas:
    - REQUEST  schemas -> validate incoming JSON, raise 422 if invalid
    - RESPONSE schemas -> define exactly what JSON gets sent back
    - INTERNAL schemas -> passed between service layers (not HTTP-facing)

WHY PYDANTIC?
  - Automatic JSON validation (wrong types -> 422 error with clear message)
  - Auto-generates OpenAPI docs (the Swagger UI you saw at /docs)
  - Type safety — IDE knows exactly what fields exist
  - .model_dump() converts schema to dict for DB saving
  - from_attributes=True lets us build schemas directly from ORM objects

SCHEMA FLOW FOR A TICKET:
  HTTP POST body
      -> TicketCreate           (validate incoming request)
      -> classify_ticket()
      -> TicketClassification   (NLP service output)
      -> find_similar_tickets()
      -> SimilarTicket[]        (similarity service output)
      -> compute_confidence()
      -> governance_decision()
      -> GovernanceDecision     (governance service output)
      -> TicketResponse         (final HTTP response)
================================================================================
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ==============================================================================
# SECTION 1: TICKET SCHEMAS
# ==============================================================================

class TicketCreate(BaseModel):
    """
    REQUEST schema — body for POST /api/v1/tickets/

    What the user sends when submitting a new ticket.
    Pydantic validates:
      - title must be at least 5 characters
      - description must be at least 10 characters
      - is_vip defaults to False if not provided
    """
    title: str = Field(
        ...,
        min_length=5,
        max_length=500,
        example="Database connection timeout on payment-svc",
        description="Short summary of the issue",
    )
    description: str = Field(
        ...,
        min_length=10,
        example=(
            "Users are experiencing connection timeouts. "
            "Error logs show max_connections exceeded. "
            "Pool exhausted after 50 retries."
        ),
        description="Full description of the issue with as much detail as possible",
    )
    reporter: Optional[str] = Field(
        None,
        example="engineer@company.com",
        description="Email or name of the person reporting the issue",
    )
    source: Optional[str] = Field(
        "api",
        example="api",
        description="Where this ticket came from: api | servicenow | jira | email",
    )
    is_vip: Optional[bool] = Field(
        False,
        description="Is the reporter a VIP (enterprise client / executive)?",
    )

    @field_validator("title", "description")
    @classmethod
    def must_not_be_blank(cls, v: str) -> str:
        """Reject titles/descriptions that are only whitespace."""
        if not v.strip():
            raise ValueError("Field cannot be blank or whitespace only")
        return v.strip()


class TicketClassification(BaseModel):
    """
    INTERNAL schema — output of nlp_service.classify()

    Contains both the predicted labels AND the full probability
    distributions for transparency / explainability.

    Example:
      predicted_category     = "database"
      category_confidence    = 0.82
      category_probabilities = {
          "application": 0.05, "database": 0.82,
          "infrastructure": 0.08, "network": 0.03, "security": 0.02
      }
    """
    predicted_category: str = Field(
        ..., example="database",
        description="Predicted category: database|application|infrastructure|network|security",
    )
    predicted_priority: str = Field(
        ..., example="P1",
        description="Predicted priority: P1|P2|P3|P4",
    )
    category_confidence: float = Field(
        ..., ge=0.0, le=1.0, example=0.82,
        description="Confidence for category prediction (0-1)",
    )
    priority_confidence: float = Field(
        ..., ge=0.0, le=1.0, example=0.74,
        description="Confidence for priority prediction (0-1)",
    )
    category_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        example={"application": 0.05, "database": 0.82, "infrastructure": 0.08},
        description="Full probability distribution across all categories",
    )
    priority_probabilities: Dict[str, float] = Field(
        default_factory=dict,
        example={"P1": 0.74, "P2": 0.20, "P3": 0.06},
        description="Full probability distribution across all priorities",
    )


class SimilarTicket(BaseModel):
    """
    INTERNAL schema — one result from similarity_service.find_similar()

    Represents a past ticket that is similar to the new one.
    The resolution field is what gets surfaced as the recommended fix.
    """
    ticket_id: str = Field(..., example="TKT-A1B2C3D4")
    title:     str = Field(..., example="Database connection timeout on order-svc")

    similarity_score: float = Field(
        ..., ge=0.0, le=1.0, example=0.87,
        description="Cosine similarity score (0=different, 1=identical)",
    )
    category: str = Field(..., example="database")
    priority: str = Field(..., example="P1")

    resolution: str = Field(
        ...,
        example=(
            "Increased connection pool size from 50 to 200. "
            "Restarted DB service. Added connection timeout=30s."
        ),
        description="What fixed the similar past ticket",
    )
    resolution_time_minutes: int = Field(
        ..., example=45,
        description="How long it took to resolve the similar ticket",
    )


class GovernanceDecision(BaseModel):
    """
    INTERNAL schema — output of governance_service.decide()

    Contains the final routing decision and all the reasons behind it.
    """
    routing_decision: str = Field(
        ..., example="human_review",
        description="auto_resolve | human_review",
    )
    risk_score: float = Field(
        ..., ge=0.0, le=1.0, example=0.75,
        description="Computed risk score (0=safe, 1=very risky)",
    )
    risk_reasons: List[str] = Field(
        default_factory=list,
        example=["P1 critical incident", "High risk score (0.75)"],
        description="List of reasons why human review was triggered (empty if auto_resolve)",
    )
    assigned_to: str = Field(
        ..., example="team-db",
        description="Team or system assigned: team-db|team-app|team-oncall-p1|system-bot",
    )
    requires_approval: bool = Field(
        ..., example=True,
        description="Whether human approval is required before applying the fix",
    )
    sla_deadline: str = Field(
        ..., example="2026-03-23T14:30:00",
        description="ISO datetime by which ticket must be resolved to meet SLA",
    )


class TicketResponse(BaseModel):
    """
    RESPONSE schema — returned by POST /api/v1/tickets/

    This is the full AI pipeline output that the caller receives
    after submitting a ticket. Contains:
      - Original ticket fields
      - NLP classification result
      - Top similar past tickets
      - Overall confidence score
      - AI-generated fix recommendation
      - Governance routing decision
    """
    ticket_id:   str  = Field(..., example="TKT-A1B2C3D4")
    title:       str  = Field(..., example="Database connection timeout on payment-svc")
    description: str  = Field(..., example="Connection pool exhausted...")
    reporter:    Optional[str]  = Field(None, example="engineer@company.com")
    source:      str  = Field(..., example="api")
    is_vip:      bool = Field(..., example=False)

    # AI Pipeline Outputs
    classification:    TicketClassification = Field(..., description="NLP classification result")
    similar_tickets:   List[SimilarTicket]  = Field(..., description="Top-5 similar past tickets")
    confidence_score:  float = Field(..., ge=0.0, le=1.0, example=0.74)
    recommended_fix:   Optional[str] = Field(None, description="AI-generated fix recommendation")
    governance:        GovernanceDecision   = Field(..., description="Routing and risk decision")

    # Status
    status:     str      = Field(..., example="open")
    created_at: datetime = Field(..., example="2026-03-23T10:00:00")

    class Config:
        from_attributes = True
        # Allows building this schema directly from a SQLAlchemy ORM object


class TicketResolve(BaseModel):
    """
    REQUEST schema — body for POST /api/v1/tickets/{ticket_id}/resolve

    Called by an engineer when they close a ticket.
    ai_recommendation_was_correct is used by the learning pipeline:
      - True  = AI was right -> reinforce this pattern
      - False = AI was wrong -> teach model to correct this case
    """
    resolution_text: str = Field(
        ..., min_length=10,
        example=(
            "Increased connection pool size from 50 to 200. "
            "Restarted DB service. Added connection timeout=30s."
        ),
        description="What actually fixed the issue",
    )
    resolved_by: str = Field(
        ..., example="john.doe@company.com",
        description="Name or email of the engineer who resolved it",
    )
    ai_recommendation_was_correct: bool = Field(
        ..., example=True,
        description=(
            "Was the AI's suggested fix correct? "
            "This feeds the continuous learning pipeline."
        ),
    )


class TicketListItem(BaseModel):
    """
    RESPONSE schema — one item in GET /api/v1/tickets/ list.

    Compact version of TicketResponse — only the fields needed
    for a list view (no similar_tickets, no full classification probs).
    """
    ticket_id:          str            = Field(..., example="TKT-A1B2C3D4")
    title:              str            = Field(..., example="DB connection timeout")
    predicted_category: Optional[str]  = Field(None, example="database")
    predicted_priority: Optional[str]  = Field(None, example="P1")
    confidence_score:   Optional[float]= Field(None, example=0.74)
    routing_decision:   Optional[str]  = Field(None, example="human_review")
    assigned_to:        Optional[str]  = Field(None, example="team-db")
    status:             str            = Field(..., example="open")
    sla_breached:       bool           = Field(..., example=False)
    created_at:         datetime       = Field(..., example="2026-03-23T10:00:00")

    class Config:
        from_attributes = True


class TicketResolveResponse(BaseModel):
    """RESPONSE schema — returned by POST /api/v1/tickets/{id}/resolve"""
    success:                  bool = Field(..., example=True)
    ticket_id:                str  = Field(..., example="TKT-A1B2C3D4")
    resolution_time_minutes:  int  = Field(..., example=45)
    ai_recommendation_correct: Optional[bool] = Field(None, example=True)
    message:                  str  = Field(..., example="Ticket resolved. 12/50 resolutions collected.")


# ==============================================================================
# SECTION 2: LOG / ANOMALY SCHEMAS
# ==============================================================================

class MetricsInput(BaseModel):
    """
    REQUEST schema — body for POST /api/v1/logs/detect

    The 5 core system metrics the anomaly model needs.
    These must match EXACTLY what the model was trained on.

    Validators ensure values are in sensible ranges before
    the data reaches the ML model.
    """
    service: str = Field(
        ..., example="payment-svc",
        description="Name of the service being monitored",
    )
    response_time_ms: float = Field(
        ..., ge=0, example=120.5,
        description="Average response time in milliseconds",
    )
    error_rate: float = Field(
        ..., ge=0.0, le=1.0, example=0.02,
        description="Error rate as a fraction: 0.02 = 2% errors",
    )
    cpu_usage_pct: float = Field(
        ..., ge=0.0, le=100.0, example=45.0,
        description="CPU usage percentage (0-100)",
    )
    memory_usage_pct: float = Field(
        ..., ge=0.0, le=100.0, example=60.0,
        description="Memory usage percentage (0-100)",
    )
    request_count: int = Field(
        ..., ge=0, example=1000,
        description="Number of requests in the monitoring window",
    )
    log_message: Optional[str] = Field(
        None, example="Connection pool at 90% capacity",
        description="Optional log message that triggered this check",
    )

    @field_validator("response_time_ms")
    @classmethod
    def reasonable_latency(cls, v: float) -> float:
        """60 seconds is an unreasonably high latency — likely a data error."""
        if v > 60_000:
            raise ValueError("response_time_ms > 60000ms — likely a data error")
        return v


class AnomalyResult(BaseModel):
    """
    RESPONSE schema — returned by POST /api/v1/logs/detect

    Full anomaly detection output including ML score, severity,
    and rule-based root cause analysis.
    """
    service:    str  = Field(..., example="payment-svc")
    is_anomaly: bool = Field(..., example=True)

    anomaly_score: float = Field(
        ..., example=-0.65,
        description=(
            "Raw IsolationForest score. "
            "Lower = more anomalous. Typical range: -0.8 to 0.0"
        ),
    )
    severity: str = Field(
        ..., example="high",
        description="critical | high | medium | low | normal",
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0, example=0.85,
        description="How confident is the anomaly classification",
    )

    root_cause_hint: str = Field(
        ...,
        example="Critical latency (2800ms); CPU exhaustion (95%)",
        description="Human-readable explanation of what looks wrong",
    )
    recommended_action: str = Field(
        ...,
        example="Check DB connection pool; Kill runaway process",
        description="Suggested remediation steps",
    )
    metrics: Dict[str, float] = Field(
        ...,
        example={
            "response_time_ms": 2800.0,
            "error_rate": 0.45,
            "cpu_usage_pct": 95.0,
            "memory_usage_pct": 92.0,
            "request_count": 900,
        },
        description="The metrics snapshot that was analyzed",
    )


class BulkLogRequest(BaseModel):
    """
    REQUEST schema — body for POST /api/v1/logs/analyze

    Send up to 1000 log entries for batch anomaly analysis.
    """
    logs: List[MetricsInput] = Field(
        ...,
        max_length=1000,
        description="List of metrics snapshots to analyze (max 1000)",
    )

    @model_validator(mode="after")
    def must_have_at_least_one(self) -> "BulkLogRequest":
        if len(self.logs) == 0:
            raise ValueError("logs list cannot be empty")
        return self


class BulkLogResponse(BaseModel):
    """
    RESPONSE schema — returned by POST /api/v1/logs/analyze

    Summary + list of anomalous entries only (normals filtered out).
    """
    total_logs:         int   = Field(..., example=100)
    anomalies_detected: int   = Field(..., example=8)
    anomaly_rate:       float = Field(..., example=0.08)
    anomalies:          List[AnomalyResult] = Field(..., description="Anomalous entries sorted by severity")
    summary:            str = Field(
        ...,
        example="Found 8 anomalies in 100 entries (8.0%). Affected: payment-svc, auth-svc",
    )


class AnomalyAcknowledge(BaseModel):
    """REQUEST schema — body for POST /api/v1/logs/anomalies/{id}/acknowledge"""
    acknowledged_by: str = Field(..., example="ops-team@company.com")
    notes: Optional[str] = Field(None, example="Investigating high CPU — likely batch job")


# ==============================================================================
# SECTION 3: DASHBOARD SCHEMAS
# ==============================================================================

class ModelStatus(BaseModel):
    """Health status of each loaded ML model."""
    ticket_classifier: str = Field(..., example="ready")
    anomaly_detector:  str = Field(..., example="ready")
    similarity_index:  str = Field(..., example="ready (500 tickets)")


class DashboardMetrics(BaseModel):
    """
    RESPONSE schema — returned by GET /api/v1/dashboard/metrics

    Aggregated KPIs for the DevOps dashboard.
    Shows: ticket volumes, AI performance, SLA status, anomaly counts.
    """
    # Ticket volumes
    total_tickets:    int = Field(..., example=342)
    open_tickets:     int = Field(..., example=28)
    auto_resolved:    int = Field(..., example=180)
    human_review:     int = Field(..., example=162)

    # AI performance
    avg_confidence_score:          float = Field(..., example=0.74)
    ai_recommendation_accuracy:    float = Field(..., example=0.88)
    avg_resolution_time_minutes:   float = Field(..., example=65.5)

    # SLA
    sla_breach_rate: float = Field(..., example=0.04)

    # Distributions
    top_categories: Dict[str, int] = Field(
        ..., example={"database": 120, "application": 95, "infrastructure": 80}
    )
    top_priorities: Dict[str, int] = Field(
        ..., example={"P1": 45, "P2": 150, "P3": 100, "P4": 47}
    )

    # Anomalies
    anomalies_last_24h: int = Field(..., example=12)

    # Learning pipeline
    resolutions_since_retrain: int = Field(..., example=23)


class HealthResponse(BaseModel):
    """
    RESPONSE schema — returned by GET /api/v1/dashboard/health

    Reports the health of the API, database, and all ML models.
    """
    status:    str      = Field(..., example="healthy")
    timestamp: datetime = Field(..., example="2026-03-23T10:00:00")
    database:  str      = Field(..., example="connected")
    models:    ModelStatus
    tickets_in_db:   int = Field(..., example=342)
    anomalies_in_db: int = Field(..., example=58)
    resolutions_counter: int = Field(..., example=23)


# ==============================================================================
# SECTION 4: GENERIC RESPONSE SCHEMAS
# ==============================================================================

class SuccessResponse(BaseModel):
    """Generic success response used by simple endpoints."""
    success: bool = Field(default=True)
    message: str  = Field(..., example="Operation completed successfully")
    data:    Optional[Any] = Field(None, description="Optional payload")


class ErrorResponse(BaseModel):
    """Generic error response — returned by the global exception handler."""
    success: bool = Field(default=False)
    error:   str  = Field(..., example="Ticket TKT-A1B2C3D4 not found")
    detail:  Optional[str] = Field(None, example="No record found with that ID")


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""
    total:   int = Field(..., example=342, description="Total records matching filter")
    limit:   int = Field(..., example=50)
    offset:  int = Field(..., example=0)
    items:   List[Any] = Field(..., description="Page of results")