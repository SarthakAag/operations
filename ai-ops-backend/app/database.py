"""
app/database.py
================================================================================
PURPOSE:
  - Async PostgreSQL connection via SQLAlchemy + asyncpg driver
  - Defines all ORM table models
  - Provides get_db() session dependency for FastAPI routes
  - Provides init_db() to create all tables at startup

TABLES:
  tickets        -> every ticket processed + all AI predictions
  audit_logs     -> immutable AI decision trail (compliance)
  resolutions    -> human-confirmed fixes (feeds ML retraining)
  anomaly_events -> anomalies detected from system logs

INSTALL REQUIREMENTS:
  pip install sqlalchemy asyncpg greenlet

.env for PostgreSQL:
  DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/ai_ops_db

.env for SQLite (dev only):
  DATABASE_URL=sqlite+aiosqlite:///./ai_support.db
================================================================================
"""

from datetime import datetime
from typing import AsyncGenerator

from sqlalchemy import (
    JSON, Boolean, Column, DateTime,
    Float, Integer, String, Text,
    text, Index
)
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.config import get_settings

settings = get_settings()


# ==============================================================================
# SECTION 1: ENGINE + SESSION FACTORY
# ==============================================================================
# create_async_engine:
#   pool_size=10        -> keep 10 connections open permanently
#   max_overflow=20     -> allow 20 extra connections under heavy load
#   pool_pre_ping=True  -> test connection before using (handles dropped connections)
#   echo=False          -> set True in dev to print all SQL queries
#
# async_sessionmaker:
#   expire_on_commit=False -> object attributes stay accessible after commit
#                             (important in async — no lazy loading)
# ==============================================================================

engine = create_async_engine(
    settings.database_url,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.debug,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ==============================================================================
# SECTION 2: BASE CLASS
# ==============================================================================

class Base(DeclarativeBase):
    pass


# ==============================================================================
# SECTION 3: ORM TABLE MODELS
# ==============================================================================

class TicketDB(Base):
    """
    TABLE: tickets

    Stores every ticket submitted to the system.
    AI predictions (category, priority, confidence, governance)
    are stored alongside the original ticket fields.

    This means you can:
      - Query "all P1 tickets where AI was wrong"
      - Query "all tickets routed to human_review"
      - Track SLA breaches over time
      - See which categories have lowest AI confidence
    """
    __tablename__ = "tickets"

    # ── Primary Key ──────────────────────────────────────────────
    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(20), unique=True, nullable=False, index=True)
    # e.g., TKT-A1B2C3D4

    # ── Original Ticket Fields ───────────────────────────────────
    title       = Column(String(500), nullable=False)
    description = Column(Text,        nullable=False)
    reporter    = Column(String(200), nullable=True)
    source      = Column(String(50),  default="api")
    # source: api | servicenow | jira | email
    is_vip      = Column(Boolean,     default=False)

    # ── AI Classification Results ────────────────────────────────
    predicted_category  = Column(String(50),  nullable=True)
    predicted_priority  = Column(String(5),   nullable=True)
    category_confidence = Column(Float,       nullable=True)
    priority_confidence = Column(Float,       nullable=True)

    # Full probability distributions stored as JSON
    # e.g., {"application": 0.05, "database": 0.82, "security": 0.13}
    category_probabilities = Column(JSON, nullable=True)
    priority_probabilities = Column(JSON, nullable=True)

    # ── Similarity Engine Results ────────────────────────────────
    # Top-5 similar past tickets stored as JSON list
    # e.g., [{"ticket_id": "TKT-XYZ", "score": 0.87, "resolution": "..."}, ...]
    similar_tickets = Column(JSON, nullable=True)

    # ── Confidence + Recommendation ──────────────────────────────
    confidence_score = Column(Float, nullable=True)
    # Overall confidence: 0.0 -> 1.0

    recommended_fix = Column(Text, nullable=True)
    # AI-generated fix recommendation text

    # ── Governance Decision ──────────────────────────────────────
    routing_decision = Column(String(20), nullable=True)
    # auto_resolve | human_review

    risk_score  = Column(Float,       nullable=True)
    assigned_to = Column(String(100), nullable=True)
    # e.g., team-db | system-bot | team-oncall-p1

    risk_reasons = Column(JSON, nullable=True)
    # List of reasons why it was routed to human (if applicable)

    # ── SLA Tracking ─────────────────────────────────────────────
    sla_deadline = Column(DateTime(timezone=True), nullable=True)
    sla_breached = Column(Boolean, default=False)

    # ── Resolution ───────────────────────────────────────────────
    status = Column(String(20), default="open", index=True)
    # open | in_progress | auto_resolved | resolved | closed

    actual_resolution = Column(Text,         nullable=True)
    resolved_by       = Column(String(100),  nullable=True)
    # resolved_by = "system-bot" for auto_resolved, human name otherwise

    resolved_at               = Column(DateTime(timezone=True), nullable=True)
    resolution_time_minutes   = Column(Integer, nullable=True)
    ai_recommendation_correct = Column(Boolean, nullable=True)
    # Was the AI's suggested fix correct? Set when human resolves the ticket.

    # ── Timestamps ───────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Indexes for common queries ────────────────────────────────
    __table_args__ = (
        Index("ix_tickets_status_priority",   "status", "predicted_priority"),
        Index("ix_tickets_category_created",  "predicted_category", "created_at"),
        Index("ix_tickets_routing_created",   "routing_decision", "created_at"),
    )

    def __repr__(self):
        return (
            f"<Ticket {self.ticket_id} | "
            f"{self.predicted_category}/{self.predicted_priority} | "
            f"{self.routing_decision} | {self.status}>"
        )


class AuditLogDB(Base):
    """
    TABLE: audit_logs

    Immutable record of EVERY AI decision made on every ticket.
    One ticket can have multiple audit entries:
      - ticket_submitted  -> classification + routing decision
      - ticket_escalated  -> manually escalated by engineer
      - ticket_resolved   -> resolution recorded

    WHY AUDIT LOGS MATTER:
      - Regulatory compliance (GDPR, SOX, ISO 27001)
      - Traceability: "why did the AI route this to team-db?"
      - Model debugging: find patterns in wrong decisions
      - Performance review: track AI accuracy over time
    """
    __tablename__ = "audit_logs"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(20), nullable=False, index=True)

    event_type = Column(String(50), nullable=False)
    # ticket_submitted | ticket_escalated | ticket_resolved |
    # ticket_auto_resolved | model_retrained

    model_version    = Column(String(20),  nullable=True)
    confidence_score = Column(Float,       nullable=True)
    risk_score       = Column(Float,       nullable=True)

    decision = Column(String(200), nullable=True)
    # Human-readable summary: "auto_resolve -> system-bot"

    decision_path = Column(JSON, nullable=True)
    # Full step-by-step reasoning:
    # {
    #   "classification": {"category": "database", "confidence": 0.82},
    #   "similarity":     {"best_score": 0.87, "tickets_found": 5},
    #   "confidence":     {"overall": 0.74},
    #   "governance":     {"routing": "human_review", "reasons": [...]}
    # }

    performed_by = Column(String(100), nullable=True)
    # "ai-system" for automated decisions, engineer name for manual ones

    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_audit_ticket_event", "ticket_id", "event_type"),
    )

    def __repr__(self):
        return f"<AuditLog {self.ticket_id} | {self.event_type} | {self.performed_by}>"


class ResolutionDB(Base):
    """
    TABLE: resolutions

    Stores every human-confirmed resolution.
    This is the TRAINING DATA for the Continuous Learning Pipeline.

    When an engineer resolves a ticket, we record:
      - The original ticket text (input to model)
      - The correct category + priority (correct labels)
      - The resolution that actually worked (fix knowledge)
      - Whether the AI recommendation was correct

    After RETRAIN_AFTER_N_RESOLUTIONS new entries accumulate,
    the models are retrained on this enriched dataset.
    """
    __tablename__ = "resolutions"

    id        = Column(Integer, primary_key=True, autoincrement=True)
    ticket_id = Column(String(20), nullable=False, index=True)

    # ── Ticket content (copied for training) ─────────────────────
    ticket_title       = Column(String(500), nullable=False)
    ticket_description = Column(Text,        nullable=False)

    # ── Correct labels (ground truth for retraining) ─────────────
    category = Column(String(50), nullable=False)
    priority = Column(String(5),  nullable=False)

    # ── Resolution knowledge ──────────────────────────────────────
    resolution_text         = Column(Text,    nullable=False)
    resolution_time_minutes = Column(Integer, nullable=True)

    # ── AI performance tracking ───────────────────────────────────
    ai_recommendation_correct = Column(Boolean, nullable=True)
    # True  = AI suggested the right fix -> positive training signal
    # False = AI was wrong -> also useful: tells us where model fails

    resolved_by = Column(String(100), nullable=True)
    created_at  = Column(DateTime(timezone=True), default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<Resolution {self.ticket_id} | "
            f"{self.category}/{self.priority} | "
            f"AI correct: {self.ai_recommendation_correct}>"
        )


class AnomalyEventDB(Base):
    """
    TABLE: anomaly_events

    Stores every anomaly detected by the IsolationForest model.

    Used for:
      - Dashboard: "how many anomalies in the last 24h?"
      - Incident correlation: "were there anomalies before this P1 ticket?"
      - Model evaluation: review false positives/negatives
      - SLA reporting: document system health history
    """
    __tablename__ = "anomaly_events"

    id       = Column(Integer, primary_key=True, autoincrement=True)
    event_id = Column(String(40), unique=True, nullable=False, index=True)
    service  = Column(String(100), nullable=False, index=True)

    # ── ML Model Output ───────────────────────────────────────────
    anomaly_score = Column(Float,      nullable=False)
    # Lower score = more anomalous. Range: -0.8 (very bad) to 0.0 (normal)

    severity = Column(String(20), nullable=False, index=True)
    # critical | high | medium | low | normal

    confidence = Column(Float, nullable=True)

    # ── Evidence ──────────────────────────────────────────────────
    metrics_snapshot = Column(JSON, nullable=True)
    # The exact metrics that triggered the anomaly:
    # {"response_time_ms": 2800, "error_rate": 0.45, "cpu_usage_pct": 95, ...}

    log_message = Column(Text, nullable=True)

    # ── Root Cause Analysis (rule-based) ──────────────────────────
    root_cause_hint    = Column(Text, nullable=True)
    # e.g., "Critical latency (2800ms); CPU exhaustion (95%)"

    recommended_action = Column(Text, nullable=True)
    # e.g., "Check DB connection pool; Kill runaway process"

    # ── Status Tracking ───────────────────────────────────────────
    status = Column(String(20), default="open", index=True)
    # open | acknowledged | resolved | false_positive

    acknowledged_by = Column(String(100), nullable=True)
    acknowledged_at = Column(DateTime(timezone=True), nullable=True)
    resolved_at     = Column(DateTime(timezone=True), nullable=True)

    # ── Timestamps ───────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)

    __table_args__ = (
        Index("ix_anomaly_service_severity", "service", "severity"),
        Index("ix_anomaly_created",          "created_at"),
    )

    def __repr__(self):
        return (
            f"<Anomaly {self.event_id} | "
            f"{self.service} | {self.severity} | "
            f"score={self.anomaly_score:.3f}>"
        )


# ==============================================================================
# SECTION 4: SESSION DEPENDENCY FOR FASTAPI ROUTES
# ==============================================================================

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    FastAPI dependency — injects an async DB session into every route.

    HOW IT WORKS:
      1. Opens a new async session for the request
      2. Yields it to the route handler
      3. Commits on success
      4. Rolls back on any exception
      5. Always closes the session when done

    USAGE IN ROUTES:
      @router.post("/tickets/")
      async def create_ticket(
          body: TicketCreate,
          db: AsyncSession = Depends(get_db)   <- injected here
      ):
          db.add(new_ticket)
          await db.flush()
          ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# ==============================================================================
# SECTION 5: DATABASE INITIALIZATION
# ==============================================================================

async def init_db():
    """
    Creates all tables in PostgreSQL if they don't exist yet.
    Called ONCE at application startup (in api.py lifespan).

    Uses CREATE TABLE IF NOT EXISTS — safe to call on every restart.
    Does NOT drop existing tables or migrate data.
    For schema migrations, use Alembic.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables():
    """
    Drops ALL tables. USE ONLY IN DEVELOPMENT / TESTING.
    Never call this in production.
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_db_connection() -> bool:
    """
    Tests that the database is reachable.
    Called by the /health endpoint.
    Returns True if connected, False if not.
    """
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False