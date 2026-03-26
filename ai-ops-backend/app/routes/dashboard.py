"""
app/routes/dashboard.py
================================================================================
PURPOSE:
  Real-time DevOps dashboard endpoints.
  Aggregates data across tickets, anomalies, and resolutions into
  the DashboardMetrics and HealthResponse schemas defined in schemas.py.

ENDPOINTS:
  GET  /api/v1/dashboard/metrics       Full dashboard data (tickets + anomalies + SLA)
  GET  /api/v1/dashboard/health        System health summary (HealthResponse)
  GET  /api/v1/dashboard/live          Lightweight poll endpoint (for 30s auto-refresh)
  GET  /api/v1/dashboard/sla-breaches  Tickets that missed or are near SLA deadline
================================================================================
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AnomalyEventDB, ResolutionDB, TicketDB, get_db

router = APIRouter(
    prefix="/api/v1/dashboard",
    tags=["Dashboard"],
)


# ==============================================================================
# ENDPOINT 1: GET /api/v1/dashboard/metrics
# ==============================================================================

@router.get(
    "/metrics",
    summary="Dashboard Metrics",
    description="""
Full dashboard data snapshot used by the DevOps dashboard UI.

Returns:
- Ticket counts by status, category, priority, routing
- Anomaly counts by severity and status
- AI performance stats (accuracy, auto-resolve rate)
- SLA compliance rate
- Mean time to resolution (MTTR) per priority
- Last-updated timestamp
    """,
)
async def get_dashboard_metrics(
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregates all key metrics into a single dashboard payload.
    Designed to be called on page load and every 30s via /live.
    """

    # ── Ticket counts ─────────────────────────────────────────────
    total_tickets = await db.scalar(select(func.count(TicketDB.id))) or 0

    status_rows = await db.execute(
        select(TicketDB.status, func.count(TicketDB.id))
        .group_by(TicketDB.status)
    )
    tickets_by_status = {row[0]: row[1] for row in status_rows}

    category_rows = await db.execute(
        select(TicketDB.predicted_category, func.count(TicketDB.id))
        .group_by(TicketDB.predicted_category)
    )
    tickets_by_category = {row[0]: row[1] for row in category_rows if row[0]}

    priority_rows = await db.execute(
        select(TicketDB.predicted_priority, func.count(TicketDB.id))
        .group_by(TicketDB.predicted_priority)
    )
    tickets_by_priority = {row[0]: row[1] for row in priority_rows if row[0]}

    routing_rows = await db.execute(
        select(TicketDB.routing_decision, func.count(TicketDB.id))
        .group_by(TicketDB.routing_decision)
    )
    tickets_by_routing = {row[0]: row[1] for row in routing_rows if row[0]}

    # ── AI performance ────────────────────────────────────────────
    auto_resolved = tickets_by_status.get("auto_resolved", 0)
    auto_resolve_rate = round(auto_resolved / total_tickets, 4) if total_tickets else 0.0

    # AI accuracy: fraction of resolved tickets where AI was correct
    accuracy_result = await db.execute(
        select(
            func.count(TicketDB.id).label("total"),
            func.sum(
                func.cast(TicketDB.ai_recommendation_correct, type_=func.count().type)
            ).label("correct"),
        ).where(TicketDB.ai_recommendation_correct.isnot(None))
    )
    acc_row = accuracy_result.one_or_none()
    if acc_row and acc_row[0]:
        ai_accuracy = round((acc_row[1] or 0) / acc_row[0], 4)
    else:
        ai_accuracy = None

    # Avg confidence score across all tickets
    avg_conf = await db.scalar(
        select(func.avg(TicketDB.confidence_score)).where(
            TicketDB.confidence_score.isnot(None)
        )
    )

    # ── SLA compliance ────────────────────────────────────────────
    now = datetime.utcnow()
    sla_breached_count = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.sla_deadline < now,
            TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
        )
    ) or 0

    open_tickets = tickets_by_status.get("open", 0) + tickets_by_status.get("in_progress", 0)
    sla_compliance_rate = (
        round(1 - sla_breached_count / open_tickets, 4) if open_tickets > 0 else 1.0
    )

    # ── MTTR per priority ─────────────────────────────────────────
    mttr_rows = await db.execute(
        select(
            TicketDB.predicted_priority,
            func.avg(TicketDB.resolution_time_minutes),
        )
        .where(TicketDB.resolution_time_minutes.isnot(None))
        .group_by(TicketDB.predicted_priority)
    )
    mttr_by_priority = {
        row[0]: round(float(row[1]), 1)
        for row in mttr_rows
        if row[0] and row[1] is not None
    }

    # ── Anomaly counts ────────────────────────────────────────────
    total_anomalies = await db.scalar(select(func.count(AnomalyEventDB.id))) or 0

    anomaly_severity_rows = await db.execute(
        select(AnomalyEventDB.severity, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.severity)
    )
    anomalies_by_severity = {row[0]: row[1] for row in anomaly_severity_rows if row[0]}

    anomaly_status_rows = await db.execute(
        select(AnomalyEventDB.status, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.status)
    )
    anomalies_by_status = {row[0]: row[1] for row in anomaly_status_rows if row[0]}

    open_anomalies = anomalies_by_status.get("open", 0)

    # Last 24h anomalies
    anomalies_24h = await db.scalar(
        select(func.count(AnomalyEventDB.id)).where(
            AnomalyEventDB.created_at >= sa_text("NOW() - INTERVAL '24 hours'")
        )
    ) or 0

    # ── Tickets last 24h ──────────────────────────────────────────
    tickets_24h = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.created_at >= sa_text("NOW() - INTERVAL '24 hours'")
        )
    ) or 0

    return {
        "generated_at": now.isoformat(),
        "tickets": {
            "total":            total_tickets,
            "last_24h":         tickets_24h,
            "by_status":        tickets_by_status,
            "by_category":      tickets_by_category,
            "by_priority":      tickets_by_priority,
            "by_routing":       tickets_by_routing,
            "sla_breached":     sla_breached_count,
            "sla_compliance_rate": sla_compliance_rate,
            "mttr_by_priority": mttr_by_priority,
        },
        "ai_performance": {
            "auto_resolve_rate": auto_resolve_rate,
            "ai_accuracy":       ai_accuracy,
            "avg_confidence":    round(float(avg_conf), 4) if avg_conf else None,
        },
        "anomalies": {
            "total":       total_anomalies,
            "open":        open_anomalies,
            "last_24h":    anomalies_24h,
            "by_severity": anomalies_by_severity,
            "by_status":   anomalies_by_status,
        },
    }


# ==============================================================================
# ENDPOINT 2: GET /api/v1/dashboard/health
# ==============================================================================

@router.get(
    "/health",
    summary="System Health",
    description="""
High-level system health summary. Maps to the HealthResponse schema.

Returns:
- overall_status: healthy | degraded | critical
- open_p1_tickets count
- open critical anomalies count
- SLA breach count
- AI model status (confidence level)
    """,
)
async def get_health(
    db: AsyncSession = Depends(get_db),
):
    """
    Determines system health from current open P1 tickets,
    critical anomalies, and SLA breaches.
    Used by the top-of-dashboard health banner.
    """
    now = datetime.utcnow()

    # Open P1 tickets
    open_p1 = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.predicted_priority == "P1",
            TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
        )
    ) or 0

    # Critical open anomalies
    critical_anomalies = await db.scalar(
        select(func.count(AnomalyEventDB.id)).where(
            AnomalyEventDB.severity == "critical",
            AnomalyEventDB.status == "open",
        )
    ) or 0

    # SLA breaches right now
    sla_breaches = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.sla_deadline < now,
            TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
        )
    ) or 0

    # Avg confidence last 100 tickets
    avg_conf = await db.scalar(
        select(func.avg(TicketDB.confidence_score))
        .order_by(TicketDB.created_at.desc())
        .limit(100)
    )

    # Determine overall health
    if open_p1 >= 3 or critical_anomalies >= 2 or sla_breaches >= 5:
        overall_status = "critical"
    elif open_p1 >= 1 or critical_anomalies >= 1 or sla_breaches >= 2:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return {
        "overall_status":      overall_status,
        "open_p1_tickets":     open_p1,
        "critical_anomalies":  critical_anomalies,
        "sla_breaches":        sla_breaches,
        "avg_ai_confidence":   round(float(avg_conf), 4) if avg_conf else None,
        "checked_at":          now.isoformat(),
    }


# ==============================================================================
# ENDPOINT 3: GET /api/v1/dashboard/live
# ==============================================================================

@router.get(
    "/live",
    summary="Live Poll",
    description="""
Lightweight endpoint for the dashboard 30-second auto-refresh.

Returns only the numbers that change frequently:
- open ticket count
- open anomaly count
- P1 count
- SLA breach count
- system status
    """,
)
async def get_live_snapshot(
    db: AsyncSession = Depends(get_db),
):
    """
    Minimal query set for frequent polling.
    Avoids the heavier aggregations in /metrics.
    """
    now = datetime.utcnow()

    open_tickets = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.status.in_(["open", "in_progress"])
        )
    ) or 0

    open_anomalies = await db.scalar(
        select(func.count(AnomalyEventDB.id)).where(
            AnomalyEventDB.status == "open"
        )
    ) or 0

    open_p1 = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.predicted_priority == "P1",
            TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
        )
    ) or 0

    sla_breaches = await db.scalar(
        select(func.count(TicketDB.id)).where(
            TicketDB.sla_deadline < now,
            TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
        )
    ) or 0

    # Lightweight health check
    if open_p1 >= 3 or sla_breaches >= 5:
        status = "critical"
    elif open_p1 >= 1 or sla_breaches >= 2 or open_anomalies >= 3:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "status":         status,
        "open_tickets":   open_tickets,
        "open_anomalies": open_anomalies,
        "open_p1":        open_p1,
        "sla_breaches":   sla_breaches,
        "polled_at":      now.isoformat(),
    }


# ==============================================================================
# ENDPOINT 4: GET /api/v1/dashboard/sla-breaches
# ==============================================================================

@router.get(
    "/sla-breaches",
    summary="SLA Breaches",
    description="List tickets that have missed or are within 30 minutes of breaching their SLA deadline.",
)
async def get_sla_breaches(
    include_near: bool = Query(True, description="Include tickets within 30 min of SLA deadline"),
    limit:        int  = Query(50, ge=1, le=200),
    db: AsyncSession   = Depends(get_db),
):
    """
    Returns tickets that are breached or near-breach.
    Used by the SLA panel in the dashboard.
    """
    now = datetime.utcnow()
    near_cutoff = now + timedelta(minutes=30)

    active_statuses = ["open", "in_progress", "escalated"]

    # Already breached
    breached_result = await db.execute(
        select(TicketDB)
        .where(
            TicketDB.sla_deadline < now,
            TicketDB.status.in_(active_statuses),
        )
        .order_by(TicketDB.sla_deadline.asc())
        .limit(limit)
    )
    breached = breached_result.scalars().all()

    near_breach = []
    if include_near:
        near_result = await db.execute(
            select(TicketDB)
            .where(
                TicketDB.sla_deadline >= now,
                TicketDB.sla_deadline <= near_cutoff,
                TicketDB.status.in_(active_statuses),
            )
            .order_by(TicketDB.sla_deadline.asc())
            .limit(limit)
        )
        near_breach = near_result.scalars().all()

    def _format(t: TicketDB, is_breached: bool) -> dict:
        minutes_overdue = None
        minutes_remaining = None
        deadline = t.sla_deadline
        if deadline:
            delta = (now - deadline).total_seconds() / 60
            if is_breached:
                minutes_overdue = round(delta)
            else:
                minutes_remaining = round(-delta)
        return {
            "ticket_id":          t.ticket_id,
            "title":              t.title,
            "predicted_priority": t.predicted_priority,
            "predicted_category": t.predicted_category,
            "assigned_to":        t.assigned_to,
            "status":             t.status,
            "is_vip":             t.is_vip or False,
            "sla_deadline":       deadline.isoformat() if deadline else None,
            "minutes_overdue":    minutes_overdue,
            "minutes_remaining":  minutes_remaining,
            "created_at":         t.created_at.isoformat() if t.created_at else None,
        }

    return {
        "checked_at":      now.isoformat(),
        "breached_count":  len(breached),
        "near_breach_count": len(near_breach),
        "breached":        [_format(t, True)  for t in breached],
        "near_breach":     [_format(t, False) for t in near_breach],
    }