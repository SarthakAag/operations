"""
app/routes/anomalies.py
================================================================================
PURPOSE:
  Standalone anomaly CRUD router — extracted from logs.py to keep
  responsibilities clean. logs.py was 500+ lines doing TWO things:
  detection pipeline AND anomaly record management.

  This file owns: list, get, acknowledge, resolve, stats for AnomalyEventDB.
  logs.py keeps: detect (single) + analyze (bulk) — the ML pipeline.

MIGRATION NOTE:
  To migrate without breaking existing clients:
    Option A (recommended): Keep logs.py anomaly endpoints as thin proxies
      that call the same DB logic, and add this router under /api/v1/anomalies.
    Option B: Register this router at /api/v1/logs (same prefix) and
      remove the anomaly endpoints from logs.py — no client URL changes.

  These endpoints are registered at /api/v1/anomalies by default.

ENDPOINTS:
  GET   /api/v1/anomalies/                              List anomaly events
  GET   /api/v1/anomalies/stats                         Aggregate stats
  GET   /api/v1/anomalies/{event_id}                    Get single anomaly
  POST  /api/v1/anomalies/{event_id}/acknowledge        Acknowledge anomaly
  PUT   /api/v1/anomalies/{event_id}/resolve            Resolve / close anomaly
  GET   /api/v1/anomalies/{event_id}/similar            Find similar past anomalies
================================================================================
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AnomalyEventDB, get_db
from app.schemas import AnomalyAcknowledge, SuccessResponse

router = APIRouter(
    prefix="/api/v1/anomalies",
    tags=["Anomalies"],
)


# ==============================================================================
# ENDPOINT 1: GET /api/v1/anomalies/
# ==============================================================================

@router.get(
    "/",
    summary="List Anomaly Events",
    description="""
Paginated list of all anomaly events with optional filters.

Supports filtering by:
- service name
- severity: critical | high | medium | low
- status: open | acknowledged | resolved | false_positive
- last N days
    """,
)
async def list_anomalies(
    service:  Optional[str] = Query(None, description="Filter by service name"),
    severity: Optional[str] = Query(None, description="critical|high|medium|low"),
    status:   Optional[str] = Query(None, description="open|acknowledged|resolved|false_positive"),
    days:     Optional[int] = Query(None, ge=1, le=365, description="Last N days"),
    limit:    int           = Query(50, ge=1, le=200),
    offset:   int           = Query(0,  ge=0),
    db: AsyncSession        = Depends(get_db),
):
    """
    Returns a paginated list of anomaly events, newest first.
    """
    query       = select(AnomalyEventDB).order_by(AnomalyEventDB.created_at.desc())
    count_query = select(func.count(AnomalyEventDB.id))

    if service:
        query       = query.where(AnomalyEventDB.service == service)
        count_query = count_query.where(AnomalyEventDB.service == service)
    if severity:
        query       = query.where(AnomalyEventDB.severity == severity)
        count_query = count_query.where(AnomalyEventDB.severity == severity)
    if status:
        query       = query.where(AnomalyEventDB.status == status)
        count_query = count_query.where(AnomalyEventDB.status == status)
    if days:
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days)
        query       = query.where(AnomalyEventDB.created_at >= cutoff)
        count_query = count_query.where(AnomalyEventDB.created_at >= cutoff)

    total  = await db.scalar(count_query) or 0
    result = await db.execute(query.limit(limit).offset(offset))
    events = result.scalars().all()

    return {
        "total":     total,
        "limit":     limit,
        "offset":    offset,
        "anomalies": [_event_to_dict(e) for e in events],
    }


# ==============================================================================
# ENDPOINT 2: GET /api/v1/anomalies/stats
# ==============================================================================

@router.get(
    "/stats",
    summary="Anomaly Stats",
    description="""
Aggregated anomaly statistics used by the DevOps dashboard health panel.

Returns:
- total_anomalies (all time)
- last_24h count
- by_severity: {critical: N, high: N, medium: N, low: N}
- by_status: {open: N, acknowledged: N, resolved: N, false_positive: N}
- top_services: 5 most affected services
- false_positive_rate: fraction of closed anomalies that were FP
    """,
)
async def get_anomaly_stats(
    db: AsyncSession = Depends(get_db),
):
    """
    Aggregate anomaly counts for the dashboard summary panel.
    Identical to /api/v1/logs/anomalies/stats but at the new path.
    """
    from sqlalchemy import text as sa_text

    total = await db.scalar(select(func.count(AnomalyEventDB.id))) or 0

    severity_rows = await db.execute(
        select(AnomalyEventDB.severity, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.severity)
    )
    by_severity = {row[0]: row[1] for row in severity_rows if row[0]}

    status_rows = await db.execute(
        select(AnomalyEventDB.status, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.status)
    )
    by_status = {row[0]: row[1] for row in status_rows if row[0]}

    service_rows = await db.execute(
        select(AnomalyEventDB.service, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.service)
        .order_by(func.count(AnomalyEventDB.id).desc())
        .limit(5)
    )
    top_services = {row[0]: row[1] for row in service_rows}

    last_24h = await db.scalar(
        select(func.count(AnomalyEventDB.id)).where(
            AnomalyEventDB.created_at >= sa_text("NOW() - INTERVAL '24 hours'")
        )
    ) or 0

    # False positive rate among closed anomalies
    closed = (by_status.get("resolved", 0) + by_status.get("false_positive", 0))
    fp_rate = round(
        by_status.get("false_positive", 0) / closed, 4
    ) if closed else 0.0

    return {
        "total_anomalies":   total,
        "last_24h":          last_24h,
        "by_severity":       by_severity,
        "by_status":         by_status,
        "top_services":      top_services,
        "false_positive_rate": fp_rate,
    }


# ==============================================================================
# ENDPOINT 3: GET /api/v1/anomalies/{event_id}
# ==============================================================================

@router.get(
    "/{event_id}",
    summary="Get Anomaly",
    description="Get full details of a single anomaly event.",
)
async def get_anomaly(
    event_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns the full anomaly record including metrics snapshot,
    root cause hint, recommended action, and current status.
    """
    event = await _get_anomaly_or_404(event_id, db)
    return _event_to_dict(event)


# ==============================================================================
# ENDPOINT 4: POST /api/v1/anomalies/{event_id}/acknowledge
# ==============================================================================

@router.post(
    "/{event_id}/acknowledge",
    response_model=SuccessResponse,
    summary="Acknowledge Anomaly",
    description="Mark an anomaly as acknowledged by an engineer.",
)
async def acknowledge_anomaly(
    event_id: str,
    body: AnomalyAcknowledge,
    db: AsyncSession = Depends(get_db),
):
    """
    Marks an anomaly as acknowledged.
    Rejects if already resolved or false_positive.
    """
    event = await _get_anomaly_or_404(event_id, db)

    if event.status in ("resolved", "false_positive"):
        raise HTTPException(
            status_code=400,
            detail=f"Cannot acknowledge anomaly with status '{event.status}'"
        )
    if event.status == "acknowledged":
        raise HTTPException(
            status_code=400,
            detail=f"Anomaly {event_id} is already acknowledged"
        )

    await db.execute(
        update(AnomalyEventDB)
        .where(AnomalyEventDB.event_id == event_id)
        .values(
            status          = "acknowledged",
            acknowledged_by = body.acknowledged_by,
            acknowledged_at = datetime.utcnow(),
        )
    )

    print(f"  [AnomaliesRoute] {event_id} acknowledged by {body.acknowledged_by}")

    return SuccessResponse(
        message=f"Anomaly {event_id} acknowledged by {body.acknowledged_by}",
        data={
            "event_id":        event_id,
            "acknowledged_by": body.acknowledged_by,
            "acknowledged_at": datetime.utcnow().isoformat(),
            "notes":           body.notes,
        },
    )


# ==============================================================================
# ENDPOINT 5: PUT /api/v1/anomalies/{event_id}/resolve
# ==============================================================================

@router.put(
    "/{event_id}/resolve",
    response_model=SuccessResponse,
    summary="Resolve Anomaly",
    description="""
Close an anomaly as resolved or false_positive.

- resolved       → confirmed anomaly, fix was applied
- false_positive → ML model was wrong (feeds into model improvement tracking)
    """,
)
async def resolve_anomaly(
    event_id:    str,
    resolved_by: str = Query(..., description="Engineer name or email"),
    resolution:  str = Query("resolved", description="resolved | false_positive"),
    db: AsyncSession = Depends(get_db),
):
    """
    Closes an anomaly event with a resolution status.
    """
    valid = {"resolved", "false_positive"}
    if resolution not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution '{resolution}'. Valid: {valid}"
        )

    event = await _get_anomaly_or_404(event_id, db)

    if event.status in ("resolved", "false_positive"):
        raise HTTPException(
            status_code=400,
            detail=f"Anomaly {event_id} is already '{event.status}'"
        )

    await db.execute(
        update(AnomalyEventDB)
        .where(AnomalyEventDB.event_id == event_id)
        .values(
            status      = resolution,
            resolved_at = datetime.utcnow(),
        )
    )

    print(f"  [AnomaliesRoute] {event_id} marked as '{resolution}' by {resolved_by}")

    return SuccessResponse(
        message=f"Anomaly {event_id} marked as '{resolution}' by {resolved_by}",
        data={
            "event_id":    event_id,
            "resolution":  resolution,
            "resolved_by": resolved_by,
            "resolved_at": datetime.utcnow().isoformat(),
        },
    )


# ==============================================================================
# ENDPOINT 6: GET /api/v1/anomalies/{event_id}/similar
# ==============================================================================

@router.get(
    "/{event_id}/similar",
    summary="Similar Anomalies",
    description="""
Find past anomalies from the same service with similar severity.

Useful when investigating an anomaly — lets the engineer quickly see
if this service has had this pattern before and how it was resolved.
    """,
)
async def get_similar_anomalies(
    event_id: str,
    limit:    int = Query(5, ge=1, le=20),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns past anomalies from the same service, sorted by created_at desc.
    Excludes the queried event itself.
    """
    event = await _get_anomaly_or_404(event_id, db)

    result = await db.execute(
        select(AnomalyEventDB)
        .where(
            AnomalyEventDB.service  == event.service,
            AnomalyEventDB.severity == event.severity,
            AnomalyEventDB.event_id != event_id,
        )
        .order_by(AnomalyEventDB.created_at.desc())
        .limit(limit)
    )
    similar = result.scalars().all()

    return {
        "event_id":      event_id,
        "service":       event.service,
        "severity":      event.severity,
        "similar_count": len(similar),
        "similar":       [_event_to_dict(e) for e in similar],
    }


# ==============================================================================
# HELPERS
# ==============================================================================

async def _get_anomaly_or_404(
    event_id: str,
    db: AsyncSession,
) -> AnomalyEventDB:
    """
    Fetches an AnomalyEventDB by event_id or raises 404.
    Reused across all endpoints in this router.
    """
    result = await db.execute(
        select(AnomalyEventDB).where(AnomalyEventDB.event_id == event_id)
    )
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(
            status_code=404,
            detail=f"Anomaly event '{event_id}' not found"
        )
    return event


def _event_to_dict(event: AnomalyEventDB) -> dict:
    """
    Converts an AnomalyEventDB ORM row to a clean response dict.
    Mirrors the same helper in logs.py — kept in sync intentionally.
    If you change one, change both (or extract to app/utils/formatters.py).
    """
    return {
        "event_id":           event.event_id,
        "service":            event.service,
        "anomaly_score":      event.anomaly_score,
        "severity":           event.severity,
        "confidence":         event.confidence,
        "root_cause_hint":    event.root_cause_hint,
        "recommended_action": event.recommended_action,
        "metrics_snapshot":   event.metrics_snapshot,
        "log_message":        event.log_message,
        "status":             event.status,
        "acknowledged_by":    event.acknowledged_by,
        "acknowledged_at":    event.acknowledged_at.isoformat() if event.acknowledged_at else None,
        "resolved_at":        event.resolved_at.isoformat() if event.resolved_at else None,
        "created_at":         event.created_at.isoformat() if event.created_at else None,
    }