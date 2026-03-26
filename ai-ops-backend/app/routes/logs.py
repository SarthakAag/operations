"""
app/routes/logs.py
================================================================================
PURPOSE:
  All anomaly detection and log analysis endpoints.
  Wraps the ML anomaly detection pipeline (IsolationForest) + AnomalyEventDB.

ENDPOINTS:
  POST   /api/v1/logs/detect                              Single anomaly check
  POST   /api/v1/logs/analyze                             Bulk log analysis
  GET    /api/v1/logs/anomalies                           List anomaly events
  GET    /api/v1/logs/anomalies/stats                     Anomaly stats summary
  GET    /api/v1/logs/anomalies/{event_id}                Get one anomaly
  POST   /api/v1/logs/anomalies/{event_id}/acknowledge    Acknowledge anomaly
  PUT    /api/v1/logs/anomalies/{event_id}/resolve        Resolve/close anomaly

PIPELINE FLOW (POST /logs/detect):
  MetricsInput (request)
      ↓
  ml_service.detect_anomaly()      Step A: IsolationForest scoring
      ↓
  AnomalyEventDB.save()            Step B: persist only if anomaly detected
      ↓
  AnomalyResult (response)

PIPELINE FLOW (POST /logs/analyze):
  BulkLogRequest (request)
      ↓
  ml_service.detect_anomaly() x N  Step A: score each log entry
      ↓
  AnomalyEventDB.save() x anomalies Step B: persist anomalies only
      ↓
  BulkLogResponse (response)
================================================================================
"""

import uuid
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select, text as sa_text, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AnomalyEventDB, get_db
from app.schemas import (
    AnomalyAcknowledge,
    AnomalyResult,
    BulkLogRequest,
    BulkLogResponse,
    MetricsInput,
    SuccessResponse,
)
from app.services.anomaly_service import get_anomaly_service

# ── Router ────────────────────────────────────────────────────────────────────
# prefix : all routes start with /api/v1/logs
# tags   : groups endpoints under "Logs & Anomaly Detection" in Swagger UI
router = APIRouter(
    prefix="/api/v1/logs",
    tags=["Logs & Anomaly Detection"],
)


# ==============================================================================
# ENDPOINT 1: POST /api/v1/logs/detect
# ==============================================================================

@router.post(
    "/detect",
    response_model=AnomalyResult,
    status_code=status.HTTP_200_OK,
    summary="Detect Anomaly",
    description="""
Run a single metrics snapshot through the IsolationForest anomaly detection model.

**Pipeline stages:**
1. **ML Scoring**     — IsolationForest computes anomaly score
2. **Severity**       — maps score to critical|high|medium|low|normal
3. **Root Cause**     — rule-based hint from metric thresholds
4. **DB Persist**     — saves to anomaly_events only if anomaly detected
    """,
)
async def detect_anomaly(
    metrics: MetricsInput,
    db: AsyncSession = Depends(get_db),
):
    """
    Runs the anomaly detection pipeline on a single metrics snapshot.
    Persists to DB only when is_anomaly=True.
    """
    print(f"\n  [LogsRoute] Detecting anomaly for service='{metrics.service}'")

    # ── STEP A: ML Anomaly Detection ──────────────────────────────
    # Input : MetricsInput (5 core system metrics)
    # Output: AnomalyResult (score, severity, root_cause, recommended_action)
    anomaly_svc = get_anomaly_service()
    result: AnomalyResult = anomaly_svc.detect(metrics)

    # ── STEP B: Persist to DB (only if anomaly detected) ──────────
    if result.is_anomaly:
        event_id = f"ANM-{uuid.uuid4().hex[:8].upper()}"

        db_event = AnomalyEventDB(
            event_id           = event_id,
            service            = metrics.service,
            anomaly_score      = result.anomaly_score,
            severity           = result.severity,
            confidence         = result.confidence,
            metrics_snapshot   = result.metrics,
            log_message        = metrics.log_message,
            root_cause_hint    = result.root_cause_hint,
            recommended_action = result.recommended_action,
            status             = "open",
        )
        db.add(db_event)
        await db.flush()

        print(
            f"  [LogsRoute] Anomaly saved: {event_id} | "
            f"service={metrics.service} | "
            f"severity={result.severity} | "
            f"score={result.anomaly_score:.3f}"
        )
    else:
        print(
            f"  [LogsRoute] Normal — service={metrics.service} | "
            f"score={result.anomaly_score:.3f}"
        )

    return result


# ==============================================================================
# ENDPOINT 2: POST /api/v1/logs/analyze
# ==============================================================================

@router.post(
    "/analyze",
    response_model=BulkLogResponse,
    status_code=status.HTTP_200_OK,
    summary="Bulk Log Analysis",
    description="""
Submit up to 1000 log entries for batch anomaly analysis.

Each entry is scored individually by the IsolationForest model.
Only anomalous entries are persisted to DB and returned in the response.
Results are sorted by severity: critical → high → medium → low.
    """,
)
async def analyze_logs(
    body: BulkLogRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Batch anomaly detection across multiple log entries.

    - Scores every entry with the ML model
    - Saves anomalies to anomaly_events table
    - Returns summary + anomalous entries sorted by severity
    """
    print(f"\n  [LogsRoute] Bulk analysis — {len(body.logs)} log entries")

    anomaly_svc = get_anomaly_service()

    # ── STEP A: Batch ML Detection ────────────────────────────────
    # detect_bulk() builds one feature matrix for ALL entries and calls
    # score_samples() ONCE — much faster than N individual calls.
    # It also filters out normal entries and sorts by severity internally.
    anomalies: List[AnomalyResult] = anomaly_svc.detect_bulk(body.logs)

    # ── STEP B: Persist anomalies to DB ──────────────────────────
    # Build a lookup of service → log_message from original input
    # so we can attach the log_message to each saved event.
    log_message_map = {log.service: log.log_message for log in body.logs}
    affected_services = set()

    for result in anomalies:
        affected_services.add(result.service)
        event_id = f"ANM-{uuid.uuid4().hex[:8].upper()}"
        db.add(AnomalyEventDB(
            event_id           = event_id,
            service            = result.service,
            anomaly_score      = result.anomaly_score,
            severity           = result.severity,
            confidence         = result.confidence,
            metrics_snapshot   = result.metrics,
            log_message        = log_message_map.get(result.service),
            root_cause_hint    = result.root_cause_hint,
            recommended_action = result.recommended_action,
            status             = "open",
        ))

    await db.flush()

    total     = len(body.logs)
    n_anomaly = len(anomalies)
    rate      = round(n_anomaly / total, 4) if total > 0 else 0.0

    affected_str = (
        f"Affected: {', '.join(sorted(affected_services))}"
        if affected_services else "No anomalies detected"
    )

    print(
        f"  [LogsRoute] Bulk done: {n_anomaly}/{total} anomalies | "
        f"{affected_str}"
    )

    return BulkLogResponse(
        total_logs         = total,
        anomalies_detected = n_anomaly,
        anomaly_rate       = rate,
        anomalies          = anomalies,
        summary            = (
            f"Found {n_anomaly} anomalies in {total} entries "
            f"({rate:.1%}). {affected_str}"
        ),
    )


# ==============================================================================
# ENDPOINT 3: GET /api/v1/logs/anomalies
# ==============================================================================

@router.get(
    "/anomalies",
    summary="List Anomaly Events",
    description="List all persisted anomaly events with optional filters. Returns paginated results.",
)
async def list_anomalies(
    service:  Optional[str] = Query(None, description="Filter by service name"),
    severity: Optional[str] = Query(None, description="Filter by severity: critical|high|medium|low"),
    status:   Optional[str] = Query(None, description="Filter by status: open|acknowledged|resolved|false_positive"),
    limit:    int           = Query(50, ge=1, le=200, description="Max results per page"),
    offset:   int           = Query(0,  ge=0,         description="Records to skip"),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a paginated list of anomaly events, newest first.
    Supports filtering by service, severity, and status.
    """
    query = (
        select(AnomalyEventDB)
        .order_by(AnomalyEventDB.created_at.desc())
    )

    if service:
        query = query.where(AnomalyEventDB.service == service)
    if severity:
        query = query.where(AnomalyEventDB.severity == severity)
    if status:
        query = query.where(AnomalyEventDB.status == status)

    # Total count for pagination metadata
    count_query = select(func.count(AnomalyEventDB.id))
    if service:
        count_query = count_query.where(AnomalyEventDB.service == service)
    if severity:
        count_query = count_query.where(AnomalyEventDB.severity == severity)
    if status:
        count_query = count_query.where(AnomalyEventDB.status == status)

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
# ENDPOINT 4: GET /api/v1/logs/anomalies/stats
# ==============================================================================

@router.get(
    "/anomalies/stats",
    summary="Anomaly Stats",
    description="Aggregated anomaly statistics. Used by the DevOps dashboard.",
)
async def get_anomaly_stats(
    db: AsyncSession = Depends(get_db),
):
    """
    Returns aggregate counts by severity, status, and top affected services.
    Also returns last-24h anomaly count for the dashboard health panel.
    """
    # Total anomalies ever
    total = await db.scalar(select(func.count(AnomalyEventDB.id))) or 0

    # Count by severity
    severity_rows = await db.execute(
        select(AnomalyEventDB.severity, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.severity)
    )
    by_severity = {row[0]: row[1] for row in severity_rows}

    # Count by status
    status_rows = await db.execute(
        select(AnomalyEventDB.status, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.status)
    )
    by_status = {row[0]: row[1] for row in status_rows}

    # Top 5 most affected services
    service_rows = await db.execute(
        select(AnomalyEventDB.service, func.count(AnomalyEventDB.id))
        .group_by(AnomalyEventDB.service)
        .order_by(func.count(AnomalyEventDB.id).desc())
        .limit(5)
    )
    top_services = {row[0]: row[1] for row in service_rows}

    # Last 24h count
    last_24h = await db.scalar(
        select(func.count(AnomalyEventDB.id)).where(
            AnomalyEventDB.created_at >= sa_text("NOW() - INTERVAL '24 hours'")
        )
    ) or 0

    return {
        "total_anomalies": total,
        "last_24h":        last_24h,
        "by_severity":     by_severity,
        "by_status":       by_status,
        "top_services":    top_services,
    }


# ==============================================================================
# ENDPOINT 5: GET /api/v1/logs/anomalies/{event_id}
# ==============================================================================

@router.get(
    "/anomalies/{event_id}",
    summary="Get Anomaly",
    description="Get full details of a single anomaly event by event_id.",
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
# ENDPOINT 6: POST /api/v1/logs/anomalies/{event_id}/acknowledge
# ==============================================================================

@router.post(
    "/anomalies/{event_id}/acknowledge",
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
    Used when an engineer picks it up for investigation.

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

    print(
        f"  [LogsRoute] {event_id} acknowledged by {body.acknowledged_by}"
        + (f" — notes: {body.notes}" if body.notes else "")
    )

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
# ENDPOINT 7: PUT /api/v1/logs/anomalies/{event_id}/resolve
# ==============================================================================

@router.put(
    "/anomalies/{event_id}/resolve",
    response_model=SuccessResponse,
    summary="Resolve Anomaly",
    description="Close an anomaly as resolved or false_positive.",
)
async def resolve_anomaly(
    event_id:    str,
    resolved_by: str = Query(..., description="Name or email of the engineer"),
    resolution:  str = Query("resolved", description="resolved | false_positive"),
    db: AsyncSession = Depends(get_db),
):
    """
    Closes an anomaly event.

    resolution options:
      resolved       → confirmed anomaly, fix was applied
      false_positive → ML model was wrong, feeds back into model improvement
    """
    valid_resolutions = {"resolved", "false_positive"}
    if resolution not in valid_resolutions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid resolution '{resolution}'. Valid: {valid_resolutions}"
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

    print(
        f"  [LogsRoute] {event_id} marked as '{resolution}' by {resolved_by}"
    )

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
# HELPERS
# ==============================================================================

async def _get_anomaly_or_404(
    event_id: str,
    db: AsyncSession,
) -> AnomalyEventDB:
    """
    Fetches an AnomalyEventDB by event_id or raises 404.
    Reused across GET, POST /acknowledge, PUT /resolve.
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
    Used by list and get endpoints.
    Mirrors the pattern of _ticket_to_list_item() in tickets.py.
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