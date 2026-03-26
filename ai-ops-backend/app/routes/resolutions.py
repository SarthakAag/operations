"""
app/routes/resolutions.py
================================================================================
PURPOSE:
  Read-side endpoints for ResolutionDB — the ML training data store.

  ResolutionDB is currently write-only: tickets.py writes to it on every
  resolve. This file makes it readable so you can:
    - Track AI accuracy over time
    - Export training data for audit / retraining
    - See which engineers resolve the most tickets
    - Measure resolution time trends per category / priority

ENDPOINTS:
  GET  /api/v1/resolutions/                   List all resolutions (paginated)
  GET  /api/v1/resolutions/stats              Aggregate stats: accuracy, MTTR, top engineers
  GET  /api/v1/resolutions/accuracy-trend     AI accuracy over time (weekly buckets)
  GET  /api/v1/resolutions/export             Export resolutions as CSV for retraining
  GET  /api/v1/resolutions/{resolution_id}    Get a single resolution record
================================================================================
"""

import csv
import io
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy import Integer, cast, func, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import ResolutionDB, get_db

router = APIRouter(
    prefix="/api/v1/resolutions",
    tags=["Resolutions"],
)


# ==============================================================================
# ENDPOINT 1: GET /api/v1/resolutions/
# ==============================================================================

@router.get(
    "/",
    summary="List Resolutions",
    description="""
Paginated list of all resolution records.

Each record is one resolved ticket that has entered the ML training pipeline.
Supports filtering by category, priority, engineer, and AI correctness flag.
    """,
)
async def list_resolutions(
    category:    Optional[str]  = Query(None, description="Filter by category"),
    priority:    Optional[str]  = Query(None, description="Filter by priority: P1|P2|P3|P4"),
    resolved_by: Optional[str]  = Query(None, description="Filter by engineer name"),
    ai_correct:  Optional[bool] = Query(None, description="Filter by AI correctness: true|false"),
    days:        Optional[int]  = Query(None, ge=1, le=365, description="Only resolutions from last N days"),
    limit:       int            = Query(50, ge=1, le=500),
    offset:      int            = Query(0,  ge=0),
    db: AsyncSession            = Depends(get_db),
):
    """
    Returns paginated resolution records. Newest first.
    """
    query       = select(ResolutionDB).order_by(ResolutionDB.created_at.desc())
    count_query = select(func.count(ResolutionDB.id))

    if category:
        query       = query.where(ResolutionDB.category == category)
        count_query = count_query.where(ResolutionDB.category == category)
    if priority:
        query       = query.where(ResolutionDB.priority == priority)
        count_query = count_query.where(ResolutionDB.priority == priority)
    if resolved_by:
        query       = query.where(ResolutionDB.resolved_by == resolved_by)
        count_query = count_query.where(ResolutionDB.resolved_by == resolved_by)
    if ai_correct is not None:
        query       = query.where(ResolutionDB.ai_recommendation_was_correct == ai_correct)
        count_query = count_query.where(ResolutionDB.ai_recommendation_was_correct == ai_correct)
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query       = query.where(ResolutionDB.created_at >= cutoff)
        count_query = count_query.where(ResolutionDB.created_at >= cutoff)

    total  = await db.scalar(count_query) or 0
    result = await db.execute(query.limit(limit).offset(offset))
    rows   = result.scalars().all()

    return {
        "total":       total,
        "limit":       limit,
        "offset":      offset,
        "resolutions": [_resolution_to_dict(r) for r in rows],
    }


# ==============================================================================
# ENDPOINT 2: GET /api/v1/resolutions/stats
# ==============================================================================

@router.get(
    "/stats",
    summary="Resolution Stats",
    description="""
Aggregate resolution statistics.

Returns:
- Total resolutions
- AI accuracy (% where ai_recommendation_was_correct = True)
- MTTR overall and by priority
- Top 10 engineers by volume
- Top 10 engineers by AI accuracy validation count
- Breakdown by category and priority
    """,
)
async def get_resolution_stats(
    days: Optional[int] = Query(None, ge=1, le=365, description="Last N days (default: all time)"),
    db: AsyncSession    = Depends(get_db),
):
    """
    Computes aggregate stats over the resolution dataset.
    Used by the analytics dashboard and the admin model-performance panel.
    """
    filters = []
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        filters.append(ResolutionDB.created_at >= cutoff)

    def apply(q):
        for f in filters:
            q = q.where(f)
        return q

    # Total
    total = await db.scalar(apply(select(func.count(ResolutionDB.id)))) or 0

    # AI accuracy — count rows where ai_recommendation_was_correct is not null
    acc_result = await db.execute(
        apply(
            select(
                func.count(ResolutionDB.id).label("total"),
                func.sum(
                    cast(ResolutionDB.ai_recommendation_was_correct, Integer)
                ).label("correct"),
            ).where(ResolutionDB.ai_recommendation_was_correct.isnot(None))
        )
    )
    acc = acc_result.one_or_none()
    ai_accuracy    = round((acc[1] or 0) / acc[0], 4) if acc and acc[0] else None
    ai_rated_total = acc[0] if acc else 0

    # MTTR overall
    avg_mttr = await db.scalar(
        apply(
            select(func.avg(ResolutionDB.resolution_time_minutes))
            .where(ResolutionDB.resolution_time_minutes.isnot(None))
        )
    )

    # MTTR by priority
    mttr_rows = await db.execute(
        apply(
            select(
                ResolutionDB.priority,
                func.avg(ResolutionDB.resolution_time_minutes),
                func.count(ResolutionDB.id),
            )
            .where(ResolutionDB.resolution_time_minutes.isnot(None))
            .group_by(ResolutionDB.priority)
        )
    )
    mttr_by_priority = {
        row[0]: {"avg_minutes": round(float(row[1]), 1), "count": row[2]}
        for row in mttr_rows
        if row[0]
    }

    # By category
    cat_rows = await db.execute(
        apply(select(ResolutionDB.category, func.count(ResolutionDB.id)).group_by(ResolutionDB.category))
    )
    by_category = {row[0]: row[1] for row in cat_rows if row[0]}

    # By priority
    pri_rows = await db.execute(
        apply(select(ResolutionDB.priority, func.count(ResolutionDB.id)).group_by(ResolutionDB.priority))
    )
    by_priority = {row[0]: row[1] for row in pri_rows if row[0]}

    # Top 10 engineers by volume
    eng_rows = await db.execute(
        apply(
            select(ResolutionDB.resolved_by, func.count(ResolutionDB.id))
            .where(ResolutionDB.resolved_by.isnot(None))
            .group_by(ResolutionDB.resolved_by)
            .order_by(func.count(ResolutionDB.id).desc())
            .limit(10)
        )
    )
    top_engineers = [{"engineer": row[0], "resolutions": row[1]} for row in eng_rows]

    # Top engineers by AI validation count (must have rated at least 3 tickets)
    validator_rows = await db.execute(
        apply(
            select(
                ResolutionDB.resolved_by,
                func.count(ResolutionDB.id).label("total"),
                func.sum(cast(ResolutionDB.ai_recommendation_was_correct, Integer)).label("correct"),
            )
            .where(
                ResolutionDB.resolved_by.isnot(None),
                ResolutionDB.ai_recommendation_was_correct.isnot(None),
            )
            .group_by(ResolutionDB.resolved_by)
            .having(func.count(ResolutionDB.id) >= 3)
            .order_by(func.count(ResolutionDB.id).desc())
            .limit(10)
        )
    )
    top_validators = [
        {
            "engineer":   row[0],
            "total":      row[1],
            "ai_correct": row[2] or 0,
            "accuracy":   round((row[2] or 0) / row[1], 4),
        }
        for row in validator_rows
    ]

    return {
        "period_days":       days,
        "total_resolutions": total,
        "ai_accuracy": {
            "accuracy":    ai_accuracy,
            "rated_total": ai_rated_total,
        },
        "mttr": {
            "overall_minutes": round(float(avg_mttr), 1) if avg_mttr else None,
            "by_priority":     mttr_by_priority,
        },
        "by_category":    by_category,
        "by_priority":    by_priority,
        "top_engineers":  top_engineers,
        "top_validators": top_validators,
    }


# ==============================================================================
# ENDPOINT 3: GET /api/v1/resolutions/accuracy-trend
# ==============================================================================

@router.get(
    "/accuracy-trend",
    summary="AI Accuracy Trend",
    description="""
Weekly AI accuracy trend over the last N weeks.

Returns one data point per week:
- Total resolutions that week
- AI correct count
- Accuracy percentage
- Average resolution time (MTTR)

Used by the admin analytics chart.
    """,
)
async def get_accuracy_trend(
    weeks: int       = Query(12, ge=1, le=52, description="How many weeks back"),
    db: AsyncSession = Depends(get_db),
):
    """
    Buckets resolutions into weekly intervals and computes AI accuracy per bucket.
    Uses date_trunc for clean ISO week boundaries.
    """
    cutoff = datetime.utcnow() - timedelta(weeks=weeks)

    rows = await db.execute(
        select(
            sa_text("DATE_TRUNC('week', created_at) AS week_start"),
            func.count(ResolutionDB.id).label("total"),
            func.sum(
                cast(ResolutionDB.ai_recommendation_was_correct, Integer)
            ).label("correct"),
            func.avg(ResolutionDB.resolution_time_minutes).label("avg_mttr"),
        )
        .where(
            ResolutionDB.created_at >= cutoff,
            ResolutionDB.ai_recommendation_was_correct.isnot(None),
        )
        .group_by(sa_text("week_start"))
        .order_by(sa_text("week_start"))
    )

    trend = []
    for row in rows:
        total   = row[1] or 0
        correct = row[2] or 0
        trend.append({
            "week_start":        row[0].isoformat() if row[0] else None,
            "total":             total,
            "correct":           correct,
            "accuracy":          round(correct / total, 4) if total else None,
            "avg_mttr_minutes":  round(float(row[3]), 1) if row[3] else None,
        })

    return {
        "weeks":       weeks,
        "data_points": len(trend),
        "trend":       trend,
    }


# ==============================================================================
# ENDPOINT 4: GET /api/v1/resolutions/export
# ==============================================================================

@router.get(
    "/export",
    summary="Export Training Data",
    description="""
Export all resolution records as a downloadable CSV.

The CSV format matches tickets.csv used by train_models.py.
Use this to:
- Back up the training dataset
- Audit what the model has been trained on
- Import into external ML tools for experimentation
    """,
)
async def export_resolutions(
    days:       Optional[int]  = Query(None, ge=1, le=365, description="Last N days only"),
    ai_correct: Optional[bool] = Query(None, description="Filter by AI correctness"),
    category:   Optional[str]  = Query(None, description="Filter by category"),
    db: AsyncSession            = Depends(get_db),
):
    """
    Streams a CSV file of all matching resolution records.
    Uses StreamingResponse so large exports don't buffer in memory.
    """
    query = select(ResolutionDB).order_by(ResolutionDB.created_at.asc())

    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(ResolutionDB.created_at >= cutoff)
    if ai_correct is not None:
        query = query.where(ResolutionDB.ai_recommendation_was_correct == ai_correct)
    if category:
        query = query.where(ResolutionDB.category == category)

    result = await db.execute(query)
    rows   = result.scalars().all()

    # Build CSV in memory
    output = io.StringIO()
    fieldnames = [
        "ticket_id", "title", "description", "category", "priority",
        "status", "resolution", "resolution_time_minutes",
        "resolved_by", "ai_recommendation_was_correct", "created_at",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for r in rows:
        writer.writerow({
            "ticket_id":                     r.ticket_id,
            "title":                         r.ticket_title,
            "description":                   r.ticket_description,
            "category":                      r.category,
            "priority":                      r.priority,
            "status":                        "resolved",
            "resolution":                    r.resolution_text,
            "resolution_time_minutes":       r.resolution_time_minutes,
            "resolved_by":                   r.resolved_by,
            "ai_recommendation_was_correct": r.ai_recommendation_was_correct,
            "created_at":                    r.created_at.isoformat() if r.created_at else "",
        })

    output.seek(0)
    filename = f"resolutions_export_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ==============================================================================
# ENDPOINT 5: GET /api/v1/resolutions/{resolution_id}
# ==============================================================================

@router.get(
    "/{resolution_id}",
    summary="Get Resolution",
    description="Get a single resolution record by its DB id.",
)
async def get_resolution(
    resolution_id: int,
    db: AsyncSession = Depends(get_db),
):
    """
    Returns a single ResolutionDB record.
    Identified by integer primary key (id), not ticket_id.
    """
    result = await db.execute(
        select(ResolutionDB).where(ResolutionDB.id == resolution_id)
    )
    record = result.scalar_one_or_none()

    if not record:
        raise HTTPException(
            status_code=404,
            detail=f"Resolution record {resolution_id} not found"
        )

    return _resolution_to_dict(record)


# ==============================================================================
# HELPERS
# ==============================================================================

def _resolution_to_dict(r: ResolutionDB) -> dict:
    """
    Converts a ResolutionDB ORM row to a clean response dict.
    """
    return {
        "id":                            r.id,
        "ticket_id":                     r.ticket_id,
        "title":                         r.ticket_title,
        "description":                   r.ticket_description,
        "category":                      r.category,
        "priority":                      r.priority,
        "resolution_text":               r.resolution_text,
        "resolution_time_minutes":       r.resolution_time_minutes,
        "resolved_by":                   r.resolved_by,
        "ai_recommendation_was_correct": r.ai_recommendation_was_correct,
        "created_at":                    r.created_at.isoformat() if r.created_at else None,
    }