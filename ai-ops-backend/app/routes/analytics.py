"""
app/routes/analytics.py
================================================================================
PURPOSE:
  Reporting layer for the DevOps analytics dashboard.
  All the aggregations that the dashboard needs but that are too heavy
  for the /dashboard/metrics live-poll endpoint.

  These endpoints are designed for chart generation — each returns
  an ordered list of data points ready to pass to a charting library.

ENDPOINTS:
  GET  /api/v1/analytics/sla-trends          SLA breach rate over time
  GET  /api/v1/analytics/volume-over-time    Ticket volume by day/week
  GET  /api/v1/analytics/category-breakdown  Category distribution with AI accuracy per category
  GET  /api/v1/analytics/priority-breakdown  Priority distribution with MTTR per priority
  GET  /api/v1/analytics/ai-drift            AI confidence + accuracy trend — detect model drift
  GET  /api/v1/analytics/mttr-by-team        Mean time to resolution per assigned team
  GET  /api/v1/analytics/anomaly-trends      Anomaly volume and severity distribution over time
================================================================================
"""

from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import Integer, cast, func, select, text as sa_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import AnomalyEventDB, ResolutionDB, TicketDB, get_db

router = APIRouter(
    prefix="/api/v1/analytics",
    tags=["Analytics"],
)


# ==============================================================================
# ENDPOINT 1: GET /api/v1/analytics/sla-trends
# ==============================================================================

@router.get(
    "/sla-trends",
    summary="SLA Breach Trends",
    description="""
SLA compliance rate over time, bucketed by day or week.

Each data point has:
- period_start: bucket start timestamp
- total_tickets: tickets created in that bucket
- breached: tickets that missed SLA
- compliance_rate: 1 - (breached / total)

Use this to answer: "Is SLA compliance getting better or worse over time?"
    """,
)
async def get_sla_trends(
    days:    int = Query(30, ge=7,  le=365, description="Lookback window in days"),
    bucket:  str = Query("day",             description="Bucket size: day | week"),
    db: AsyncSession = Depends(get_db),
):
    """
    Buckets tickets by creation date and computes SLA compliance rate per bucket.
    """
    if bucket not in ("day", "week"):
        bucket = "day"

    cutoff = datetime.utcnow() - timedelta(days=days)

    rows = await db.execute(
        select(
            sa_text(f"DATE_TRUNC('{bucket}', created_at) AS period_start"),
            func.count(TicketDB.id).label("total"),
            func.sum(
                cast(
                    (TicketDB.sla_deadline < sa_text("NOW()"))
                    & TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
                    Integer,
                )
            ).label("breached"),
        )
        .where(TicketDB.created_at >= cutoff)
        .group_by(sa_text("period_start"))
        .order_by(sa_text("period_start"))
    )

    data = []
    for row in rows:
        total   = row[1] or 0
        breached = row[2] or 0
        data.append({
            "period_start":    row[0].isoformat() if row[0] else None,
            "total_tickets":   total,
            "breached":        breached,
            "compliance_rate": round(1 - breached / total, 4) if total else 1.0,
        })

    return {
        "bucket":      bucket,
        "days":        days,
        "data_points": len(data),
        "data":        data,
    }


# ==============================================================================
# ENDPOINT 2: GET /api/v1/analytics/volume-over-time
# ==============================================================================

@router.get(
    "/volume-over-time",
    summary="Ticket Volume Over Time",
    description="""
Ticket submission volume bucketed by day or week.

Optionally split by routing decision (auto_resolve vs human_review)
so you can see how the auto-resolve rate is trending.
    """,
)
async def get_volume_over_time(
    days:       int  = Query(30, ge=7, le=365),
    bucket:     str  = Query("day", description="day | week"),
    split_routing: bool = Query(False, description="Split counts by routing decision"),
    db: AsyncSession    = Depends(get_db),
):
    """
    Returns ticket volume time series. Used by the volume trend chart.
    """
    if bucket not in ("day", "week"):
        bucket = "day"

    cutoff = datetime.utcnow() - timedelta(days=days)

    if split_routing:
        rows = await db.execute(
            select(
                sa_text(f"DATE_TRUNC('{bucket}', created_at) AS period_start"),
                TicketDB.routing_decision,
                func.count(TicketDB.id).label("count"),
            )
            .where(TicketDB.created_at >= cutoff)
            .group_by(sa_text("period_start"), TicketDB.routing_decision)
            .order_by(sa_text("period_start"))
        )

        # Pivot into {period_start, auto_resolve, human_review, total}
        buckets: dict = {}
        for row in rows:
            key = row[0].isoformat() if row[0] else "unknown"
            if key not in buckets:
                buckets[key] = {"period_start": key, "auto_resolve": 0, "human_review": 0}
            if row[1]:
                buckets[key][row[1]] = row[2]

        data = []
        for v in sorted(buckets.values(), key=lambda x: x["period_start"]):
            v["total"] = v.get("auto_resolve", 0) + v.get("human_review", 0)
            v["auto_resolve_rate"] = round(
                v["auto_resolve"] / v["total"], 4
            ) if v["total"] else 0.0
            data.append(v)
    else:
        rows = await db.execute(
            select(
                sa_text(f"DATE_TRUNC('{bucket}', created_at) AS period_start"),
                func.count(TicketDB.id).label("count"),
            )
            .where(TicketDB.created_at >= cutoff)
            .group_by(sa_text("period_start"))
            .order_by(sa_text("period_start"))
        )
        data = [
            {
                "period_start": row[0].isoformat() if row[0] else None,
                "total":        row[1],
            }
            for row in rows
        ]

    return {
        "bucket":        bucket,
        "days":          days,
        "split_routing": split_routing,
        "data_points":   len(data),
        "data":          data,
    }


# ==============================================================================
# ENDPOINT 3: GET /api/v1/analytics/category-breakdown
# ==============================================================================

@router.get(
    "/category-breakdown",
    summary="Category Breakdown",
    description="""
Category distribution enriched with per-category AI accuracy and MTTR.

For each category returns:
- ticket_count
- auto_resolve_rate
- avg_confidence_score
- ai_accuracy (from resolutions where correctness was rated)
- avg_resolution_time_minutes
    """,
)
async def get_category_breakdown(
    days: Optional[int] = Query(None, ge=1, le=365),
    db: AsyncSession    = Depends(get_db),
):
    """
    Joins ticket and resolution data to build a per-category performance table.
    """
    ticket_filter  = []
    resolve_filter = []
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        ticket_filter.append(TicketDB.created_at >= cutoff)
        resolve_filter.append(ResolutionDB.created_at >= cutoff)

    def apply_ticket(q):
        for f in ticket_filter:
            q = q.where(f)
        return q

    def apply_resolve(q):
        for f in resolve_filter:
            q = q.where(f)
        return q

    # Ticket stats per category
    ticket_rows = await db.execute(
        apply_ticket(
            select(
                TicketDB.predicted_category,
                func.count(TicketDB.id).label("total"),
                func.avg(TicketDB.confidence_score).label("avg_conf"),
                func.sum(
                    cast(TicketDB.routing_decision == "auto_resolve", Integer)
                ).label("auto_resolve_count"),
                func.avg(TicketDB.resolution_time_minutes).label("avg_mttr"),
            )
            .where(TicketDB.predicted_category.isnot(None))
            .group_by(TicketDB.predicted_category)
        )
    )

    # AI accuracy per category from resolutions
    accuracy_rows = await db.execute(
        apply_resolve(
            select(
                ResolutionDB.category,
                func.count(ResolutionDB.id).label("rated"),
                func.sum(
                    cast(ResolutionDB.ai_recommendation_was_correct, Integer)
                ).label("correct"),
            )
            .where(ResolutionDB.ai_recommendation_was_correct.isnot(None))
            .group_by(ResolutionDB.category)
        )
    )
    accuracy_map = {
        row[0]: round((row[2] or 0) / row[1], 4)
        for row in accuracy_rows
        if row[0] and row[1]
    }

    categories = []
    for row in ticket_rows:
        cat   = row[0]
        total = row[1] or 0
        auto  = row[3] or 0
        categories.append({
            "category":                    cat,
            "ticket_count":                total,
            "auto_resolve_rate":           round(auto / total, 4) if total else 0.0,
            "avg_confidence_score":        round(float(row[2]), 4) if row[2] else None,
            "ai_accuracy":                 accuracy_map.get(cat),
            "avg_resolution_time_minutes": round(float(row[4]), 1) if row[4] else None,
        })

    # Sort by ticket count descending
    categories.sort(key=lambda x: x["ticket_count"], reverse=True)

    return {
        "period_days": days,
        "categories":  categories,
    }


# ==============================================================================
# ENDPOINT 4: GET /api/v1/analytics/priority-breakdown
# ==============================================================================

@router.get(
    "/priority-breakdown",
    summary="Priority Breakdown",
    description="""
Priority distribution with SLA compliance and MTTR per priority.

For each priority returns:
- ticket_count
- sla_breach_count + sla_compliance_rate
- avg_confidence_score
- avg_resolution_time_minutes
- p1_p2_share: fraction of tickets that are high-priority
    """,
)
async def get_priority_breakdown(
    days: Optional[int] = Query(None, ge=1, le=365),
    db: AsyncSession    = Depends(get_db),
):
    now = datetime.utcnow()
    ticket_filter = []
    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        ticket_filter.append(TicketDB.created_at >= cutoff)

    def apply(q):
        for f in ticket_filter:
            q = q.where(f)
        return q

    rows = await db.execute(
        apply(
            select(
                TicketDB.predicted_priority,
                func.count(TicketDB.id).label("total"),
                func.avg(TicketDB.confidence_score).label("avg_conf"),
                func.avg(TicketDB.resolution_time_minutes).label("avg_mttr"),
            )
            .where(TicketDB.predicted_priority.isnot(None))
            .group_by(TicketDB.predicted_priority)
        )
    )

    # SLA breaches per priority
    breach_rows = await db.execute(
        apply(
            select(
                TicketDB.predicted_priority,
                func.count(TicketDB.id).label("breached"),
            )
            .where(
                TicketDB.sla_deadline < now,
                TicketDB.status.notin_(["resolved", "closed", "auto_resolved"]),
            )
            .group_by(TicketDB.predicted_priority)
        )
    )
    breach_map = {row[0]: row[1] for row in breach_rows if row[0]}

    priorities = []
    total_tickets = 0
    for row in rows:
        total_tickets += row[1] or 0

    for row in rows:
        pri   = row[0]
        total = row[1] or 0
        breach = breach_map.get(pri, 0)
        priorities.append({
            "priority":                    pri,
            "ticket_count":                total,
            "share":                       round(total / total_tickets, 4) if total_tickets else 0.0,
            "sla_breach_count":            breach,
            "sla_compliance_rate":         round(1 - breach / total, 4) if total else 1.0,
            "avg_confidence_score":        round(float(row[2]), 4) if row[2] else None,
            "avg_resolution_time_minutes": round(float(row[3]), 1) if row[3] else None,
        })

    # Sort P1 → P4
    priority_order = {"P1": 0, "P2": 1, "P3": 2, "P4": 3}
    priorities.sort(key=lambda x: priority_order.get(x["priority"], 9))

    return {
        "period_days": days,
        "priorities":  priorities,
    }


# ==============================================================================
# ENDPOINT 5: GET /api/v1/analytics/ai-drift
# ==============================================================================

@router.get(
    "/ai-drift",
    summary="AI Drift Detection",
    description="""
Tracks AI confidence and accuracy over time to detect model drift.

Returns weekly data points showing:
- avg_confidence: average confidence score for tickets that week
- auto_resolve_rate: fraction routed to auto-resolve
- ai_accuracy: fraction where AI was confirmed correct (from resolutions)
- drift_alert: True if accuracy dropped >10pp vs the previous week

Use this to know WHEN to retrain: look for weeks where accuracy
or confidence trends downward consistently.
    """,
)
async def get_ai_drift(
    weeks: int       = Query(12, ge=2, le=52),
    db: AsyncSession = Depends(get_db),
):
    """
    Computes weekly AI performance metrics. Key signal for retraining decisions.
    """
    cutoff = datetime.utcnow() - timedelta(weeks=weeks)

    # Weekly confidence + auto-resolve rate from tickets
    ticket_rows = await db.execute(
        select(
            sa_text("DATE_TRUNC('week', created_at) AS week_start"),
            func.count(TicketDB.id).label("total"),
            func.avg(TicketDB.confidence_score).label("avg_conf"),
            func.sum(
                cast(TicketDB.routing_decision == "auto_resolve", Integer)
            ).label("auto_resolved"),
        )
        .where(TicketDB.created_at >= cutoff)
        .group_by(sa_text("week_start"))
        .order_by(sa_text("week_start"))
    )

    ticket_data = {}
    for row in ticket_rows:
        key = row[0].isoformat() if row[0] else "unknown"
        total = row[1] or 0
        auto  = row[3] or 0
        ticket_data[key] = {
            "week_start":       key,
            "ticket_count":     total,
            "avg_confidence":   round(float(row[2]), 4) if row[2] else None,
            "auto_resolve_rate": round(auto / total, 4) if total else 0.0,
        }

    # Weekly accuracy from resolutions
    resolution_rows = await db.execute(
        select(
            sa_text("DATE_TRUNC('week', created_at) AS week_start"),
            func.count(ResolutionDB.id).label("rated"),
            func.sum(
                cast(ResolutionDB.ai_recommendation_was_correct, Integer)
            ).label("correct"),
        )
        .where(
            ResolutionDB.created_at >= cutoff,
            ResolutionDB.ai_recommendation_was_correct.isnot(None),
        )
        .group_by(sa_text("week_start"))
        .order_by(sa_text("week_start"))
    )

    accuracy_data = {}
    for row in resolution_rows:
        key   = row[0].isoformat() if row[0] else "unknown"
        rated = row[1] or 0
        correct = row[2] or 0
        accuracy_data[key] = round(correct / rated, 4) if rated else None

    # Merge and compute drift alerts
    all_weeks = sorted(set(list(ticket_data.keys()) + list(accuracy_data.keys())))
    data      = []
    prev_accuracy = None

    for week in all_weeks:
        td  = ticket_data.get(week, {})
        acc = accuracy_data.get(week)

        drift_alert = False
        if prev_accuracy is not None and acc is not None:
            drift_alert = (prev_accuracy - acc) > 0.10   # >10pp drop

        data.append({
            "week_start":       week,
            "ticket_count":     td.get("ticket_count", 0),
            "avg_confidence":   td.get("avg_confidence"),
            "auto_resolve_rate": td.get("auto_resolve_rate"),
            "ai_accuracy":      acc,
            "drift_alert":      drift_alert,
        })

        if acc is not None:
            prev_accuracy = acc

    return {
        "weeks":       weeks,
        "data_points": len(data),
        "data":        data,
    }


# ==============================================================================
# ENDPOINT 6: GET /api/v1/analytics/mttr-by-team
# ==============================================================================

@router.get(
    "/mttr-by-team",
    summary="MTTR by Team",
    description="""
Mean time to resolution grouped by assigned_to team.

Returns per team:
- total resolved tickets
- avg_resolution_time_minutes (MTTR)
- p1_count: how many P1 tickets they handled
- avg_confidence: average AI confidence score on their tickets
    """,
)
async def get_mttr_by_team(
    days: Optional[int] = Query(None, ge=1, le=365),
    db: AsyncSession    = Depends(get_db),
):
    """
    Aggregates resolution performance per team.
    Useful for identifying slow teams and for capacity planning.
    """
    query = (
        select(
            TicketDB.assigned_to,
            func.count(TicketDB.id).label("total"),
            func.avg(TicketDB.resolution_time_minutes).label("avg_mttr"),
            func.avg(TicketDB.confidence_score).label("avg_conf"),
            func.sum(
                cast(TicketDB.predicted_priority == "P1", Integer)
            ).label("p1_count"),
        )
        .where(
            TicketDB.assigned_to.isnot(None),
            TicketDB.resolution_time_minutes.isnot(None),
        )
        .group_by(TicketDB.assigned_to)
        .order_by(func.count(TicketDB.id).desc())
    )

    if days:
        cutoff = datetime.utcnow() - timedelta(days=days)
        query = query.where(TicketDB.created_at >= cutoff)

    rows = await db.execute(query)

    teams = [
        {
            "team":                        row[0],
            "resolved_count":              row[1] or 0,
            "avg_mttr_minutes":            round(float(row[2]), 1) if row[2] else None,
            "avg_confidence_score":        round(float(row[3]), 4) if row[3] else None,
            "p1_count":                    row[4] or 0,
        }
        for row in rows
    ]

    return {
        "period_days": days,
        "teams":       teams,
    }


# ==============================================================================
# ENDPOINT 7: GET /api/v1/analytics/anomaly-trends
# ==============================================================================

@router.get(
    "/anomaly-trends",
    summary="Anomaly Trends",
    description="""
Anomaly volume and severity breakdown over time.

Returns weekly data points:
- total_anomalies detected
- by_severity counts (critical, high, medium, low)
- false_positive_rate: fraction marked as false_positive

Use this to see if the anomaly detector is becoming more/less sensitive over time.
    """,
)
async def get_anomaly_trends(
    weeks: int       = Query(12, ge=1, le=52),
    db: AsyncSession = Depends(get_db),
):
    """
    Computes weekly anomaly volume and severity distribution.
    """
    cutoff = datetime.utcnow() - timedelta(weeks=weeks)

    rows = await db.execute(
        select(
            sa_text("DATE_TRUNC('week', created_at) AS week_start"),
            AnomalyEventDB.severity,
            AnomalyEventDB.status,
            func.count(AnomalyEventDB.id).label("count"),
        )
        .where(AnomalyEventDB.created_at >= cutoff)
        .group_by(sa_text("week_start"), AnomalyEventDB.severity, AnomalyEventDB.status)
        .order_by(sa_text("week_start"))
    )

    # Pivot into per-week objects
    weeks_map: dict = {}
    for row in rows:
        key = row[0].isoformat() if row[0] else "unknown"
        if key not in weeks_map:
            weeks_map[key] = {
                "week_start":     key,
                "total":          0,
                "critical":       0,
                "high":           0,
                "medium":         0,
                "low":            0,
                "false_positive": 0,
            }
        sev   = row[1] or "unknown"
        status = row[2] or "unknown"
        count  = row[3] or 0

        weeks_map[key]["total"] += count
        if sev in ("critical", "high", "medium", "low"):
            weeks_map[key][sev] = weeks_map[key].get(sev, 0) + count
        if status == "false_positive":
            weeks_map[key]["false_positive"] += count

    data = []
    for v in sorted(weeks_map.values(), key=lambda x: x["week_start"]):
        total = v["total"] or 0
        fp    = v["false_positive"] or 0
        v["false_positive_rate"] = round(fp / total, 4) if total else 0.0
        data.append(v)

    return {
        "weeks":       weeks,
        "data_points": len(data),
        "data":        data,
    }