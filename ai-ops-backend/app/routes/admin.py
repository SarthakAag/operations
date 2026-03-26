from fastapi import APIRouter
from typing import Dict, List
from datetime import datetime

router = APIRouter(prefix="/api/v1/admin", tags=["admin"])

# ✅ SIMPLE IN-MEMORY STORE (NO DB DEPENDENCY)
TICKET_STORE: Dict[str, dict] = {}


# ─────────────────────────────────────────────
# 1. GET ALL TICKETS
# ─────────────────────────────────────────────
@router.get("/tickets")
def get_all_tickets():
    return {
        "count": len(TICKET_STORE),
        "tickets": list(TICKET_STORE.values())
    }


# ─────────────────────────────────────────────
# 2. GET SINGLE TICKET
# ─────────────────────────────────────────────
@router.get("/tickets/{ticket_id}")
def get_ticket(ticket_id: str):
    ticket = TICKET_STORE.get(ticket_id)

    if not ticket:
        return {"error": "Ticket not found"}

    return ticket


# ─────────────────────────────────────────────
# 3. OVERRIDE DECISION
# ─────────────────────────────────────────────
@router.post("/override/{ticket_id}")
def override_ticket(ticket_id: str, decision: str):
    ticket = TICKET_STORE.get(ticket_id)

    if not ticket:
        return {"error": "Ticket not found"}

    ticket["governance"]["routing_decision"] = decision
    ticket["status"] = "resolved" if decision == "auto_resolve" else "open"
    ticket["updated_at"] = datetime.utcnow().isoformat()

    return {
        "message": "Decision overridden",
        "ticket_id": ticket_id,
        "new_decision": decision
    }


# ─────────────────────────────────────────────
# 4. METRICS DASHBOARD
# ─────────────────────────────────────────────
@router.get("/metrics")
def get_metrics():
    total = len(TICKET_STORE)

    if total == 0:
        return {
            "total_tickets": 0,
            "auto_resolved": 0,
            "human_review": 0,
            "auto_resolve_rate": 0,
            "avg_confidence": 0
        }

    auto = sum(
        1 for t in TICKET_STORE.values()
        if t.get("governance", {}).get("routing_decision") == "auto_resolve"
    )

    human = sum(
        1 for t in TICKET_STORE.values()
        if t.get("governance", {}).get("routing_decision") == "human_review"
    )

    avg_conf = sum(
        t.get("confidence_score", 0)
        for t in TICKET_STORE.values()
    ) / total

    return {
        "total_tickets": total,
        "auto_resolved": auto,
        "human_review": human,
        "auto_resolve_rate": round(auto / total, 2),
        "avg_confidence": round(avg_conf, 2)
    }