from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.incident import Incident

from app.services.anomaly_detection import detect_anomaly
from app.services.ticket_classifier import classify_ticket
from app.services.recommendation import recommend_fix
from app.services.auto_healing import apply_fix
from app.services.rca_engine import find_root_cause   # ✅ NEW

router = APIRouter()

# DB Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 🚀 PROCESS INCIDENT (Main API)
@router.post("/process")
def process(data: dict, db: Session = Depends(get_db)):
    
    # 🔍 Step 1: Detect anomaly (ML)
    anomaly = detect_anomaly(data)

    # 🧠 Step 2: Classify ticket (ML NLP)
    category = classify_ticket(data.get("text", ""))

    # 🧩 Step 3: Root Cause Analysis
    root_cause = find_root_cause(data)

    # 💡 Step 4: Recommend fix (AI)
    recommendation = recommend_fix(data.get("error", ""))

    # 💾 Step 5: Save to DB
    incident = Incident(
        issue=data.get("error"),
        category=category,
        recommendation=recommendation,
        status="Pending"
    )

    db.add(incident)
    db.commit()
    db.refresh(incident)

    # 📤 Response
    return {
        "anomaly": anomaly,
        "category": category,
        "root_cause": root_cause,
        "recommendation": recommendation,
        "incident_id": incident.id
    }


# 🔧 APPLY FIX (Auto-Healing)
@router.post("/fix/{incident_id}")
def fix_incident(incident_id: int, db: Session = Depends(get_db)):
    
    incident = db.query(Incident).filter(Incident.id == incident_id).first()

    if not incident:
        return {"error": "Incident not found"}

    result = apply_fix(incident.recommendation)

    incident.status = "Resolved"
    db.commit()

    return {
        "message": "Fix applied",
        "result": result
    }


# 📊 GET ALL INCIDENTS
@router.get("/incidents")
def get_incidents(db: Session = Depends(get_db)):
    return db.query(Incident).all()