"""
api.py
================================================================================
AI-Driven Application Support & Operations — Main API

This is the SINGLE entry point for the entire backend.
It loads both trained ML models at startup and exposes REST endpoints
for ticket processing, anomaly detection, and the DevOps dashboard.

HOW TO RUN:
  python api.py

ENDPOINTS:
  POST   /api/v1/tickets/            Submit ticket → full AI pipeline
  GET    /api/v1/tickets/            List all tickets
  GET    /api/v1/tickets/{id}        Get one ticket by ID
  POST   /api/v1/tickets/{id}/resolve  Resolve ticket (triggers learning)

  POST   /api/v1/logs/detect         Single anomaly check
  POST   /api/v1/logs/analyze        Bulk log analysis

  GET    /api/v1/dashboard/metrics   DevOps dashboard KPIs
  GET    /api/v1/dashboard/health    Model health check

  GET    /docs                       Swagger UI
================================================================================
"""
from fastapi.middleware.cors import CORSMiddleware
import json
import os
import pickle
import re
import string
import time
import uuid
import warnings
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from app.routes.admin import router as admin_router


import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# Paths
TICKET_MODEL_PATH  = "models/ticket_model.pkl"
ANOMALY_MODEL_PATH = "models/anomaly_model.pkl"
TICKETS_CSV_PATH   = "data/tickets.csv"
LOGS_JSON_PATH     = "data/logs.json"

# Governance thresholds
AUTO_RESOLVE_CONFIDENCE = 0.80   # confidence >= this -> auto resolve
HIGH_RISK_THRESHOLD     = 0.70   # risk >= this -> always human review

# SLA (minutes)
SLA_MAP = {"P1": 60, "P2": 240, "P3": 1440, "P4": 4320}

# Category -> team routing
CATEGORY_TEAM = {
    "security":       "team-security",
    "database":       "team-db",
    "infrastructure": "team-infra",
    "network":        "team-network",
    "application":    "team-app",
}

# Risk weights per category and priority
CATEGORY_RISK = {
    "security": 1.0, "database": 0.75,
    "infrastructure": 0.65, "network": 0.55, "application": 0.40,
}
PRIORITY_RISK = {"P1": 1.0, "P2": 0.65, "P3": 0.35, "P4": 0.15}

# In-memory ticket store (replace with DB in production)
TICKET_STORE: Dict[str, dict] = {}
ANOMALY_STORE: List[dict] = []
RESOLUTION_COUNTER = 0


# ==============================================================================
# SECTION 2: PYDANTIC SCHEMAS (Request / Response models)
# ==============================================================================

class TicketCreate(BaseModel):
    title:       str  = Field(..., min_length=5,  example="Database connection timeout on payment-svc")
    description: str  = Field(..., min_length=10, example="Users seeing timeout errors. Pool exhausted.")
    reporter:    Optional[str]  = Field(None, example="engineer@company.com")
    is_vip:      Optional[bool] = Field(False)

    @field_validator("title", "description")
    @classmethod
    def strip_blank(cls, v):
        if not v.strip():
            raise ValueError("Field cannot be blank")
        return v.strip()


class MetricsInput(BaseModel):
    service:          str   = Field(..., example="payment-svc")
    response_time_ms: float = Field(..., ge=0,   example=120.5)
    error_rate:       float = Field(..., ge=0, le=1, example=0.02)
    cpu_usage_pct:    float = Field(..., ge=0, le=100, example=45.0)
    memory_usage_pct: float = Field(..., ge=0, le=100, example=60.0)
    request_count:    int   = Field(..., ge=0, example=1000)
    log_message:      Optional[str] = None


class TicketResolve(BaseModel):
    resolution_text:              str  = Field(..., min_length=10)
    resolved_by:                  str
    ai_recommendation_was_correct: bool


class BulkLogRequest(BaseModel):
    logs: List[MetricsInput] = Field(..., max_length=1000)


# ==============================================================================
# SECTION 3: TEXT PREPROCESSING
# (Must match exactly what was used in train_models.py)
# ==============================================================================

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", " ", text)
    text = re.sub(r"tkt-[a-z0-9]+", " ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ==============================================================================
# SECTION 4: MODEL LOADING (runs once at startup)
# ==============================================================================
# Both models are loaded into memory when the server starts.
# Why? Loading a pickle file takes ~0.5s. If we loaded on every request,
# every ticket submission would be 0.5s slower. Singleton = load once, reuse.
# ==============================================================================

ticket_model_bundle  = None   # contains pipeline + label_encoders + metrics
anomaly_model_bundle = None   # contains model + scaler + threshold + feature_cols
historical_df        = None   # DataFrame of past tickets for similarity search
tfidf_vectorizer     = None   # TF-IDF for similarity search
ticket_matrix        = None   # TF-IDF matrix of all historical tickets


def load_models():
    """
    Called once at startup. Loads both ML models and builds
    the similarity search index from historical tickets.
    """
    global ticket_model_bundle, anomaly_model_bundle
    global historical_df, tfidf_vectorizer, ticket_matrix

    print("\n  Loading ML models...")

    # ── Ticket model ──────────────────────────────────────────────
    if Path(TICKET_MODEL_PATH).exists():
        with open(TICKET_MODEL_PATH, "rb") as f:
            ticket_model_bundle = pickle.load(f)
        print(f"  Ticket model loaded  v{ticket_model_bundle['version']}"
              f"  Category F1={ticket_model_bundle['metrics']['category_f1']:.3f}")
    else:
        print(f"  WARNING: {TICKET_MODEL_PATH} not found. Run train_models.py first.")

    # ── Anomaly model ─────────────────────────────────────────────
    if Path(ANOMALY_MODEL_PATH).exists():
        with open(ANOMALY_MODEL_PATH, "rb") as f:
            anomaly_model_bundle = pickle.load(f)
        print(f"  Anomaly model loaded v{anomaly_model_bundle['version']}"
              f"  threshold={anomaly_model_bundle['threshold']:.4f}")
    else:
        print(f"  WARNING: {ANOMALY_MODEL_PATH} not found. Run train_models.py first.")

    # ── Similarity index ──────────────────────────────────────────
    # We build a TF-IDF index over all resolved historical tickets.
    # When a new ticket arrives, we compare it against this index
    # using cosine similarity to find the most similar past tickets.
    if Path(TICKETS_CSV_PATH).exists():
        from sklearn.feature_extraction.text import TfidfVectorizer as TF

        df = pd.read_csv(TICKETS_CSV_PATH)
        df = df[df["status"].isin(["resolved", "closed"])].copy()
        df = df.dropna(subset=["title", "description", "resolution"])
        df = df.reset_index(drop=True)

        df["indexed_text"] = (
            df["title"].apply(preprocess_text) + " " +
            df["title"].apply(preprocess_text) + " " +
            df["description"].apply(preprocess_text)
        )

        tfidf_vectorizer = TF(ngram_range=(1, 2), max_features=10000,
                              min_df=1, sublinear_tf=True)
        ticket_matrix    = tfidf_vectorizer.fit_transform(df["indexed_text"])
        historical_df    = df

        print(f"  Similarity index built: {len(df)} tickets  "
              f"vocab={len(tfidf_vectorizer.vocabulary_)}")
    else:
        print(f"  WARNING: {TICKETS_CSV_PATH} not found.")

    print("  All models ready.\n")


# ==============================================================================
# SECTION 5: AI PIPELINE FUNCTIONS
# Each function = one module of the architecture diagram
# ==============================================================================

# ── Module A: NLP Classification ──────────────────────────────────────────────
def classify_ticket(title: str, description: str) -> dict:
    """
    Uses the trained TF-IDF + RandomForest pipeline to classify the ticket.

    Flow:
      1. Preprocess text (clean + normalize)
      2. Repeat title 3x (title has stronger signal than description)
      3. Feed through pipeline
      4. Call predict_proba() to get probability for each class
      5. Best probability = prediction, probability value = confidence

    Returns dict with:
      predicted_category     : "database" | "application" | ...
      predicted_priority     : "P1" | "P2" | "P3" | "P4"
      category_confidence    : float 0-1
      priority_confidence    : float 0-1
      category_probabilities : { "application": 0.05, "database": 0.82, ... }
      priority_probabilities : { "P1": 0.70, "P2": 0.20, "P3": 0.10 }
    """
    if ticket_model_bundle is None:
        return {
            "predicted_category": "application", "predicted_priority": "P3",
            "category_confidence": 0.0, "priority_confidence": 0.0,
            "category_probabilities": {}, "priority_probabilities": {},
        }

    pipeline   = ticket_model_bundle["pipeline"]
    le_cat     = ticket_model_bundle["label_encoders"]["category"]
    le_pri     = ticket_model_bundle["label_encoders"]["priority"]

    clean_title = preprocess_text(title)
    clean_desc  = preprocess_text(description)
    combined    = f"{clean_title} {clean_title} {clean_title} {clean_desc}"

    # predict_proba() returns list[array] — one array per output
    proba    = pipeline.predict_proba([combined])
    cat_proba = proba[0][0]
    pri_proba = proba[1][0]

    cat_idx = int(cat_proba.argmax())
    pri_idx = int(pri_proba.argmax())

    return {
        "predicted_category":     le_cat.classes_[cat_idx],
        "predicted_priority":     le_pri.classes_[pri_idx],
        "category_confidence":    round(float(cat_proba[cat_idx]), 4),
        "priority_confidence":    round(float(pri_proba[pri_idx]), 4),
        "category_probabilities": {le_cat.classes_[i]: round(float(p), 4) for i, p in enumerate(cat_proba)},
        "priority_probabilities": {le_pri.classes_[i]: round(float(p), 4) for i, p in enumerate(pri_proba)},
    }


# ── Module B: Semantic Similarity Search ──────────────────────────────────────
def find_similar_tickets(title: str, description: str,
                         category: str, top_k: int = 5) -> list:
    """
    Finds the top-K most similar past tickets using TF-IDF cosine similarity.

    Algorithm:
      1. Build query vector from new ticket text
      2. Compute cosine similarity vs ALL historical ticket vectors
         (cosine similarity = dot product of normalized vectors)
         Result: similarity score per historical ticket (0=different, 1=identical)
      3. Boost same-category tickets by 20%
         (a DB ticket is more likely to have a useful DB resolution)
      4. Sort by score, return top K

    Cosine similarity works well here because:
      - Magnitude doesn't matter (long vs short descriptions)
      - Only direction (word pattern) matters
      - "connection timeout database pool" is similar to
        "database connection pool exhausted timeout" even with different word order
    """
    if historical_df is None or tfidf_vectorizer is None:
        return []

    query_text = preprocess_text(title + " " + title + " " + description)
    query_vec  = tfidf_vectorizer.transform([query_text])
    scores     = cosine_similarity(query_vec, ticket_matrix).flatten()

    # Category boost
    if "category" in historical_df.columns:
        same_cat_mask = (historical_df["category"] == category).values
        scores = scores + (same_cat_mask * 0.20 * scores)
        scores = np.clip(scores, 0, 1.0)

    top_indices = scores.argsort()[::-1][:top_k + 5]
    results     = []
    seen        = set()

    for idx in top_indices:
        score = float(scores[idx])
        if score < 0.10:
            break
        row        = historical_df.iloc[idx]
        resolution = str(row.get("resolution", ""))
        key        = resolution[:80]
        if key in seen:
            continue
        seen.add(key)
        results.append({
            "ticket_id":               str(row.get("ticket_id", f"HIST-{idx}")),
            "title":                   str(row.get("title", "")),
            "similarity_score":        round(score, 4),
            "category":                str(row.get("category", "")),
            "priority":                str(row.get("priority", "")),
            "resolution":              resolution,
            "resolution_time_minutes": int(row.get("resolution_time_minutes", 0)),
        })
        if len(results) >= top_k:
            break

    return results


# ── Module C: Confidence Scoring ───────────────────────────────────────────────
def compute_confidence(classification: dict, similar_tickets: list) -> float:
    """
    Computes overall confidence by combining 3 signals.

    ┌────────────────────────────┬────────┐
    │ Signal                     │ Weight │
    ├────────────────────────────┼────────┤
    │ Classification confidence  │  35%   │
    │ Similarity score           │  40%   │
    │ Historical resolution rate │  25%   │
    └────────────────────────────┴────────┘

    Similarity gets the highest weight (0.40) because finding a nearly
    identical past ticket with a known resolution is the strongest signal
    that the same fix will work again.
    """
    # Signal 1: classification (category weighted more than priority)
    cat_conf = classification["category_confidence"]
    pri_conf = classification["priority_confidence"]
    class_signal = cat_conf * 0.60 + pri_conf * 0.40

    # Signal 2: similarity
    if similar_tickets:
        best  = similar_tickets[0]["similarity_score"]
        top3  = [t["similarity_score"] for t in similar_tickets[:3]]
        sim_signal = 0.70 * best + 0.30 * (sum(top3) / len(top3))
    else:
        sim_signal = 0.0

    # Signal 3: historical resolution time (proxy for difficulty)
    if len(similar_tickets) >= 3:
        times       = [t["resolution_time_minutes"] for t in similar_tickets[:5] if t["resolution_time_minutes"] > 0]
        avg_time    = sum(times) / len(times) if times else 1440
        hist_signal = max(0.3, min(1.0, 1.0 - avg_time / 2000))
    else:
        hist_signal = 0.4

    score = 0.35 * class_signal + 0.40 * sim_signal + 0.25 * hist_signal

    # Penalty: very uncertain category lowers overall confidence
    if cat_conf < 0.40:
        score *= 0.80

    return round(max(0.05, min(1.0, score)), 4)


def build_recommended_fix(similar_tickets: list, confidence: float) -> Optional[str]:
    """Builds the AI fix recommendation text from most similar past tickets."""
    if not similar_tickets or confidence < 0.25:
        return None

    best = similar_tickets[0]

    if confidence >= 0.70:
        return (
            f"Based on {len(similar_tickets)} similar past incidents "
            f"(best match: {best['similarity_score']:.0%} similarity):\n\n"
            f"{best['resolution']}"
        )
    else:
        text = (
            f"Suggested fix (confidence {confidence:.0%}) — review before applying:\n\n"
            f"Primary: {best['resolution']}"
        )
        if len(similar_tickets) > 1:
            text += f"\n\nAlternative: {similar_tickets[1]['resolution']}"
        return text


# ── Module D: Risk Scoring ─────────────────────────────────────────────────────
def compute_risk(category: str, priority: str,
                 confidence: float, similar_tickets: list) -> float:
    """
    Computes risk score for this ticket's proposed resolution.

    Components:
      Priority risk  (40%): P1 = 1.0, P4 = 0.15
      Category risk  (35%): security = 1.0, application = 0.40
      Uncertainty    (25%): 1 - confidence (low confidence = high risk)

    Boost: No similar tickets found = +25% risk (no precedent).
    """
    pri_risk = PRIORITY_RISK.get(priority, 0.35)
    cat_risk = CATEGORY_RISK.get(category, 0.50)
    unc_risk = 1.0 - confidence

    risk = 0.40 * pri_risk + 0.35 * cat_risk + 0.25 * unc_risk

    if not similar_tickets:
        risk = min(1.0, risk * 1.25)

    return round(risk, 4)


# ── Module E: Governance Decision ──────────────────────────────────────────────
def governance_decision(classification: dict, confidence: float,
                        is_vip: bool, similar_tickets: list) -> dict:
    """
    Makes the final routing decision.

    MUST GO TO HUMAN if ANY of these:
      1. VIP reporter
      2. P1 priority (critical incident)
      3. Security category
      4. Risk score >= 0.70
      5. Confidence < 0.80

    Otherwise -> AUTO_RESOLVE

    Returns full governance dict with routing, risk, team, SLA deadline.
    """
    category = classification["predicted_category"]
    priority = classification["predicted_priority"]

    risk_score = compute_risk(category, priority, confidence, similar_tickets)

    reasons = []
    if is_vip:
        reasons.append("VIP reporter — manual review required")
    if priority == "P1":
        reasons.append("P1 priority — critical incident needs human oversight")
    if category == "security":
        reasons.append("Security category — all fixes require approval")
    if risk_score >= HIGH_RISK_THRESHOLD:
        reasons.append(f"High risk score ({risk_score:.2f} >= {HIGH_RISK_THRESHOLD})")
    if confidence < AUTO_RESOLVE_CONFIDENCE:
        reasons.append(f"Low confidence ({confidence:.0%} < {AUTO_RESOLVE_CONFIDENCE:.0%})")

    routing   = "human_review" if reasons else "auto_resolve"
    assigned  = "system-bot" if routing == "auto_resolve" else (
                "team-oncall-p1" if priority == "P1" else CATEGORY_TEAM.get(category, "team-l2"))

    sla_minutes  = SLA_MAP.get(priority, 1440)
    sla_deadline = datetime.utcnow() + timedelta(minutes=sla_minutes)

    return {
        "routing_decision":  routing,
        "risk_score":        risk_score,
        "risk_reasons":      reasons,
        "assigned_to":       assigned,
        "requires_approval": bool(reasons),
        "sla_deadline":      sla_deadline.isoformat(),
    }


# ── Module F: Anomaly Detection ────────────────────────────────────────────────
def detect_anomaly(metrics: MetricsInput) -> dict:
    """
    Detects anomaly in a single metrics snapshot.

    Flow:
      1. Build feature vector (5 raw + 2 engineered = 7 features)
      2. Normalize using fitted StandardScaler (same as training)
      3. Get anomaly score from IsolationForest.score_samples()
      4. Compare score vs learned threshold -> is_anomaly
      5. Rule-based root cause analysis -> explains what is wrong

    Score interpretation:
      Below threshold by > 0.3 -> CRITICAL
      Below threshold by > 0.15 -> HIGH
      Below threshold           -> MEDIUM
      Above threshold           -> NORMAL
    """
    m = {
        "response_time_ms":  metrics.response_time_ms,
        "error_rate":        metrics.error_rate,
        "cpu_usage_pct":     metrics.cpu_usage_pct,
        "memory_usage_pct":  metrics.memory_usage_pct,
        "request_count":     metrics.request_count,
    }

    # Fallback: rule-based only if model not loaded
    if anomaly_model_bundle is None:
        is_anomaly = (
            metrics.error_rate > 0.20 or
            metrics.response_time_ms > 2000 or
            metrics.cpu_usage_pct > 90 or
            metrics.memory_usage_pct > 90
        )
        return _build_anomaly_result(metrics.service, 0.0, is_anomaly, "medium", 0.5, m)

    model     = anomaly_model_bundle["model"]
    scaler    = anomaly_model_bundle["scaler"]
    threshold = anomaly_model_bundle["threshold"]
    feat_cols = anomaly_model_bundle["feature_cols"]

    # Build feature vector in exact training order
    error_x_latency   = m["error_rate"] * m["response_time_ms"]
    resource_pressure = (m["cpu_usage_pct"] + m["memory_usage_pct"]) / 2
    X = np.array([[
        m["response_time_ms"], m["error_rate"],
        m["cpu_usage_pct"],    m["memory_usage_pct"],
        m["request_count"],    error_x_latency, resource_pressure,
    ]])
    X_scaled     = scaler.transform(X)
    score        = float(model.score_samples(X_scaled)[0])
    is_anomaly   = score < threshold
    gap          = threshold - score

    if not is_anomaly:
        severity, confidence = "normal", 0.0
    elif gap > 0.3:
        severity, confidence = "critical", 0.95
    elif gap > 0.15:
        severity, confidence = "high", 0.85
    else:
        severity, confidence = "medium", 0.70

    return _build_anomaly_result(metrics.service, score, is_anomaly, severity, confidence, m)


def _build_anomaly_result(service, score, is_anomaly, severity, confidence, m) -> dict:
    """Builds anomaly result dict with root cause analysis."""
    hint   = "System operating normally"
    action = "No action required"

    if is_anomaly:
        issues  = []
        actions = []

        if m["response_time_ms"] > 2000:
            issues.append(f"Critical latency ({m['response_time_ms']:.0f}ms)")
            actions.append("Check DB connection pool and downstream services")
        elif m["response_time_ms"] > 800:
            issues.append(f"High latency ({m['response_time_ms']:.0f}ms)")
            actions.append("Investigate slow queries and network")

        if m["error_rate"] > 0.30:
            issues.append(f"Critical error rate ({m['error_rate']:.1%})")
            actions.append("Check logs for exceptions, consider rollback")
        elif m["error_rate"] > 0.10:
            issues.append(f"Elevated error rate ({m['error_rate']:.1%})")
            actions.append("Review recent deployments")

        if m["cpu_usage_pct"] > 90:
            issues.append(f"CPU exhaustion ({m['cpu_usage_pct']:.0f}%)")
            actions.append("Kill runaway process, enable auto-scaling")
        if m["memory_usage_pct"] > 90:
            issues.append(f"Memory critical ({m['memory_usage_pct']:.0f}%)")
            actions.append("Check for memory leak, restart service")
        if m["request_count"] > 5000:
            issues.append(f"Traffic spike ({m['request_count']} req/s)")
            actions.append("Scale horizontally, enable rate limiting")

        hint   = "; ".join(issues)  if issues  else "Anomalous pattern detected"
        action = "; ".join(actions) if actions else "Investigate metrics dashboard"

    return {
        "service":            service,
        "is_anomaly":         is_anomaly,
        "anomaly_score":      round(score, 4),
        "severity":           severity,
        "confidence":         round(confidence, 4),
        "root_cause_hint":    hint,
        "recommended_action": action,
        "metrics":            m,
    }


# ==============================================================================
# SECTION 6: FASTAPI APP
# ==============================================================================

app = FastAPI(
    title="AI-Driven Application Support & Operations",
    version="1.0.0",
    description="""
## AI Support Ops API

Transforms reactive IT support into intelligent, predictive operations.

### Complete AI Pipeline
1. NLP Classification
2. Similarity Search
3. Confidence Scoring
4. Governance Engine
5. Anomaly Detection
    """,
    docs_url="/docs",
)

# ✅ ADD CORS HERE (OUTSIDE FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(admin_router)






# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    print("\n" + "=" * 60)
    print("  Starting AI Support Ops API v1.0.0")
    print("=" * 60)
    load_models()
    print("  API ready at http://localhost:8000")
    print("  Swagger docs: http://localhost:8000/docs\n")


# ── Request logging middleware ─────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start    = time.time()
    response = await call_next(request)
    ms       = round((time.time() - start) * 1000, 1)
    print(f"  {request.method:6s} {request.url.path}  ->  {response.status_code}  [{ms}ms]")
    return response


# ── Global error handler ───────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def error_handler(request: Request, exc: Exception):
    print(f"  ERROR on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": str(exc)},
    )


# ==============================================================================
# SECTION 7: TICKET ENDPOINTS
# ==============================================================================

@app.post("/api/v1/tickets/", status_code=201)
async def submit_ticket(ticket: TicketCreate):
    """
    Submit a ticket through the complete AI pipeline.

    Pipeline stages:
      A) NLP Classification  -> category + priority prediction
      B) Similarity Search   -> top-5 most similar past tickets
      C) Confidence Scoring  -> overall AI confidence (0-1)
      D) Governance Decision -> auto_resolve or human_review
      E) Save to store       -> persist with all AI metadata
    """
    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"
    print(f"\n  Processing ticket: {ticket_id} — {ticket.title[:50]}...")

     # ── A: Classify ────────────────────────────────────────────────
    classification = classify_ticket(ticket.title, ticket.description)

    # ── B: Find similar ────────────────────────────────────────────
    from app.services.similarity_service import SimilarityService

    sim_svc = SimilarityService()

    similar = sim_svc.find_similar(
        title=ticket.title,
        description=ticket.description,
        predicted_category=classification["predicted_category"]
    )

    # ── C: Confidence ──────────────────────────────────────────────
    from app.services.confidence_ml_service import ConfidenceMLService

    conf_svc = ConfidenceMLService()

    # 🔥 Convert dict → object (IMPORTANT FIX)
    class Obj:
        pass

    cls_obj = Obj()
    cls_obj.category_confidence = classification["category_confidence"]
    cls_obj.priority_confidence = classification["priority_confidence"]
    cls_obj.predicted_category = classification["predicted_category"]

    confidence = conf_svc.compute(
        classification=cls_obj,
        similar_tickets=similar
    )

    # Build recommended fix
    recommended_fix = None
    if similar:
        recommended_fix = (
            f"Suggested fix (confidence {int(confidence * 100)}%) — review before applying:\n\n"
            f"Primary: {similar[0]['resolution']}"
        )

    # ── D: Governance ──────────────────────────────────────────────
    governance = governance_decision(
        classification,
        confidence,
        ticket.is_vip or False,
        similar
    )

    # ── E: Store ticket ────────────────────────────────────────────
    status = "auto_resolved" if governance["routing_decision"] == "auto_resolve" else "open"
    record = {
        "ticket_id":         ticket_id,
        "title":             ticket.title,
        "description":       ticket.description,
        "reporter":          ticket.reporter,
        "is_vip":            ticket.is_vip or False,
        "classification":    classification,
        "similar_tickets":   similar,
        "confidence_score":  confidence,
        "recommended_fix":   recommended_fix,
        "governance":        governance,
        "status":            status,
        "created_at":        datetime.utcnow().isoformat(),
        "resolved_at":       datetime.utcnow().isoformat() if status == "auto_resolved" else None,
        "actual_resolution": recommended_fix if status == "auto_resolved" else None,
    }
    TICKET_STORE[ticket_id] = record

    print(f"  Done: category={classification['predicted_category']}  "
          f"priority={classification['predicted_priority']}  "
          f"confidence={confidence:.0%}  "
          f"decision={governance['routing_decision']}")

    return record


@app.get("/api/v1/tickets/")
async def list_tickets(
    status:   Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    limit:    int = 50,
    offset:   int = 0,
):
    """List all tickets with optional filters."""
    tickets = list(TICKET_STORE.values())

    if status:
        tickets = [t for t in tickets if t["status"] == status]
    if priority:
        tickets = [t for t in tickets if
                   t["classification"]["predicted_priority"] == priority]
    if category:
        tickets = [t for t in tickets if
                   t["classification"]["predicted_category"] == category]

    # Sort newest first
    tickets.sort(key=lambda t: t["created_at"], reverse=True)

    return {
        "total": len(tickets),
        "tickets": tickets[offset: offset + limit],
    }


@app.get("/api/v1/tickets/{ticket_id}")
async def get_ticket(ticket_id: str):
    """Get one ticket by its ID."""
    ticket = TICKET_STORE.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")
    return ticket


@app.post("/api/v1/tickets/{ticket_id}/resolve")
async def resolve_ticket(ticket_id: str, body: TicketResolve):
    """
    Resolve a ticket with a human-provided resolution.
    Records whether the AI recommendation was correct.
    This data feeds the Continuous Learning Pipeline.
    """
    global RESOLUTION_COUNTER

    ticket = TICKET_STORE.get(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    if ticket["status"] in ("resolved", "closed", "auto_resolved"):
        raise HTTPException(status_code=400, detail=f"Ticket already {ticket['status']}")

    # Calculate resolution time
    created     = datetime.fromisoformat(ticket["created_at"])
    res_minutes = int((datetime.utcnow() - created).total_seconds() / 60)

    ticket["status"]           = "resolved"
    ticket["actual_resolution"] = body.resolution_text
    ticket["resolved_by"]      = body.resolved_by
    ticket["resolved_at"]      = datetime.utcnow().isoformat()
    ticket["resolution_time_minutes"] = res_minutes
    ticket["ai_was_correct"]   = body.ai_recommendation_was_correct

    RESOLUTION_COUNTER += 1
    print(f"\n  Ticket {ticket_id} resolved by {body.resolved_by} "
          f"in {res_minutes}min. AI correct: {body.ai_recommendation_was_correct}")
    print(f"  Resolutions since last retrain: {RESOLUTION_COUNTER}/50")

    return {
        "success": True,
        "ticket_id": ticket_id,
        "resolution_time_minutes": res_minutes,
        "ai_was_correct": body.ai_recommendation_was_correct,
        "message": f"Ticket resolved. {RESOLUTION_COUNTER}/50 resolutions collected for retraining.",
    }


# ==============================================================================
# SECTION 8: ANOMALY DETECTION ENDPOINTS
# ==============================================================================

@app.post("/api/v1/logs/detect")
async def detect_single(metrics: MetricsInput):
    """
    Real-time anomaly detection for one metrics snapshot.

    Call this on every monitoring poll (e.g., every 30 seconds per service).
    Returns is_anomaly, severity, root_cause_hint, and recommended_action.
    """
    result = detect_anomaly(metrics)

    if result["is_anomaly"]:
        result["event_id"] = str(uuid.uuid4())
        ANOMALY_STORE.append({
            **result,
            "timestamp": datetime.utcnow().isoformat(),
        })
        print(f"\n  ANOMALY DETECTED: {metrics.service}  "
              f"severity={result['severity']}  "
              f"hint={result['root_cause_hint'][:60]}")

    return result


@app.post("/api/v1/logs/analyze")
async def analyze_bulk(request: BulkLogRequest):
    """
    Analyze a batch of log entries (up to 1000) at once.
    Returns only anomalous entries, sorted by severity.
    """
    anomalies = []
    for log in request.logs:
        result = detect_anomaly(log)
        if result["is_anomaly"]:
            anomalies.append(result)

    # Sort: critical first
    order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    anomalies.sort(key=lambda r: order.get(r["severity"], 4))

    total      = len(request.logs)
    n_anomalies = len(anomalies)
    rate       = n_anomalies / max(1, total)

    services = list({a["service"] for a in anomalies})
    summary  = (
        f"All {total} entries normal." if not anomalies else
        f"Found {n_anomalies} anomalies in {total} entries ({rate:.1%}). "
        f"Affected: {', '.join(services)}"
    )

    return {
        "total_logs":        total,
        "anomalies_detected": n_anomalies,
        "anomaly_rate":      round(rate, 4),
        "anomalies":         anomalies,
        "summary":           summary,
    }


@app.get("/api/v1/logs/anomalies")
async def list_anomalies(severity: Optional[str] = None):
    """List stored anomaly events."""
    events = ANOMALY_STORE.copy()
    if severity:
        events = [e for e in events if e["severity"] == severity]
    events.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return {"total": len(events), "anomalies": events[:100]}


# ==============================================================================
# SECTION 9: DASHBOARD ENDPOINTS
# ==============================================================================

@app.get("/api/v1/dashboard/metrics")
async def dashboard_metrics():
    """
    Aggregated KPIs for the DevOps dashboard.
    Shows ticket volumes, AI performance, SLA status, anomaly counts.
    """
    tickets = list(TICKET_STORE.values())

    total        = len(tickets)
    open_count   = sum(1 for t in tickets if t["status"] == "open")
    auto_resolved = sum(1 for t in tickets if t["routing_decision_cached"] == "auto_resolve"
                       if "routing_decision_cached" in t)

    # Simpler approach — use governance routing_decision
    auto_count   = sum(1 for t in tickets if t.get("governance", {}).get("routing_decision") == "auto_resolve")
    human_count  = sum(1 for t in tickets if t.get("governance", {}).get("routing_decision") == "human_review")
    avg_conf     = sum(t.get("confidence_score", 0) for t in tickets) / max(1, total)

    resolved     = [t for t in tickets if t.get("resolved_at")]
    avg_res_time = sum(t.get("resolution_time_minutes", 0) for t in resolved) / max(1, len(resolved))

    cat_counts   = Counter(t.get("classification", {}).get("predicted_category", "") for t in tickets)
    pri_counts   = Counter(t.get("classification", {}).get("predicted_priority", "") for t in tickets)

    anomalies_24h = sum(
        1 for a in ANOMALY_STORE
        if datetime.fromisoformat(a.get("timestamp", "2000-01-01"))
           > datetime.utcnow() - timedelta(hours=24)
    )

    ai_correct   = [t for t in tickets if "ai_was_correct" in t]
    ai_accuracy  = sum(1 for t in ai_correct if t["ai_was_correct"]) / max(1, len(ai_correct))

    return {
        "total_tickets":             total,
        "open_tickets":              open_count,
        "auto_resolved":             auto_count,
        "human_review":              human_count,
        "avg_confidence_score":      round(avg_conf, 4),
        "avg_resolution_time_minutes": round(avg_res_time, 1),
        "top_categories":            dict(cat_counts.most_common(5)),
        "top_priorities":            dict(pri_counts),
        "anomalies_last_24h":        anomalies_24h,
        "ai_recommendation_accuracy": round(ai_accuracy, 4),
        "resolutions_since_retrain": RESOLUTION_COUNTER,
        "model_status": {
            "ticket_classifier": "loaded" if ticket_model_bundle else "not_loaded",
            "anomaly_detector":  "loaded" if anomaly_model_bundle else "not_loaded",
            "similarity_index":  "loaded" if historical_df is not None else "not_loaded",
        },
    }


@app.get("/api/v1/dashboard/health")
async def health():
    """Health check — shows model load status."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "models": {
            "ticket_classifier":  "ready" if ticket_model_bundle else "not_loaded",
            "anomaly_detector":   "ready" if anomaly_model_bundle else "not_loaded",
            "similarity_index":   f"ready ({len(historical_df)} tickets)" if historical_df is not None else "not_loaded",
        },
        "tickets_in_memory":    len(TICKET_STORE),
        "anomalies_in_memory":  len(ANOMALY_STORE),
        "resolutions_counter":  RESOLUTION_COUNTER,
    }


@app.get("/")
async def root():
    return {
        "name":    "AI-Driven Application Support & Operations",
        "version": "1.0.0",
        "docs":    "/docs",
        "health":  "/api/v1/dashboard/health",
    }


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)