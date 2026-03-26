# AI-Driven Application Support & Operations (AIOps)

An intelligent support system that transforms **reactive IT operations** into **automated, data-driven, and proactive incident management** using AI.

## Overview

This project simulates a real-world **AIOps platform** that:

* Classifies incoming support tickets using NLP
* Finds similar past incidents using embeddings
* Recommends fixes based on historical resolutions
* Calculates confidence using ML
* Applies governance rules for safe automation
* Provides an admin dashboard for monitoring and control

## Key Features

### 1.NLP-Based Ticket Classification

* Predicts **category** (database, application, network, etc.)
* Predicts **priority** (P1, P2, P3)
* Uses trained ML models

### 2.Semantic Similarity Search

* Uses **transformer embeddings** (`sentence-transformers`)
* Finds similar historical incidents
* Reuses proven resolutions

---

### 3️.Confidence Scoring (ML-Based)

* Combines:

  * Classification confidence
  * Similarity score
  * Historical patterns
* Outputs a single **decision confidence score**


### 4️.Governance Engine (Core Innovation )

* Decides:

  * `auto_resolve`
  * `human_review`
  * `escalate`
* Based on:

  * Confidence
  * Risk score
  * Priority (P1 handling)
  * VIP users
  * Security constraints

### 5️.Admin Dashboard

* View all tickets
* Monitor AI decisions
* Override system decisions
* Track system performance metrics

## Architecture

User → API → NLP Model → Similarity Engine → Confidence Model → Governance Engine → Response
                                           ↓
                                    Admin Dashboard

## Tech Stack

* **Backend:** FastAPI
* **ML/NLP:** Scikit-learn, Sentence Transformers
* **Similarity:** Cosine similarity (embeddings)
* **Frontend:** HTML + CSS
* **Data:** CSV (tickets dataset)
* **Deployment:** Local (Uvicorn)


## Project Structure
<img width="370" height="687" alt="Screenshot 2026-03-26 192801" src="https://github.com/user-attachments/assets/98b56ece-38d5-46a3-b8e5-a984467370a1" />
<img width="379" height="610" alt="Screenshot 2026-03-26 192829" src="https://github.com/user-attachments/assets/e4d0b60e-0467-4e39-b99b-62bddf6a213a" />




## Setup Instructions

### 1️.Clone the Repository

git clone <your-repo-url>
cd ai-ops-backend


### 2️.Create Virtual Environment


python -m venv .venv
.venv\Scripts\activate


### 3️.Install Dependencies

pip install -r requirements.txt


### 4️.Run the Backend

python api.py


### 5️.Access API Docs

http://localhost:8000/docs

### 6️.Open Admin Dashboard

Open:
frontend/admin.html


## Example Workflow

1. Submit a ticket:

json
{
  "title": "Database connection timeout",
  "description": "Connection pool exhausted"
}

2. System performs:

* Classification → `database`, `P1`
* Similarity → finds past incidents
* Confidence → ~85%
* Governance → `auto_resolve`

3. Returns:

json
{
  "decision": "auto_resolve",
  "recommended_fix": "...",
  "confidence_score": 0.85
}


## Governance Logic (Highlight)

* High confidence + strong similarity → **Auto Resolve**
* Medium confidence → **Human Review**
* Low confidence → **Escalation**
* Security issues → **Always Human**
* P1 → Smart override based on confidence


## Security & Best Practices

* Sensitive data excluded via `.gitignore`
* No hardcoded credentials
* Modular architecture for scalability
* Human-in-the-loop safety

## Impact

* Reduces manual workload
* Speeds up incident resolution
* Improves consistency using historical knowledge
* Enables safe AI automation with governance

## Future Enhancements

* Real-time streaming dashboard
* Feedback loop learning (self-improving AI)
* Integration with cloud monitoring tools
* Advanced anomaly detection

## Author

Developed as part of an AI-driven operations system project.

---

## ⭐ Final Note

This project demonstrates how AI can **augment IT operations**, balancing **automation with safety**, making it suitable for real-world enterprise environments.

---
