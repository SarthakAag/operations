"""
app/services/learning_service.py
================================================================================
PURPOSE:
  Implements the Continuous Learning Pipeline.

  Every time a human engineer resolves a ticket, this service:
    1. Saves the resolution to the DB (ResolutionDB)
    2. Appends the ticket + resolution to tickets.csv
    3. Adds the ticket to the in-memory similarity index immediately
    4. Increments a resolution counter
    5. When counter >= RETRAIN_THRESHOLD:
         Runs train_models.py as a background subprocess
         On completion: hot-reloads all ML models

WHY CONTINUOUS LEARNING MATTERS:
  ML models trained on a fixed snapshot drift over time:
    - New services are deployed (new service names appear in tickets)
    - New failure modes emerge (new patterns not in training data)
    - New engineers join with different resolution styles

  Without retraining:
    Year 1: model accuracy = 95%
    Year 2: model accuracy = 78%  (drift)
    Year 3: model accuracy = 61%  (degraded)

  With continuous learning:
    Model accuracy stays high as it learns from every resolution.

TWO-TIER UPDATE STRATEGY:
  Tier 1 — IMMEDIATE (per resolution):
    similarity_service.add_resolved_ticket()
    The new resolution is searchable for the NEXT ticket instantly.
    No retraining needed — just update the TF-IDF index in memory.

  Tier 2 — BATCH (every N resolutions):
    Full model retraining via subprocess (train_models.py)
    Updates the NLP classifier and anomaly detector.
    Takes ~30 seconds — runs in background, API stays live.
    Hot-reload replaces models without restarting the server.

THREAD SAFETY:
  _retrain_lock prevents two retraining jobs from running simultaneously.
  If a retrain is already in progress when the counter hits threshold,
  the new trigger is ignored until the running job completes.

PERSISTENCE:
  _resolution_counter is in-memory only.
  If the server restarts, the counter resets to 0.
  For production, store counter in Redis or DB.
  (Shown as TODO in the code below.)
================================================================================
"""

import asyncio
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from app.config import get_settings

settings = get_settings()


# ==============================================================================
# SECTION 1: MODULE-LEVEL STATE
# ==============================================================================
# These live at module level so they persist across requests
# for the lifetime of the server process.
# ==============================================================================

# Counts resolved tickets since the last model retraining.
# Reset to 0 after each successful retraining.
_resolution_counter: int = 0

# asyncio.Lock prevents two retraining jobs running simultaneously.
# Created lazily in _get_retrain_lock() because asyncio.Lock()
# must be created inside an async context.
_retrain_lock: Optional[asyncio.Lock] = None

# Tracks whether a retrain is currently running.
_retrain_in_progress: bool = False


def _get_retrain_lock() -> asyncio.Lock:
    """
    Returns the singleton retrain lock.
    Creates it on first call (lazy init inside async context).
    """
    global _retrain_lock
    if _retrain_lock is None:
        _retrain_lock = asyncio.Lock()
    return _retrain_lock


# ==============================================================================
# SECTION 2: CSV PERSISTENCE
# ==============================================================================

def _append_to_csv(
    ticket_id:               str,
    title:                   str,
    description:             str,
    category:                str,
    priority:                str,
    resolution:              str,
    resolution_time_minutes: int,
):
    """
    Appends a newly resolved ticket to tickets.csv.

    WHY append to CSV?
      train_models.py reads from tickets.csv.
      Every new resolved ticket that is appended becomes
      part of the NEXT training run's dataset.
      Over time the dataset grows and the model improves.

    File handling:
      - Creates the CSV file if it doesn't exist (with header)
      - Appends a new row if it already exists (no header)
      - extrasaction="ignore": extra fields are silently skipped

    Args:
        ticket_id               : e.g., "TKT-A1B2C3D4"
        title                   : ticket title
        description             : ticket description
        category                : confirmed category (ground truth)
        priority                : confirmed priority (ground truth)
        resolution              : what actually fixed it
        resolution_time_minutes : how long it took
    """
    csv_path   = Path(settings.tickets_csv_path)
    file_exists = csv_path.exists()

    fieldnames = [
        "ticket_id", "title", "description",
        "category", "priority", "status",
        "resolution", "resolution_time_minutes",
        "created_at", "resolved_at",
    ]

    try:
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()

            writer.writerow({
                "ticket_id":               ticket_id,
                "title":                   title,
                "description":             description,
                "category":                category,
                "priority":                priority,
                "status":                  "resolved",
                "resolution":              resolution,
                "resolution_time_minutes": resolution_time_minutes,
                "created_at":              datetime.utcnow().isoformat(),
                "resolved_at":             datetime.utcnow().isoformat(),
            })

        print(
            f"  [LearningService] Appended {ticket_id} to {csv_path} "
            f"(category={category}, priority={priority})"
        )

    except Exception as e:
        # CSV append failure should not block the API response
        print(f"  [LearningService] WARNING: CSV append failed: {e}")


# ==============================================================================
# SECTION 3: HOT MODEL RELOAD
# ==============================================================================

def _hot_reload_all_models():
    """
    Reloads all ML models from disk WITHOUT restarting the server.

    Called after train_models.py completes successfully.

    Order matters:
      1. NLP model first      (most used, highest priority)
      2. Anomaly model second
      3. Similarity index last (rebuilds from updated CSV)

    Each service's reload_model() / rebuild_index() reads the
    new .pkl file from disk and replaces the in-memory bundle.
    In-flight requests using the old model complete normally.
    New requests after reload use the new model.
    """
    print("  [LearningService] Hot-reloading all models...")

    try:
        # Import here to avoid circular imports at module load time
        from app.services.nlp_service import get_nlp_service
        from app.services.anomaly_service import get_anomaly_service
        from app.services.similarity_service import get_similarity_service

        get_nlp_service().reload_model()
        get_anomaly_service().reload_model()
        get_similarity_service().rebuild_index()

        print("  [LearningService] All models hot-reloaded successfully")

    except Exception as e:
        print(f"  [LearningService] ERROR during hot-reload: {e}")


# ==============================================================================
# SECTION 4: BACKGROUND RETRAINING
# ==============================================================================

async def _run_retrain_background():
    """
    Runs train_models.py as an async subprocess.

    WHY subprocess instead of calling train functions directly?
      train_models.py takes ~30 seconds.
      Running it directly in the async event loop would BLOCK
      the entire FastAPI server for 30 seconds.
      Running as subprocess lets it use a separate process
      while the event loop continues handling requests.

    Flow:
      asyncio.create_subprocess_exec()  → spawn train_models.py
      wait_for(communicate(), 600)      → wait up to 10 minutes
      returncode == 0                   → success → hot_reload
      returncode != 0                   → log error → keep old model

    asyncio.Lock prevents two retraining processes running simultaneously.
    If the lock is already held, this call returns immediately.
    """
    global _retrain_in_progress, _resolution_counter

    lock = _get_retrain_lock()

    # If retrain already running, skip this trigger
    if lock.locked():
        print(
            "  [LearningService] Retrain already in progress — skipping trigger"
        )
        return

    async with lock:
        _retrain_in_progress = True
        print(
            f"\n  [LearningService] Starting background retraining...\n"
            f"  Python: {sys.executable}\n"
            f"  Script: train_models.py"
        )

        try:
            # Spawn train_models.py as a subprocess
            proc = await asyncio.create_subprocess_exec(
                sys.executable,           # same python as the server
                "train_models.py",
                stdout = asyncio.subprocess.PIPE,
                stderr = asyncio.subprocess.PIPE,
            )

            # Wait up to 10 minutes for training to complete
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=600,
            )

            if proc.returncode == 0:
                # ── SUCCESS ───────────────────────────────────────
                print(
                    "  [LearningService] Retraining completed successfully!\n"
                    + stdout.decode()[-500:]   # last 500 chars of output
                )
                _hot_reload_all_models()
                _resolution_counter = 0       # reset counter
                print(
                    f"  [LearningService] Resolution counter reset to 0. "
                    f"Threshold: {settings.retrain_after_n_resolutions}"
                )
            else:
                # ── FAILURE ───────────────────────────────────────
                print(
                    f"  [LearningService] ERROR: Retraining failed "
                    f"(exit code {proc.returncode})\n"
                    + stderr.decode()[-500:]
                )
                # Keep old models — don't reload

        except asyncio.TimeoutError:
            print(
                "  [LearningService] ERROR: Retraining timed out after 10 minutes"
            )
        except Exception as e:
            print(f"  [LearningService] ERROR: Retraining exception: {e}")
        finally:
            _retrain_in_progress = False


# ==============================================================================
# SECTION 5: MAIN RECORD RESOLUTION FUNCTION
# ==============================================================================

async def record_resolution(
    ticket_id:                    str,
    title:                        str,
    description:                  str,
    category:                     str,
    priority:                     str,
    resolution_text:              str,
    resolved_by:                  str,
    resolution_time_minutes:      int,
    ai_recommendation_was_correct: bool,
    db=None,                              # AsyncSession (optional — for DB save)
):
    """
    Records a human-confirmed resolution and triggers the learning pipeline.

    Called by POST /api/v1/tickets/{ticket_id}/resolve

    Args:
        ticket_id                    : e.g., "TKT-A1B2C3D4"
        title                        : ticket title
        description                  : ticket description
        category                     : confirmed category (from DB record)
        priority                     : confirmed priority (from DB record)
        resolution_text              : what the engineer actually did
        resolved_by                  : engineer name/email
        resolution_time_minutes      : total time from open to close
        ai_recommendation_was_correct: did AI suggest the right fix?
        db                           : AsyncSession for DB persistence

    Pipeline steps:
      1. Save to ResolutionDB (if db session provided)
      2. Append to tickets.csv (always)
      3. Update similarity index (immediate, in-memory)
      4. Increment counter
      5. Trigger retrain if counter >= threshold
    """
    global _resolution_counter

    print(
        f"\n  [LearningService] Recording resolution for {ticket_id}\n"
        f"  category={category} | priority={priority} | "
        f"time={resolution_time_minutes}min | "
        f"ai_correct={ai_recommendation_was_correct}"
    )

    # ── Step 1: Save to ResolutionDB ──────────────────────────────
    if db is not None:
        try:
            from app.database import ResolutionDB
            resolution_record = ResolutionDB(
                ticket_id                 = ticket_id,
                ticket_title              = title,
                ticket_description        = description,
                category                  = category,
                priority                  = priority,
                resolution_text           = resolution_text,
                resolution_time_minutes   = resolution_time_minutes,
                ai_recommendation_correct = ai_recommendation_was_correct,
                resolved_by               = resolved_by,
            )
            db.add(resolution_record)
            await db.flush()
            print(f"  [LearningService] Saved to ResolutionDB")
        except Exception as e:
            print(f"  [LearningService] WARNING: DB save failed: {e}")

    # ── Step 2: Append to tickets.csv ─────────────────────────────
    _append_to_csv(
        ticket_id               = ticket_id,
        title                   = title,
        description             = description,
        category                = category,
        priority                = priority,
        resolution              = resolution_text,
        resolution_time_minutes = resolution_time_minutes,
    )

    # ── Step 3: Update similarity index immediately ───────────────
    # This makes the new resolution searchable for the NEXT ticket
    # without waiting for full retraining.
    try:
        from app.services.similarity_service import get_similarity_service
        get_similarity_service().add_resolved_ticket(
            ticket_id               = ticket_id,
            title                   = title,
            description             = description,
            category                = category,
            priority                = priority,
            resolution              = resolution_text,
            resolution_time_minutes = resolution_time_minutes,
        )
    except Exception as e:
        print(f"  [LearningService] WARNING: Similarity index update failed: {e}")

    # ── Step 4: Increment counter ─────────────────────────────────
    _resolution_counter += 1
    threshold = settings.retrain_after_n_resolutions
    print(
        f"  [LearningService] Resolution counter: "
        f"{_resolution_counter}/{threshold}"
    )

    # ── Step 5: Trigger retraining if threshold reached ───────────
    if _resolution_counter >= threshold:
        print(
            f"  [LearningService] Threshold reached! "
            f"Triggering background model retraining..."
        )
        # create_task: fire-and-forget background job
        # The API response is NOT blocked waiting for training to finish
        asyncio.create_task(_run_retrain_background())

    return {
        "recorded":           True,
        "ticket_id":          ticket_id,
        "counter":            _resolution_counter,
        "threshold":          threshold,
        "retrain_triggered":  _resolution_counter >= threshold,
    }


# ==============================================================================
# SECTION 6: MANUAL RETRAIN (admin endpoint)
# ==============================================================================

async def manual_retrain() -> dict:
    """
    Manually triggers model retraining.
    Called by POST /api/v1/admin/retrain

    Does not require the counter to be at threshold.
    Useful for:
      - After bulk-importing historical tickets
      - After major system changes
      - Admin-initiated quality refresh

    Returns immediately with status.
    Retraining runs in background.
    """
    if _retrain_in_progress:
        return {
            "status":  "already_running",
            "message": "Retraining is already in progress",
        }

    asyncio.create_task(_run_retrain_background())
    return {
        "status":  "started",
        "message": "Retraining started in background. Models will hot-reload on completion.",
        "counter": _resolution_counter,
    }


# ==============================================================================
# SECTION 7: STATUS GETTERS
# ==============================================================================

def get_resolution_counter() -> int:
    """Returns current resolution counter (for dashboard)."""
    return _resolution_counter


def get_learning_status() -> dict:
    """
    Returns full continuous learning pipeline status.
    Used by GET /api/v1/dashboard/health
    """
    threshold = settings.retrain_after_n_resolutions
    return {
        "resolution_counter":    _resolution_counter,
        "retrain_threshold":     threshold,
        "resolutions_remaining": max(0, threshold - _resolution_counter),
        "progress_pct":          round(_resolution_counter / threshold * 100, 1),
        "retrain_in_progress":   _retrain_in_progress,
        "csv_path":              settings.tickets_csv_path,
        "csv_exists":            Path(settings.tickets_csv_path).exists(),
    }


def reset_counter_for_testing():
    """
    Resets the counter to 0.
    ONLY for use in tests — never call in production.
    """
    global _resolution_counter
    _resolution_counter = 0