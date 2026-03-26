"""
app/config.py
================================================================================
PURPOSE:
  Central configuration for the entire application.
  All settings are loaded from the .env file ONCE at startup.
  Every other module imports get_settings() — never reads .env directly.

WHY THIS APPROACH?
  - Single source of truth for all config values
  - Change a value in .env -> all modules pick it up automatically
  - Easy to swap values for dev vs production environments
  - @lru_cache means .env is read only once, not on every request

HOW TO USE IN OTHER MODULES:
  from app.config import get_settings
  settings = get_settings()
  print(settings.ticket_model_path)
================================================================================
"""

import os
from functools import lru_cache
from dotenv import load_dotenv

# Load .env file into environment variables
load_dotenv()


class Settings:
    """
    All settings with defaults.
    Values in .env file override these defaults automatically.
    """

    def __init__(self):
        # ── App ───────────────────────────────────────────────────
        self.app_name    = os.getenv("APP_NAME", "AI-Driven Application Support & Operations")
        self.app_version = os.getenv("APP_VERSION", "1.0.0")
        self.debug       = os.getenv("DEBUG", "True").lower() == "true"

        # ── Database ──────────────────────────────────────────────
        # Dev  : SQLite  (no setup, just works)
        # Prod : PostgreSQL URL from .env
        self.database_url = os.getenv(
            "DATABASE_URL",
            "sqlite+aiosqlite:///./ai_support.db"
        )

        # ── ML Model Paths ────────────────────────────────────────
        # These point to .pkl files created by train_models.py
        self.ticket_model_path  = os.getenv("TICKET_MODEL_PATH",  "models/ticket_model.pkl")
        self.anomaly_model_path = os.getenv("ANOMALY_MODEL_PATH", "models/anomaly_model.pkl")
        self.tickets_csv_path   = os.getenv("TICKETS_CSV_PATH",   "data/tickets.csv")
        self.logs_json_path     = os.getenv("LOGS_JSON_PATH",     "data/logs.json")

        # ── Governance Thresholds ─────────────────────────────────
        # confidence >= auto_resolve_confidence  ->  AI resolves automatically
        # confidence <  auto_resolve_confidence  ->  routes to human team
        self.auto_resolve_confidence = float(os.getenv("AUTO_RESOLVE_CONFIDENCE", "0.80"))

        # risk score >= high_risk_threshold  ->  always human review
        self.high_risk_threshold = float(os.getenv("HIGH_RISK_THRESHOLD", "0.70"))

        # False = VIP tickets always go to human review (safer default)
        self.vip_auto_resolve = os.getenv("VIP_AUTO_RESOLVE", "False").lower() == "true"

        # ── SLA Thresholds (in minutes) ───────────────────────────
        # P1 = Critical  60 min
        # P2 = High      4 hours
        # P3 = Medium    24 hours
        # P4 = Low       72 hours
        self.sla_p1_minutes = int(os.getenv("SLA_P1_MINUTES", "60"))
        self.sla_p2_minutes = int(os.getenv("SLA_P2_MINUTES", "240"))
        self.sla_p3_minutes = int(os.getenv("SLA_P3_MINUTES", "1440"))
        self.sla_p4_minutes = int(os.getenv("SLA_P4_MINUTES", "4320"))

        # ── Anomaly Detection ─────────────────────────────────────
        # Must match contamination used during model training (0.10)
        self.anomaly_contamination = float(os.getenv("ANOMALY_CONTAMINATION", "0.10"))

        # ── Continuous Learning ───────────────────────────────────
        # Trigger model retraining after this many new human resolutions
        self.retrain_after_n_resolutions = int(os.getenv("RETRAIN_AFTER_N_RESOLUTIONS", "50"))

        # ── External Integrations (optional) ─────────────────────
        self.servicenow_url    = os.getenv("SERVICENOW_URL", "")
        self.jira_url          = os.getenv("JIRA_URL", "")
        self.slack_webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")

    # ─────────────────────────────────────────────────────────────
    # HELPER PROPERTIES
    # These convert individual settings into useful dicts
    # so other modules don't have to rebuild them each time
    # ─────────────────────────────────────────────────────────────

    @property
    def sla_map(self) -> dict:
        """
        SLA thresholds as a dict — used by governance engine.
        Example: settings.sla_map["P1"] -> 60
        """
        return {
            "P1": self.sla_p1_minutes,
            "P2": self.sla_p2_minutes,
            "P3": self.sla_p3_minutes,
            "P4": self.sla_p4_minutes,
        }

    @property
    def category_risk(self) -> dict:
        """
        Risk weight per fix category.
        Higher = riskier = more likely to need human review.
        Used by governance_service.py to compute risk score.
        """
        return {
            "security":       1.0,   # security fixes always need approval
            "database":       0.75,  # DB changes can cause data loss
            "infrastructure": 0.65,  # infra changes affect all services
            "network":        0.55,  # network changes have large blast radius
            "application":    0.40,  # app fixes are easiest to rollback
        }

    @property
    def priority_risk(self) -> dict:
        """
        Risk weight per ticket priority.
        P1 = any auto-fix mistake worsens an active outage.
        """
        return {
            "P1": 1.0,
            "P2": 0.65,
            "P3": 0.35,
            "P4": 0.15,
        }

    @property
    def category_team(self) -> dict:
        """
        Maps ticket category to responsible support team.
        Used by governance_service.py for routing decisions.
        """
        return {
            "security":       "team-security",
            "database":       "team-db",
            "infrastructure": "team-infra",
            "network":        "team-network",
            "application":    "team-app",
        }


@lru_cache()
def get_settings() -> Settings:
    """
    Returns cached Settings instance.
    .env is read only ONCE — on first call.
    All subsequent calls return the same cached object.

    Usage in any module:
        from app.config import get_settings
        settings = get_settings()
    """
    return Settings()