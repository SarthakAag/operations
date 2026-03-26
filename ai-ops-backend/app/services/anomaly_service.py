"""
app/services/anomaly_service.py
================================================================================
PURPOSE:
  Loads the trained anomaly_model.pkl and detects anomalies in
  system metrics snapshots.

  Given 5 metrics for a service:
    response_time_ms, error_rate, cpu_usage_pct,
    memory_usage_pct, request_count

  Returns:
    is_anomaly         : True | False
    anomaly_score      : float (lower = more anomalous, range ~-0.8 to 0.0)
    severity           : critical | high | medium | low | normal
    confidence         : how certain is the anomaly call (0-1)
    root_cause_hint    : which metrics are abnormal and by how much
    recommended_action : what to do about it

HOW THE ML MODEL WORKS (IsolationForest):
  Core idea: anomalies are EASIER TO ISOLATE than normal points.

  During training, the model built 200 random decision trees.
  Each tree randomly picks a feature and a split value, recursively
  partitioning data until each point is alone.

  Normal points: dense clusters → need MANY splits to isolate
  Anomaly points: outliers       → isolated with VERY FEW splits

  anomaly_score = average path length across all 200 trees
    Short path (few splits needed) = ANOMALY  → score is very negative
    Long path  (many splits needed) = NORMAL  → score is close to 0

  score < threshold → is_anomaly = True

TWO-LAYER DETECTION:
  Layer 1 (ML): IsolationForest detects the anomaly based on learned patterns
  Layer 2 (Rules): After ML says "anomaly", rules explain WHICH metric caused it

  Why two layers?
    ML answers "IS something wrong?" very well
    ML cannot answer "WHAT is wrong?" — it has no semantic understanding
    Rules are needed for the human-readable explanation
================================================================================
"""

import pickle
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.config import get_settings
from app.schemas import AnomalyResult, MetricsInput

settings = get_settings()


# ==============================================================================
# SECTION 1: SEVERITY THRESHOLDS
# ==============================================================================
# How far BELOW the optimal threshold determines severity.
#
# Example: if optimal_threshold = -0.52
#   score = -0.85  -> gap = 0.33  -> CRITICAL  (far below threshold)
#   score = -0.70  -> gap = 0.18  -> HIGH
#   score = -0.55  -> gap = 0.03  -> MEDIUM     (just crossed threshold)
#   score = -0.40  -> above threshold -> NORMAL
# ==============================================================================

SEVERITY_GAPS = {
    "critical": 0.30,   # gap > 0.30  -> CRITICAL
    "high":     0.15,   # gap > 0.15  -> HIGH
    "medium":   0.00,   # gap > 0.00  -> MEDIUM  (any anomaly that isn't high/critical)
}

# Thresholds for rule-based root cause analysis
# These are the per-metric thresholds that trigger a rule
METRIC_THRESHOLDS = {
    "response_time_ms": {
        "critical": 2000,   # > 2000ms = severe latency problem
        "high":     800,    # > 800ms  = significant latency
        "warn":     400,    # > 400ms  = elevated latency
    },
    "error_rate": {
        "critical": 0.30,   # > 30% errors = severe incident
        "high":     0.10,   # > 10% errors = significant issue
        "warn":     0.05,   # > 5%  errors = elevated errors
    },
    "cpu_usage_pct": {
        "critical": 90,     # > 90% CPU = near exhaustion
        "high":     75,     # > 75% CPU = high load
        "warn":     65,
    },
    "memory_usage_pct": {
        "critical": 90,     # > 90% memory = OOM risk
        "high":     80,     # > 80% memory = high pressure
        "warn":     70,
    },
    "request_count": {
        "spike":    5000,   # > 5000 req/s = traffic spike
        "drop":     50,     # < 50 req/s   = traffic drop (also anomalous)
    },
}


# ==============================================================================
# SECTION 2: FEATURE ENGINEERING
# ==============================================================================

def build_feature_vector(metrics: Dict[str, float]) -> np.ndarray:
    """
    Builds the 7-feature vector in EXACT same order as training.

    CRITICAL: Feature order must match train_models.py exactly.
    The StandardScaler was fit on this specific column order.
    If order changes, scaler applies wrong normalization to wrong feature.

    5 RAW features:
      1. response_time_ms   — latency
      2. error_rate         — error fraction
      3. cpu_usage_pct      — CPU load
      4. memory_usage_pct   — memory load
      5. request_count      — traffic volume

    2 ENGINEERED features:
      6. error_x_latency    = error_rate * response_time_ms
         Why? This is ONLY high when BOTH error AND latency are high.
         A cascading failure has high errors AND high latency together.
         This single feature captures that combined signal powerfully.

      7. resource_pressure  = (cpu + memory) / 2
         Why? CPU and memory often spike together during resource exhaustion.
         This average gives one clean signal instead of two correlated features.
    """
    rt  = metrics.get("response_time_ms",  120.0)
    er  = metrics.get("error_rate",         0.01)
    cpu = metrics.get("cpu_usage_pct",      45.0)
    mem = metrics.get("memory_usage_pct",   55.0)
    req = metrics.get("request_count",    1000.0)

    error_x_latency   = er * rt           # combined failure signal
    resource_pressure = (cpu + mem) / 2   # combined resource signal

    return np.array([[rt, er, cpu, mem, req, error_x_latency, resource_pressure]])


# ==============================================================================
# SECTION 3: RULE-BASED ROOT CAUSE ANALYSIS
# ==============================================================================

def analyze_root_cause(
    metrics: Dict[str, float],
    is_anomaly: bool
) -> Tuple[str, str]:
    """
    Identifies WHICH metrics are abnormal and generates human-readable
    root cause hint + recommended action.

    This runs AFTER the IsolationForest identifies an anomaly.
    ML answers "IS something wrong?" — these rules answer "WHAT is wrong?"

    Logic:
      For each metric, check if it exceeds thresholds.
      Collect all issues found.
      Build a combined hint string.
      Pair each issue with an action.

    Combined signals are also detected:
      high error + high latency together → cascading failure
      high CPU + high memory together   → resource exhaustion

    Returns:
      (root_cause_hint, recommended_action)
    """
    if not is_anomaly:
        return "System operating within normal parameters", "No action required"

    rt  = metrics.get("response_time_ms",  0)
    er  = metrics.get("error_rate",        0)
    cpu = metrics.get("cpu_usage_pct",     0)
    mem = metrics.get("memory_usage_pct",  0)
    req = metrics.get("request_count",     0)

    issues  = []
    actions = []

    # ── Response time ─────────────────────────────────────────────
    if rt > METRIC_THRESHOLDS["response_time_ms"]["critical"]:
        issues.append(f"Critical latency spike ({rt:.0f}ms > 2000ms threshold)")
        actions.append("Check DB connection pool and downstream service health")
    elif rt > METRIC_THRESHOLDS["response_time_ms"]["high"]:
        issues.append(f"High latency ({rt:.0f}ms > 800ms threshold)")
        actions.append("Investigate slow queries and network latency")
    elif rt > METRIC_THRESHOLDS["response_time_ms"]["warn"]:
        issues.append(f"Elevated latency ({rt:.0f}ms > 400ms threshold)")
        actions.append("Monitor latency trend — check for slow queries")

    # ── Error rate ────────────────────────────────────────────────
    if er > METRIC_THRESHOLDS["error_rate"]["critical"]:
        issues.append(f"Critical error rate ({er:.1%} > 30% threshold)")
        actions.append("Check application logs for exceptions — consider rollback")
    elif er > METRIC_THRESHOLDS["error_rate"]["high"]:
        issues.append(f"High error rate ({er:.1%} > 10% threshold)")
        actions.append("Review recent deployments and dependency health")
    elif er > METRIC_THRESHOLDS["error_rate"]["warn"]:
        issues.append(f"Elevated error rate ({er:.1%} > 5% threshold)")
        actions.append("Monitor error trend and check logs")

    # ── CPU ───────────────────────────────────────────────────────
    if cpu > METRIC_THRESHOLDS["cpu_usage_pct"]["critical"]:
        issues.append(f"CPU exhaustion ({cpu:.0f}% > 90% threshold)")
        actions.append("Identify runaway process via top/htop — enable auto-scaling")
    elif cpu > METRIC_THRESHOLDS["cpu_usage_pct"]["high"]:
        issues.append(f"High CPU usage ({cpu:.0f}% > 75% threshold)")
        actions.append("Check for CPU-intensive operations — consider scaling out")

    # ── Memory ────────────────────────────────────────────────────
    if mem > METRIC_THRESHOLDS["memory_usage_pct"]["critical"]:
        issues.append(f"Memory pressure critical ({mem:.0f}% > 90% threshold)")
        actions.append("Check for memory leak via heap dump — restart service if OOM risk")
    elif mem > METRIC_THRESHOLDS["memory_usage_pct"]["high"]:
        issues.append(f"High memory usage ({mem:.0f}% > 80% threshold)")
        actions.append("Monitor heap growth — schedule maintenance window")

    # ── Traffic ───────────────────────────────────────────────────
    if req > METRIC_THRESHOLDS["request_count"]["spike"]:
        issues.append(f"Traffic spike ({req:.0f} req/s > 5000 threshold)")
        actions.append("Scale horizontally — enable rate limiting — check for DDoS")
    elif req < METRIC_THRESHOLDS["request_count"]["drop"] and req >= 0:
        issues.append(f"Traffic drop ({req:.0f} req/s < 50 threshold)")
        actions.append("Verify load balancer health and upstream routing")

    # ── Combined signals (cascading failures) ─────────────────────
    if er > 0.20 and rt > 1000:
        # Insert at front — most important
        issues.insert(0, "Multi-signal anomaly: high errors + high latency (cascading failure)")
        actions.insert(0, "Initiate incident response — likely cascading failure across services")

    if cpu > 85 and mem > 85:
        issues.insert(0, "Resource exhaustion: CPU and memory both critical")
        actions.insert(0, "Immediate scale-out required — OOM kill risk is imminent")

    # ── Fallback if nothing specific found ────────────────────────
    if not issues:
        hint   = "Anomalous pattern detected — specific metric not identified"
        action = "Review metrics dashboard — investigate recent deployments"
    else:
        hint   = " | ".join(issues)
        action = " | ".join(actions)

    return hint, action


# ==============================================================================
# SECTION 4: SEVERITY COMPUTATION
# ==============================================================================

def compute_severity(
    score: float,
    threshold: float
) -> Tuple[str, float]:
    """
    Maps anomaly score to severity + confidence.

    The IsolationForest gives a raw score.
    We need to convert it to a human-readable severity label.

    Logic:
      If score >= threshold  -> NORMAL (not an anomaly)
      If score < threshold:
        gap = threshold - score   (how far below threshold)
        gap > 0.30  -> CRITICAL   (very clearly anomalous)
        gap > 0.15  -> HIGH
        gap > 0.00  -> MEDIUM     (just crossed threshold)

    Confidence scales with gap:
      Larger gap = more confident it's an anomaly
      Just crossing threshold = 55% confidence (uncertain)
      Far below threshold = 95% confidence (certain)

    Returns:
      (severity_label, confidence_float)
    """
    if score >= threshold:
        return "normal", 0.0

    gap = threshold - score   # positive: how far below threshold

    if gap > SEVERITY_GAPS["critical"]:
        severity   = "critical"
        confidence = min(1.0, 0.95 + (gap - 0.30) * 0.1)

    elif gap > SEVERITY_GAPS["high"]:
        severity   = "high"
        confidence = 0.80 + (gap - 0.15) * 1.0

    else:
        severity   = "medium"
        confidence = 0.55 + gap * 1.5

    return severity, round(min(1.0, confidence), 4)


# ==============================================================================
# SECTION 5: ANOMALY SERVICE CLASS
# ==============================================================================

class AnomalyService:
    """
    Singleton service that loads anomaly_model.pkl once and
    detects anomalies in system metrics.

    Bundle contents:
      model       : trained IsolationForest
      scaler      : fitted StandardScaler (MUST use same scaler as training)
      feature_cols: list of feature names in correct order
      threshold   : optimal anomaly score threshold (found during training)
      metrics     : model performance info (F1, contamination)
      version     : "1.0.0"
    """

    def __init__(self):
        self._bundle = None
        self._load_model()

    # ── Model Loading ─────────────────────────────────────────────

    def _load_model(self):
        """
        Loads anomaly_model.pkl from disk into memory.
        Called once at startup.
        """
        model_path = Path(settings.anomaly_model_path)

        if not model_path.exists():
            print(
                f"  [AnomalyService] WARNING: Model not found at {model_path}\n"
                f"  Run: python train_models.py"
            )
            return

        try:
            with open(model_path, "rb") as f:
                self._bundle = pickle.load(f)

            print(
                f"  [AnomalyService] Loaded anomaly_model.pkl"
                f"  v{self._bundle['version']}"
                f"  | threshold={self._bundle['threshold']:.4f}"
                f"  | F1={self._bundle['metrics']['optimal_f1']:.3f}"
            )
        except Exception as e:
            print(f"  [AnomalyService] ERROR loading model: {e}")
            self._bundle = None

    def is_ready(self) -> bool:
        return self._bundle is not None

    def get_model_info(self) -> dict:
        if not self.is_ready():
            return {"status": "not_loaded", "version": None}
        return {
            "status":    "ready",
            "version":   self._bundle["version"],
            "threshold": self._bundle["threshold"],
            "metrics":   self._bundle["metrics"],
            "score_stats": self._bundle.get("score_stats", {}),
        }

    # ── Single Anomaly Detection ──────────────────────────────────

    def detect(self, input_data: MetricsInput) -> AnomalyResult:
        """
        Detects anomaly in ONE metrics snapshot.
        Called by POST /api/v1/logs/detect

        Args:
            input_data : MetricsInput with 5 system metrics

        Returns:
            AnomalyResult with full detection output

        Flow:
          MetricsInput
              -> build_feature_vector()    5 raw + 2 engineered = 7 features
              -> scaler.transform()        normalize to z-scores
              -> model.score_samples()     raw IsolationForest score
              -> compare vs threshold      is_anomaly bool
              -> compute_severity()        critical/high/medium/normal
              -> analyze_root_cause()      which metric spiked + action
              -> AnomalyResult
        """
        metrics_dict = {
            "response_time_ms":  input_data.response_time_ms,
            "error_rate":        input_data.error_rate,
            "cpu_usage_pct":     input_data.cpu_usage_pct,
            "memory_usage_pct":  input_data.memory_usage_pct,
            "request_count":     float(input_data.request_count),
        }

        # ── Fallback: rule-based only if model not loaded ─────────
        if not self.is_ready():
            print("  [AnomalyService] Model not loaded — using rule-based fallback")
            is_anomaly = self._rule_based_check(metrics_dict)
            hint, action = analyze_root_cause(metrics_dict, is_anomaly)
            return AnomalyResult(
                service            = input_data.service,
                is_anomaly         = is_anomaly,
                anomaly_score      = 0.0,
                severity           = "high" if is_anomaly else "normal",
                confidence         = 0.50  if is_anomaly else 0.0,
                root_cause_hint    = hint,
                recommended_action = action,
                metrics            = metrics_dict,
            )

        # ── ML-based detection ────────────────────────────────────
        model     = self._bundle["model"]
        scaler    = self._bundle["scaler"]
        threshold = self._bundle["threshold"]

        # Build + scale feature vector
        X        = build_feature_vector(metrics_dict)
        X_scaled = scaler.transform(X)

        # score_samples(): lower score = more anomalous
        # Returns array of shape (1,) — one score per sample
        score      = float(model.score_samples(X_scaled)[0])
        is_anomaly = score < threshold

        # Map score to severity + confidence
        severity, confidence = compute_severity(score, threshold)

        # Rule-based root cause explanation
        hint, action = analyze_root_cause(metrics_dict, is_anomaly)

        print(
            f"  [AnomalyService] {input_data.service}: "
            f"score={score:.4f} threshold={threshold:.4f} "
            f"anomaly={is_anomaly} severity={severity}"
        )

        return AnomalyResult(
            service            = input_data.service,
            is_anomaly         = is_anomaly,
            anomaly_score      = round(score, 4),
            severity           = severity,
            confidence         = confidence,
            root_cause_hint    = hint,
            recommended_action = action,
            metrics            = metrics_dict,
        )

    # ── Bulk Detection ────────────────────────────────────────────

    def detect_bulk(self, logs: List[MetricsInput]) -> List[AnomalyResult]:
        """
        Analyzes a batch of log entries.
        Used by POST /api/v1/logs/analyze

        Optimized: builds one feature matrix for all entries,
        then calls score_samples() once instead of N times.
        This is significantly faster for large batches.

        Returns ONLY anomalous results sorted by severity (critical first).
        Normal entries are filtered out to keep the response concise.
        """
        if not logs:
            return []

        # ── Batch processing (ML path) ────────────────────────────
        if self.is_ready():
            scaler    = self._bundle["scaler"]
            model     = self._bundle["model"]
            threshold = self._bundle["threshold"]

            # Build full feature matrix: shape (N, 7)
            metrics_list = []
            for log in logs:
                m = {
                    "response_time_ms": log.response_time_ms,
                    "error_rate":       log.error_rate,
                    "cpu_usage_pct":    log.cpu_usage_pct,
                    "memory_usage_pct": log.memory_usage_pct,
                    "request_count":    float(log.request_count),
                }
                metrics_list.append(m)

            # Build matrix for ALL entries at once
            X_all    = np.vstack([build_feature_vector(m) for m in metrics_list])
            X_scaled = scaler.transform(X_all)

            # Single model call for all N entries  (O(N log N) vs O(N^2))
            scores = model.score_samples(X_scaled)

            # Build results for anomalies only
            results = []
            for i, (log, score, m) in enumerate(zip(logs, scores, metrics_list)):
                score      = float(score)
                is_anomaly = score < threshold
                if not is_anomaly:
                    continue

                severity, confidence = compute_severity(score, threshold)
                hint, action         = analyze_root_cause(m, True)

                results.append(AnomalyResult(
                    service            = log.service,
                    is_anomaly         = True,
                    anomaly_score      = round(score, 4),
                    severity           = severity,
                    confidence         = confidence,
                    root_cause_hint    = hint,
                    recommended_action = action,
                    metrics            = m,
                ))

        else:
            # Fallback: process one by one
            results = [
                self.detect(log)
                for log in logs
                if self.detect(log).is_anomaly
            ]

        # Sort: critical first, then high, medium, low
        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        results.sort(key=lambda r: severity_order.get(r.severity, 4))

        print(
            f"  [AnomalyService] Bulk: {len(logs)} entries → "
            f"{len(results)} anomalies "
            f"({len(results)/max(1,len(logs)):.1%} rate)"
        )
        return results

    # ── Rule-based Fallback ───────────────────────────────────────

    def _rule_based_check(self, metrics: Dict[str, float]) -> bool:
        """
        Simple threshold-based anomaly detection.
        Used when ML model is not loaded.
        Less accurate but better than nothing.
        """
        return (
            metrics.get("error_rate",       0) > 0.20 or
            metrics.get("response_time_ms", 0) > 2000 or
            metrics.get("cpu_usage_pct",    0) > 90   or
            metrics.get("memory_usage_pct", 0) > 90
        )

    # ── Model Reload (continuous learning) ───────────────────────

    def reload_model(self):
        """
        Hot-reloads model from disk after retraining.
        Called by learning_service.py after new anomaly_model.pkl is written.
        """
        print("  [AnomalyService] Reloading model from disk...")
        self._bundle = None
        self._load_model()
        print("  [AnomalyService] Reload complete.")


# ==============================================================================
# SECTION 6: SINGLETON ACCESSOR
# ==============================================================================

_anomaly_service: Optional[AnomalyService] = None


def get_anomaly_service() -> AnomalyService:
    """
    Returns the singleton AnomalyService instance.

    Usage:
        from app.services.anomaly_service import get_anomaly_service
        svc    = get_anomaly_service()
        result = svc.detect(metrics_input)
    """
    global _anomaly_service
    if _anomaly_service is None:
        _anomaly_service = AnomalyService()
    return _anomaly_service