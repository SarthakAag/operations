"""
scripts/generate_sample_data.py
================================================================================
PURPOSE:
  Generates two realistic training datasets:
    1. data/tickets.csv  -> 500 IT support tickets
    2. data/logs.json    -> 2000 system metric snapshots

HOW TO RUN:
  cd ai_support_ops
  python scripts/generate_sample_data.py
================================================================================
"""

import csv
import json
import random
import uuid
from collections import Counter
from datetime import datetime, timedelta
import os

# Create data folder if not exists
os.makedirs("data", exist_ok=True)


# ==============================================================================
# TICKET TEMPLATES
# Format: (title, description, category, priority, resolution)
# ==============================================================================

TICKET_TEMPLATES = [

    # ── CATEGORY: database ────────────────────────────────────────────────────
    (
        "Database connection timeout on {service}",
        "Users experiencing connection timeouts on {service}. "
        "Error logs show max_connections exceeded. Pool exhausted after {n} retries. "
        "JDBC pool showing all {n} connections in use with no idle slots.",
        "database", "P1",
        "Increased connection pool size from 50 to 200 in application config. "
        "Restarted DB service to clear stale connections. "
        "Added connection timeout=30s and validation query to pool settings."
    ),
    (
        "Slow query performance degrading {service} response times",
        "Query response times jumped from 50ms to {n}ms on {service}. "
        "EXPLAIN ANALYZE shows full table scan on users table. "
        "Missing index causing {n} row scan per request. DB CPU at 95 percent.",
        "database", "P2",
        "Added composite index on (user_id, created_at). "
        "Query time reduced to 45ms. Ran ANALYZE to update table statistics. "
        "Added slow query alerting at 200ms threshold."
    ),
    (
        "Database replication lag on {service} read replica",
        "Replication lag growing on read replica for {service}. Currently at {n} seconds. "
        "Read queries returning stale data to users. "
        "Primary write load very high since {time}.",
        "database", "P2",
        "Network saturation was causing replication lag. "
        "Prioritized replication traffic via QoS rules. "
        "Lag cleared in 20 minutes. Added replication lag alert at 30 seconds."
    ),
    (
        "Cache miss rate spike in {service} Redis cluster",
        "Redis cache miss rate jumped from 5 percent to {n} percent for {service}. "
        "Database load increased tenfold due to cache bypass. "
        "All read endpoints degraded. Response time now {n}ms.",
        "database", "P3",
        "Cache TTL was accidentally set to zero after config deployment. "
        "Reset TTL to 3600 seconds and ran cache warm-up job. "
        "Miss rate back to 4 percent within 15 minutes."
    ),
    (
        "Disk space critical on {service} database node",
        "Disk usage at {n} percent on {service} database production node. "
        "Write operations will fail when disk is full. "
        "Binary logs consuming 80GB with auto-purge disabled.",
        "database", "P1",
        "Purged binary logs older than 7 days, freed 90GB of space. "
        "Re-enabled auto-purge with 3-day retention policy. "
        "Extended EBS volume from 500GB to 1TB as permanent solution."
    ),
    (
        "Primary database failover triggered unexpectedly on {service}",
        "Automatic failover triggered on {service} primary database at {time}. "
        "Read replica promoted to primary. Application reconnecting. "
        "Root cause unknown. {n} transactions may be affected.",
        "database", "P1",
        "Primary node experienced kernel OOM and was killed by OS. "
        "Failover completed successfully in 45 seconds. "
        "Increased primary node memory from 16GB to 32GB to prevent recurrence."
    ),

    # ── CATEGORY: application ─────────────────────────────────────────────────
    (
        "API endpoint {endpoint} returning 500 internal server errors",
        "The {endpoint} endpoint throwing HTTP 500 errors after deployment at {time}. "
        "Affects {n} percent of all requests. "
        "Stack trace shows NullPointerException in payment processor.",
        "application", "P1",
        "Null pointer thrown when discount code field is absent in request. "
        "Added null check before accessing discount code value. "
        "Deployed hotfix. Error rate returned to zero percent."
    ),
    (
        "Login service authentication failures for {service}",
        "Users unable to login to {service} since {time}. "
        "Auth service returning HTTP 401 for all login attempts. "
        "Affects {n} users based on error log count.",
        "application", "P1",
        "JWT secret key rotation during deployment caused token verification failures. "
        "Redeployed auth-service with correct HMAC secret. "
        "All active sessions invalidated and users prompted to re-login."
    ),
    (
        "Memory leak detected in {service} application",
        "Heap memory growing steadily in {service} over {n} hours without releasing. "
        "Memory at 95 percent of 4GB allocation. OOM kill imminent. "
        "Heap dump shows large number of unclosed database connections.",
        "application", "P2",
        "Profiling identified unclosed database connections in request handler. "
        "Fixed connection cleanup in finally block. Deployed patch version. "
        "Scheduled rolling restart as immediate mitigation."
    ),
    (
        "Message queue backlog growing in {service}",
        "Kafka consumer lag grown to {n} unprocessed messages for {service}. "
        "Processing {n} minutes behind real-time. "
        "Downstream services reporting stale data and delayed notifications.",
        "application", "P2",
        "Consumer group had 3 instances for 8 partitions causing under-consumption. "
        "Scaled consumer instances to 8 to match partition count. "
        "Backlog cleared within 45 minutes. Auto-scaling configured."
    ),
    (
        "Scheduled batch job {job} failed to execute",
        "Nightly job {job} did not run at scheduled time {time}. "
        "Data pipeline stalled and reports delayed by 24 hours. "
        "No error in logs, job simply never triggered.",
        "application", "P3",
        "Cron expression broke after daylight saving time transition. "
        "Updated schedule and manually triggered job to catch up. "
        "Added job execution monitoring with alert on missed runs."
    ),
    (
        "{service} service returning incorrect calculation results",
        "Users reporting wrong totals from {service} billing calculation. "
        "Off by {n} percent in some edge cases. "
        "Issue started after version update deployed at {time}.",
        "application", "P2",
        "Floating point rounding error introduced in refactored calculation method. "
        "Reverted to integer arithmetic with explicit decimal handling. "
        "Deployed fix and re-processed affected transactions."
    ),
    (
        "Third party API integration timing out in {service}",
        "Integration with external API inside {service} returning timeout. "
        "Error rate at {n} percent for affected workflow. "
        "Circuit breaker not triggering despite repeated failures.",
        "application", "P3",
        "External API rate limit hit due to missing exponential backoff. "
        "Added retry logic with exponential backoff and jitter. "
        "Configured circuit breaker with proper threshold settings."
    ),

    # ── CATEGORY: infrastructure ──────────────────────────────────────────────
    (
        "High CPU usage on {service} server",
        "CPU utilization above 90 percent for {n} minutes on {service}. "
        "Memory also at 85 percent. All user-facing operations degraded. "
        "Top process shows {service} worker consuming 94 percent CPU.",
        "infrastructure", "P2",
        "Identified runaway worker process stuck in infinite retry loop. "
        "Killed process and restarted service gracefully. "
        "Added auto-scaling rule to provision new instance at 75 percent CPU."
    ),
    (
        "{service} pod crash loop in Kubernetes cluster",
        "Kubernetes pod for {service} in CrashLoopBackOff state. "
        "Restarting every {n} seconds. Events show OOMKilled reason. "
        "Pod has restarted {n} times in the last hour.",
        "infrastructure", "P1",
        "Memory limit of 512Mi was too low for current workload. "
        "Updated deployment memory limit to 2Gi and request to 1Gi. "
        "Pod stable after update. Added memory usage alert at 80 percent."
    ),
    (
        "Load balancer health check failures on {service}",
        "Load balancer reporting {n} of 10 instances unhealthy for {service}. "
        "Traffic unevenly distributed across remaining instances. "
        "Some users intermittently getting connection errors.",
        "infrastructure", "P2",
        "Health check path changed in deployment but ALB config was not updated. "
        "Updated target group health check path from /ping to /health. "
        "All 10 instances healthy again. Added health check path to deploy checklist."
    ),
    (
        "SSL certificate expiring for {service} domain in {n} days",
        "SSL certificate for {service} production domain expires in {n} days. "
        "Users will see browser security warning after expiry. "
        "Certificate was manually provisioned without auto-renewal setup.",
        "infrastructure", "P2",
        "Renewed certificate via Let's Encrypt certbot. "
        "Configured auto-renewal cron job. "
        "Added 30-day expiry alert to monitoring."
    ),
    (
        "Deployment pipeline failing for {service}",
        "CI/CD pipeline for {service} failing at docker build stage. "
        "Build has failed {n} times since {time}. "
        "All deployments blocked. No new releases possible.",
        "infrastructure", "P2",
        "Base Docker image update broke build due to deprecated OS package. "
        "Pinned base image to previous working version. "
        "Added image version locking policy to prevent silent updates."
    ),

    # ── CATEGORY: network ─────────────────────────────────────────────────────
    (
        "Network packet loss between {service} and database cluster",
        "Intermittent {n} percent packet loss between app servers and DB cluster. "
        "Latency increased from 2ms to 450ms on all database queries. "
        "Traceroute shows drops at node-3 hop.",
        "network", "P1",
        "Faulty NIC on node-3 causing packet drops at hardware level. "
        "Replaced NIC. Network bonding handled failover automatically. "
        "No data loss. Latency back to 1.8ms average."
    ),
    (
        "CDN cache not invalidating for {service} after deployment",
        "Users seeing stale content {n} hours after deployment to {service}. "
        "CloudFront serving old version from edge cache. "
        "Cache-Control header set to max-age=86400 causing long TTL.",
        "network", "P3",
        "Manually purged CloudFront cache via invalidation API. "
        "Reduced Cache-Control to max-age=300 for HTML responses. "
        "Added cache invalidation step to deployment pipeline."
    ),
    (
        "SMTP outbound email delivery failures from {service}",
        "Transactional emails not delivered from {service} since {time}. "
        "SMTP server returning 550 rejection error. "
        "{n} emails queued and undelivered.",
        "network", "P3",
        "Sending IP blacklisted after bulk campaign triggered spam filters. "
        "Requested removal from Spamhaus blacklist. "
        "Added SPF, DKIM, DMARC records. Implemented hourly send rate limit."
    ),
    (
        "VPN connectivity issues for remote access to {service}",
        "Remote engineers unable to connect to internal {service} via VPN. "
        "Connection drops after {n} seconds. "
        "Issue started at {time} affecting {n} users.",
        "network", "P2",
        "VPN gateway certificate expired causing TLS handshake failure. "
        "Renewed gateway certificate and restarted VPN service. "
        "Configured certificate expiry monitoring for 30-day advance warning."
    ),

    # ── CATEGORY: security ────────────────────────────────────────────────────
    (
        "Brute force login attempts detected against {service}",
        "Brute force pattern detected on {service} login endpoint. "
        "{n} failed attempts in 5 minutes from same IP range. "
        "Some attempts using valid usernames obtained from prior data breach.",
        "security", "P1",
        "Blocked offending IP range at WAF level immediately. "
        "Enforced MFA for all accounts that received failed attempts. "
        "Security team notified. SIEM correlation rule updated."
    ),
    (
        "Critical security vulnerability found in {service} dependency",
        "CVE-2024-{n} found in third-party library used by {service}. "
        "CVSS score 8.5 High severity. Remote code execution possible. "
        "Patch available in next minor version release.",
        "security", "P2",
        "Updated vulnerable dependency to patched version immediately. "
        "Re-ran security scan. No other critical CVEs found. "
        "No evidence of exploitation found in access logs."
    ),
    (
        "Unauthorized data access attempt detected in {service}",
        "Audit logs show user account accessing {n} records outside normal patterns. "
        "Access at {time} which is outside business hours. "
        "Data accessed includes PII fields not normally required for this role.",
        "security", "P1",
        "Suspended user account pending investigation. "
        "Exported full data access logs for forensic review. "
        "Security team and DPO notified per GDPR breach response policy."
    ),
    (
        "API keys exposed in {service} public repository",
        "Production API keys for {service} found committed to public GitHub repo. "
        "Keys have been publicly visible for {n} hours. "
        "Potential unauthorized access to external services.",
        "security", "P1",
        "Rotated all exposed API keys immediately. "
        "Audited API usage logs for unauthorized calls. "
        "Added git-secrets pre-commit hook to prevent future exposure."
    ),
]

# ==============================================================================
# VARIABLE POOLS for template substitution
# ==============================================================================

SERVICES = [
    "payment-svc", "user-svc", "order-svc", "inventory-svc",
    "auth-svc", "notification-svc", "analytics-svc", "api-gateway",
    "search-svc", "billing-svc", "reporting-svc", "checkout-svc"
]

ENDPOINTS = [
    "/api/v1/payment/charge", "/api/v1/users/login",
    "/api/v1/orders/create", "/api/v1/checkout/process",
    "/api/v1/inventory/update", "/api/v1/billing/invoice"
]

JOBS = [
    "daily-report-generator", "data-sync-job",
    "nightly-cleanup", "invoice-batch-processor",
    "analytics-aggregator", "db-backup-job"
]

STATUSES       = ["resolved", "resolved", "closed"]
TEAMS          = ["team-infra", "team-app", "team-security", "team-network", "team-db"]
VIP_REPORTERS  = ["enterprise@bigcorp.com", "vip-exec@company.com"]
NORMAL_REPORTERS = [f"engineer{i}@company.com" for i in range(1, 20)]


# ==============================================================================
# TICKET GENERATOR
# ==============================================================================

def generate_ticket() -> dict:
    """
    Creates one realistic ticket:
      1. Pick a random template
      2. Substitute all {placeholders} with random values
      3. Add timestamps, team assignments, VIP flag
    """
    title_tmpl, desc_tmpl, category, priority, resolution = random.choice(TICKET_TEMPLATES)

    # Build substitution map
    subs = {
        "{service}":  random.choice(SERVICES),
        "{endpoint}": random.choice(ENDPOINTS),
        "{job}":      random.choice(JOBS),
        "{time}":     f"{random.randint(0, 23):02d}:{random.randint(0, 59):02d}",
        "{n}":        str(random.randint(10, 999)),
    }

    title       = title_tmpl
    description = desc_tmpl
    for placeholder, value in subs.items():
        title       = title.replace(placeholder, value)
        description = description.replace(placeholder, value)

    # Timestamps
    created_at = datetime.now() - timedelta(days=random.randint(1, 365))
    resolution_minutes = {
        "P1": random.randint(15, 180),
        "P2": random.randint(60, 480),
        "P3": random.randint(120, 1440),
        "P4": random.randint(480, 4320),
    }[priority]
    resolved_at = created_at + timedelta(minutes=resolution_minutes)

    is_vip   = random.random() < 0.10
    reporter = random.choice(VIP_REPORTERS if is_vip else NORMAL_REPORTERS)

    return {
        "ticket_id":                f"TKT-{uuid.uuid4().hex[:8].upper()}",
        "title":                    title,
        "description":              description,
        "category":                 category,
        "priority":                 priority,
        "status":                   random.choice(STATUSES),
        "assigned_to":              random.choice(TEAMS),
        "reporter":                 reporter,
        "is_vip":                   is_vip,
        "resolution":               resolution,
        "created_at":               created_at.isoformat(),
        "resolved_at":              resolved_at.isoformat(),
        "resolution_time_minutes":  resolution_minutes,
        "reopen_count":             random.choices([0, 1, 2], weights=[0.80, 0.15, 0.05])[0],
    }


# ==============================================================================
# LOG / METRICS GENERATOR
# ==============================================================================
#
# NORMAL baseline (from typical production observations):
#   response_time : Gaussian(mean=120ms,  std=30ms)
#   error_rate    : Gaussian(mean=1%,     std=0.5%)
#   cpu_usage     : Gaussian(mean=45%,    std=10%)
#   memory_usage  : Gaussian(mean=55%,    std=10%)
#   request_count : Gaussian(mean=1000/s, std=200)
#
# ANOMALY types — each type spikes DIFFERENT metrics:
#
#   high_latency        -> response_time spikes to ~2500ms
#                          Real cause: slow DB query, downstream timeout
#
#   high_error_rate     -> error_rate spikes to ~45%
#                          Real cause: bad deployment, config error
#
#   resource_exhaustion -> cpu + memory both near 90-95%
#                          Real cause: memory leak, runaway process
#
#   traffic_spike       -> request_count spikes to ~8000/s
#                          Real cause: DDoS, viral event, bot traffic
# ==============================================================================

ANOMALY_TYPES = ["high_latency", "high_error_rate", "resource_exhaustion", "traffic_spike"]


def generate_log_entry(timestamp: datetime, service: str, is_anomaly: bool) -> dict:
    """
    Generates one system metrics snapshot.
    Normal entries cluster around baseline. Anomalies have spiked metrics.
    """

    # ── Normal baseline metrics ───────────────────────────────────
    response_time = random.gauss(120, 30)
    error_rate    = random.gauss(0.01, 0.005)
    cpu           = random.gauss(45, 10)
    memory        = random.gauss(55, 10)
    requests      = random.gauss(1000, 200)

    # ── Inject anomaly patterns ───────────────────────────────────
    anomaly_type = None
    if is_anomaly:
        anomaly_type = random.choice(ANOMALY_TYPES)

        if anomaly_type == "high_latency":
            response_time = random.gauss(2500, 500)   # 20x normal
            error_rate    = random.gauss(0.15, 0.05)  # errors follow latency

        elif anomaly_type == "high_error_rate":
            error_rate    = random.gauss(0.45, 0.10)  # 45x normal
            response_time = random.gauss(800, 200)    # some latency too

        elif anomaly_type == "resource_exhaustion":
            cpu           = random.gauss(92, 5)       # near 100%
            memory        = random.gauss(95, 3)       # near 100%
            response_time = random.gauss(600, 100)    # slower under pressure

        elif anomaly_type == "traffic_spike":
            requests      = random.gauss(8000, 1000)  # 8x normal
            cpu           = random.gauss(88, 8)       # CPU rises with traffic
            response_time = random.gauss(400, 100)    # slight latency increase

    # ── Log level — more WARN/ERROR during anomalies ──────────────
    level_weights = (
        [0.70, 0.20, 0.07, 0.03] if not is_anomaly
        else [0.15, 0.20, 0.35, 0.30]
    )
    level = random.choices(["INFO", "DEBUG", "WARN", "ERROR"], weights=level_weights)[0]

    messages = {
        "INFO":  [
            f"Request processed in {int(response_time)}ms",
            f"Health check OK",
            f"Cache hit ratio {random.randint(85, 99)} percent",
        ],
        "DEBUG": [
            f"DB query executed in {random.randint(5, 50)}ms",
            f"Token validated for user_id {random.randint(1000, 9999)}",
        ],
        "WARN":  [
            f"Response time {int(response_time)}ms exceeds 500ms threshold",
            f"Connection pool at {random.randint(70, 90)} percent capacity",
            f"Retry attempt {random.randint(1, 3)} for downstream service",
        ],
        "ERROR": [
            f"Connection timeout after {int(response_time)}ms",
            f"NullPointerException in PaymentHandler",
            f"DB connection refused — pool exhausted",
            f"HTTP 500 on {random.choice(ENDPOINTS)}",
        ],
    }

    return {
        "log_id":       str(uuid.uuid4()),
        "timestamp":    timestamp.isoformat(),
        "service":      service,
        "level":        level,
        "message":      random.choice(messages[level]),
        "metrics": {
            "response_time_ms":  max(0.0,   round(response_time, 2)),
            "error_rate":        min(1.0, max(0.0, round(error_rate, 4))),
            "cpu_usage_pct":     min(100.0, max(0.0, round(cpu, 2))),
            "memory_usage_pct":  min(100.0, max(0.0, round(memory, 2))),
            "request_count":     max(0,     int(requests)),
        },
        "is_anomaly":    is_anomaly,
        "anomaly_type":  anomaly_type,  # None for normal entries
        "host":          f"node-{random.randint(1, 5)}.{service}.internal",
        "trace_id":      uuid.uuid4().hex[:16],
    }


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("=" * 60)
    print("  AI Support Ops — Sample Data Generator")
    print("=" * 60)

    # ── 1. Generate tickets.csv ───────────────────────────────────
    print("\n[1/2] Generating data/tickets.csv ...")

    NUM_TICKETS = 500
    tickets = [generate_ticket() for _ in range(NUM_TICKETS)]

    cat_counts = Counter(t["category"] for t in tickets)
    pri_counts = Counter(t["priority"] for t in tickets)

    print(f"  Total tickets  : {NUM_TICKETS}")
    print(f"  Categories     : {dict(cat_counts)}")
    print(f"  Priorities     : {dict(pri_counts)}")

    fieldnames = list(tickets[0].keys())
    with open("data/tickets.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(tickets)

    print("  Saved -> data/tickets.csv")

    # ── 2. Generate logs.json ─────────────────────────────────────
    print("\n[2/2] Generating data/logs.json ...")

    NUM_LOGS     = 2000
    ANOMALY_RATE = 0.10   # 10% anomalies

    logs       = []
    start_time = datetime.now() - timedelta(days=30)

    for i in range(NUM_LOGS):
        timestamp  = start_time + timedelta(minutes=i * 20)
        service    = random.choice(SERVICES)
        is_anomaly = random.random() < ANOMALY_RATE
        logs.append(generate_log_entry(timestamp, service, is_anomaly))

    anomaly_count = sum(1 for l in logs if l["is_anomaly"])
    type_counts   = Counter(l["anomaly_type"] for l in logs if l["is_anomaly"])

    print(f"  Total entries  : {NUM_LOGS}")
    print(f"  Normal entries : {NUM_LOGS - anomaly_count}")
    print(f"  Anomalies      : {anomaly_count}  ({anomaly_count / NUM_LOGS:.1%})")
    print(f"  Anomaly types  : {dict(type_counts)}")

    with open("data/logs.json", "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2)

    print("  Saved -> data/logs.json")

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Sample data generation complete!")
    print("  Next step -> python train_models.py")
    print("=" * 60)


if __name__ == "__main__":
    main()