"""Audit trail: one JSON line per request.

Logs what each guardrail decided, never the raw text. The question is stored
as a SHA-256 hash, so you can match repeat attacks without keeping PII in logs.
"""

import hashlib
import json
import time
import uuid
from pathlib import Path


class AuditLog:
    def __init__(self, path: str = "audit_log.jsonl"):
        self.path = Path(path)

    def record(self, question: str, events: list[dict], outcome: str) -> str:
        request_id = uuid.uuid4().hex[:12]
        entry = {
            "request_id": request_id,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "question_sha256": hashlib.sha256(question.encode()).hexdigest(),
            "outcome": outcome,  # answered | blocked_input | blocked_output
            "events": events,
        }
        with self.path.open("a") as f:
            f.write(json.dumps(entry) + "\n")
        return request_id
