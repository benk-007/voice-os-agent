"""The agent loop: utterance -> T5 plan -> guardrails -> (optional) execution.

The model is called **once** per command: :meth:`VoiceOSAgent.plan` returns a
validated plan that can be previewed, confirmed and then executed without
re-running the model.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

from voice_os_agent.executor.actions import perform_execution
from voice_os_agent.safety.validator import needs_confirmation, validate_parsed

log = logging.getLogger(__name__)


@dataclass
class Plan:
    utterance: str
    raw: str
    parsed: dict | None
    confidence: float | None
    latency_ms: float
    status: str  # "ready" | "needs_confirmation" | "validation_failed"
    reason: str = "ok"
    result: dict = field(default_factory=dict)

    @property
    def intent(self) -> str | None:
        return (self.parsed or {}).get("intent")

    def to_dict(self) -> dict:
        return asdict(self)


class VoiceOSAgent:
    def __init__(self, parser=None) -> None:
        if parser is None:
            from voice_os_agent.nlu.parser import SemanticParser

            parser = SemanticParser()
        self.parser = parser

    def plan(self, utterance: str) -> Plan:
        """Parse and validate an utterance (no side effect on the OS)."""
        res = self.parser.parse(utterance)
        ok, reason, validated = validate_parsed(res.parsed)
        if not ok:
            status = "validation_failed"
        elif needs_confirmation(validated["intent"], res.confidence):
            status = "needs_confirmation"
        else:
            status = "ready"
        plan = Plan(utterance, res.raw, validated if ok else res.parsed, res.confidence, res.latency_ms, status, reason)
        log.info("plan %r -> %s (%s, conf=%s)", utterance, res.raw, status, plan.confidence)
        return plan

    def execute(self, plan: Plan, execute: bool = True, confirm: bool = False) -> Plan:
        """Execute a plan. Refused if validation failed, or if it needs confirmation and none was given."""
        if plan.status == "validation_failed":
            return plan
        if plan.status == "needs_confirmation" and not confirm:
            return plan
        result = perform_execution(plan.intent, plan.parsed, execute=execute, confirm=confirm)
        plan.result = result
        plan.status = {"ok": "executed", "simulated": "simulated"}.get(result.get("status"), "failed")
        return plan

    def run_and_execute(self, utterance: str, execute: bool = False, confirm: bool = False) -> dict:
        """Backward-compatible one-shot API of the original script."""
        return self.execute(self.plan(utterance), execute=execute, confirm=confirm).to_dict()
