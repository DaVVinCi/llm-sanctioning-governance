from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any


@dataclass
class FixedContributionAgent:
    agent_id: int
    contribution: float = 30.0
    penalty_amount: float = 0.0
    targets: list[int] | None = None

    def make_decision(self, round_num: int, observation: dict[str, Any]) -> dict[str, Any]:
        return {
            "contribution": self.contribution,
            "penalty_amount": self.penalty_amount,
            "targets": list(self.targets or []),
            "reasoning_text": "fixed rule",
            "raw_response": None,
        }


@dataclass
class MockLLMAgent:
    """Deterministic offline agent for smoke tests and README demos."""

    agent_id: int
    model: str = "mock"
    punishment_mode: str = "broadcast"
    local_history: list[dict[str, Any]] | None = None

    def reset(self) -> None:
        self.local_history = []

    def make_decision(self, round_num: int, observation: dict[str, Any]) -> dict[str, Any]:
        self.local_history = list(observation.get("private_history", []))
        contribution = self._contribution(round_num)
        penalty_amount = self._penalty(round_num)
        targets = self._targets() if self.punishment_mode == "targeted" and penalty_amount > 0 else []
        reasoning_text = (
            f"mock policy: Agent {self.agent_id} contributes {contribution} "
            f"and spends {penalty_amount} under {self.punishment_mode} mode."
        )
        raw_response = json.dumps(
            {
                "contribution": contribution,
                "penalty_amount": penalty_amount,
                "targets": targets,
                "reasoning": reasoning_text,
            },
            ensure_ascii=False,
        )
        return {
            "contribution": contribution,
            "penalty_amount": penalty_amount,
            "targets": targets,
            "reasoning_text": reasoning_text,
            "raw_response": raw_response,
            "invalid_response": False,
        }

    def _contribution(self, round_num: int) -> float:
        base = 12 + 3 * self.agent_id
        if round_num % 3 == 0:
            base += 3
        return float(min(30, base))

    def _penalty(self, round_num: int) -> float:
        if self.punishment_mode == "none":
            return 0.0
        return 1.0 if round_num > 1 and self.agent_id in {2, 3} else 0.0

    def _targets(self) -> list[int]:
        return [4] if self.agent_id in {2, 3} else []
