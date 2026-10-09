from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pg_probe.llm.client import LLMClient


@dataclass
class LLMAgent:
    agent_id: int
    model: str
    punishment_mode: str
    temperature: float = 0.7
    prompt_dir: Path = Path("prompts")
    client: LLMClient = field(default_factory=LLMClient)
    local_history: list[dict[str, Any]] = field(default_factory=list)

    MODE_SPECIFIC_RULES: ClassVar[dict[str, str]] = {
        "none": (
            "There is no punishment phase in this condition.\n"
            "You can only affect the game through your contribution.\n"
            "The only decision field for this condition is contribution.\n"
            "Punishment-related fields are not part of this condition's decision task.\n"
            "Your payoff is your remaining endowment plus your share of the public-good dividend."
        ),
        "broadcast": (
            "After choosing your contribution, you may spend p points on punishment.\n"
            "If you spend p, you pay p as a cost.\n"
            "Every other agent loses 1.5 * p points.\n"
            "Targets, if provided, are treated only as intended targets and do not change payoff calculation.\n"
            "Because punishment affects all other agents, it may also harm agents who contributed highly."
        ),
        "targeted": (
            "After choosing your contribution, you may spend p points on punishment and select target agents.\n"
            "You pay p as a cost.\n"
            "Only selected targets lose punishment points.\n"
            "If multiple targets are selected, your punishment budget is divided equally among them.\n"
            "Each selected target loses 1.5 * p / number_of_targets points.\n"
            "Targets must be other existing agents only. Do not target yourself. "
            "Do not target nonexistent agents."
        ),
    }
    RESPONSE_SCHEMAS: ClassVar[dict[str, str]] = {
        "none": (
            "{\n"
            '  "contribution": number,\n'
            '  "reasoning": "short explanation"\n'
            "}"
        ),
        "broadcast": (
            "{\n"
            '  "contribution": number,\n'
            '  "penalty_amount": number,\n'
            '  "targets": [agent ids],\n'
            '  "reasoning": "short explanation"\n'
            "}"
        ),
        "targeted": (
            "{\n"
            '  "contribution": number,\n'
            '  "penalty_amount": number,\n'
            '  "targets": [agent ids],\n'
            '  "reasoning": "short explanation"\n'
            "}"
        ),
    }
    INFORMATION_RULES: ClassVar[dict[str, str]] = {
        "none": (
            "Information rule:\n"
            "You can observe previous public contribution outcomes.\n"
            "You cannot observe other agents' private feedback, payoff, cumulative score, "
            "raw responses, or reasoning.\n"
            "You cannot observe other agents' current-round decisions before making your own decision."
        ),
        "broadcast": (
            "Information rule:\n"
            "You can observe previous public contribution outcomes.\n"
            "You cannot observe other agents' private punishment actions, targets, punishment received, "
            "payoff, cumulative score, raw responses, or reasoning.\n"
            "You cannot observe other agents' current-round decisions before making your own decision."
        ),
        "targeted": (
            "Information rule:\n"
            "You can observe previous public contribution outcomes.\n"
            "You cannot observe other agents' private punishment actions, targets, punishment received, "
            "payoff, cumulative score, raw responses, or reasoning.\n"
            "You cannot observe other agents' current-round decisions before making your own decision."
        ),
    }

    def reset(self) -> None:
        self.local_history = []

    def make_decision(self, round_num: int, observation: dict[str, Any]) -> dict[str, Any]:
        self.local_history = list(observation.get("private_history", []))
        prompt = self._build_prompt(round_num, observation)
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            raw_response = response.content
        except Exception as exc:
            return self._invalid_action(raw_response=str(exc), reason=f"LLM call failed: {exc}")

        return self.parse_response(raw_response)

    def parse_response(self, raw_response: str) -> dict[str, Any]:
        try:
            parsed = self._extract_json(raw_response)
            if self.punishment_mode == "none":
                penalty_amount = 0.0
                targets: list[int] = []
            else:
                penalty_amount = parsed.get("penalty_amount", 0.0)
                targets = parsed.get("targets", [])
            return {
                "contribution": parsed.get("contribution", 0),
                "penalty_amount": penalty_amount,
                "targets": targets,
                "reasoning_text": str(
                    parsed.get("reasoning_text", parsed.get("reasoning", parsed.get("reason", "")))
                    or ""
                ),
                "raw_response": raw_response,
                "invalid_response": False,
            }
        except Exception as exc:
            return self._invalid_action(
                raw_response=raw_response,
                reason=f"Could not parse response as decision JSON: {exc}",
            )

    def _build_prompt(self, round_num: int, observation: dict[str, Any]) -> str:
        template = self._read_prompt("public_goods_decision_template.txt")
        punishment_mode = str(observation.get("punishment_mode", self.punishment_mode))
        public_history_text = self._format_public_history(observation.get("public_history", []))
        private_history_text = self._format_private_history(observation.get("private_history", []))
        return template.format(
            punishment_mode=punishment_mode,
            mode_specific_rule=self._mode_specific_rule(punishment_mode),
            response_schema=self._response_schema(punishment_mode),
            information_rule=self._information_rule(punishment_mode),
            round_num=observation.get("round", round_num),
            agent_id=observation.get("agent_id", self.agent_id),
            public_history=public_history_text,
            private_history=private_history_text,
        )

    def _format_public_history(self, public_history: list[dict[str, Any]]) -> str:
        if not public_history:
            return "No previous rounds."

        lines = []
        for row in public_history:
            lines.append(
                "Round {round}, Agent {agent_id}: contribution={contribution}, "
                "total_contribution={total_contribution}, public_dividend={public_dividend}".format(**row)
            )
        return "\n".join(lines)

    def _format_private_history(self, private_history: list[dict[str, Any]]) -> str:
        if not private_history:
            return "No previous private feedback."

        lines = []
        for row in private_history:
            if "penalty_spend" in row:
                lines.append(
                    "Round {round}: contribution={contribution}, penalty_spend={penalty_spend}, "
                    "targets={targets}, incoming_punishment={incoming_punishment}, payoff={payoff}, "
                    "cumulative_score={cumulative_score}".format(**row)
                )
            else:
                lines.append(
                    "Round {round}: contribution={contribution}, payoff={payoff}, "
                    "cumulative_score={cumulative_score}".format(**row)
                )
        return "\n".join(lines)

    def _mode_specific_rule(self, punishment_mode: str) -> str:
        try:
            return self.MODE_SPECIFIC_RULES[punishment_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported punishment_mode={punishment_mode!r}.") from exc

    def _response_schema(self, punishment_mode: str) -> str:
        try:
            return self.RESPONSE_SCHEMAS[punishment_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported punishment_mode={punishment_mode!r}.") from exc

    def _information_rule(self, punishment_mode: str) -> str:
        try:
            return self.INFORMATION_RULES[punishment_mode]
        except KeyError as exc:
            raise ValueError(f"Unsupported punishment_mode={punishment_mode!r}.") from exc

    def _read_prompt(self, filename: str) -> str:
        path = self.prompt_dir / filename
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _extract_json(text: str) -> dict[str, Any]:
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.S)
            if not match:
                raise
            data = json.loads(match.group(0))

        if not isinstance(data, dict):
            raise ValueError("top-level JSON value must be an object")
        return data

    @staticmethod
    def _invalid_action(raw_response: str, reason: str) -> dict[str, Any]:
        return {
            "contribution": 0,
            "penalty_amount": 0,
            "targets": [],
            "reasoning_text": reason,
            "raw_response": raw_response,
            "invalid_response": True,
        }
