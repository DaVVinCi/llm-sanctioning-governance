from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal, Protocol


PunishmentMode = Literal["none", "broadcast", "targeted"]


class AgentLike(Protocol):
    agent_id: int

    def make_decision(self, round_num: int, observation: dict[str, Any]) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class PublicGoodsAction:
    contribution: float
    penalty_amount: float
    targets: list[int] = field(default_factory=list)
    original_targets: Any = field(default_factory=list)
    normalized_targets: list[int] = field(default_factory=list)
    target_repaired: bool = False
    penalty_repaired: bool = False
    action_valid: bool = True
    reasoning_text: str = ""
    raw_response: Any = None
    invalid_response: bool = False


class PublicGoodsGame:
    """Repeated public-goods game with configurable punishment mechanics.

    This class is intentionally independent of file paths, TXT reporting, and
    hard-coded model names. It returns structured Python data for downstream
    JSONL/CSV recorders.
    """

    VALID_PUNISHMENT_MODES = {"none", "broadcast", "targeted"}

    def __init__(
        self,
        agents: Iterable[AgentLike | Callable[[int, dict[str, Any]], dict[str, Any]]],
        punishment_mode: PunishmentMode = "broadcast",
        num_agents: int = 4,
        initial_endowment: float = 30,
        multiplier: float = 1.5,
        penalty_factor: float = 1.5,
        rounds: int = 10,
        progress_callback: Callable[[str, int, int | None], None] | None = None,
    ) -> None:
        if punishment_mode not in self.VALID_PUNISHMENT_MODES:
            raise ValueError(
                f"Unsupported punishment_mode={punishment_mode!r}. "
                f"Expected one of {sorted(self.VALID_PUNISHMENT_MODES)}."
            )

        self.agents = list(agents)
        if len(self.agents) != num_agents:
            raise ValueError(f"Expected {num_agents} agents, got {len(self.agents)}.")
        if num_agents <= 1:
            raise ValueError("num_agents must be greater than 1.")
        if rounds <= 0:
            raise ValueError("rounds must be positive.")

        self.punishment_mode = punishment_mode
        self.num_agents = num_agents
        self.initial_endowment = float(initial_endowment)
        self.multiplier = float(multiplier)
        self.penalty_factor = float(penalty_factor)
        self.rounds = int(rounds)
        self.progress_callback = progress_callback
        self.cumulative_scores = [0.0 for _ in range(num_agents)]
        self.history: list[dict[str, Any]] = []

        for index, agent in enumerate(self.agents, start=1):
            if hasattr(agent, "agent_id"):
                try:
                    setattr(agent, "agent_id", index)
                except Exception:
                    pass

    def play_game(self) -> dict[str, Any]:
        """Run all rounds and return a structured experiment result."""
        self.cumulative_scores = [0.0 for _ in range(self.num_agents)]
        self.history = []

        round_results: list[list[dict[str, Any]]] = []
        for round_num in range(1, self.rounds + 1):
            self._notify_progress("round_start", round_num)
            round_results.append(self.play_round(round_num))

        flat_records = [record for round_records in round_results for record in round_records]
        return {
            "config": {
                "punishment_mode": self.punishment_mode,
                "num_agents": self.num_agents,
                "initial_endowment": self.initial_endowment,
                "multiplier": self.multiplier,
                "penalty_factor": self.penalty_factor,
                "rounds": self.rounds,
            },
            "rounds": round_results,
            "records": flat_records,
            "final_scores": {
                agent_id: self.cumulative_scores[agent_id - 1]
                for agent_id in range(1, self.num_agents + 1)
            },
        }

    def play_round(self, round_num: int) -> list[dict[str, Any]]:
        """Run one round and return one structured record per agent."""
        actions = []
        for agent_index, agent in enumerate(self.agents, start=1):
            self._notify_progress("agent_start", round_num, agent_index)
            actions.append(self._get_action(agent, round_num))
            self._notify_progress("agent_done", round_num, agent_index)
        contributions = [action.contribution for action in actions]
        penalty_spends = self._effective_penalty_spends(actions)
        incoming_punishments = self._calculate_incoming_punishments(actions, penalty_spends)

        total_contribution = sum(contributions)
        dividend = self.multiplier * total_contribution / self.num_agents

        records: list[dict[str, Any]] = []
        for idx, action in enumerate(actions):
            agent_id = idx + 1
            remaining = self.initial_endowment - action.contribution
            payoff = remaining + dividend - penalty_spends[idx] - incoming_punishments[idx]
            payoff = self._round_number(payoff)
            self.cumulative_scores[idx] = self._round_number(self.cumulative_scores[idx] + payoff)

            record = {
                "round": round_num,
                "agent_id": agent_id,
                "contribution": self._round_number(action.contribution),
                "remaining": self._round_number(remaining),
                "dividend": self._round_number(dividend),
                "penalty_spend": self._round_number(penalty_spends[idx]),
                "incoming_punishment": self._round_number(incoming_punishments[idx]),
                "payoff": payoff,
                "cumulative_score": self.cumulative_scores[idx],
                "targets": list(action.targets),
                "original_targets": action.original_targets,
                "normalized_targets": list(action.normalized_targets),
                "target_repaired": bool(action.target_repaired),
                "penalty_repaired": bool(action.penalty_repaired),
                "action_valid": bool(action.action_valid),
                "reasoning_text": action.reasoning_text,
                "raw_response": action.raw_response,
                "invalid_response": bool(action.invalid_response),
            }
            records.append(record)

        self.history.extend(records)
        return records

    def build_observation(self, agent_id: int, round_num: int) -> dict[str, Any]:
        """Build the information visible to one agent before a round decision."""
        public_totals = self._public_totals_by_round(round_num)

        public_history = [
            {
                "round": row["round"],
                "agent_id": row["agent_id"],
                "contribution": row["contribution"],
                "total_contribution": public_totals[row["round"]],
                "public_dividend": row["dividend"],
            }
            for row in self.history
            if row.get("round", 0) < round_num
        ]

        private_history = []
        for row in self.history:
            if row.get("round", 0) >= round_num or row.get("agent_id") != agent_id:
                continue
            private_row = {
                "round": row["round"],
                "contribution": row["contribution"],
                "payoff": row["payoff"],
                "cumulative_score": row["cumulative_score"],
            }
            if self.punishment_mode != "none":
                private_row.update(
                    {
                        "penalty_spend": row["penalty_spend"],
                        "targets": list(row["targets"]),
                        "incoming_punishment": row["incoming_punishment"],
                    }
                )
            private_history.append(private_row)

        return {
            "agent_id": agent_id,
            "round": round_num,
            "punishment_mode": self.punishment_mode,
            "public_history": public_history,
            "private_history": private_history,
        }

    def _get_action(
        self,
        agent: AgentLike | Callable[[int, dict[str, Any]], dict[str, Any]],
        round_num: int,
    ) -> PublicGoodsAction:
        agent_id = self._agent_id(agent)
        observation = self.build_observation(agent_id, round_num)
        if callable(agent) and not hasattr(agent, "make_decision"):
            raw_action = agent(round_num, observation)
        else:
            raw_action = agent.make_decision(round_num, observation)  # type: ignore[union-attr]
        return self._normalize_action(raw_action, agent_id)

    def _notify_progress(self, event: str, round_num: int, agent_index: int | None = None) -> None:
        if self.progress_callback is not None:
            self.progress_callback(event, round_num, agent_index)

    def _agent_id(
        self,
        agent: AgentLike | Callable[[int, dict[str, Any]], dict[str, Any]],
    ) -> int:
        if hasattr(agent, "agent_id"):
            return int(getattr(agent, "agent_id"))
        return self.agents.index(agent) + 1

    def _public_totals_by_round(self, round_num: int) -> dict[int, float]:
        totals: dict[int, float] = {}
        for row in self.history:
            row_round = int(row.get("round", 0))
            if row_round < round_num:
                totals[row_round] = totals.get(row_round, 0.0) + self._coerce_float(row.get("contribution"))
        return {row_round: self._round_number(total) for row_round, total in totals.items()}

    def _normalize_action(self, raw_action: Any, agent_id: int) -> PublicGoodsAction:
        if not isinstance(raw_action, dict):
            raise TypeError("Agent action must be a dict.")

        contribution = self._coerce_float(raw_action.get("contribution", 0.0))
        contribution = max(0.0, min(self.initial_endowment, contribution))

        raw_penalty_amount = self._coerce_float(raw_action.get("penalty_amount", 0.0))
        penalty_amount = self._coerce_float(raw_action.get("penalty_amount", 0.0))
        penalty_amount = max(0.0, penalty_amount)

        original_targets = raw_action.get("targets", [])
        normalized_targets = self._normalize_targets(original_targets, agent_id)
        targets = list(normalized_targets)
        target_repaired = not self._targets_equivalent(original_targets, normalized_targets)
        penalty_repaired = raw_penalty_amount != penalty_amount

        if self.punishment_mode == "none":
            if penalty_amount != 0.0:
                penalty_repaired = True
            if targets:
                target_repaired = True
            penalty_amount = 0.0
            targets = []
        elif penalty_amount <= 0.0:
            if targets:
                target_repaired = True
            targets = []
        elif self.punishment_mode == "targeted" and not targets:
            penalty_amount = 0.0
            penalty_repaired = True

        normalized_targets = list(targets)
        reasoning_text = str(raw_action.get("reasoning_text", raw_action.get("reasoning", "")) or "")
        raw_response = raw_action.get("raw_response")
        invalid_response = bool(raw_action.get("invalid_response", False))

        return PublicGoodsAction(
            contribution=contribution,
            penalty_amount=penalty_amount,
            targets=targets,
            original_targets=original_targets,
            normalized_targets=normalized_targets,
            target_repaired=target_repaired,
            penalty_repaired=penalty_repaired,
            action_valid=not invalid_response and not target_repaired and not penalty_repaired,
            reasoning_text=reasoning_text,
            raw_response=raw_response,
            invalid_response=invalid_response,
        )

    def _normalize_targets(self, targets: Any, agent_id: int) -> list[int]:
        if targets is None:
            return []
        if not isinstance(targets, list):
            return []

        normalized: list[int] = []
        for target in targets:
            try:
                target_id = int(target)
            except (TypeError, ValueError):
                continue
            if target_id == agent_id:
                continue
            if 1 <= target_id <= self.num_agents and target_id not in normalized:
                normalized.append(target_id)
        return normalized

    @staticmethod
    def _targets_equivalent(original_targets: Any, normalized_targets: list[int]) -> bool:
        if not isinstance(original_targets, list):
            return normalized_targets == []
        try:
            original_as_ints = [int(target) for target in original_targets]
        except (TypeError, ValueError):
            return False
        return original_as_ints == normalized_targets

    def _effective_penalty_spends(self, actions: list[PublicGoodsAction]) -> list[float]:
        if self.punishment_mode == "none":
            return [0.0 for _ in actions]
        return [action.penalty_amount for action in actions]

    def _calculate_incoming_punishments(
        self,
        actions: list[PublicGoodsAction],
        penalty_spends: list[float],
    ) -> list[float]:
        incoming = [0.0 for _ in actions]

        if self.punishment_mode == "none":
            return incoming

        if self.punishment_mode == "broadcast":
            for punished_idx in range(self.num_agents):
                incoming[punished_idx] = sum(
                    self.penalty_factor * penalty_spends[punisher_idx]
                    for punisher_idx in range(self.num_agents)
                    if punisher_idx != punished_idx
                )
            return incoming

        if self.punishment_mode == "targeted":
            for punisher_idx, action in enumerate(actions):
                valid_targets = [
                    target_id
                    for target_id in action.targets
                    if target_id != punisher_idx + 1
                ]
                if not valid_targets:
                    continue

                share = penalty_spends[punisher_idx] / len(valid_targets)
                for target_id in valid_targets:
                    incoming[target_id - 1] += self.penalty_factor * share
            return incoming

        raise ValueError(f"Unsupported punishment_mode={self.punishment_mode!r}.")

    @staticmethod
    def _coerce_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _round_number(value: float) -> float:
        return round(float(value), 6)
