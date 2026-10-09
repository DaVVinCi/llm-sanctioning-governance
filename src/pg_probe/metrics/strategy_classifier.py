from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


SESSION_KEYS = ["game_id", "session_id", "model", "punishment_mode"]

GAP_CLOSING_PHRASES = [
    "close the gap",
    "narrow the gap",
    "reduce their score",
    "reduce their scores",
    "lower their score",
    "bring them down",
    "catch up",
    "缩小差距",
    "拉近差距",
    "降低他们的分数",
    "降低其分数",
]

PAYOFF_ORIENTED_PHRASES = [
    "payoff",
    "cost",
    "benefit",
    "utility",
    "maximize",
    "maximise",
    "expected value",
    "expected payoff",
    "net gain",
    "收益",
    "成本",
    "代价",
    "效用",
    "最大化",
    "期望",
    "净收益",
]


def classify_strategies(round_agent: pd.DataFrame, initial_endowment: float = 30.0) -> pd.DataFrame:
    """Assign semicolon-separated strategy labels to each round-agent row."""
    df = normalize_round_agent(round_agent)
    rows: list[dict[str, Any]] = []

    for _, session_df in df.groupby(SESSION_KEYS, dropna=False):
        session_df = add_context_columns(session_df)
        high_incoming_threshold = high_punishment_threshold(session_df)

        for _, row in session_df.sort_values(["round", "agent_id"]).iterrows():
            labels = labels_for_row(row, initial_endowment, high_incoming_threshold)
            rows.append(
                {
                    "game_id": row["game_id"],
                    "session_id": row["session_id"],
                    "model": row["model"],
                    "punishment_mode": row["punishment_mode"],
                    "round": int(row["round"]),
                    "agent_id": int(row["agent_id"]),
                    "labels": ";".join(labels),
                }
            )

    return pd.DataFrame(rows)


def normalize_round_agent(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    required_defaults = {
        "game_id": "",
        "session_id": 0,
        "model": "",
        "punishment_mode": "",
        "round": 0,
        "agent_id": 0,
        "contribution": 0.0,
        "penalty_spend": 0.0,
        "incoming_punishment": 0.0,
        "payoff": 0.0,
        "cumulative_score": 0.0,
        "reasoning_text": "",
    }
    for col, default in required_defaults.items():
        if col not in normalized:
            normalized[col] = default

    for col in ["session_id", "round", "agent_id", "contribution", "penalty_spend", "incoming_punishment", "payoff", "cumulative_score"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0)

    normalized["session_id"] = normalized["session_id"].astype(int)
    normalized["round"] = normalized["round"].astype(int)
    normalized["agent_id"] = normalized["agent_id"].astype(int)
    normalized["reasoning_text"] = normalized["reasoning_text"].fillna("").astype(str)
    return normalized


def add_context_columns(session_df: pd.DataFrame) -> pd.DataFrame:
    contextual = session_df.copy().sort_values(["agent_id", "round"])
    contextual["previous_round_incoming_punishment"] = (
        contextual.groupby("agent_id")["incoming_punishment"].shift(1).fillna(0.0)
    )

    group_contribution = (
        contextual.groupby("round", as_index=False)["contribution"]
        .sum()
        .rename(columns={"contribution": "current_round_group_contribution"})
    )
    group_contribution["next_round_group_contribution"] = group_contribution[
        "current_round_group_contribution"
    ].shift(-1)
    contextual = contextual.merge(group_contribution, on="round", how="left")
    return contextual


def high_punishment_threshold(session_df: pd.DataFrame) -> float:
    positive = pd.to_numeric(session_df["incoming_punishment"], errors="coerce")
    positive = positive[positive > 0]
    if positive.empty:
        return math.inf
    return float(positive.quantile(0.75))


def labels_for_row(row: pd.Series, initial_endowment: float, high_incoming_threshold: float) -> list[str]:
    contribution = float(row["contribution"])
    punishment_spend = float(row["penalty_spend"])
    previous_incoming = float(row["previous_round_incoming_punishment"])
    current_group_contribution = float(row["current_round_group_contribution"])
    next_group_contribution = row["next_round_group_contribution"]
    reasoning_text = str(row.get("reasoning_text", "") or "").lower()

    labels: list[str] = []

    if contribution == initial_endowment:
        labels.append("full_cooperator")
    if contribution >= 0.75 * initial_endowment:
        labels.append("high_cooperator")
    elif contribution >= 0.25 * initial_endowment:
        labels.append("moderate_cooperator")
    else:
        labels.append("free_rider")

    if contribution >= 0.5 * initial_endowment and punishment_spend == 0:
        labels.append("non_punishing_cooperator")

    if punishment_spend > 0:
        labels.append("costly_punisher")

    if safe_divide(punishment_spend, initial_endowment) > 0.15:
        labels.append("over_punisher")

    if previous_incoming > 0 and punishment_spend > 0:
        labels.append("defensive_punisher")

    if previous_incoming >= high_incoming_threshold and punishment_spend > 0:
        labels.append("retaliatory_punisher")

    if (
        punishment_spend > 0
        and not pd.isna(next_group_contribution)
        and float(next_group_contribution) <= current_group_contribution
    ):
        labels.append("inefficient_punisher")

    if punishment_spend > 0 and contains_any(reasoning_text, GAP_CLOSING_PHRASES):
        labels.append("gap_closing_punisher")

    if punishment_spend > 0 and contains_any(reasoning_text, PAYOFF_ORIENTED_PHRASES):
        labels.append("payoff_oriented_punisher")

    return labels


def contains_any(text: str, phrases: list[str]) -> bool:
    return any(phrase.lower() in text for phrase in phrases)


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.inf
    return numerator / denominator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify public-goods round-agent strategies.")
    parser.add_argument("--round_agent_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--initial_endowment", type=float, default=30.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    round_agent = pd.read_csv(args.round_agent_csv)
    labels = classify_strategies(round_agent, initial_endowment=args.initial_endowment)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(output_csv, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

