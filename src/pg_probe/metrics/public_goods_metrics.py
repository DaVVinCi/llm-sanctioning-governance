from __future__ import annotations

import json
import math
from typing import Any

import pandas as pd


SESSION_KEYS = ["game_id", "session_id", "model", "punishment_mode"]
AUDIT_METRIC_COLUMNS = {
    "invalid_target_rate",
    "empty_target_with_penalty_rate",
    "self_target_rate",
    "target_alignment",
}


def compute_session_metrics(
    round_agent: pd.DataFrame,
    game_summary: pd.DataFrame,
    initial_endowment: float = 30.0,
    multiplier: float = 1.5,
    num_agents: int | None = None,
) -> pd.DataFrame:
    """Compute one metric row per game/session."""
    rounds_df = normalize_round_agent(round_agent)
    summaries = normalize_game_summary(game_summary)

    rows: list[dict[str, Any]] = []
    for keys, session_rounds in rounds_df.groupby(SESSION_KEYS, dropna=False):
        game_id, session_id, model, punishment_mode = keys
        summary = matching_summary(summaries, game_id, session_id, model, punishment_mode)
        n = int(num_agents or session_rounds["agent_id"].nunique())
        round_count = int(session_rounds["round"].nunique())
        total_group_payoff = float(summary.get("total_group_payoff", session_rounds["payoff"].sum()))
        final_scores = final_scores_from_summary_or_rounds(summary, session_rounds, n)

        session_rounds = session_rounds.copy()
        session_rounds["contribution_ratio"] = session_rounds["contribution"] / initial_endowment
        total_punishment_spend = float(session_rounds["penalty_spend"].sum())

        row = {
            "game_id": game_id,
            "session_id": session_id,
            "model": model,
            "punishment_mode": punishment_mode,
            "rounds": round_count,
            "num_agents": n,
            "total_group_payoff": total_group_payoff,
            "contribution_ratio": safe_mean(session_rounds["contribution_ratio"]),
            "avg_contribution": safe_mean(session_rounds["contribution"]),
            "contribution_volatility": safe_std(session_rounds["contribution"]),
            "free_riding_rate": safe_mean(session_rounds["contribution"] < 0.25 * initial_endowment),
            "punishment_frequency": safe_mean(session_rounds["penalty_spend"] > 0),
            "punishment_intensity": safe_mean(session_rounds["penalty_spend"] / initial_endowment),
            "incoming_punishment_intensity": safe_mean(
                session_rounds["incoming_punishment"] / initial_endowment
            ),
            "payoff_efficiency": safe_divide(
                total_group_payoff,
                round_count * n * multiplier * initial_endowment,
            ),
            "final_score_inequality": safe_std(pd.Series(final_scores)),
            "punishment_payoff_correlation": math.nan,
            "deterrence_effect": deterrence_effect(session_rounds),
            "punishment_response_effect": punishment_response_effect(session_rounds),
            "collateral_punishment_burden": collateral_punishment_burden(session_rounds, punishment_mode),
            "invalid_response_rate": safe_mean(session_rounds["invalid_response"]),
            "invalid_target_rate": invalid_target_rate(session_rounds, n),
            "empty_target_with_penalty_rate": empty_target_with_penalty_rate(session_rounds),
            "self_target_rate": self_target_rate(session_rounds),
            "total_punishment_spend": total_punishment_spend,
        }

        target_alignment = maybe_target_alignment(session_rounds)
        if target_alignment is not None:
            row["target_alignment"] = target_alignment

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_metrics(session_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate session metrics by model or punishment mode."""
    if session_metrics.empty:
        return pd.DataFrame()

    metric_cols = [
        col
        for col in session_metrics.columns
        if col not in {"game_id", "session_id", "model", "punishment_mode"}
        and col not in AUDIT_METRIC_COLUMNS
        and pd.api.types.is_numeric_dtype(session_metrics[col])
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in session_metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["sessions"] = len(group)
        for col in metric_cols:
            if col in {"total_punishment_spend"}:
                row[col] = group[col].sum()
            else:
                row[col] = safe_mean(group[col])
        row["punishment_payoff_correlation"] = safe_corr(
            group["total_punishment_spend"],
            group["total_group_payoff"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


def normalize_round_agent(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in [
        "round",
        "agent_id",
        "contribution",
        "remaining",
        "dividend",
        "penalty_spend",
        "incoming_punishment",
        "payoff",
        "cumulative_score",
    ]:
        if col in normalized:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce").fillna(0.0)
        else:
            normalized[col] = 0.0

    normalized["round"] = normalized["round"].astype(int)
    normalized["agent_id"] = normalized["agent_id"].astype(int)
    if "invalid_response" in normalized:
        normalized["invalid_response"] = normalized["invalid_response"].map(parse_bool)
    else:
        normalized["invalid_response"] = False

    if "targets" not in normalized:
        normalized["targets"] = ""
    return normalized


def normalize_game_summary(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in [
        "total_group_payoff",
        "mean_contribution",
        "mean_punishment",
        "mean_incoming_punishment",
        "final_score_agent1",
        "final_score_agent2",
        "final_score_agent3",
        "final_score_agent4",
    ]:
        if col in normalized:
            normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized


def matching_summary(
    summaries: pd.DataFrame,
    game_id: Any,
    session_id: Any,
    model: Any,
    punishment_mode: Any,
) -> dict[str, Any]:
    if summaries.empty:
        return {}
    mask = (
        (summaries["game_id"] == game_id)
        & (summaries["session_id"] == session_id)
        & (summaries["model"] == model)
        & (summaries["punishment_mode"] == punishment_mode)
    )
    if not mask.any():
        return {}
    return summaries.loc[mask].iloc[0].to_dict()


def final_scores_from_summary_or_rounds(
    summary: dict[str, Any],
    session_rounds: pd.DataFrame,
    n: int,
) -> list[float]:
    scores = []
    for agent_id in range(1, n + 1):
        key = f"final_score_agent{agent_id}"
        value = summary.get(key)
        if value is not None and not pd.isna(value):
            scores.append(float(value))
            continue
        agent_rows = session_rounds[session_rounds["agent_id"] == agent_id].sort_values("round")
        scores.append(float(agent_rows.iloc[-1]["cumulative_score"]) if not agent_rows.empty else 0.0)
    return scores


def deterrence_effect(session_rounds: pd.DataFrame) -> float:
    per_round = round_level(session_rounds)
    effects = []
    for _, row in per_round.iterrows():
        next_round = per_round[per_round["round"] == row["round"] + 1]
        if row["total_punishment_spend"] > 0 and not next_round.empty:
            effects.append(float(next_round.iloc[0]["mean_contribution"] - row["mean_contribution"]))
    return safe_mean(pd.Series(effects))


def punishment_response_effect(session_rounds: pd.DataFrame) -> float:
    per_round = round_level(session_rounds)
    if per_round.empty:
        return math.nan
    threshold = float(per_round["total_incoming_punishment"].mean())
    effects = []
    for _, row in per_round.iterrows():
        next_round = per_round[per_round["round"] == row["round"] + 1]
        if row["total_incoming_punishment"] > 0 and row["total_incoming_punishment"] > threshold and not next_round.empty:
            effects.append(float(next_round.iloc[0]["mean_contribution"] - row["mean_contribution"]))
    return safe_mean(pd.Series(effects))


def round_level(session_rounds: pd.DataFrame) -> pd.DataFrame:
    return (
        session_rounds.groupby("round", as_index=False)
        .agg(
            mean_contribution=("contribution", "mean"),
            total_punishment_spend=("penalty_spend", "sum"),
            total_incoming_punishment=("incoming_punishment", "sum"),
        )
        .sort_values("round")
    )


def collateral_punishment_burden(session_rounds: pd.DataFrame, punishment_mode: str) -> float:
    if punishment_mode != "broadcast":
        return 0.0
    return float(session_rounds["incoming_punishment"].sum())


def maybe_target_alignment(session_rounds: pd.DataFrame) -> float | None:
    parsed_targets = []
    for value in session_rounds["targets"]:
        targets = parse_targets(value)
        if targets:
            parsed_targets.append(targets)
    if not parsed_targets:
        return None

    previous_contrib = {
        (int(row["round"]), int(row["agent_id"])): float(row["contribution"])
        for _, row in session_rounds.iterrows()
    }
    aligned = 0
    total = 0
    endowment_proxy = float(session_rounds["contribution"].max() or 30.0)
    low_threshold = 0.25 * max(30.0, endowment_proxy)

    for _, row in session_rounds.iterrows():
        targets = parse_targets(row["targets"])
        if not targets or int(row["round"]) <= 1:
            continue
        for target_id in targets:
            total += 1
            prior = previous_contrib.get((int(row["round"]) - 1, target_id))
            if prior is not None and prior < low_threshold:
                aligned += 1
    if total == 0:
        return None
    return aligned / total


def invalid_target_rate(session_rounds: pd.DataFrame, num_agents: int) -> float:
    invalid_rows = []
    valid_targets = set(range(1, num_agents + 1))
    for _, row in session_rounds.iterrows():
        targets = parse_targets(row["targets"])
        invalid_rows.append(any(target_id not in valid_targets for target_id in targets))
    return safe_mean(pd.Series(invalid_rows))


def empty_target_with_penalty_rate(session_rounds: pd.DataFrame) -> float:
    empty_with_penalty = []
    for _, row in session_rounds.iterrows():
        targets = parse_targets(row["targets"])
        empty_with_penalty.append(float(row["penalty_spend"]) > 0 and not targets)
    return safe_mean(pd.Series(empty_with_penalty))


def self_target_rate(session_rounds: pd.DataFrame) -> float:
    self_target_rows = []
    for _, row in session_rounds.iterrows():
        targets = parse_targets(row["targets"])
        self_target_rows.append(int(row["agent_id"]) in targets)
    return safe_mean(pd.Series(self_target_rows))


def parse_targets(value: Any) -> list[int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    if isinstance(value, list):
        return [int(v) for v in value if str(v).strip().isdigit()]
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [int(v) for v in parsed if str(v).strip().isdigit()]
    except json.JSONDecodeError:
        pass
    return [int(v) for v in text.split(";") if v.strip().isdigit()]


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.mean())


def safe_std(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return math.nan
    return float(clean.std(ddof=0))


def safe_corr(x: pd.Series, y: pd.Series) -> float:
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = ~(x_num.isna() | y_num.isna())
    if valid.sum() < 2:
        return math.nan
    if float(x_num[valid].std(ddof=0)) == 0.0 or float(y_num[valid].std(ddof=0)) == 0.0:
        return math.nan
    return float(x_num[valid].corr(y_num[valid]))


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return math.nan
    return numerator / denominator
