from __future__ import annotations

import importlib.util
import math
from typing import Any

import numpy as np
import pandas as pd

from common import (
    INITIAL_ENDOWMENT,
    MULTIPLIER,
    NUM_AGENTS,
    OUTPUT_DIR,
    PENALTY_FACTOR,
    ROUNDS,
    bool_to_float,
    ensure_output_dir,
    load_all_round_agent,
    parse_targets,
    safe_divide,
    target_column,
)


PHENOTYPE_THRESHOLDS: dict[str, float] = {
    "stable_cooperator_min_cr": 0.55,
    "stable_cooperator_min_pe": 0.80,
    "free_rider_max_cr": 0.15,
    "punishment_restraint_max_pf": 0.05,
    "punishment_restraint_max_harm": 0.02,
    "punishment_restraint_min_pe": 0.75,
    "punitive_inefficient_min_pf": 0.15,
    "punitive_inefficient_min_harm": 0.05,
    "punitive_inefficient_max_pe": 0.75,
    "retaliatory_anti_social_min_rate": 0.40,
    "precise_norm_min_alignment": 0.85,
    "precise_norm_max_anti_social": 0.20,
    "precise_norm_min_pf": 0.05,
}

REFERENCE_VALUES: dict[str, dict[str, float]] = {
    "none": {
        "contribution_ratio": 0.248,
        "payoff_efficiency": 0.749,
        "punishment_frequency": 0.0,
    },
    "broadcast": {
        "contribution_ratio": 0.460,
        "payoff_efficiency": 0.739,
        "punishment_frequency": 0.179,
        "punishment_harm_per_agent_round": 0.100,
        "collateral_share": 0.593,
    },
    "targeted": {
        "contribution_ratio": 0.471,
        "payoff_efficiency": 0.757,
        "punishment_frequency": 0.124,
        "punishment_harm_per_agent_round": 0.060,
        "target_alignment": 0.856,
        "anti_social_punishment_rate": 0.254,
    },
}


def main() -> int:
    ensure_output_dir()
    df = load_all_round_agent()
    if df.empty:
        raise RuntimeError("No round_agent.csv files found under clean_analysis_data/.")

    df = add_analysis_columns(df)
    write_merged_round_agent(df)

    target_events = build_targeted_target_events(df)
    target_events.to_csv(OUTPUT_DIR / "targeted_target_events.csv", index=False, encoding="utf-8-sig")

    session_metrics = compute_session_metrics(df, target_events)
    session_metrics.to_csv(OUTPUT_DIR / "session_metrics.csv", index=False, encoding="utf-8-sig")

    condition_summary = summarize_sessions(session_metrics, ["condition"])
    condition_summary = add_reference_checks(condition_summary)
    condition_summary.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False, encoding="utf-8-sig")

    model_condition_summary = summarize_sessions(session_metrics, ["model", "condition"])
    model_condition_summary["behavioral_phenotypes"] = model_condition_summary.apply(phenotype_labels, axis=1)
    model_condition_summary.to_csv(OUTPUT_DIR / "model_condition_summary.csv", index=False, encoding="utf-8-sig")

    print("Metric computation complete.")
    return 0


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out = out.sort_values(["condition", "model", "game_id", "agent_id", "round"]).reset_index(drop=True)
    session_cols = ["condition", "model", "game_id"]
    agent_cols = session_cols + ["agent_id"]

    out["contribution_ratio_row"] = out["contribution"] / INITIAL_ENDOWMENT
    out["punishment_frequency_row"] = (out["penalty_spend"] > 0).astype(float)
    out["punishment_cost_row"] = out["penalty_spend"] / INITIAL_ENDOWMENT
    out["punishment_harm_row"] = out["incoming_punishment"] / INITIAL_ENDOWMENT

    out["next_contribution"] = out.groupby(agent_cols)["contribution"].shift(-1)
    out["delta_next_contribution"] = out["next_contribution"] - out["contribution"]
    out["has_next_round"] = out["next_contribution"].notna()
    out["prev_contribution"] = out.groupby(agent_cols)["contribution"].shift(1)

    prev_round = (
        out.groupby(session_cols + ["round"], as_index=False)["contribution"]
        .mean()
        .rename(columns={"contribution": "prev_round_mean_contribution"})
    )
    prev_round["round"] = prev_round["round"] + 1
    out = out.merge(prev_round, on=session_cols + ["round"], how="left")
    out["prev_low_contributor"] = out["prev_contribution"] < out["prev_round_mean_contribution"]
    out["prev_non_low_contributor"] = (~out["prev_low_contributor"]) & out["prev_contribution"].notna()
    return out


def write_merged_round_agent(df: pd.DataFrame) -> None:
    parquet_path = OUTPUT_DIR / "merged_round_agent.parquet"
    csv_path = OUTPUT_DIR / "merged_round_agent.csv"
    if parquet_path.exists():
        parquet_path.unlink()
    if csv_path.exists():
        csv_path.unlink()

    parquet_available = importlib.util.find_spec("pyarrow") is not None or importlib.util.find_spec("fastparquet") is not None
    if parquet_available:
        try:
            df.to_parquet(parquet_path, index=False)
            print(f"Wrote {parquet_path}")
            return
        except Exception as exc:
            print(f"Parquet write failed; falling back to CSV: {exc}")

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"Wrote {csv_path}")


def compute_session_metrics(df: pd.DataFrame, target_events: pd.DataFrame) -> pd.DataFrame:
    session_cols = ["condition", "model", "game_id"]
    rows: list[dict[str, Any]] = []
    for keys, group in df.groupby(session_cols, dropna=False):
        condition, model, game_id = keys
        row: dict[str, Any] = {
            "condition": condition,
            "model": model,
            "game_id": game_id,
            "session_id": first_value(group, "session_id"),
            "session_dir": first_value(group, "session_dir"),
            "rounds": int(group["round"].nunique()),
            "num_agents": int(group["agent_id"].nunique()),
            "rows": int(len(group)),
            "contribution_ratio": float(group["contribution_ratio_row"].mean()),
            "payoff_efficiency": safe_divide(group["payoff"].sum(), ROUNDS * NUM_AGENTS * MULTIPLIER * INITIAL_ENDOWMENT),
            "punishment_frequency": float(group["punishment_frequency_row"].mean()),
            "punishment_cost_per_agent_round": float(group["punishment_cost_row"].mean()),
            "punishment_harm_per_agent_round": float(group["punishment_harm_row"].mean()),
        }

        row["punishment_response"] = punishment_response(group)
        row["collateral_burden_share"] = collateral_burden_share(group)
        row["collateral_share"] = row["collateral_burden_share"] if condition == "broadcast" else math.nan
        rows.append(row)

    session_metrics = pd.DataFrame(rows)
    if not target_events.empty:
        event_summary = (
            target_events.groupby(session_cols, as_index=False)
            .agg(
                target_event_count=("target_agent_id", "count"),
                target_alignment=("target_alignment", "mean"),
                anti_social_punishment_rate=("anti_social_punishment", "mean"),
            )
        )
        session_metrics = session_metrics.merge(event_summary, on=session_cols, how="left")
    else:
        session_metrics["target_event_count"] = 0
        session_metrics["target_alignment"] = math.nan
        session_metrics["anti_social_punishment_rate"] = math.nan

    session_metrics["target_event_count"] = session_metrics["target_event_count"].fillna(0).astype(int)
    return session_metrics


def build_targeted_target_events(df: pd.DataFrame) -> pd.DataFrame:
    targeted = df[(df["condition"] == "targeted") & (df["round"] > 1) & (df["penalty_spend"] > 0)].copy()
    if targeted.empty:
        return pd.DataFrame()

    targets_col = target_column(targeted)
    prev_lookup = {
        (row.condition, row.model, row.game_id, int(row.round), int(row.agent_id)): float(row.prev_contribution)
        for row in targeted.itertuples()
        if not pd.isna(row.prev_contribution)
    }
    full_prev_lookup = {
        (row.condition, row.model, row.game_id, int(row.round), int(row.agent_id)): float(row.prev_contribution)
        for row in df.itertuples()
        if not pd.isna(row.prev_contribution)
    }
    prev_mean_lookup = {
        (row.condition, row.model, row.game_id, int(row.round)): float(row.prev_round_mean_contribution)
        for row in df[["condition", "model", "game_id", "round", "prev_round_mean_contribution"]]
        .dropna()
        .drop_duplicates()
        .itertuples(index=False)
    }

    events: list[dict[str, Any]] = []
    for row in targeted.itertuples():
        targets = parse_targets(getattr(row, targets_col, ""))
        if not targets:
            continue
        punisher_key = (row.condition, row.model, row.game_id, int(row.round), int(row.agent_id))
        punisher_prev = prev_lookup.get(punisher_key)
        prev_mean = prev_mean_lookup.get((row.condition, row.model, row.game_id, int(row.round)))
        allocated_cost = float(row.penalty_spend) / len(targets)
        allocated_harm = PENALTY_FACTOR * allocated_cost
        for target_id in targets:
            target_key = (row.condition, row.model, row.game_id, int(row.round), int(target_id))
            target_prev = full_prev_lookup.get(target_key)
            target_alignment = (
                target_prev < prev_mean
                if target_prev is not None and prev_mean is not None and not pd.isna(prev_mean)
                else math.nan
            )
            anti_social = (
                punisher_prev <= target_prev
                if punisher_prev is not None and target_prev is not None
                else math.nan
            )
            events.append(
                {
                    "condition": row.condition,
                    "model": row.model,
                    "game_id": row.game_id,
                    "session_id": row.session_id,
                    "round": int(row.round),
                    "punisher_agent_id": int(row.agent_id),
                    "target_agent_id": int(target_id),
                    "targets_count": len(targets),
                    "target_source_column": targets_col,
                    "penalty_spend": float(row.penalty_spend),
                    "allocated_punishment_cost": allocated_cost,
                    "allocated_punishment_harm": allocated_harm,
                    "punisher_prev_contribution": punisher_prev,
                    "target_prev_contribution": target_prev,
                    "prev_round_mean_contribution": prev_mean,
                    "target_alignment": bool_to_float(target_alignment),
                    "anti_social_punishment": bool_to_float(anti_social),
                    "reasoning_text": row.reasoning_text,
                }
            )
    return pd.DataFrame(events)


def punishment_response(group: pd.DataFrame) -> float:
    valid = group[group["has_next_round"]]
    if valid.empty:
        return math.nan
    punished = valid.loc[valid["incoming_punishment"] > 0, "delta_next_contribution"]
    unpunished = valid.loc[valid["incoming_punishment"] == 0, "delta_next_contribution"]
    if punished.empty or unpunished.empty:
        return math.nan
    return float(punished.mean() - unpunished.mean())


def collateral_burden_share(group: pd.DataFrame) -> float:
    exposed = group[(group["round"] > 1) & (group["incoming_punishment"] > 0)].copy()
    if exposed.empty:
        return math.nan
    denominator = exposed["incoming_punishment"].sum()
    if denominator <= 0:
        return math.nan
    numerator = exposed.loc[exposed["prev_non_low_contributor"], "incoming_punishment"].sum()
    return float(numerator / denominator)


def summarize_sessions(session_metrics: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = [
        "contribution_ratio",
        "payoff_efficiency",
        "punishment_frequency",
        "punishment_cost_per_agent_round",
        "punishment_harm_per_agent_round",
        "punishment_response",
        "collateral_share",
        "collateral_burden_share",
        "target_alignment",
        "anti_social_punishment_rate",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in session_metrics.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_cols, keys))
        row["sessions"] = int(len(group))
        row["models"] = int(group["model"].nunique()) if "model" in group.columns else math.nan
        row["target_event_count"] = int(group["target_event_count"].sum()) if "target_event_count" in group.columns else 0
        for col in metric_cols:
            row[col] = float(group[col].mean(skipna=True)) if col in group.columns else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def add_reference_checks(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    for condition, refs in REFERENCE_VALUES.items():
        mask = out["condition"] == condition
        for metric, reference in refs.items():
            if metric in out.columns:
                out.loc[mask, f"{metric}_reference"] = reference
                out.loc[mask, f"{metric}_reference_diff"] = out.loc[mask, metric] - reference
    return out


def phenotype_labels(row: pd.Series) -> str:
    labels: list[str] = []
    condition = str(row.get("condition", ""))
    cr = value(row, "contribution_ratio")
    pe = value(row, "payoff_efficiency")
    pf = value(row, "punishment_frequency")
    harm = value(row, "punishment_harm_per_agent_round")
    alignment = value(row, "target_alignment")
    anti = value(row, "anti_social_punishment_rate")

    if cr >= PHENOTYPE_THRESHOLDS["stable_cooperator_min_cr"] and pe >= PHENOTYPE_THRESHOLDS["stable_cooperator_min_pe"]:
        labels.append("stable_cooperator")
    if cr <= PHENOTYPE_THRESHOLDS["free_rider_max_cr"]:
        labels.append("free_rider")
    if (
        condition in {"broadcast", "targeted"}
        and pf <= PHENOTYPE_THRESHOLDS["punishment_restraint_max_pf"]
        and harm <= PHENOTYPE_THRESHOLDS["punishment_restraint_max_harm"]
        and pe >= PHENOTYPE_THRESHOLDS["punishment_restraint_min_pe"]
    ):
        labels.append("punishment_restraint")
    if (
        condition in {"broadcast", "targeted"}
        and pf >= PHENOTYPE_THRESHOLDS["punitive_inefficient_min_pf"]
        and harm >= PHENOTYPE_THRESHOLDS["punitive_inefficient_min_harm"]
        and pe < PHENOTYPE_THRESHOLDS["punitive_inefficient_max_pe"]
    ):
        labels.append("punitive_but_inefficient")
    if condition == "targeted" and anti >= PHENOTYPE_THRESHOLDS["retaliatory_anti_social_min_rate"]:
        labels.append("retaliatory_anti_social_punisher")
    if (
        condition == "targeted"
        and alignment >= PHENOTYPE_THRESHOLDS["precise_norm_min_alignment"]
        and anti <= PHENOTYPE_THRESHOLDS["precise_norm_max_anti_social"]
        and pf >= PHENOTYPE_THRESHOLDS["precise_norm_min_pf"]
    ):
        labels.append("precise_norm_enforcer")

    return ";".join(labels) if labels else "unclassified"


def first_value(df: pd.DataFrame, column: str) -> Any:
    if column not in df.columns or df.empty:
        return ""
    values = df[column].dropna()
    return values.iloc[0] if not values.empty else ""


def value(row: pd.Series, column: str) -> float:
    val = row.get(column, math.nan)
    return float(val) if not pd.isna(val) else math.nan


if __name__ == "__main__":
    raise SystemExit(main())
