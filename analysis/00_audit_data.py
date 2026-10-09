from __future__ import annotations

import math
from typing import Any

import pandas as pd

from common import (
    CONDITIONS,
    DATA_DIR,
    EXPECTED_MODELS,
    EXPECTED_SESSION_ROWS,
    EXPECTED_SESSIONS,
    EXPECTED_TOTAL_ROWS,
    FLOAT_TOLERANCE,
    INITIAL_ENDOWMENT,
    MULTIPLIER,
    NUM_AGENTS,
    OUTPUT_DIR,
    PENALTY_FACTOR,
    REQUIRED_SESSION_FILES,
    ROUNDS,
    ensure_output_dir,
    iter_session_dirs,
    load_json,
    normalize_round_agent,
    parse_targets,
    read_csv,
    target_column,
)


def main() -> int:
    ensure_output_dir()
    records: list[dict[str, Any]] = []
    sessions = list(iter_session_dirs(DATA_DIR))

    observed_conditions = sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()) if DATA_DIR.exists() else []
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "conditions_exact",
        ",".join(CONDITIONS),
        ",".join(observed_conditions),
        set(observed_conditions) == set(CONDITIONS) and len(observed_conditions) == len(CONDITIONS),
    )

    models = sorted({model for _, model, _, _ in sessions})
    add_record(records, "global", "", "", "", "", "model_count", EXPECTED_MODELS, len(models), len(models) == EXPECTED_MODELS)
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "session_count",
        EXPECTED_SESSIONS,
        len(sessions),
        len(sessions) == EXPECTED_SESSIONS,
    )

    total_rows = 0
    game_locations: dict[str, list[str]] = {}
    invalid_response_count = 0
    payoff_global_max_error = 0.0
    incoming_global_max_error = 0.0
    missing_files: list[str] = []

    for condition, model, session, session_path in sessions:
        missing = [name for name in REQUIRED_SESSION_FILES if not (session_path / name).exists()]
        missing_files.extend(str(session_path / name) for name in missing)
        add_record(
            records,
            "session",
            condition,
            model,
            session,
            "",
            "required_files_present",
            "all required files",
            ";".join(missing) if missing else "all present",
            not missing,
        )
        round_path = session_path / "round_agent.csv"
        if not round_path.exists():
            continue

        df = normalize_round_agent(read_csv(round_path))
        game_id = str(df["game_id"].dropna().iloc[0]) if "game_id" in df.columns and not df.empty else ""
        if game_id:
            game_locations.setdefault(game_id, []).append(f"{condition}/{model}/{session}")
        total_rows += len(df)
        invalid_response_count += int(df["invalid_response"].sum())

        add_record(
            records,
            "session",
            condition,
            model,
            session,
            game_id,
            "session_row_count",
            EXPECTED_SESSION_ROWS,
            len(df),
            len(df) == EXPECTED_SESSION_ROWS,
        )

        audit_config(records, condition, model, session, session_path, df, game_id)

        payoff_error = payoff_formula_error(df)
        payoff_global_max_error = max(payoff_global_max_error, payoff_error)
        add_record(
            records,
            "session",
            condition,
            model,
            session,
            game_id,
            "payoff_formula_max_abs_error",
            f"<={FLOAT_TOLERANCE}",
            f"{payoff_error:.12g}",
            payoff_error <= FLOAT_TOLERANCE,
        )

        incoming_error = incoming_punishment_error(df, condition)
        incoming_global_max_error = max(incoming_global_max_error, incoming_error)
        add_record(
            records,
            "session",
            condition,
            model,
            session,
            game_id,
            "incoming_punishment_formula_max_abs_error",
            f"<={FLOAT_TOLERANCE}",
            f"{incoming_error:.12g}",
            incoming_error <= FLOAT_TOLERANCE,
        )

    duplicate_game_ids = {gid: locs for gid, locs in game_locations.items() if len(locs) > 1}
    add_record(records, "global", "", "", "", "", "missing_files", 0, len(missing_files), len(missing_files) == 0, ";".join(missing_files[:20]))
    add_record(records, "global", "", "", "", "", "total_row_count", EXPECTED_TOTAL_ROWS, total_rows, total_rows == EXPECTED_TOTAL_ROWS)
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "duplicate_game_id_count",
        0,
        len(duplicate_game_ids),
        len(duplicate_game_ids) == 0,
        ";".join(list(duplicate_game_ids)[:20]),
    )
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "invalid_response_count",
        0,
        invalid_response_count,
        invalid_response_count == 0,
    )
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "payoff_formula_global_max_abs_error",
        f"<={FLOAT_TOLERANCE}",
        f"{payoff_global_max_error:.12g}",
        payoff_global_max_error <= FLOAT_TOLERANCE,
    )
    add_record(
        records,
        "global",
        "",
        "",
        "",
        "",
        "incoming_punishment_formula_global_max_abs_error",
        f"<={FLOAT_TOLERANCE}",
        f"{incoming_global_max_error:.12g}",
        incoming_global_max_error <= FLOAT_TOLERANCE,
    )

    audit = pd.DataFrame(records)
    audit.to_csv(OUTPUT_DIR / "audit_report.csv", index=False, encoding="utf-8-sig")
    failed = audit[~audit["passed"]]
    if failed.empty:
        print("Audit complete: all checks passed.")
    else:
        print(f"Audit complete: {len(failed)} checks failed. See outputs/audit_report.csv.")
    return 0


def audit_config(
    records: list[dict[str, Any]],
    condition: str,
    model: str,
    session: str,
    session_path,
    df: pd.DataFrame,
    game_id: str,
) -> None:
    config_path = session_path / "config.json"
    if not config_path.exists():
        return
    cfg = load_json(config_path)
    checks = {
        "config_model": (str(cfg.get("model")), unique_value(df, "model")),
        "config_punishment_mode": (str(cfg.get("punishment_mode")), unique_value(df, "punishment_mode")),
        "config_rounds": (int(cfg.get("rounds", -1)), int(df["round"].max()) if not df.empty else -1),
        "config_num_agents": (int(cfg.get("num_agents", -1)), int(df["agent_id"].nunique()) if not df.empty else -1),
        "config_game_id": (str(cfg.get("game_id")), unique_value(df, "game_id")),
    }
    for check_name, (expected, observed) in checks.items():
        add_record(records, "session", condition, model, session, game_id, check_name, expected, observed, expected == observed)


def unique_value(df: pd.DataFrame, column: str) -> str:
    if column not in df.columns or df.empty:
        return ""
    values = [str(value) for value in df[column].dropna().unique()]
    return values[0] if len(values) == 1 else "|".join(values)


def payoff_formula_error(df: pd.DataFrame) -> float:
    if df.empty:
        return math.nan
    round_total = df.groupby("round")["contribution"].transform("sum")
    expected = (
        INITIAL_ENDOWMENT
        - df["contribution"]
        + MULTIPLIER * round_total / NUM_AGENTS
        - df["penalty_spend"]
        - df["incoming_punishment"]
    )
    return float((expected - df["payoff"]).abs().max())


def incoming_punishment_error(df: pd.DataFrame, condition: str) -> float:
    if df.empty:
        return math.nan
    expected = recompute_incoming_punishment(df, condition)
    return float((expected - df["incoming_punishment"]).abs().max())


def recompute_incoming_punishment(df: pd.DataFrame, condition: str) -> pd.Series:
    expected = pd.Series(0.0, index=df.index)
    if condition == "none":
        return expected

    if condition == "broadcast":
        for _, group in df.groupby("round"):
            total_penalty = group["penalty_spend"].sum()
            expected.loc[group.index] = PENALTY_FACTOR * (total_penalty - group["penalty_spend"])
        return expected

    if condition == "targeted":
        targets_col = target_column(df)
        for _, group in df.groupby("round"):
            by_agent = {int(row.agent_id): row.Index for row in group[["agent_id"]].itertuples()}
            for idx, row in group.iterrows():
                targets = parse_targets(row.get(targets_col, ""))
                if not targets:
                    continue
                allocation = PENALTY_FACTOR * float(row["penalty_spend"]) / len(targets)
                for target_id in targets:
                    target_idx = by_agent.get(int(target_id))
                    if target_idx is not None:
                        expected.loc[target_idx] += allocation
        return expected

    return expected


def add_record(
    records: list[dict[str, Any]],
    scope: str,
    condition: str,
    model: str,
    session: str,
    game_id: str,
    check_name: str,
    expected: Any,
    observed: Any,
    passed: bool,
    details: str = "",
) -> None:
    records.append(
        {
            "scope": scope,
            "condition": condition,
            "model": model,
            "session": session,
            "game_id": game_id,
            "check_name": check_name,
            "expected": expected,
            "observed": observed,
            "passed": bool(passed),
            "details": details,
        }
    )


if __name__ == "__main__":
    raise SystemExit(main())
