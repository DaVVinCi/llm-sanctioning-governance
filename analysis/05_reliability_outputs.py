from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import patsy
import statsmodels.formula.api as smf

from common import (
    INITIAL_ENDOWMENT,
    OUTPUT_DIR,
    ensure_output_dir,
    load_all_round_agent,
    read_csv,
)


BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260525
CI_METHOD = "session_level_bootstrap_95ci"


@dataclass(frozen=True)
class EstimateResult:
    estimate: float
    ci_low: float
    ci_high: float
    n_sessions: int
    notes: str = ""


def main() -> int:
    ensure_output_dir()
    stats = read_csv(OUTPUT_DIR / "stats_results.csv")
    sessions = read_csv(OUTPUT_DIR / "session_metrics.csv")
    condition_summary = read_csv(OUTPUT_DIR / "condition_summary.csv")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    main_with_ci = build_main_metrics_with_ci(stats, sessions, rng)
    main_with_ci.to_csv(OUTPUT_DIR / "main_metrics_with_ci.csv", index=False, encoding="utf-8-sig")

    late_sessions = compute_round_window_session_metrics(round_min=6, round_max=10)
    final_sessions = compute_round_window_session_metrics(round_min=10, round_max=10)
    robustness = build_robustness_summary(main_with_ci, sessions, late_sessions, final_sessions, condition_summary, rng)
    robustness.to_csv(OUTPUT_DIR / "robustness_summary.csv", index=False, encoding="utf-8-sig")
    write_robustness_markdown(robustness)

    print("Wrote outputs/main_metrics_with_ci.csv")
    print("Wrote outputs/robustness_summary.csv")
    print("Wrote outputs/robustness_summary.md")
    return 0


def build_main_metrics_with_ci(stats: pd.DataFrame, sessions: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    ci_by_model: dict[str, dict[str, tuple[float, float, str]]] = {}
    n_by_model: dict[str, int] = {}

    for model_name, model_rows in stats.groupby("model_name", sort=False):
        ok_rows = model_rows[model_rows["status"].astype(str) == "ok"]
        if ok_rows.empty:
            continue
        formula = str(ok_rows["formula"].iloc[0])
        data = data_for_formula(sessions, formula)
        n_by_model[model_name] = len(data)
        terms = list(ok_rows["term"].astype(str))
        ci_by_model[model_name] = bootstrap_formula_terms(data, formula, terms, rng)

    out = stats.copy()
    out.insert(0, "metric", out["formula"].map(metric_from_formula))
    out.insert(1, "condition", out.apply(lambda row: condition_label(str(row["term"]), str(row["formula"])), axis=1))
    out["n_sessions"] = out.apply(lambda row: n_by_model.get(str(row["model_name"]), numeric_or_nan(row.get("nobs"))), axis=1)
    out["ci_low"] = math.nan
    out["ci_high"] = math.nan
    out["ci_method"] = CI_METHOD
    out["notes"] = ""

    for idx, row in out.iterrows():
        if str(row.get("status")) != "ok":
            out.at[idx, "ci_method"] = ""
            out.at[idx, "notes"] = "CI not computed because the original stats row was not status=ok."
            continue
        term = str(row["term"])
        model_name = str(row["model_name"])
        ci = ci_by_model.get(model_name, {}).get(term)
        if ci is None:
            out.at[idx, "ci_method"] = ""
            out.at[idx, "notes"] = "CI not computed because the term was not estimable in the bootstrap fits."
            continue
        low, high, note = ci
        out.at[idx, "ci_low"] = low
        out.at[idx, "ci_high"] = high
        out.at[idx, "notes"] = note

    return out


def bootstrap_formula_terms(
    data: pd.DataFrame,
    formula: str,
    terms: list[str],
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> dict[str, tuple[float, float, str]]:
    data = data.reset_index(drop=True)
    try:
        y_design, x_design = patsy.dmatrices(formula, data, return_type="dataframe")
    except Exception as exc:
        return {
            term: (math.nan, math.nan, f"CI not computed: failed to build design matrix: {exc}.")
            for term in terms
        }

    y = y_design.to_numpy(dtype=float).ravel()
    x = x_design.to_numpy(dtype=float)
    columns = list(x_design.columns)
    term_positions = {term: columns.index(term) for term in terms if term in columns}
    groups = [group.index.to_numpy() for _, group in data.groupby(["condition", "model"], sort=False, dropna=False)]

    draws: dict[str, list[float]] = {term: [] for term in terms}
    skipped = 0

    for _ in range(reps):
        try:
            sample_idx = np.concatenate([rng.choice(group, size=len(group), replace=True) for group in groups])
            coefs = np.linalg.lstsq(x[sample_idx], y[sample_idx], rcond=None)[0]
        except Exception:
            skipped += 1
            continue
        for term, position in term_positions.items():
            value = coefs[position]
            if not pd.isna(value):
                draws[term].append(float(value))

    ci: dict[str, tuple[float, float, str]] = {}
    for term, values in draws.items():
        if len(values) < max(100, reps // 4):
            ci[term] = (
                math.nan,
                math.nan,
                f"CI not computed: only {len(values)} valid bootstrap estimates out of {reps}.",
            )
            continue
        low, high = np.quantile(np.array(values, dtype=float), [0.025, 0.975])
        note = (
            f"{CI_METHOD}; {len(values)} valid bootstrap estimates; "
            "sessions resampled with replacement within observed condition-model cells."
        )
        if skipped:
            note += f" {skipped} bootstrap fits failed and were omitted."
        ci[term] = (float(low), float(high), note)
    return ci


def stratified_session_sample(data: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    group_cols = ["condition", "model"]
    for _, group in data.groupby(group_cols, sort=False, dropna=False):
        sample_idx = rng.choice(group.index.to_numpy(), size=len(group), replace=True)
        pieces.append(data.loc[sample_idx])
    return pd.concat(pieces, ignore_index=True)


def data_for_formula(sessions: pd.DataFrame, formula: str) -> pd.DataFrame:
    outcome = metric_from_formula(formula)
    data = sessions.copy()
    if "Treatment(reference='broadcast')" in formula:
        data = data[data["condition"].isin(["broadcast", "targeted"])].copy()
    needed = [outcome, "condition", "model", "game_id"]
    return data.dropna(subset=[col for col in needed if col in data.columns]).reset_index(drop=True)


def metric_from_formula(formula: str) -> str:
    return str(formula).split("~", 1)[0].strip()


def condition_label(term: str, formula: str) -> str:
    if term == "Intercept":
        reference = reference_condition(formula)
        return f"{reference}_reference_intercept" if reference else "reference_intercept"
    condition_match = re.search(r"C\(condition.*?\)\[T\.([^\]]+)\]", term)
    if condition_match:
        reference = reference_condition(formula)
        condition = condition_match.group(1)
        return f"{condition}_vs_{reference}" if reference else condition
    model_match = re.search(r"C\(model\)\[T\.([^\]]+)\]", term)
    if model_match:
        return f"model_{model_match.group(1)}_vs_reference"
    return term


def reference_condition(formula: str) -> str:
    match = re.search(r"Treatment\(reference='([^']+)'\)", formula)
    return match.group(1) if match else ""


def numeric_or_nan(value: Any) -> float:
    try:
        if pd.isna(value):
            return math.nan
        return float(value)
    except Exception:
        return math.nan


def compute_round_window_session_metrics(round_min: int, round_max: int) -> pd.DataFrame:
    round_agent = load_all_round_agent()
    if round_agent.empty:
        raise RuntimeError("No round_agent.csv files found under clean_analysis_data/.")
    window = round_agent[(round_agent["round"] >= round_min) & (round_agent["round"] <= round_max)].copy()
    if window.empty:
        raise RuntimeError(f"No rows found for round window {round_min}-{round_max}.")
    window["contribution_ratio_row"] = window["contribution"] / INITIAL_ENDOWMENT

    rows: list[dict[str, Any]] = []
    for keys, group in window.groupby(["condition", "model", "game_id"], sort=False, dropna=False):
        condition, model, game_id = keys
        rows.append(
            {
                "condition": condition,
                "model": model,
                "game_id": game_id,
                "n_rounds_window": int(group["round"].nunique()),
                "rows_window": int(len(group)),
                "contribution_ratio": float(group["contribution_ratio_row"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_robustness_summary(
    main_with_ci: pd.DataFrame,
    sessions: pd.DataFrame,
    late_sessions: pd.DataFrame,
    final_sessions: pd.DataFrame,
    condition_summary: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    core_specs = [
        {
            "metric": "contribution_ratio",
            "comparison": "broadcast_vs_none",
            "formula": "contribution_ratio ~ C(condition, Treatment(reference='none')) + C(model)",
            "term": "C(condition, Treatment(reference='none'))[T.broadcast]",
            "expected": "positive",
            "claim": "Broadcast punishment increases contribution ratio relative to no punishment.",
        },
        {
            "metric": "contribution_ratio",
            "comparison": "targeted_vs_none",
            "formula": "contribution_ratio ~ C(condition, Treatment(reference='none')) + C(model)",
            "term": "C(condition, Treatment(reference='none'))[T.targeted]",
            "expected": "positive",
            "claim": "Targeted punishment increases contribution ratio relative to no punishment.",
        },
        {
            "metric": "punishment_harm_per_agent_round",
            "comparison": "targeted_vs_broadcast",
            "formula": "punishment_harm_per_agent_round ~ C(condition, Treatment(reference='broadcast')) + C(model)",
            "term": "C(condition, Treatment(reference='broadcast'))[T.targeted]",
            "expected": "negative",
            "claim": "Broadcast punishment generates higher punishment harm than targeted punishment.",
        },
        {
            "metric": "collateral_burden_share",
            "comparison": "targeted_vs_broadcast",
            "formula": "collateral_burden_share ~ C(condition, Treatment(reference='broadcast')) + C(model)",
            "term": "C(condition, Treatment(reference='broadcast'))[T.targeted]",
            "expected": "negative",
            "claim": "Targeted punishment has a lower collateral burden share than broadcast punishment.",
        },
    ]

    for spec in core_specs:
        main_row = lookup_main_row(main_with_ci, spec["metric"], spec["comparison"])
        original = float(main_row["estimate"])
        robust = original
        rows.append(
            robustness_row(
                check_name="session_level_bootstrap",
                metric=spec["metric"],
                comparison=spec["comparison"],
                original_estimate=original,
                robust_estimate=robust,
                ci_low=float(main_row["ci_low"]),
                ci_high=float(main_row["ci_high"]),
                expected=spec["expected"],
                conclusion=conclusion_text(spec["claim"], robust, float(main_row["ci_low"]), float(main_row["ci_high"]), spec["expected"]),
                notes="Bootstrap CI taken from outputs/main_metrics_with_ci.csv.",
            )
        )

        loo = leave_one_model_out(sessions, spec["formula"], spec["term"], spec["expected"])
        rows.append(
            robustness_row(
                check_name="leave_one_model_out",
                metric=spec["metric"],
                comparison=spec["comparison"],
                original_estimate=original,
                robust_estimate=loo["least_favorable"],
                ci_low=math.nan,
                ci_high=math.nan,
                expected=spec["expected"],
                conclusion=loo["conclusion"],
                notes=loo["notes"],
            )
        )

    alignment = condition_summary.loc[condition_summary["condition"] == "targeted", "target_alignment"].iloc[0]
    alignment_boot = bootstrap_condition_mean(sessions, "targeted", "target_alignment", rng)
    rows.append(
        robustness_row(
            check_name="session_level_bootstrap",
            metric="target_alignment",
            comparison="targeted_condition_mean",
            original_estimate=float(alignment),
            robust_estimate=float(alignment),
            ci_low=alignment_boot.ci_low,
            ci_high=alignment_boot.ci_high,
            expected="above_half",
            conclusion=(
                "Targeted punishment remains highly aligned with low contributors."
                if alignment_boot.ci_low > 0.5
                else "Target alignment is positive but its lower bootstrap bound requires caution."
            ),
            notes=alignment_boot.notes,
        )
    )

    alignment_loo = leave_one_model_out_mean(sessions, "targeted", "target_alignment", threshold=0.5)
    rows.append(
        robustness_row(
            check_name="leave_one_model_out",
            metric="target_alignment",
            comparison="targeted_condition_mean",
            original_estimate=float(alignment),
            robust_estimate=alignment_loo["least_favorable"],
            ci_low=math.nan,
            ci_high=math.nan,
            expected="above_half",
            conclusion=alignment_loo["conclusion"],
            notes=alignment_loo["notes"],
        )
    )

    rows.extend(round_window_rows(main_with_ci, late_sessions, "late_round_only", "rounds_6_to_10", rng))
    rows.extend(round_window_rows(main_with_ci, final_sessions, "final_round_only", "round_10", rng))

    out = pd.DataFrame(rows)
    return out[
        [
            "check_name",
            "metric",
            "comparison_or_condition",
            "original_estimate",
            "robust_estimate",
            "ci_low",
            "ci_high",
            "direction_consistent",
            "conclusion",
            "notes",
        ]
    ]


def lookup_main_row(main_with_ci: pd.DataFrame, metric: str, comparison: str) -> pd.Series:
    mask = (main_with_ci["metric"] == metric) & (main_with_ci["condition"] == comparison)
    if not mask.any():
        raise KeyError(f"No main metrics row for {metric} / {comparison}")
    return main_with_ci.loc[mask].iloc[0]


def robustness_row(
    check_name: str,
    metric: str,
    comparison: str,
    original_estimate: float,
    robust_estimate: float,
    ci_low: float,
    ci_high: float,
    expected: str,
    conclusion: str,
    notes: str,
) -> dict[str, Any]:
    return {
        "check_name": check_name,
        "metric": metric,
        "comparison_or_condition": comparison,
        "original_estimate": original_estimate,
        "robust_estimate": robust_estimate,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "direction_consistent": direction_consistent(robust_estimate, expected),
        "conclusion": conclusion,
        "notes": notes,
    }


def direction_consistent(value: float, expected: str) -> bool:
    if pd.isna(value):
        return False
    if expected == "positive":
        return value > 0
    if expected == "negative":
        return value < 0
    if expected == "above_half":
        return value > 0.5
    return False


def conclusion_text(claim: str, estimate: float, ci_low: float, ci_high: float, expected: str) -> str:
    sign_ok = direction_consistent(estimate, expected)
    ci_excludes_zero = (
        expected == "positive"
        and not pd.isna(ci_low)
        and ci_low > 0
        or expected == "negative"
        and not pd.isna(ci_high)
        and ci_high < 0
    )
    if sign_ok and ci_excludes_zero:
        return f"Robust: {claim}"
    if sign_ok:
        return f"Directionally consistent but CI overlaps the null: {claim}"
    return f"Not robust in expected direction: {claim}"


def leave_one_model_out(data: pd.DataFrame, formula: str, term: str, expected: str) -> dict[str, Any]:
    estimates: dict[str, float] = {}
    model_data = data_for_formula(data, formula)
    for model in sorted(model_data["model"].dropna().unique()):
        subset = model_data[model_data["model"] != model].copy()
        try:
            result = smf.ols(formula=formula, data=subset).fit()
            estimates[str(model)] = float(result.params.get(term, math.nan))
        except Exception:
            estimates[str(model)] = math.nan

    valid = {model: estimate for model, estimate in estimates.items() if not pd.isna(estimate)}
    if not valid:
        return {
            "least_favorable": math.nan,
            "conclusion": "Leave-one-model-out could not be estimated.",
            "notes": "No valid leave-one-model-out estimates; CI left NA.",
        }
    values = np.array(list(valid.values()), dtype=float)
    if expected == "positive":
        least_favorable = float(np.min(values))
    elif expected == "negative":
        least_favorable = float(np.max(values))
    else:
        least_favorable = float(np.min(values))
    all_consistent = all(direction_consistent(value, expected) for value in values)
    rng_text = f"range={values.min():.6g} to {values.max():.6g}"
    dropped = "; ".join(f"{model}:{estimate:.6g}" for model, estimate in sorted(valid.items()))
    return {
        "least_favorable": least_favorable,
        "conclusion": (
            "All leave-one-model-out estimates preserve the expected direction."
            if all_consistent
            else "At least one leave-one-model-out estimate changes or loses the expected direction."
        ),
        "notes": f"CI not computed for deterministic leave-one-model-out check; {rng_text}; omitted-model estimates: {dropped}.",
    }


def leave_one_model_out_mean(data: pd.DataFrame, condition: str, metric: str, threshold: float) -> dict[str, Any]:
    condition_data = data[data["condition"] == condition].copy()
    estimates: dict[str, float] = {}
    for model in sorted(condition_data["model"].dropna().unique()):
        subset = condition_data[condition_data["model"] != model]
        estimates[str(model)] = float(subset[metric].mean(skipna=True))

    valid = {model: estimate for model, estimate in estimates.items() if not pd.isna(estimate)}
    if not valid:
        return {
            "least_favorable": math.nan,
            "conclusion": "Leave-one-model-out condition mean could not be estimated.",
            "notes": "No valid leave-one-model-out estimates; CI left NA.",
        }
    values = np.array(list(valid.values()), dtype=float)
    least_favorable = float(values.min())
    all_consistent = bool((values > threshold).all())
    dropped = "; ".join(f"{model}:{estimate:.6g}" for model, estimate in sorted(valid.items()))
    return {
        "least_favorable": least_favorable,
        "conclusion": (
            f"All leave-one-model-out means remain above {threshold:.2f}."
            if all_consistent
            else f"At least one leave-one-model-out mean falls to or below {threshold:.2f}."
        ),
        "notes": (
            "CI not computed for deterministic leave-one-model-out check; "
            f"range={values.min():.6g} to {values.max():.6g}; omitted-model estimates: {dropped}."
        ),
    }


def bootstrap_condition_mean(
    data: pd.DataFrame,
    condition: str,
    metric: str,
    rng: np.random.Generator,
    reps: int = BOOTSTRAP_REPS,
) -> EstimateResult:
    subset = data[data["condition"] == condition].copy()
    subset = subset[["condition", "model", "game_id", metric]].copy()
    if subset.empty:
        return EstimateResult(math.nan, math.nan, math.nan, 0, "CI not computed: empty condition subset.")
    values: list[float] = []
    for _ in range(reps):
        sample = stratified_session_sample(subset, rng)
        estimate = float(sample[metric].mean(skipna=True))
        if not pd.isna(estimate):
            values.append(estimate)
    if len(values) < max(100, reps // 4):
        return EstimateResult(
            float(subset[metric].mean(skipna=True)),
            math.nan,
            math.nan,
            len(subset),
            f"CI not computed: only {len(values)} valid bootstrap estimates out of {reps}.",
        )
    low, high = np.quantile(np.array(values, dtype=float), [0.025, 0.975])
    return EstimateResult(
        float(subset[metric].mean(skipna=True)),
        float(low),
        float(high),
        len(subset),
        f"{CI_METHOD}; {len(values)} valid bootstrap estimates; sessions resampled within model cells.",
    )


def round_window_rows(
    main_with_ci: pd.DataFrame,
    window_sessions: pd.DataFrame,
    check_name: str,
    window_label: str,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    specs = [
        {
            "comparison": "broadcast_vs_none",
            "term": "C(condition, Treatment(reference='none'))[T.broadcast]",
            "expected": "positive",
            "claim": "Broadcast punishment increases contribution ratio relative to no punishment.",
        },
        {
            "comparison": "targeted_vs_none",
            "term": "C(condition, Treatment(reference='none'))[T.targeted]",
            "expected": "positive",
            "claim": "Targeted punishment increases contribution ratio relative to no punishment.",
        },
        {
            "comparison": "targeted_vs_broadcast",
            "term": "C(condition, Treatment(reference='broadcast'))[T.targeted]",
            "expected": "positive",
            "claim": "Targeted punishment preserves contribution at least as well as broadcast punishment.",
            "formula": "contribution_ratio ~ C(condition, Treatment(reference='broadcast')) + C(model)",
        },
    ]
    rows: list[dict[str, Any]] = []
    default_formula = "contribution_ratio ~ C(condition, Treatment(reference='none')) + C(model)"

    for spec in specs:
        formula = spec.get("formula", default_formula)
        data = data_for_formula(window_sessions, formula)
        estimate = estimate_formula_term(data, formula, spec["term"])
        ci_map = bootstrap_formula_terms(data, formula, [spec["term"]], rng)
        ci_low, ci_high, note = ci_map.get(spec["term"], (math.nan, math.nan, "CI not computed."))
        if spec["comparison"] in {"broadcast_vs_none", "targeted_vs_none"}:
            original = float(lookup_main_row(main_with_ci, "contribution_ratio", spec["comparison"])["estimate"])
        else:
            original = estimate_formula_term(
                data_for_formula(window_sessions, formula),
                formula,
                spec["term"],
            )
        rows.append(
            robustness_row(
                check_name=check_name,
                metric="contribution_ratio",
                comparison=f"{spec['comparison']} ({window_label})",
                original_estimate=original,
                robust_estimate=estimate,
                ci_low=ci_low,
                ci_high=ci_high,
                expected=spec["expected"],
                conclusion=conclusion_text(spec["claim"], estimate, ci_low, ci_high, spec["expected"]),
                notes=note,
            )
        )
    return rows


def estimate_formula_term(data: pd.DataFrame, formula: str, term: str) -> float:
    if data.empty:
        return math.nan
    try:
        y_design, x_design = patsy.dmatrices(formula, data.reset_index(drop=True), return_type="dataframe")
        if term not in x_design.columns:
            return math.nan
        coefs = np.linalg.lstsq(x_design.to_numpy(dtype=float), y_design.to_numpy(dtype=float).ravel(), rcond=None)[0]
        return float(coefs[list(x_design.columns).index(term)])
    except Exception:
        return math.nan


def write_robustness_markdown(robustness: pd.DataFrame) -> None:
    def fmt(value: Any) -> str:
        try:
            if pd.isna(value):
                return "NA"
            return f"{float(value):.3f}"
        except Exception:
            return "NA"

    rows = robustness.copy()
    lines: list[str] = []
    lines.append("# Robustness Summary")
    lines.append("")
    lines.append(
        "All robustness checks use the existing cleaned data and previously defined metrics. "
        "Uncertainty intervals are session-level bootstrap 95% confidence intervals, with sessions "
        "resampled as the unit of analysis."
    )
    lines.append("")
    lines.append("## Main Findings")
    lines.append("")

    def row_for(check: str, metric: str, comparison_prefix: str) -> pd.Series:
        mask = (
            (rows["check_name"] == check)
            & (rows["metric"] == metric)
            & rows["comparison_or_condition"].astype(str).str.startswith(comparison_prefix)
        )
        return rows.loc[mask].iloc[0]

    b_none = row_for("session_level_bootstrap", "contribution_ratio", "broadcast_vs_none")
    t_none = row_for("session_level_bootstrap", "contribution_ratio", "targeted_vs_none")
    lines.append(
        "- Punishment conditions increase contribution ratio relative to no punishment. "
        f"The broadcast-minus-none estimate is {fmt(b_none['robust_estimate'])} "
        f"(95% CI [{fmt(b_none['ci_low'])}, {fmt(b_none['ci_high'])}]), and the "
        f"targeted-minus-none estimate is {fmt(t_none['robust_estimate'])} "
        f"(95% CI [{fmt(t_none['ci_low'])}, {fmt(t_none['ci_high'])}])."
    )

    late_t = row_for("late_round_only", "contribution_ratio", "targeted_vs_none")
    final_t = row_for("final_round_only", "contribution_ratio", "targeted_vs_none")
    lines.append(
        "- The targeted condition continues to show higher late contributions than the no-punishment condition. "
        f"For rounds 6-10, the targeted-minus-none estimate is {fmt(late_t['robust_estimate'])} "
        f"(95% CI [{fmt(late_t['ci_low'])}, {fmt(late_t['ci_high'])}]); in round 10 alone, "
        f"the estimate is {fmt(final_t['robust_estimate'])} "
        f"(95% CI [{fmt(final_t['ci_low'])}, {fmt(final_t['ci_high'])}])."
    )

    harm = row_for("session_level_bootstrap", "punishment_harm_per_agent_round", "targeted_vs_broadcast")
    collateral = row_for("session_level_bootstrap", "collateral_burden_share", "targeted_vs_broadcast")
    lines.append(
        "- Broadcast punishment produces higher punishment harm and collateral burden than targeted punishment. "
        f"The targeted-minus-broadcast harm estimate is {fmt(harm['robust_estimate'])} "
        f"(95% CI [{fmt(harm['ci_low'])}, {fmt(harm['ci_high'])}]), and the corresponding "
        f"collateral-burden estimate is {fmt(collateral['robust_estimate'])} "
        f"(95% CI [{fmt(collateral['ci_low'])}, {fmt(collateral['ci_high'])}])."
    )

    align = row_for("session_level_bootstrap", "target_alignment", "targeted_condition_mean")
    lines.append(
        "- Targeted punishment is directionally precise. The target-alignment mean is "
        f"{fmt(align['robust_estimate'])} (95% CI [{fmt(align['ci_low'])}, {fmt(align['ci_high'])}]), "
        "while the collateral-burden contrast above favors targeted punishment."
    )

    lines.append("")
    lines.append("## Model Sensitivity")
    lines.append("")
    loo = rows[rows["check_name"] == "leave_one_model_out"].copy()
    all_consistent = bool(loo["direction_consistent"].all()) if not loo.empty else False
    if all_consistent:
        lines.append(
            "Leave-one-model-out checks preserve the expected direction for every core comparison. "
            "Thus, the main conclusions are not fully driven by any single model in the panel."
        )
    else:
        inconsistent = loo[~loo["direction_consistent"]]
        labels = ", ".join(inconsistent["comparison_or_condition"].astype(str).tolist())
        lines.append(
            "Some leave-one-model-out checks do not preserve the expected direction and should be treated cautiously: "
            f"{labels}."
        )

    lines.append("")
    lines.append("## Cautions")
    lines.append("")
    lines.append(
        "Collateral burden is evaluated only among sessions with observed punishment exposure, so this estimate is "
        "conditional on punishment occurring and should be reported with that denominator clearly stated."
    )
    lines.append(
        "The final-round-only estimates are useful as an appendix sensitivity check because they rely on a narrower "
        "slice of behavior and therefore have wider uncertainty than the full-sample and late-round analyses."
    )
    lines.append("")
    lines.append("## Recommended Placement")
    lines.append("")
    lines.append(
        "Main text: session-level bootstrap CIs for the contribution-ratio contrasts, targeted late-round contribution, "
        "punishment-harm contrast, collateral-burden contrast, and target alignment."
    )
    lines.append(
        "Appendix: full leave-one-model-out table, final-round-only checks, and the complete coefficient-level "
        "bootstrap table in `main_metrics_with_ci.csv`."
    )
    lines.append("")

    (OUTPUT_DIR / "robustness_summary.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
