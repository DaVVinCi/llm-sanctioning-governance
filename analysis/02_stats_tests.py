from __future__ import annotations

import importlib.util
from typing import Any

import pandas as pd

from common import OUTPUT_DIR, ensure_output_dir, read_csv


EXPLORATORY_NOTE = (
    "Exploratory statistics only; final paper claims require fixed cleaning rules, "
    "fixed metrics, and confirmatory re-analysis."
)


def main() -> int:
    ensure_output_dir()
    session_path = OUTPUT_DIR / "session_metrics.csv"
    if not session_path.exists():
        raise FileNotFoundError("Run 01_compute_metrics.py before 02_stats_tests.py.")
    session_metrics = read_csv(session_path)

    rows: list[dict[str, Any]] = []
    if importlib.util.find_spec("statsmodels") is None:
        rows.append(skipped_row("statsmodels_unavailable"))
    else:
        import statsmodels.formula.api as smf

        full_models = [
            ("contribution_ratio_condition_model", "contribution_ratio ~ C(condition, Treatment(reference='none')) + C(model)"),
            ("payoff_efficiency_condition_model", "payoff_efficiency ~ C(condition, Treatment(reference='none')) + C(model)"),
            ("punishment_frequency_condition_model", "punishment_frequency ~ C(condition, Treatment(reference='none')) + C(model)"),
        ]
        for name, formula in full_models:
            rows.extend(run_ols(smf, session_metrics, name, formula))

        bt = session_metrics[session_metrics["condition"].isin(["broadcast", "targeted"])].copy()
        bt_models = [
            (
                "broadcast_vs_targeted_payoff_efficiency",
                "payoff_efficiency ~ C(condition, Treatment(reference='broadcast')) + C(model)",
            ),
            (
                "broadcast_vs_targeted_punishment_harm",
                "punishment_harm_per_agent_round ~ C(condition, Treatment(reference='broadcast')) + C(model)",
            ),
            (
                "broadcast_vs_targeted_collateral_burden",
                "collateral_burden_share ~ C(condition, Treatment(reference='broadcast')) + C(model)",
            ),
        ]
        for name, formula in bt_models:
            data = bt.dropna(subset=[formula.split("~", 1)[0].strip()])
            rows.extend(run_ols(smf, data, name, formula))

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "stats_results.csv", index=False, encoding="utf-8-sig")
    print("Statistical models complete.")
    return 0


def run_ols(smf: Any, data: pd.DataFrame, model_name: str, formula: str) -> list[dict[str, Any]]:
    if data.empty:
        return [skipped_row("empty_data", model_name, formula)]
    try:
        result = smf.ols(formula=formula, data=data).fit()
    except Exception as exc:
        return [
            {
                "model_name": model_name,
                "formula": formula,
                "term": "",
                "estimate": "",
                "std_error": "",
                "t_value": "",
                "p_value": "",
                "nobs": len(data),
                "r_squared": "",
                "status": "failed",
                "note": f"{EXPLORATORY_NOTE} Error: {exc}",
            }
        ]

    rows: list[dict[str, Any]] = []
    for term in result.params.index:
        rows.append(
            {
                "model_name": model_name,
                "formula": formula,
                "term": term,
                "estimate": float(result.params[term]),
                "std_error": float(result.bse[term]),
                "t_value": float(result.tvalues[term]),
                "p_value": float(result.pvalues[term]),
                "nobs": int(result.nobs),
                "r_squared": float(result.rsquared),
                "status": "ok",
                "note": EXPLORATORY_NOTE,
            }
        )
    return rows


def skipped_row(reason: str, model_name: str = "", formula: str = "") -> dict[str, Any]:
    return {
        "model_name": model_name,
        "formula": formula,
        "term": "",
        "estimate": "",
        "std_error": "",
        "t_value": "",
        "p_value": "",
        "nobs": "",
        "r_squared": "",
        "status": "skipped",
        "note": f"{EXPLORATORY_NOTE} Skipped: {reason}.",
    }


if __name__ == "__main__":
    raise SystemExit(main())
