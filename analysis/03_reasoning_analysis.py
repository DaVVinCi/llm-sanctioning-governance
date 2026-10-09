from __future__ import annotations

from typing import Any

import pandas as pd

from common import OUTPUT_DIR, ensure_output_dir, load_merged_round_agent, read_csv


REASONING_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fairness_norm_free_rider": (
        "fair",
        "unfair",
        "norm",
        "free-rider",
        "free rider",
        "free-riding",
        "freerider",
        "defect",
        "defection",
        "exploit",
        "reciprocity",
    ),
    "deterrence_incentive_enforce": (
        "deter",
        "deterrence",
        "incentive",
        "incentivize",
        "enforce",
        "enforcement",
        "sanction",
        "discipline",
    ),
    "retaliation_revenge_punish_back": (
        "retaliat",
        "revenge",
        "punish back",
        "punishment back",
        "payback",
        "backfire",
        "escalat",
    ),
    "restraint_costly_no_need": (
        "restrain",
        "costly",
        "cost",
        "wasteful",
        "no need",
        "unnecessary",
        "avoid punishment",
        "conserve",
        "preserve",
    ),
    "cooperation_public_good_collective": (
        "cooperat",
        "public good",
        "collective",
        "group return",
        "mutual",
        "sustain",
        "collaborat",
    ),
    "self_interest_preserve_maximize_payoff": (
        "self-interest",
        "self interested",
        "maximize payoff",
        "maximise payoff",
        "maximize",
        "maximise",
        "payoff",
        "preserve",
        "conserve",
        "immediate payoff",
        "net gain",
    ),
}


def main() -> int:
    ensure_output_dir()
    df = load_merged_round_agent(OUTPUT_DIR)
    if df.empty:
        raise RuntimeError("Merged round-agent data is empty.")

    rows: list[dict[str, Any]] = []
    df = add_reasoning_flags(df)
    total_by_condition = df.groupby("condition").size().to_dict()
    df["punished"] = df["penalty_spend"] > 0

    for condition, group in df.groupby("condition"):
        for category in REASONING_KEYWORDS:
            flagged = group[group[category]]
            rows.append(
                {
                    "scope": "round_agent",
                    "condition": condition,
                    "reasoning_category": category,
                    "n": int(len(flagged)),
                    "denominator": int(total_by_condition.get(condition, 0)),
                    "share": safe_share(len(flagged), total_by_condition.get(condition, 0)),
                    "punishment_rate_given_category": float(flagged["punished"].mean()) if not flagged.empty else float("nan"),
                    "target_alignment": float("nan"),
                    "anti_social_punishment_rate": float("nan"),
                    "note": "Keyword-based exploratory reasoning analysis; not final human annotation.",
                }
            )

    events_path = OUTPUT_DIR / "targeted_target_events.csv"
    if events_path.exists():
        events = read_csv(events_path)
        if not events.empty:
            events = add_reasoning_flags(events)
            total_events = len(events)
            for category in REASONING_KEYWORDS:
                flagged = events[events[category]]
                rows.append(
                    {
                        "scope": "targeted_target_events",
                        "condition": "targeted",
                        "reasoning_category": category,
                        "n": int(len(flagged)),
                        "denominator": int(total_events),
                        "share": safe_share(len(flagged), total_events),
                        "punishment_rate_given_category": float("nan"),
                        "target_alignment": float(flagged["target_alignment"].mean()) if not flagged.empty else float("nan"),
                        "anti_social_punishment_rate": float(flagged["anti_social_punishment"].mean()) if not flagged.empty else float("nan"),
                        "note": "Keyword-based exploratory reasoning analysis; not final human annotation.",
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "reasoning_summary.csv", index=False, encoding="utf-8-sig")
    print("Reasoning analysis complete.")
    return 0


def add_reasoning_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    text = out.get("reasoning_text", pd.Series("", index=out.index)).fillna("").astype(str).str.lower()
    for category, keywords in REASONING_KEYWORDS.items():
        out[category] = text.map(lambda value: any(keyword in value for keyword in keywords))
    return out


def safe_share(numerator: int, denominator: int | None) -> float:
    if not denominator:
        return float("nan")
    return float(numerator) / float(denominator)


if __name__ == "__main__":
    raise SystemExit(main())
