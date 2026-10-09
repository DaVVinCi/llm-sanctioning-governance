from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from common import CONDITIONS, OUTPUT_DIR, ensure_output_dir, load_merged_round_agent, read_csv


CONDITION_COLORS = {
    "none": "#4C78A8",
    "broadcast": "#F58518",
    "targeted": "#54A24B",
}


def main() -> int:
    ensure_output_dir()
    merged = load_merged_round_agent(OUTPUT_DIR)
    session_metrics = read_csv(OUTPUT_DIR / "session_metrics.csv")
    model_summary = read_csv(OUTPUT_DIR / "model_condition_summary.csv")

    make_fig1_contribution_trajectory(merged)
    make_fig2_payoff_efficiency(session_metrics)
    make_fig3_punishment_cost_harm(session_metrics)
    make_fig4_targeted_model_phenotypes(model_summary)
    print("Figures complete.")
    return 0


def make_fig1_contribution_trajectory(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    trajectory = df.groupby(["condition", "round"], as_index=False)["contribution"].mean()
    for condition in CONDITIONS:
        group = trajectory[trajectory["condition"] == condition]
        ax.plot(
            group["round"],
            group["contribution"],
            marker="o",
            linewidth=2,
            label=condition,
            color=CONDITION_COLORS[condition],
        )
    ax.set_xlabel("Round")
    ax.set_ylabel("Mean contribution")
    ax.set_title("Contribution trajectory by condition")
    ax.set_xticks(range(1, 11))
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_contribution_trajectory.pdf")
    plt.close(fig)


def make_fig2_payoff_efficiency(session_metrics: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    data = [session_metrics.loc[session_metrics["condition"] == condition, "payoff_efficiency"].dropna() for condition in CONDITIONS]
    box = ax.boxplot(data, labels=CONDITIONS, patch_artist=True, showfliers=False)
    for patch, condition in zip(box["boxes"], CONDITIONS):
        patch.set_facecolor(CONDITION_COLORS[condition])
        patch.set_alpha(0.65)
    ax.set_ylabel("Session payoff efficiency")
    ax.set_title("Payoff efficiency by condition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_payoff_efficiency.pdf")
    plt.close(fig)


def make_fig3_punishment_cost_harm(session_metrics: pd.DataFrame) -> None:
    subset = session_metrics[session_metrics["condition"].isin(["broadcast", "targeted"])]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    positions = [1, 2, 4, 5]
    data = [
        subset.loc[subset["condition"] == "broadcast", "punishment_cost_per_agent_round"].dropna(),
        subset.loc[subset["condition"] == "targeted", "punishment_cost_per_agent_round"].dropna(),
        subset.loc[subset["condition"] == "broadcast", "punishment_harm_per_agent_round"].dropna(),
        subset.loc[subset["condition"] == "targeted", "punishment_harm_per_agent_round"].dropna(),
    ]
    box = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True, showfliers=False)
    colors = [CONDITION_COLORS["broadcast"], CONDITION_COLORS["targeted"], CONDITION_COLORS["broadcast"], CONDITION_COLORS["targeted"]]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["Punishment cost", "Punishment harm"])
    ax.set_ylabel("Per agent-round ratio")
    ax.set_title("Punishment cost and harm")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=CONDITION_COLORS["broadcast"], lw=8, alpha=0.65, label="broadcast"),
            plt.Line2D([0], [0], color=CONDITION_COLORS["targeted"], lw=8, alpha=0.65, label="targeted"),
        ],
        frameon=False,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_punishment_cost_harm.pdf")
    plt.close(fig)


def make_fig4_targeted_model_phenotypes(model_summary: pd.DataFrame) -> None:
    targeted = model_summary[model_summary["condition"] == "targeted"].copy()
    targeted = targeted.dropna(subset=["target_alignment", "payoff_efficiency", "anti_social_punishment_rate"])
    fig, ax = plt.subplots(figsize=(7, 4.8))
    if targeted.empty:
        ax.text(0.5, 0.5, "No targeted punishment events to plot", ha="center", va="center")
        ax.set_axis_off()
    else:
        sizes = 60 + targeted["punishment_frequency"].fillna(0) * 800
        scatter = ax.scatter(
            targeted["target_alignment"],
            targeted["payoff_efficiency"],
            s=sizes,
            c=targeted["anti_social_punishment_rate"],
            cmap="viridis_r",
            alpha=0.82,
            edgecolor="black",
            linewidth=0.5,
        )
        for _, row in targeted.iterrows():
            ax.annotate(short_model_name(row["model"]), (row["target_alignment"], row["payoff_efficiency"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Anti-social punishment rate")
        ax.set_xlabel("Target alignment")
        ax.set_ylabel("Payoff efficiency")
        ax.set_title("Targeted model phenotypes")
        ax.set_xlim(0, 1.05)
        ymin = max(0, math.floor(targeted["payoff_efficiency"].min() * 10) / 10 - 0.05)
        ax.set_ylim(ymin, 1.02)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_targeted_model_phenotypes.pdf")
    plt.close(fig)


def short_model_name(model: str) -> str:
    replacements = {
        "claude-sonnet-4-5-20250929": "sonnet",
        "claude-opus-4-5-20251101": "opus",
        "gemini-2.5-flash": "gemini-flash",
        "gemini-2.5-pro": "gemini-pro",
        "deepseek-v3.2": "ds-v3.2",
    }
    return replacements.get(str(model), str(model))


if __name__ == "__main__":
    raise SystemExit(main())
