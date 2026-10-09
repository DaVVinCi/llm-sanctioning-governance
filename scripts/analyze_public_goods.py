from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pg_probe.metrics.public_goods_metrics import aggregate_metrics, compute_session_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze structured public-goods experiment outputs.")
    parser.add_argument("--round_agent_csv", help="Path to round_agent.csv.")
    parser.add_argument("--game_summary_csv", help="Path to game_summary.csv.")
    parser.add_argument("--input_dir", help="Directory containing round_agent.csv and game_summary.csv.")
    parser.add_argument("--output_dir", required=True, help="Directory for metrics CSV outputs.")
    parser.add_argument("--initial_endowment", type=float, default=30.0)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--num_agents", type=int, default=None)
    return parser.parse_args()


def resolve_inputs(args: argparse.Namespace) -> tuple[Path | None, Path | None]:
    if args.input_dir:
        input_dir = Path(args.input_dir)
        return input_dir / "round_agent.csv", input_dir / "game_summary.csv"
    if not args.round_agent_csv or not args.game_summary_csv:
        raise ValueError("Provide either --input_dir or both --round_agent_csv and --game_summary_csv.")
    return Path(args.round_agent_csv), Path(args.game_summary_csv)


def read_experiment_tables(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    round_agent_csv, game_summary_csv = resolve_inputs(args)
    if round_agent_csv and game_summary_csv and round_agent_csv.exists() and game_summary_csv.exists():
        return pd.read_csv(round_agent_csv), pd.read_csv(game_summary_csv)

    if not args.input_dir:
        raise FileNotFoundError("Could not find round_agent.csv and game_summary.csv.")

    input_dir = Path(args.input_dir)
    round_paths = sorted(input_dir.rglob("round_agent.csv"))
    summary_paths = sorted(input_dir.rglob("game_summary.csv"))
    if not round_paths or not summary_paths:
        raise FileNotFoundError(f"No round_agent.csv/game_summary.csv files found under {input_dir}")

    round_agent = pd.concat([pd.read_csv(path) for path in round_paths], ignore_index=True)
    game_summary = pd.concat([pd.read_csv(path) for path in summary_paths], ignore_index=True)
    return round_agent, game_summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    round_agent, game_summary = read_experiment_tables(args)

    metrics_by_session = compute_session_metrics(
        round_agent=round_agent,
        game_summary=game_summary,
        initial_endowment=args.initial_endowment,
        multiplier=args.multiplier,
        num_agents=args.num_agents,
    )
    metrics_by_model = aggregate_metrics(metrics_by_session, ["model"])
    metrics_by_mode = aggregate_metrics(metrics_by_session, ["punishment_mode"])

    metrics_by_session.to_csv(output_dir / "metrics_by_session.csv", index=False)
    metrics_by_model.to_csv(output_dir / "metrics_by_model.csv", index=False)
    metrics_by_mode.to_csv(output_dir / "metrics_by_mode.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
