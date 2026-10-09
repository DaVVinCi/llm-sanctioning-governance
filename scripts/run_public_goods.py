from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import logging
import os
import random
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from pg_probe.agents.llm_agent import LLMAgent
from pg_probe.agents.rule_agents import MockLLMAgent
from pg_probe.games.public_goods_game import PublicGoodsGame
from pg_probe.llm.client import LLMClient


ROUND_AGENT_FIELDS = [
    "game_id",
    "session_id",
    "model",
    "punishment_mode",
    "round",
    "agent_id",
    "contribution",
    "remaining",
    "dividend",
    "penalty_spend",
    "incoming_punishment",
    "payoff",
    "cumulative_score",
    "targets",
    "original_targets",
    "normalized_targets",
    "target_repaired",
    "penalty_repaired",
    "action_valid",
    "reasoning_text",
    "raw_response",
    "invalid_response",
]

GAME_SUMMARY_FIELDS = [
    "game_id",
    "session_id",
    "model",
    "punishment_mode",
    "total_group_payoff",
    "mean_contribution",
    "mean_punishment",
    "mean_incoming_punishment",
    "final_score_agent1",
    "final_score_agent2",
    "final_score_agent3",
    "final_score_agent4",
    "winner",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run public-goods punishment probe experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--model", required=True, help="LLM model name.")
    parser.add_argument("--punishment_mode", choices=["none", "broadcast", "targeted"])
    parser.add_argument("--sessions", type=int, default=1)
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--provider_config",
        default="configs/model_providers.yaml",
        help="Path to model-provider routing YAML.",
    )
    parser.add_argument(
        "--secrets_config",
        default="configs/api_keys.local.yaml",
        help="Path to local provider API key/base URL YAML.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def effective_config(args: argparse.Namespace) -> dict[str, Any]:
    load_env_file(PROJECT_ROOT / ".env")
    config_path = resolve_path(args.config)
    config = load_yaml(config_path)
    config["config_path"] = args.config
    config["model"] = args.model
    config["punishment_mode"] = args.punishment_mode or config.get("punishment_mode", "broadcast")
    config["sessions"] = int(args.sessions)
    config["rounds"] = int(args.rounds if args.rounds is not None else config.get("rounds", 10))
    config["temperature"] = float(args.temperature)
    config["seed"] = int(args.seed)
    config["output_dir"] = args.output_dir
    config["num_agents"] = int(config.get("num_agents", 4))
    config["initial_endowment"] = float(config.get("initial_endowment", 30))
    config["multiplier"] = float(config.get("multiplier", 1.5))
    config["penalty_factor"] = float(config.get("penalty_factor", 1.5))
    config.update(resolve_provider_config(args.model, args.provider_config, args.secrets_config))
    return config


def resolve_provider_config(
    model: str,
    provider_config_path: str,
    secrets_config_path: str = "configs/api_keys.local.yaml",
) -> dict[str, Any]:
    if model == "mock":
        return {
            "provider_config_path": provider_config_path,
            "secrets_config_path": secrets_config_path,
            "provider": "mock",
            "api_key_env": "",
            "base_url_env": "",
            "base_url": "",
        }

    path = resolve_path(provider_config_path)
    if not path.exists():
        return legacy_provider_config(provider_config_path, secrets_config_path)

    routing = load_yaml(path)
    providers = routing.get("providers", {})
    provider_name = match_provider_name(model, routing)
    if provider_name is None:
        return legacy_provider_config(provider_config_path, secrets_config_path)
    if provider_name not in providers:
        raise ValueError(f"Provider {provider_name!r} matched for model {model!r}, but it is not defined.")

    provider = providers[provider_name] or {}
    api_key_env = str(provider.get("api_key_env") or "OPENAI_API_KEY")
    base_url_env = str(provider.get("base_url_env") or "OPENAI_BASE_URL")
    secrets = load_provider_secrets(secrets_config_path)
    provider_secrets = secrets.get(provider_name, {})
    api_key = provider_secrets.get("api_key") or provider.get("api_key") or os.getenv(api_key_env)
    base_url = provider_secrets.get("base_url") or provider.get("base_url") or os.getenv(base_url_env)
    if not api_key:
        raise RuntimeError(
            f"Model {model!r} matched provider {provider_name!r}, but environment variable "
            f"{api_key_env} is not set and no api_key is configured in {secrets_config_path}."
        )

    return {
        "provider_config_path": provider_config_path,
        "secrets_config_path": secrets_config_path,
        "provider": provider_name,
        "api_key_env": api_key_env,
        "base_url_env": base_url_env,
        "api_key": api_key,
        "base_url": base_url or "",
    }


def load_provider_secrets(secrets_config_path: str) -> dict[str, dict[str, str]]:
    path = resolve_path(secrets_config_path)
    if not path.exists():
        return {}

    data = load_yaml(path)
    providers = data.get("providers", data)
    if not isinstance(providers, dict):
        raise ValueError(f"Secrets config must contain a provider mapping: {path}")

    normalized: dict[str, dict[str, str]] = {}
    for provider_name, provider_data in providers.items():
        if not isinstance(provider_data, dict):
            continue
        normalized[str(provider_name)] = {
            "api_key": str(provider_data.get("api_key") or ""),
            "base_url": str(provider_data.get("base_url") or ""),
        }
    return normalized


def legacy_provider_config(provider_config_path: str, secrets_config_path: str) -> dict[str, Any]:
    secrets = load_provider_secrets(secrets_config_path)
    legacy_secrets = secrets.get("openai_compatible", {})
    return {
        "provider_config_path": provider_config_path,
        "secrets_config_path": secrets_config_path,
        "provider": "openai_compatible",
        "api_key_env": "OPENAI_API_KEY",
        "base_url_env": "OPENAI_BASE_URL",
        "api_key": legacy_secrets.get("api_key") or os.getenv("OPENAI_API_KEY"),
        "base_url": legacy_secrets.get("base_url") or os.getenv("OPENAI_BASE_URL") or "",
    }


def match_provider_name(model: str, routing: dict[str, Any]) -> str | None:
    exact_map = routing.get("model_provider_map", {}) or {}
    if model in exact_map:
        return str(exact_map[model])

    model_lower = model.lower()
    pattern_map = routing.get("model_provider_patterns", {}) or {}
    for provider_name, patterns in pattern_map.items():
        for pattern in patterns or []:
            if fnmatch.fnmatchcase(model_lower, str(pattern).lower()):
                return str(provider_name)
    return None


def resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def create_agents(config: dict[str, Any]) -> list[LLMAgent | MockLLMAgent]:
    if config["model"] == "mock":
        return [
            MockLLMAgent(
                agent_id=agent_id,
                model="mock",
                punishment_mode=config["punishment_mode"],
            )
            for agent_id in range(1, config["num_agents"] + 1)
        ]

    prompt_dir = PROJECT_ROOT / "prompts"
    return [
        LLMAgent(
            agent_id=agent_id,
            model=config["model"],
            punishment_mode=config["punishment_mode"],
            temperature=config["temperature"],
            prompt_dir=prompt_dir,
            client=LLMClient(
                api_key=config.get("api_key"),
                base_url=config.get("base_url"),
            ),
        )
        for agent_id in range(1, config["num_agents"] + 1)
    ]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(f"pg_probe.{log_path.parent.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def format_number(value: Any, digits: int = 3) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.{digits}f}"


def records_by_round(records: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record["round"]), []).append(record)
    for round_records in grouped.values():
        round_records.sort(key=lambda row: int(row["agent_id"]))
    return dict(sorted(grouped.items()))


def agent_records(records: list[dict[str, Any]], agent_id: int) -> list[dict[str, Any]]:
    return [row for row in records if int(row["agent_id"]) == agent_id]


def game_time_from_id(game_id: str) -> str:
    try:
        return datetime.strptime(game_id[:15], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def target_text(targets: Any) -> str:
    if not targets:
        return "无"
    if isinstance(targets, list):
        return ", ".join(f"Agent{target}" for target in targets) or "无"
    return str(targets)


def report_strategy_text(config: dict[str, Any]) -> str:
    if config["model"] == "mock":
        return "使用 mock/rule baseline agent 进行决策"
    return f"使用LLM（{config['model']}）进行决策，基于博弈论分析做出选择"


def write_txt_report(
    path: Path,
    records: list[dict[str, Any]],
    summary_row: dict[str, Any],
    config: dict[str, Any],
    game_id: str,
    session_id: int,
) -> None:
    n_agents = int(config["num_agents"])
    grouped = records_by_round(records)
    strategy_text = report_strategy_text(config)
    winner = str(summary_row["winner"]).replace(";", ", ")
    winning_score = max(float(summary_row[f"final_score_agent{agent_id}"]) for agent_id in range(1, n_agents + 1))

    lines: list[str] = []
    lines.append("=== 游戏结束！最终结果 ===")
    lines.append("")
    for agent_id in range(1, n_agents + 1):
        rows = agent_records(records, agent_id)
        mean_contribution = sum(float(row["contribution"]) for row in rows) / max(1, len(rows))
        lines.append(f"Agent{agent_id}:")
        lines.append(f"  最终得分：{format_number(summary_row[f'final_score_agent{agent_id}'])}")
        lines.append(f"  平均贡献：{format_number(mean_contribution, 1)}")
        lines.append("")

    lines.append(f"获胜者是：{winner}，得分：{format_number(winning_score)}")
    lines.append("")
    lines.append("=== 各轮次贡献值明细 ===")
    lines.append("")

    header = "         |" + "".join(f"{f'Agent{agent_id}':^14}|" for agent_id in range(1, n_agents + 1))
    separator = "---------+" + "+".join("-" * 14 for _ in range(n_agents)) + "+"
    table_rows = [
        ("贡献点数", "contribution"),
        ("剩余点数", "remaining"),
        ("分红点数", "dividend"),
        ("惩罚点数", "penalty_spend"),
        ("被罚点数", "incoming_punishment"),
        ("收益点数", "payoff"),
        ("累计总分", "cumulative_score"),
    ]

    lines.append(header)
    lines.append(separator)
    for round_num, round_records in grouped.items():
        lines.append(f"第 {round_num} 轮   |")
        by_agent = {int(row["agent_id"]): row for row in round_records}
        for label, key in table_rows:
            values = [
                f"{format_number(by_agent[agent_id][key]):^14}"
                for agent_id in range(1, n_agents + 1)
            ]
            lines.append(f"  {label:<8}|" + "|".join(values) + "|")
        lines.append(separator)

    lines.append("")
    lines.append("=== 游戏基本信息 ===")
    lines.append(f"游戏ID: {game_id}")
    lines.append("游戏类型: Public Goods")
    lines.append(f"实验时间: {game_time_from_id(game_id)}")
    lines.append(f"Session ID: {session_id}")
    lines.append(f"惩罚模式: {config['punishment_mode']}")
    lines.append(f"轮数: {config['rounds']}")
    lines.append(f"初始资源: {format_number(config['initial_endowment'], 0)}")
    lines.append(f"公共池乘数: {format_number(config['multiplier'], 1)}")
    lines.append(f"惩罚系数: {format_number(config['penalty_factor'], 1)}")
    lines.append(f"智能体数量: {n_agents}")
    lines.append(f"使用的大模型: {config['model']}")
    lines.append("玩家: " + ", ".join(f"Agent{agent_id}" for agent_id in range(1, n_agents + 1)))
    lines.append("")
    lines.append("=== 每轮详细信息 ===")

    for round_num, round_records in grouped.items():
        total_contribution = sum(float(row["contribution"]) for row in round_records)
        public_pool_payoff = total_contribution * float(config["multiplier"])
        mean_contribution = total_contribution / max(1, n_agents)

        lines.append("")
        lines.append(f"=== 第 {round_num} 轮 ===")
        lines.append("")
        lines.append(f"轮次总贡献：{format_number(total_contribution, 0)}")
        lines.append(f"公共池收益：{format_number(public_pool_payoff, 3)}")
        lines.append(f"平均贡献：{format_number(mean_contribution, 1)}")
        lines.append("")

        for row in round_records:
            lines.append(f"Agent{row['agent_id']}:")
            lines.append(f"  策略: {strategy_text}")
            lines.append(f"  贡献值: {format_number(row['contribution'], 3)}")
            lines.append(f"  惩罚值: {format_number(row['penalty_spend'], 3)}")
            if config["punishment_mode"] != "none":
                lines.append(f"  惩罚目标: {target_text(row.get('targets'))}")
            lines.append("  推理过程:")
            reasoning = str(row.get("reasoning_text") or "").strip() or "无"
            for reason_line in reasoning.splitlines():
                lines.append(f"  {reason_line}")
            lines.append("  结果:")
            lines.append(f"    本轮得分: {format_number(row['payoff'], 3)}")
            lines.append(f"    累计总分: {format_number(row['cumulative_score'], 3)}")
            lines.append("")
        lines.append("=" * 50)

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def add_record_metadata(records: list[dict[str, Any]], config: dict[str, Any], game_id: str, session_id: int) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        row = {
            "game_id": game_id,
            "session_id": session_id,
            "model": config["model"],
            "punishment_mode": config["punishment_mode"],
            **record,
        }
        row["targets"] = json.dumps(row["targets"], ensure_ascii=False)
        row["original_targets"] = json.dumps(row.get("original_targets", []), ensure_ascii=False)
        row["normalized_targets"] = json.dumps(row.get("normalized_targets", []), ensure_ascii=False)
        rows.append({field: row.get(field, "") for field in ROUND_AGENT_FIELDS})
    return rows


def summarize_game(records: list[dict[str, Any]], config: dict[str, Any], game_id: str, session_id: int) -> dict[str, Any]:
    final_scores = {}
    for agent_id in range(1, config["num_agents"] + 1):
        agent_records = [row for row in records if row["agent_id"] == agent_id]
        final_scores[agent_id] = agent_records[-1]["cumulative_score"] if agent_records else 0.0

    max_score = max(final_scores.values())
    winners = [f"Agent{agent_id}" for agent_id, score in final_scores.items() if score == max_score]
    n_records = max(1, len(records))

    return {
        "game_id": game_id,
        "session_id": session_id,
        "model": config["model"],
        "punishment_mode": config["punishment_mode"],
        "total_group_payoff": sum(final_scores.values()),
        "mean_contribution": sum(row["contribution"] for row in records) / n_records,
        "mean_punishment": sum(row["penalty_spend"] for row in records) / n_records,
        "mean_incoming_punishment": sum(row["incoming_punishment"] for row in records) / n_records,
        "final_score_agent1": final_scores.get(1, 0.0),
        "final_score_agent2": final_scores.get(2, 0.0),
        "final_score_agent3": final_scores.get(3, 0.0),
        "final_score_agent4": final_scores.get(4, 0.0),
        "winner": ";".join(winners),
    }


def raw_response_rows(records: list[dict[str, Any]], config: dict[str, Any], game_id: str, session_id: int) -> list[dict[str, Any]]:
    return [
        {
            "game_id": game_id,
            "session_id": session_id,
            "model": config["model"],
            "punishment_mode": config["punishment_mode"],
            "round": row["round"],
            "agent_id": row["agent_id"],
            "raw_response": row["raw_response"],
            "reasoning_text": row["reasoning_text"],
            "invalid_response": row["invalid_response"],
        }
        for row in records
    ]


def public_session_config(config: dict[str, Any], game_id: str, session_id: int) -> dict[str, Any]:
    safe_config = {
        key: value
        for key, value in config.items()
        if key not in {"api_key"}
    }
    safe_config["game_id"] = game_id
    safe_config["session_id"] = session_id
    return safe_config


def print_progress(message: str) -> None:
    print(message, flush=True)


def run_session(config: dict[str, Any], session_id: int, output_root: Path) -> None:
    session_dir = output_root / f"session_{session_id:03d}"
    session_dir.mkdir(parents=True, exist_ok=True)
    game_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    logger = setup_logger(session_dir / "run.log")

    random.seed(config["seed"] + session_id)
    agents = create_agents(config)
    for agent in agents:
        agent.reset()

    session_config = public_session_config(config, game_id, session_id)
    (session_dir / "config.json").write_text(
        json.dumps(session_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info("starting session_id=%s game_id=%s", session_id, game_id)

    total_sessions = int(config["sessions"])
    total_rounds = int(config["rounds"])
    total_agents = int(config["num_agents"])

    def progress_callback(event: str, round_num: int, agent_index: int | None = None) -> None:
        prefix = f"[session {session_id}/{total_sessions}]"
        if event == "round_start":
            print_progress(f"{prefix} round {round_num}/{total_rounds}")
        elif event == "agent_start" and agent_index is not None:
            print_progress(
                f"{prefix} round {round_num}/{total_rounds} agent {agent_index}/{total_agents} calling model..."
            )
        elif event == "agent_done" and agent_index is not None:
            print_progress(f"{prefix} round {round_num}/{total_rounds} agent {agent_index}/{total_agents} done")

    print_progress(f"[session {session_id}/{total_sessions}] started")
    game = PublicGoodsGame(
        agents=agents,
        punishment_mode=config["punishment_mode"],
        num_agents=config["num_agents"],
        initial_endowment=config["initial_endowment"],
        multiplier=config["multiplier"],
        penalty_factor=config["penalty_factor"],
        rounds=config["rounds"],
        progress_callback=progress_callback,
    )
    result = game.play_game()
    records = result["records"]

    round_rows = add_record_metadata(records, config, game_id, session_id)
    summary_row = summarize_game(records, config, game_id, session_id)
    raw_rows = raw_response_rows(records, config, game_id, session_id)

    write_csv(session_dir / "round_agent.csv", ROUND_AGENT_FIELDS, round_rows)
    write_csv(session_dir / "game_summary.csv", GAME_SUMMARY_FIELDS, [summary_row])
    write_jsonl(session_dir / "raw_responses.jsonl", raw_rows)
    write_txt_report(session_dir / "report.txt", records, summary_row, config, game_id, session_id)

    invalid_count = sum(1 for row in records if row["invalid_response"])
    logger.info("finished session_id=%s invalid_responses=%s", session_id, invalid_count)
    print_progress(f"[session {session_id}/{total_sessions}] finished -> {session_dir}")


def main() -> int:
    args = parse_args()
    config = effective_config(args)
    output_root = resolve_path(config["output_dir"])
    output_root.mkdir(parents=True, exist_ok=True)

    print_progress(
        "Starting experiment: "
        f"model={config['model']} mode={config['punishment_mode']} "
        f"sessions={config['sessions']} rounds={config['rounds']}"
    )
    for session_id in range(1, config["sessions"] + 1):
        run_session(config, session_id, output_root)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
