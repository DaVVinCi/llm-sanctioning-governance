from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "clean_analysis_data"
OUTPUT_DIR = ROOT / "outputs"

CONDITIONS = ("none", "broadcast", "targeted")
REQUIRED_SESSION_FILES = (
    "config.json",
    "round_agent.csv",
    "raw_responses.jsonl",
    "game_summary.csv",
)

INITIAL_ENDOWMENT = 30.0
MULTIPLIER = 1.5
NUM_AGENTS = 4
ROUNDS = 10
PENALTY_FACTOR = 1.5
EXPECTED_SESSION_ROWS = ROUNDS * NUM_AGENTS
EXPECTED_TOTAL_ROWS = 36_000
EXPECTED_MODELS = 10
EXPECTED_SESSIONS = 900
FLOAT_TOLERANCE = 1e-8

NUMERIC_COLUMNS = [
    "round",
    "agent_id",
    "contribution",
    "remaining",
    "dividend",
    "penalty_spend",
    "incoming_punishment",
    "payoff",
    "cumulative_score",
]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_session_dirs(data_dir: Path = DATA_DIR) -> Iterable[tuple[str, str, str, Path]]:
    if not data_dir.exists():
        return
    for condition_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        for model_dir in sorted(p for p in condition_dir.iterdir() if p.is_dir()):
            for session_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
                yield condition_dir.name, model_dir.name, session_dir.name, session_dir


def normalize_round_agent(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in NUMERIC_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = 0.0
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    normalized["round"] = normalized["round"].fillna(0).astype(int)
    normalized["agent_id"] = normalized["agent_id"].fillna(0).astype(int)
    for col in NUMERIC_COLUMNS:
        if col not in {"round", "agent_id"}:
            normalized[col] = normalized[col].fillna(0.0)
    if "invalid_response" in normalized.columns:
        normalized["invalid_response"] = normalized["invalid_response"].map(parse_bool)
    else:
        normalized["invalid_response"] = False
    if "reasoning_text" not in normalized.columns:
        normalized["reasoning_text"] = ""
    normalized["reasoning_text"] = normalized["reasoning_text"].fillna("").astype(str)
    return normalized


def load_all_round_agent(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for condition, model_dir, session_dir, session_path in iter_session_dirs(data_dir):
        round_path = session_path / "round_agent.csv"
        if not round_path.exists():
            continue
        df = normalize_round_agent(read_csv(round_path))
        df["condition"] = condition
        df["model_dir"] = model_dir
        df["session_dir"] = session_dir
        df["source_path"] = str(round_path)
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_targets(value: Any) -> list[int]:
    if isinstance(value, list | tuple | set):
        return [_safe_int(v) for v in value if _safe_int(v) is not None]
    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return []

    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
        except Exception:
            continue
        if isinstance(parsed, list | tuple | set):
            return [_safe_int(v) for v in parsed if _safe_int(v) is not None]
        parsed_int = _safe_int(parsed)
        if parsed_int is not None:
            return [parsed_int]

    return [int(match) for match in re.findall(r"-?\d+", text)]


def target_column(df: pd.DataFrame) -> str:
    return "normalized_targets" if "normalized_targets" in df.columns else "targets"


def safe_mean(series: pd.Series) -> float:
    if series.empty:
        return math.nan
    value = pd.to_numeric(series, errors="coerce").mean()
    return float(value) if not pd.isna(value) else math.nan


def safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0 or pd.isna(denominator):
        return math.nan
    return float(numerator) / float(denominator)


def load_merged_round_agent(output_dir: Path = OUTPUT_DIR) -> pd.DataFrame:
    parquet_path = output_dir / "merged_round_agent.parquet"
    csv_path = output_dir / "merged_round_agent.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return read_csv(csv_path)
    raise FileNotFoundError("No merged round-agent output found.")


def bool_to_float(value: Any) -> float:
    if pd.isna(value):
        return math.nan
    return 1.0 if bool(value) else 0.0


def _safe_int(value: Any) -> int | None:
    try:
        if isinstance(value, bool):
            return None
        if isinstance(value, float) and math.isnan(value):
            return None
        return int(value)
    except Exception:
        return None
