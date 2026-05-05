"""CSV performance logging helpers for system testing."""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = PROJECT_ROOT / "assets" / "performance_results"

DIALOGUE_FIELDS = [
    "timestamp",
    "user_id",
    "session_id",
    "audio_file",
    "recognized_text_len",
    "visual_frame_count",
    "visual_emotion",
    "audio_emotion",
    "llm_emotion",
    "asr_s",
    "audio_emotion_s",
    "visual_decision_s",
    "reply_generation_s",
    "tts_s",
    "database_s",
    "total_s",
]

VISION_FIELDS = [
    "timestamp",
    "emotion",
    "confidence",
    "transfer_s",
    "preprocess_s",
    "inference_s",
    "postprocess_s",
    "total_s",
]

SUMMARY_FIELDS = [
    "updated_at",
    "record_type",
    "sample_count",
    "avg_asr_s",
    "avg_audio_emotion_s",
    "avg_visual_decision_s",
    "avg_reply_generation_s",
    "avg_tts_s",
    "avg_database_s",
    "avg_total_s",
    "avg_transfer_s",
    "avg_preprocess_s",
    "avg_inference_s",
    "avg_postprocess_s",
]

_LOCK = Lock()


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _write_row(path: Path, fields: list[str], row: dict) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0

    with path.open("a", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        normalized = {field: row.get(field, "") for field in fields}
        writer.writerow(normalized)


def _format_seconds(value: float) -> str:
    return f"{value:.4f}"


def record_dialogue_performance(row: dict) -> Path:
    """Append one full dialogue performance record and refresh the summary CSV."""
    with _LOCK:
        row = dict(row)
        row.setdefault("timestamp", _now())
        for key in [
            "asr_s",
            "audio_emotion_s",
            "visual_decision_s",
            "reply_generation_s",
            "tts_s",
            "database_s",
            "total_s",
        ]:
            if isinstance(row.get(key), (float, int)):
                row[key] = _format_seconds(float(row[key]))

        path = RESULT_DIR / "dialogue_performance.csv"
        _write_row(path, DIALOGUE_FIELDS, row)
        _refresh_summary()
        return path


def record_vision_performance(row: dict) -> Path:
    """Append one single-frame vision performance record and refresh the summary CSV."""
    with _LOCK:
        row = dict(row)
        row.setdefault("timestamp", _now())
        for key in ["transfer_s", "preprocess_s", "inference_s", "postprocess_s", "total_s"]:
            if isinstance(row.get(key), (float, int)):
                row[key] = _format_seconds(float(row[key]))

        path = RESULT_DIR / "vision_performance.csv"
        _write_row(path, VISION_FIELDS, row)
        _refresh_summary()
        return path


def _read_rows(path: Path) -> list[dict]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _average(rows: list[dict], key: str) -> str:
    values = []
    for row in rows:
        try:
            value = float(row.get(key, ""))
        except ValueError:
            continue
        values.append(value)
    if not values:
        return ""
    return _format_seconds(sum(values) / len(values))


def _refresh_summary() -> None:
    dialogue_rows = _read_rows(RESULT_DIR / "dialogue_performance.csv")
    vision_rows = _read_rows(RESULT_DIR / "vision_performance.csv")

    summary_rows = [
        {
            "updated_at": _now(),
            "record_type": "dialogue",
            "sample_count": len(dialogue_rows),
            "avg_asr_s": _average(dialogue_rows, "asr_s"),
            "avg_audio_emotion_s": _average(dialogue_rows, "audio_emotion_s"),
            "avg_visual_decision_s": _average(dialogue_rows, "visual_decision_s"),
            "avg_reply_generation_s": _average(dialogue_rows, "reply_generation_s"),
            "avg_tts_s": _average(dialogue_rows, "tts_s"),
            "avg_database_s": _average(dialogue_rows, "database_s"),
            "avg_total_s": _average(dialogue_rows, "total_s"),
        },
        {
            "updated_at": _now(),
            "record_type": "vision",
            "sample_count": len(vision_rows),
            "avg_transfer_s": _average(vision_rows, "transfer_s"),
            "avg_preprocess_s": _average(vision_rows, "preprocess_s"),
            "avg_inference_s": _average(vision_rows, "inference_s"),
            "avg_postprocess_s": _average(vision_rows, "postprocess_s"),
            "avg_total_s": _average(vision_rows, "total_s"),
        },
    ]

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULT_DIR / "performance_summary.csv"
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)
