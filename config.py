"""全局配置与共享状态。"""

import threading
import time
from typing import Dict, Optional

# 模型配置
VISION_MODEL = "/root/multimodal_emotion_chat/models/facial_emotions"
VISION_SKIP_FRAMES = 1
VISION_DEVICE = None

ASR_MODEL = "large"
SER_MODEL = "/root/multimodal_emotion_chat/models/emotion2vec"
AUDIO_SAMPLE_RATE = 16000

LLM_MODEL = "/root/multimodal_emotion_chat/models/qwen3.5-9b"
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 0.95

TTS_VOICE = "zh-CN-XiaoxiaoNeural"
ADMIN_INVITE_CODE = "GRAD-ADMIN-2026"

VISUAL_CONFIDENCE_THRESHOLD = 0.85
AUDIO_CONFIDENCE_THRESHOLD = 0.8

STATE = {
    "visual_emotion": "Neutral",
    "visual_confidence": 0.0,
    "visual_emotion_probs": {},
    "frame_counter": 0,
    "last_visual_update": 0,
    "audio_emotion": None,
    "audio_confidence": 0.0,
    "last_audio_update": 0,
    "final_emotion": "Neutral",
    "inference_times": [],
}

lock = threading.Lock()


def get_state(key: str, default=None):
    with lock:
        return STATE.get(key, default)


def set_state(key: str, value):
    with lock:
        STATE[key] = value


def update_visual_emotion(emotion: str, confidence: float, emotion_probs: Dict[str, float]):
    with lock:
        STATE["visual_emotion"] = emotion
        STATE["visual_confidence"] = confidence
        STATE["visual_emotion_probs"] = emotion_probs
        STATE["last_visual_update"] = time.time()
        STATE["frame_counter"] += 1


def update_audio_emotion(emotion: str, confidence: float):
    with lock:
        STATE["audio_emotion"] = emotion
        STATE["audio_confidence"] = confidence
        STATE["last_audio_update"] = time.time()


def update_final_emotion(emotion: str):
    with lock:
        STATE["final_emotion"] = emotion


def get_visual_state() -> Dict:
    with lock:
        return {
            "emotion": STATE["visual_emotion"],
            "confidence": STATE["visual_confidence"],
            "emotion_probs": STATE["visual_emotion_probs"],
            "frame_counter": STATE["frame_counter"],
            "last_update": STATE["last_visual_update"],
        }


def get_fusion_state() -> Dict:
    with lock:
        return {
            "visual_emotion": STATE["visual_emotion"],
            "visual_confidence": STATE["visual_confidence"],
            "audio_emotion": STATE["audio_emotion"],
            "audio_confidence": STATE["audio_confidence"],
            "final_emotion": STATE["final_emotion"],
        }


def record_inference_time(module: str, time_ms: float):
    with lock:
        STATE["inference_times"].append({
            "module": module,
            "time": time_ms,
            "timestamp": time.time(),
        })
        if len(STATE["inference_times"]) > 100:
            STATE["inference_times"] = STATE["inference_times"][-100:]


def get_avg_inference_time(module: Optional[str] = None) -> float:
    with lock:
        times = STATE["inference_times"]
        if module:
            times = [item for item in times if item["module"] == module]

        current_time = time.time()
        times = [item for item in times if current_time - item["timestamp"] < 10]
        if not times:
            return 0.0
        return sum(item["time"] for item in times) / len(times)
