"""
配置文件 - 全局配置和状态管理
"""

import threading
import time
from typing import Dict, Optional


# ==================== 全局配置 ====================

# 视觉模块配置
VISION_MODEL = "/root/multimodal_emotion_chat/models/facial_emotions"
VISION_SKIP_FRAMES = 1  # 每1帧抽1帧
VISION_DEVICE = None  # None表示自动检测GPU，或指定为0表示使用第一个GPU

# 听觉模块配置
ASR_MODEL = "large"  # Whisper模型大小: tiny/base/small/medium/large
SER_MODEL = "/root/multimodal_emotion_chat/models/emotion2vec"  # emotion2vec模型路径
AUDIO_SAMPLE_RATE = 16000

# LLM配置
LLM_MODEL = "/root/multimodal_emotion_chat/models/qwen3.5-9b"  # 使用本地模型路径
LLM_MAX_LENGTH = 512
LLM_TEMPERATURE = 0.7

# TTS配置
TTS_VOICE = "zh-CN-XiaoxiaoNeural"

# 融合配置
VISUAL_CONFIDENCE_THRESHOLD = 0.85  # 视觉高置信度阈值
AUDIO_CONFIDENCE_THRESHOLD = 0.8    # 音频高置信度阈值


# ==================== 全局状态管理 ====================

# 全局状态字典
STATE = {
    # 视觉状态
    "visual_emotion": "Neutral",
    "visual_confidence": 0.0,
    "visual_emotion_probs": {},
    "frame_counter": 0,
    "last_visual_update": 0,

    # 听觉状态
    "audio_emotion": None,
    "audio_confidence": 0.0,
    "last_audio_update": 0,

    # 融合状态
    "final_emotion": "Neutral",

    # 性能统计
    "inference_times": [],
}

# 线程锁
lock = threading.Lock()


# ==================== 状态访问函数 ====================

def get_state(key: str, default=None):
    """获取状态值"""
    with lock:
        return STATE.get(key, default)


def set_state(key: str, value):
    """设置状态值"""
    with lock:
        STATE[key] = value


def update_visual_emotion(emotion: str, confidence: float, emotion_probs: Dict[str, float]):
    """更新视觉情绪识别结果"""
    with lock:
        STATE["visual_emotion"] = emotion
        STATE["visual_confidence"] = confidence
        STATE["visual_emotion_probs"] = emotion_probs
        STATE["last_visual_update"] = time.time()
        STATE["frame_counter"] += 1


def update_audio_emotion(emotion: str, confidence: float):
    """更新音频情绪识别结果"""
    with lock:
        STATE["audio_emotion"] = emotion
        STATE["audio_confidence"] = confidence
        STATE["last_audio_update"] = time.time()


def update_final_emotion(emotion: str):
    """更新融合后的最终情绪"""
    with lock:
        STATE["final_emotion"] = emotion


def get_visual_state() -> Dict:
    """获取视觉状态（用于界面显示）"""
    with lock:
        return {
            "emotion": STATE["visual_emotion"],
            "confidence": STATE["visual_confidence"],
            "emotion_probs": STATE["visual_emotion_probs"],
            "frame_counter": STATE["frame_counter"],
            "last_update": STATE["last_visual_update"]
        }


def get_fusion_state() -> Dict:
    """获取融合状态（用于决策）"""
    with lock:
        return {
            "visual_emotion": STATE["visual_emotion"],
            "visual_confidence": STATE["visual_confidence"],
            "audio_emotion": STATE["audio_emotion"],
            "audio_confidence": STATE["audio_confidence"],
            "final_emotion": STATE["final_emotion"]
        }


def record_inference_time(module: str, time_ms: float):
    """记录推理耗时"""
    with lock:
        STATE["inference_times"].append({
            "module": module,
            "time": time_ms,
            "timestamp": time.time()
        })
        # 只保留最近100条记录
        if len(STATE["inference_times"]) > 100:
            STATE["inference_times"] = STATE["inference_times"][-100:]


def get_avg_inference_time(module: Optional[str] = None) -> float:
    """获取平均推理耗时"""
    with lock:
        times = STATE["inference_times"]
        if module:
            times = [t for t in times if t["module"] == module]

        # 只计算最近10秒的记录
        current_time = time.time()
        times = [t for t in times if current_time - t["timestamp"] < 10]

        if not times:
            return 0.0
        return sum(t["time"] for t in times) / len(times)
