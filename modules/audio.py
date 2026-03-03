"""
听觉模块 - 语音识别
使用 faster-whisper 进行语音转录
"""

import os
import time
from typing import Optional
from faster_whisper import WhisperModel

# 全局模型实例（懒加载）
_model: Optional[WhisperModel] = None
_model_config = {
    "model_path": os.path.join(
        os.path.dirname(__file__),
        "../models/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
    ),  # 本地模型路径
    "device": "cuda",
    "compute_type": "float16"
}


def _get_model() -> WhisperModel:
    """
    获取 Whisper 模型实例（单例模式）

    Returns:
        WhisperModel 实例
    """
    global _model

    if _model is None:
        print(f"🚀 加载本地 Whisper 模型")
        print(f"📁 模型路径: {os.path.abspath(_model_config['model_path'])}")

        _model = WhisperModel(
            _model_config["model_path"],
            device=_model_config["device"],
            compute_type=_model_config["compute_type"]
        )

        print("✅ Whisper 模型加载完成")

    return _model


def transcribe_audio(audio_path: str) -> str:
    """
    将音频文件转录为文本

    Args:
        audio_path: 音频文件路径（支持 wav/mp3/ogg/flac/m4a/aac）

    Returns:
        转录的文本内容

    Raises:
        FileNotFoundError: 音频文件不存在
        Exception: 转录过程中的其他错误
    """
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 获取模型
    model = _get_model()

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行转录（针对中文优化）
        segments, info = model.transcribe(
            audio_path,
            beam_size=5,
            language="zh",
            initial_prompt="以下是普通话，请转录为简体中文。"
        )

        # 拼接所有片段
        text = "".join([segment.text for segment in segments])

        # 计算耗时
        elapsed_time = time.time() - start_time

        # 打印日志
        print(f"📝 转录完成: {text}")
        print(f"⏱️  耗时: {elapsed_time:.2f}s")
        print(f"🌍 检测语言: {info.language} (置信度: {info.language_probability:.2%})")

        return text.strip()

    except Exception as e:
        print(f"❌ 转录失败: {e}")
        raise
