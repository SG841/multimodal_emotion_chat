"""
听觉模块 - 语音识别和情感识别
使用 faster-whisper 进行语音转录
使用 emotion2vec 进行语音情感识别
"""

import os
import time
import numpy as np
from typing import Optional, Tuple, Dict
from faster_whisper import WhisperModel

# 全局模型实例（懒加载）
_model: Optional[WhisperModel] = None
_emotion_model = None
_model_config = {
    "model_path": os.path.join(
        os.path.dirname(__file__),
        "../models/faster-whisper/models--Systran--faster-whisper-large-v3/snapshots/edaa852ec7e145841d8ffdb056a99866b5f0a478"
    ),  # 本地模型路径
    "device": "cuda",
    "compute_type": "float16"
}

# emotion2vec模型配置
_emotion_model_config = {
    "model_path": os.path.join(
        os.path.dirname(__file__),
        "../models/emotion2vec"
    ),  # 本地模型路径
    "device": "cuda"
}

# emotion2vec原始标签
EMOTION2VEC_LABELS = {
    0: "Angry",
    1: "Disgusted",
    2: "Fear",
    3: "Happy",
    4: "Neutral",
    5: "Other",
    6: "Sad",
    7: "Surprise",
    8: "Unknown"
}

# 项目需要预测的情绪编号
TARGET_EMOTION_IDS = [0, 2, 3, 4, 6, 7]  # angry, fearful, happy, neutral, sad, surprised


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

        # 强制检查底层模型所在的设备
        print(f"DEBUG [音频]: Whisper 模型已部署在 -> {_model.model.device}")
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


def _get_emotion_model():
    """
    获取 emotion2vec 模型实例（单例模式）

    Returns:
        emotion2vec 模型实例
    """
    global _emotion_model

    if _emotion_model is None:
        print(f"🚀 加载本地 emotion2vec 模型")
        print(f"📁 模型路径: {os.path.abspath(_emotion_model_config['model_path'])}")

        try:
            from funasr import AutoModel
            _emotion_model = AutoModel(
                model=_emotion_model_config["model_path"],
                device=_emotion_model_config["device"]
            )
            print("✅ emotion2vec 模型加载完成")
        except Exception as e:
            print(f"❌ emotion2vec 模型加载失败: {e}")
            raise

    return _emotion_model


def predict_audio_emotion(audio_path: str) -> Tuple[str, float, Dict[str, float]]:
    """
    使用 emotion2vec 预测音频的情感

    Args:
        audio_path: 音频文件路径

    Returns:
        (emotion_label, confidence, emotion_probs)
        emotion_label: 情感标签
        confidence: 置信度
        emotion_probs: 情感概率字典（六个目标情绪的概率之和为1）

    Raises:
        FileNotFoundError: 音频文件不存在
        Exception: 预测过程中的其他错误
    """
    # 检查文件是否存在
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音频文件不存在: {audio_path}")

    # 获取模型
    model = _get_emotion_model()

    # 记录开始时间
    start_time = time.time()

    try:
        # 执行情感识别
        result = model.generate(
            input=audio_path,
            granularity="utterance",
            extract_embedding=False
        )

        # 解析结果（根据emotion2vec的输出格式）
        # result 是一个列表，每个元素包含预测结果
        if result and len(result) > 0:
            pred = result[0]
            print(f"🔍 emotion2vec 原始输出: {pred}")

            # 获取情感标签和概率
            if "labels" in pred and "scores" in pred:
                labels = pred["labels"]
                scores = pred["scores"]
                print(f"🔍 labels: {labels}")
                print(f"🔍 scores: {scores}")

                # emotion2vec 返回的标签格式: ['生气/angry', '厌恶/disgusted', ...]
                # 需要映射到我们需要的英文情绪名称
                emotion_label_map = {
                    '生气/angry': 'Angry',
                    '厌恶/disgusted': None,  # 不在目标列表中
                    '恐惧/fearful': 'Fear',
                    '开心/happy': 'Happy',
                    '中立/neutral': 'Neutral',
                    '其他/other': None,  # 不在目标列表中
                    '难过/sad': 'Sad',
                    '吃惊/surprised': 'Surprise',
                    '<unk>': None  # 未知标签
                }

                # 只提取目标情绪的预测结果
                target_probs_raw = {}
                for label, score in zip(labels, scores):
                    mapped_emotion = emotion_label_map.get(label)
                    # 只保留目标情绪
                    if mapped_emotion in ["Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise"]:
                        target_probs_raw[mapped_emotion] = float(score)

                # 如果没有目标情绪概率，使用默认值
                total_raw_prob = sum(target_probs_raw.values())
                if total_raw_prob <= 0:
                    print("⚠️  emotion2vec 未预测到目标情绪，使用默认值")
                    return "Neutral", 0.0, {emo: 0.0 for emo in ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]}

                # 归一化：使六个目标情绪的概率之和为1
                target_probs_normalized = {
                    emo: prob / total_raw_prob
                    for emo, prob in target_probs_raw.items()
                }

                # 找出概率最高的情绪
                max_emotion = max(target_probs_normalized.items(), key=lambda x: x[1])
                emotion_label = max_emotion[0]
                confidence = max_emotion[1]

                # 计算耗时
                elapsed_time = time.time() - start_time
                print(f"😊 情感识别: {emotion_label} (置信度: {confidence:.2%})")
                print(f"⏱️  耗时: {elapsed_time:.2f}s")
                print(f"📊 归一化后概率分布: {target_probs_normalized}")
                print(f"✅ 六个情绪概率之和: {sum(target_probs_normalized.values()):.6f}")

                return emotion_label, confidence, target_probs_normalized
            else:
                print(f"⚠️  emotion2vec 输出格式异常: {pred}")
                return "Neutral", 0.0, {emo: 0.0 for emo in ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]}
        else:
            print("⚠️  emotion2vec 返回空结果")
            return "Neutral", 0.0, {emo: 0.0 for emo in ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]}

    except Exception as e:
        print(f"❌ 情感识别失败: {e}")
        raise
