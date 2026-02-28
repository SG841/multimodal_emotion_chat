"""
视觉感知模块
基于 Vision Transformer (ViT) 进行人脸表情情绪识别
"""

import numpy as np
from PIL import Image
from transformers import pipeline
import time
from typing import Dict, Tuple, Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    VISION_MODEL,
    VISION_SKIP_FRAMES,
    VISION_DEVICE,
    update_visual_emotion,
    record_inference_time
)


class VisionEmotionDetector:
    """视觉情绪识别器"""

    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化视觉情绪识别器

        Args:
            model_name: 模型名称，默认使用配置中的模型
            device: 设备，None表示自动检测
        """
        self.model_name = VISION_MODEL
        self.device = VISION_DEVICE
        self.classifier = None
        self.frame_count = 0
        self.last_emotion = "Neutral"
        self.last_confidence = 0.0
        self.last_emotion_probs = {}

        print(f"[视觉模块] 正在加载模型: {self.model_name}")
        self._load_model()
        print(f"[视觉模块] 模型加载完成")

    def _load_model(self):
        """加载预训练模型"""
        try:
            self.classifier = pipeline(
                "image-classification",
                model=self.model_name,
                device=self.device
            )
        except Exception as e:
            print(f"[视觉模块] 模型加载失败: {e}")
            raise

    def predict_emotion(
        self,
        image: np.ndarray,
        skip_frames: int = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        预测图像中的情绪

        Args:
            image: 输入图像 (numpy array, RGB格式)
            skip_frames: 跳帧数，None表示使用配置值

        Returns:
            (emotion, confidence, emotion_probs)
            - emotion: 情绪标签
            - confidence: 置信度 (0.0-1.0)
            - emotion_probs: 各情绪的概率分布
        """
        skip_frames = skip_frames or VISION_SKIP_FRAMES
        self.frame_count += 1

        # 跳帧检测
        if self.frame_count % skip_frames != 0:
            return (
                self.last_emotion,
                self.last_confidence,
                self.last_emotion_probs
            )

        start_time = time.time()

        # 转换图像格式
        if image is None:
            return "Neutral", 0.0, {}

        # 确保图像是 RGB 格式
        # 注意：从 Gradio 传入的图像已经是 RGB 格式，无需转换
        # 只有当输入是 BGR 格式时才需要转换
        pass

        # 转换为 PIL Image
        pil_image = Image.fromarray(image)

        # 情绪识别推理
        try:
            results = self.classifier(pil_image, top_k=7)

            # 提取最高概率的情绪
            top_result = results[0]
            emotion = top_result["label"]
            confidence = top_result["score"]

            # 构建情绪概率字典
            emotion_probs = {r["label"]: r["score"] for r in results}

            # 更新缓存
            self.last_emotion = emotion
            self.last_confidence = confidence
            self.last_emotion_probs = emotion_probs

            # 更新全局状态
            update_visual_emotion(emotion, confidence, emotion_probs)

            # 记录推理耗时
            inference_time = (time.time() - start_time) * 1000
            record_inference_time("vision", inference_time)

            return emotion, confidence, emotion_probs

        except Exception as e:
            print(f"[视觉模块] 推理错误: {e}")
            return self.last_emotion, self.last_confidence, self.last_emotion_probs

    def get_top_emotions(self, top_k: int = 3) -> list:
        """
        获取概率最高的前K个情绪

        Args:
            top_k: 返回的个数

        Returns:
            [(emotion, confidence), ...] 列表
        """
        sorted_emotions = sorted(
            self.last_emotion_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:top_k]

    def reset(self):
        """重置状态"""
        self.frame_count = 0
        self.last_emotion = "Neutral"
        self.last_confidence = 0.0
        self.last_emotion_probs = {}


# 创建全局检测器实例（懒加载）
_detector = None


def get_detector() -> VisionEmotionDetector:
    """获取全局检测器实例（懒加载）"""
    global _detector
    if _detector is None:
        _detector = VisionEmotionDetector()
    return _detector


def predict_emotion(image: np.ndarray, skip_frames: int = None):
    """
    预测图像情绪（便捷函数）

    Args:
        image: 输入图像
        skip_frames: 跳帧数

    Returns:
        (emotion, confidence, emotion_probs)
    """
    detector = get_detector()
    return detector.predict_emotion(image, skip_frames)


if __name__ == "__main__":
    # 测试代码
    print("视觉模块测试")
    detector = VisionEmotionDetector()

    # 创建测试图像（全白色）
    test_image = np.ones((224, 224, 3), dtype=np.uint8) * 255

    # 测试预测
    emotion, confidence, probs = detector.predict_emotion(test_image, skip_frames=1)
    print(f"测试结果: {emotion}, 置信度: {confidence:.2%}")
    print(f"情绪概率: {probs}")
