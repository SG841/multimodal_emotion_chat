"""
视觉感知模块
基于 Vision Transformer (ViT) 进行人脸表情情绪识别
"""

import cv2
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
        if len(image.shape) == 3 and image.shape[2] == 3:
            # OpenCV 默认是 BGR，需要转为 RGB
            if image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 转换为 PIL Image
        pil_image = Image.fromarray(image)

        # 情绪识别推理
        try:
            results = self.classifier(pil_image)

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

    def draw_emotion_label(
        self,
        image: np.ndarray,
        emotion: str,
        confidence: float,
        emotion_probs: Dict[str, float] = None
    ) -> np.ndarray:
        """
        在图像上绘制情绪标签

        Args:
            image: 输入图像 (BGR格式)
            emotion: 情绪标签
            confidence: 置信度
            emotion_probs: 情绪概率分布

        Returns:
            绘制标签后的图像 (BGR格式)
        """
        # 创建图像副本
        result = image.copy()

        # 绘制情绪标签背景
        label_text = f"Emotion: {emotion} ({confidence:.1%})"

        # 字体设置
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2

        # 计算文本大小
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, font, font_scale, thickness
        )

        # 绘制背景框
        margin = 10
        bg_width = text_width + 2 * margin
        bg_height = text_height + 2 * margin + baseline

        # 根据情绪选择颜色
        emotion_colors = {
            "Happy": (0, 255, 0),      # 绿色
            "Sad": (255, 0, 0),        # 红色
            "Angry": (0, 0, 255),      # 蓝色
            "Neutral": (128, 128, 128),  # 灰色
            "Surprise": (0, 165, 255),  # 橙色
            "Fear": (255, 0, 255),     # 紫色
            "Disgust": (0, 255, 255),  # 青色
        }
        color = emotion_colors.get(emotion, (128, 128, 128))

        # 绘制半透明背景
        overlay = result.copy()
        cv2.rectangle(
            overlay,
            (10, 10),
            (10 + bg_width, 10 + bg_height),
            color,
            -1
        )
        result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

        # 绘制边框
        cv2.rectangle(
            result,
            (10, 10),
            (10 + bg_width, 10 + bg_height),
            color,
            2
        )

        # 绘制文本
        cv2.putText(
            result,
            label_text,
            (10 + margin, 10 + margin + text_height),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA
        )

        # 如果提供了情绪概率，绘制进度条
        if emotion_probs:
            self._draw_emotion_bars(result, emotion_probs, 10 + bg_height + 10)

        return result

    def _draw_emotion_bars(
        self,
        image: np.ndarray,
        emotion_probs: Dict[str, float],
        start_y: int,
        bar_height: int = 20
    ):
        """
        绘制情绪概率条形图

        Args:
            image: 输入图像
            emotion_probs: 情绪概率字典
            start_y: 起始Y坐标
            bar_height: 条形高度
        """
        emotion_colors = {
            "Happy": (0, 255, 0),
            "Sad": (255, 0, 0),
            "Angry": (0, 0, 255),
            "Neutral": (128, 128, 128),
            "Surprise": (0, 165, 255),
            "Fear": (255, 0, 255),
            "Disgust": (0, 255, 255),
        }

        # 取前5个概率最高的情绪
        sorted_emotions = sorted(
            emotion_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        for i, (emotion, prob) in enumerate(sorted_emotions):
            y = start_y + i * (bar_height + 5)

            # 绘制背景条
            bar_width = 200
            cv2.rectangle(
                image,
                (10, y),
                (10 + bar_width, y + bar_height),
                (50, 50, 50),
                -1
            )

            # 绘制概率条
            prob_width = int(bar_width * prob)
            color = emotion_colors.get(emotion, (128, 128, 128))
            cv2.rectangle(
                image,
                (10, y),
                (10 + prob_width, y + bar_height),
                color,
                -1
            )

            # 绘制标签
            label = f"{emotion}: {prob:.1%}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                image,
                label,
                (10 + bar_width + 10, y + bar_height - 5),
                font,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

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


def draw_emotion_label(
    image: np.ndarray,
    emotion: str,
    confidence: float,
    emotion_probs: Dict[str, float] = None
):
    """
    在图像上绘制情绪标签（便捷函数）

    Args:
        image: 输入图像
        emotion: 情绪标签
        confidence: 置信度
        emotion_probs: 情绪概率分布

    Returns:
        绘制标签后的图像
    """
    detector = get_detector()
    return detector.draw_emotion_label(image, emotion, confidence, emotion_probs)


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
