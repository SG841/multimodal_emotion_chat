"""
视觉感知模块
基于 Vision Transformer (ViT) 进行人脸表情情绪识别
优化：抛弃 pipeline，使用原生 torch + FP16 + GPU预处理，显著提升 L4S 性能
"""

import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn.functional as F
from torchvision import transforms
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
    """视觉情绪识别器 - 原生 torch + FP16 + GPU预处理 优化版"""

    def __init__(self, model_name: str = None, device: str = None):
        """
        初始化视觉情绪识别器

        Args:
            model_name: 模型名称，默认使用配置中的模型
            device: 设备，None表示自动检测
        """
        self.model_name = VISION_MODEL
        self.device = torch.device(VISION_DEVICE if VISION_DEVICE else "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.processor = None
        self.gpu_transform = None  # GPU预处理transform
        self.frame_count = 0
        self.last_emotion = "Neutral"
        self.last_confidence = 0.0
        self.last_emotion_probs = {}

        print(f"[视觉模块] 正在加载模型: {self.model_name} on {self.device}")
        print(f"[视觉模块] 使用 FP16 推理模式 + GPU预处理")
        self._load_model()
        print(f"[视觉模块] 模型加载完成")

    def _load_model(self):
        """直接使用 torch 加载模型，启用 FP16 + GPU预处理"""
        try:
            # 直接加载模型和处理器，不使用 pipeline
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)

            # 尝试加载模型，不指定 attn_implementation（模型可能不支持）
            try:
                self.model = AutoModelForImageClassification.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16  # FP16 半精度推理
                )
                print(f"[视觉模块] FP16 推理模式已启用")
            except Exception as model_err:
                # 如果 FP16 加载失败，回退到 FP32
                print(f"[视觉模块] FP16 加载失败，回退到 FP32: {model_err}")
                self.model = AutoModelForImageClassification.from_pretrained(self.model_name)

            # 将模型移动到目标设备并设置为评估模式
            self.model = self.model.to(self.device)
            self.model.eval()

            # 禁用梯度计算，减少显存占用
            torch.set_grad_enabled(False)

            # 创建GPU预处理transform（替代CPU的processor）
            self._create_gpu_transform()

            print(f"[视觉模块] 模型已加载到 {self.device}")
            print(f"DEBUG [视觉]: 模型实际运行设备 -> {self.model.device}")

        except Exception as e:
            print(f"[视觉模块] 模型加载失败: {e}")
            raise

    def _create_gpu_transform(self):
        """
        创建GPU预处理transform，避免CPU瓶颈
        从processor.config提取预处理参数
        """
        # 从processor获取预处理参数
        if hasattr(self.processor, 'image_mean') and hasattr(self.processor, 'image_std'):
            mean = self.processor.image_mean
            std = self.processor.image_std
        else:
            # 默认ImageNet归一化
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]

        # 获取目标尺寸
        if hasattr(self.processor, 'size'):
            size = self.processor.size.get('shortest_edge', 224)
        else:
            size = 224

        # 创建GPU预处理pipeline - 针对tensor的transform
        # 注意：transform需要先归一化到[0,1]然后才做其他处理
        self.gpu_transform = transforms.Compose([
            transforms.Resize(size),  # GPU加速的resize
            transforms.CenterCrop(size),  # 居中裁剪
            transforms.Normalize(mean=mean, std=std)  # GPU加速的归一化
        ])

        print(f"[视觉模块] GPU预处理transform已创建 (size={size})")

    def predict_emotion(
        self,
        image: np.ndarray,
        skip_frames: int = None
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        预测图像中的情绪 - 原生 torch + FP16 + GPU预处理

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
        start = time.perf_counter()  # 【精准测速】高精度计时开始

        # 转换图像格式
        if image is None:
            return "Neutral", 0.0, {}

        # 情绪识别推理 - 原生 torch + 真正的GPU预处理 + 异步拷贝
        try:
            with torch.inference_mode():  # 比 torch.no_grad() 更高效
                # 1. 【异步拷贝】non_blocking=True 让 CPU 不等待 GPU 完成拷贝
                # 直接从 numpy 转为 tensor，不经过 PIL，不经过 CPU 缩放
                img_tensor = torch.from_numpy(image.copy()).to(self.device, non_blocking=True).permute(2, 0, 1)

                # 【测速】搬运耗时
                t1 = time.perf_counter()

                # 2. 转换为半精度并归一化到 [0, 1]
                # L4S 处理这种张量计算快如闪电
                img_tensor = img_tensor.half() / 255.0

                # 3. 真正的 GPU 预处理：Resize, Crop, Normalize
                # 这步是在 GPU 上并行的
                img_tensor = self.gpu_transform(img_tensor).unsqueeze(0)

                # 【测速】预处理耗时
                t2 = time.perf_counter()

                # 4. 推理
                outputs = self.model(pixel_values=img_tensor)

                # 【测速】推理耗时
                t3 = time.perf_counter()

                # 【调试输出】各环节耗时分析
                print(f"[Vision] 搬运: {(t1-start)*1000:.2f}ms | 预处理: {(t2-t1)*1000:.2f}ms | 推理: {(t3-t2)*1000:.2f}ms")

                # 5. 后处理 (直接在 GPU 上算 Softmax)
                logits = outputs.logits
                probs = F.softmax(logits, dim=-1)[0]

                # 获取 top-7
                top_k = min(7, len(probs))
                top_probs, top_indices = torch.topk(probs, top_k)

                # 转换为 CPU numpy 数组
                top_probs = top_probs.cpu().numpy()
                top_indices = top_indices.cpu().numpy()

                # 获取标签
                id2label = self.model.config.id2label
                top_labels = [id2label[idx] for idx in top_indices]

                # 构建结果
                emotion = top_labels[0]
                confidence = float(top_probs[0])
                emotion_probs = {label: float(prob) for label, prob in zip(top_labels, top_probs)}

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
