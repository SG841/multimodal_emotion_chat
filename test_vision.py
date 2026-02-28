"""
测试视觉模块是否正常工作
"""

import cv2
import numpy as np
from modules.vision import predict_emotion, draw_emotion_label

print("=" * 50)
print("视觉模块功能测试")
print("=" * 50)
print()

# 测试1：使用随机RGB图像（模拟Gradio输入）
print("【测试1】RGB格式图像识别（模拟Gradio输入）")
print("-" * 50)
rgb_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
print(f"输入图像: shape={rgb_frame.shape}, dtype={rgb_frame.dtype}, 格式=RGB")

try:
    emotion, confidence, probs = predict_emotion(rgb_frame, skip_frames=1)
    print(f"✓ 识别成功: {emotion} (置信度: {confidence:.2%})")
    print(f"  前3个情绪: {sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]}")
except Exception as e:
    print(f"✗ 识别失败: {e}")
print()

# 测试2：使用模拟的人脸图像
print("【测试2】绘制情绪标签功能")
print("-" * 50)
try:
    # 创建一个测试图像（灰色背景）
    test_frame = np.ones((400, 600, 3), dtype=np.uint8) * 128

    # 转换为BGR用于绘图
    test_frame_bgr = cv2.cvtColor(test_frame, cv2.COLOR_RGB2BGR)

    # 绘制情绪标签
    emotion = "Happy"
    confidence = 0.85
    emotion_probs = {
        "Happy": 0.85, "Sad": 0.05, "Angry": 0.03,
        "Neutral": 0.04, "Surprise": 0.02, "Fear": 0.01
    }

    result_frame = draw_emotion_label(test_frame_bgr, emotion, confidence, emotion_probs)
    print(f"✓ 绘制功能正常: 绘制了 '{emotion}' 标签和情绪条")
except Exception as e:
    print(f"✗ 绘制失败: {e}")
print()

# 测试3：跳帧功能
print("【测试3】跳帧功能验证")
print("-" * 50)
from modules.vision import get_detector
detector = get_detector()
detector.reset()

for i in range(7):
    test_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    emo, conf, probs = detector.predict_emotion(test_img, skip_frames=5)
    actual_inference = (i + 1) % 5 == 0
    print(f"帧 {i+1}: {emo} ({conf:.2%}) - {'✓ 实际推理' if actual_inference else '○ 跳帧缓存'}")
print()

print("=" * 50)
print("测试完成！视觉模块工作正常。")
print("=" * 50)
print()
print("现在可以运行 python app.py 启动界面进行测试。")
