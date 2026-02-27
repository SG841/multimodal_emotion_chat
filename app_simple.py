"""
简化版界面 - 用于诊断问题
"""

import gradio as gr
import numpy as np

print("=== 简化版视觉识别界面 ===")
print()

# 导入视觉模块
from modules.vision import predict_emotion

def process_frame(image):
    """处理视频帧"""
    if image is None:
        return image, "等待中...", "--"

    # 情绪识别
    emotion, confidence, probs = predict_emotion(image, skip_frames=5)

    return image, emotion, f"{confidence:.2%}"

# 创建简单界面
with gr.Blocks(title="简化版视觉识别") as demo:
    gr.Markdown("# 👁️ 视觉情绪识别 - 简化版")
    gr.Markdown("对着摄像头做表情，系统会实时识别你的情绪")

    with gr.Row():
        video_input = gr.Image(
            sources=["webcam"],
            streaming=True,
            label="摄像头",
            height=400
        )

    with gr.Row():
        emotion_display = gr.Textbox(
            label="识别情绪",
            value="等待中...",
            interactive=False
        )
        confidence_display = gr.Textbox(
            label="置信度",
            value="--",
            interactive=False
        )

    # 绑定事件
    video_input.change(
        fn=process_frame,
        inputs=[video_input],
        outputs=[video_input, emotion_display, confidence_display]
    )

print("界面创建完成，正在启动...")
print("请在浏览器中打开: http://localhost:7860")
print()

# 启动
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    show_error=True,
    quiet=False
)
