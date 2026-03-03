"""
Gradio 交互界面模块
多模态情感感知共情对话系统的前端界面
"""

import gradio as gr
import numpy as np
from typing import Optional, Tuple
import time
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入视觉模块
try:
    from modules.vision import predict_emotion
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"视觉模块导入失败: {e}")
    VISION_AVAILABLE = False

# 导入监控模块
try:
    from utils.monitor import monitor
    MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"监控模块导入失败: {e}")
    MONITOR_AVAILABLE = False


class EmotionChatInterface:
    """共情对话系统交互界面"""

    def __init__(self):
        """初始化界面组件"""
        # 全局状态（实际使用时会从 modules.shared_state 导入）
        self.current_emotion = "Neutral"
        self.visual_confidence = 0.0
        self.chat_history = []
        self.system_logs = []
        self._frame_count_internal = 0  # 内部帧计数器（整数）
        self.last_emotion_probs = {}   # 保存最后一次的情绪概率

        # 创建界面
        self.create_interface()

    def create_interface(self):
        """创建 Gradio 界面"""
        with gr.Blocks(title="多模态情感感知共情对话系统") as self.interface:

            # 标题栏
            gr.Markdown(
                """
                # 🤖 多模态情感感知共情对话系统

                基于视觉与听觉的情感识别 + 大语言模型共情对话
                """
            )

            # 主布局：左侧感知区，右侧交互区
            with gr.Row():
                # ===== 左侧：感知区域 =====
                with gr.Column(scale=3):
                    gr.Markdown("### 👁️ 视觉感知区")

                    # 摄像头视频流
                    self.video_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="实时视频流",
                        elem_id="video-stream",
                        height=400,
                        width=640
                    )

                    # 情绪状态显示卡片
                    with gr.Row():
                        with gr.Column():
                            self.emotion_display = gr.Textbox(
                                label="当前识别情绪",
                                value="等待中...",
                                interactive=False,
                                elem_id="emotion-display"
                            )
                        with gr.Column():
                            self.confidence_display = gr.Textbox(
                                label="视觉置信度",
                                value="--",
                                interactive=False
                            )

                    # 情绪进度条可视化
                    self.emotion_bars = gr.HTML(
                        value=self._get_empty_emotion_bars(),
                        label="情绪概率分布"
                    )

                    # 视觉状态统计
                    with gr.Row():
                        self.frame_count = gr.Number(
                            label="已处理帧数",
                            value=0,
                            interactive=False
                        )
                        self.last_update = gr.Textbox(
                            label="最后更新时间",
                            value="--",
                            interactive=False
                        )

                # ===== 右侧：交互区域 =====
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 对话交互区")

                    # 对话历史记录
                    self.chat_display = gr.Chatbot(
                        label="对话历史",
                        height=300,
                        avatar_images=(
                            None,  # 用户头像
                            "🤖"  # 系统头像
                        )
                    )

                    # 语音输入区域
                    with gr.Row():
                        self.audio_input = gr.Audio(
                            sources=["microphone"],
                            type="filepath",
                            label="语音输入"
                        )
                        self.recording_status = gr.Textbox(
                            value="点击录音开始说话...",
                            interactive=False,
                            scale=2
                        )

                    # 文本显示（语音识别结果）
                    self.recognized_text = gr.Textbox(
                        label="语音识别结果",
                        placeholder="您的语音将在这里显示...",
                        lines=2,
                        interactive=False
                    )

                    # 语音播放区域
                    self.audio_output = gr.Audio(
                        label="系统回复语音",
                        autoplay=True
                    )

                    # 操作按钮
                    with gr.Row():
                        self.clear_btn = gr.Button("清空对话", variant="secondary")
                        self.example_btn = gr.Button("示例对话", variant="secondary")

            # ===== 底部：系统监控区域 =====
            gr.Markdown("### 📊 系统监控日志")

            with gr.Row():
                with gr.Column(scale=1):
                    # 融合决策显示
                    self.fusion_display = gr.Textbox(
                        label="多模态融合决策",
                        value="等待数据...",
                        interactive=False,
                        lines=3
                    )

                    # 融合权重显示
                    with gr.Row():
                        self.visual_weight = gr.Textbox(
                            label="视觉权重",
                            value="--",
                            interactive=False,
                            scale=1
                        )
                        self.audio_weight = gr.Textbox(
                            label="听觉权重",
                            value="--",
                            interactive=False,
                            scale=1
                        )

                with gr.Column(scale=2):
                    # 系统日志
                    self.log_display = gr.Textbox(
                        label="系统运行日志",
                        value="系统初始化完成...",
                        interactive=False,
                        lines=8,
                        max_lines=10,
                        autoscroll=True
                    )

                    # 性能指标
                    with gr.Row():
                        self.gpu_memory = gr.Textbox(
                            label="GPU 显存占用",
                            value="-- MB",
                            interactive=False
                        )

            # 绑定事件
            self._bind_events()

    def _bind_events(self):
        """绑定界面事件"""

        # 视频流处理 - 实时情绪识别
        # 关键优化：不从outputs返回video_input，避免图像往返开销
        # 前端video_input已经在streaming显示，只需更新文本UI
        self.video_input.stream(
            fn=self.process_video_stream,
            inputs=[self.video_input],
            outputs=[
                self.emotion_display,
                self.confidence_display,
                self.emotion_bars,
                self.frame_count,
                self.last_update,
                self.log_display,
                self.gpu_memory
            ],
            show_progress="hidden"
        )

        # 录音结束触发对话处理
        self.audio_input.stop_recording(
            fn=self.process_dialogue,
            inputs=[self.audio_input, self.recognized_text],
            outputs=[
                self.chat_display,
                self.recognized_text,
                self.audio_output,
                self.fusion_display,
                self.visual_weight,
                self.audio_weight,
                self.log_display,
                self.recording_status,
                self.gpu_memory
            ]
        )

        # 清空对话按钮
        self.clear_btn.click(
            fn=self.clear_chat,
            outputs=[
                self.chat_display,
                self.recognized_text,
                self.audio_output,
                self.log_display
            ]
        )

        # 示例对话按钮
        self.example_btn.click(
            fn=self.show_example
        )

    def process_video_stream(
        self,
        video_frame: Optional[np.ndarray]
    ) -> Tuple[
        str, str, str, int, str, str, str
    ]:
        """
        处理视频流 - 情绪识别
        优化：不返回video_frame，避免图像往返开销

        Args:
            video_frame: 视频帧 (numpy array, RGB)

        Returns:
            情绪标签、置信度、情绪条HTML、帧计数、更新时间、日志
        """
        if video_frame is None:
            gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"
            return "等待中...", "--", self._get_empty_emotion_bars(), 0, "--", "等待视频流...", gpu_mem

        # 调用视觉模块进行情绪识别
        try:
            if VISION_AVAILABLE:
                emotion, confidence, emotion_probs = predict_emotion(video_frame, skip_frames=5)
            else:
                # 视觉模块不可用时使用模拟数据
                import random
                emotions = ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]
                emotion = random.choice(emotions)
                confidence = round(random.uniform(0.6, 0.95), 2)
                emotion_probs = {
                    "Happy": 0.1, "Sad": 0.1, "Angry": 0.1,
                    "Neutral": 0.1, "Surprise": 0.1, "Fear": 0.1
                }
                emotion_probs[emotion] = confidence
                for k in emotion_probs:
                    emotion_probs[k] = round(emotion_probs[k], 2)
        except Exception as e:
            print(f"视觉识别错误: {e}")
            emotion = "Neutral"
            confidence = 0.0
            emotion_probs = {}

        # 更新状态
        self.current_emotion = emotion
        self.visual_confidence = confidence

        # 生成情绪条HTML
        emotion_bars_html = self._get_emotion_bars(emotion_probs)

        # 更新日志
        log_entry = f"[{time.strftime('%H:%M:%S')}] 视觉识别: {emotion} (置信度: {confidence})"
        self.system_logs.append(log_entry)
        if len(self.system_logs) > 50:
            self.system_logs.pop(0)

        log_display = "\n".join(self.system_logs[-10:])

        # 增加帧计数
        self._frame_count_internal += 1

        # 获取 GPU 显存
        gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

        return (
            emotion,
            f"{confidence:.2%}",
            emotion_bars_html,
            self._frame_count_internal,
            time.strftime("%H:%M:%S"),
            log_display,
            gpu_mem
        )

    def process_dialogue(
        self,
        audio_path: Optional[str],
        current_text: str
    ) -> Tuple:
        """
        处理用户对话 - 多模态融合 + LLM生成 + TTS播放

        Args:
            audio_path: 录音文件路径
            current_text: 当前显示的文本

        Returns:
            对话历史、识别文本、音频输出、融合决策、视觉权重、听觉权重、日志、耗时、状态
        """
        if not audio_path:
            gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"
            return (
                self.chat_history,
                current_text,
                None,
                "等待语音输入...",
                "--",
                "--",
                "\n".join(self.system_logs[-10:]),
                "请先点击录音",
                gpu_mem
            )

        start_time = time.time()

        # TODO: 调用听觉模块进行语音识别和情感识别
        # from modules.audio import transcribe_audio, recognize_speech_emotion
        # user_text, audio_emotion, audio_confidence = ...

        # 模拟数据（实际使用时删除）
        user_text = "我今天感觉不太好"
        audio_emotion = "Sad"
        audio_confidence = 0.85

        # TODO: 调用融合模块
        # from modules.fusion import fuse_emotions
        # final_emotion, visual_weight, audio_weight = ...

        # 模拟融合结果
        final_emotion = audio_emotion if audio_confidence > 0.8 else self.current_emotion
        visual_weight = f"{(1 - audio_confidence) * 100:.0f}%"
        audio_weight = f"{audio_confidence * 100:.0f}%"

        # 更新融合显示
        if MONITOR_AVAILABLE:
            fusion_text = monitor.format_fusion_report(
                v_emo=self.current_emotion,
                v_conf=self.visual_confidence,
                a_emo=audio_emotion,
                a_conf=audio_confidence,
                final=final_emotion
            )
        else:
            fusion_text = f"视觉情绪: {self.current_emotion} ({self.visual_confidence:.0%})\n"
            fusion_text += f"听觉情绪: {audio_emotion} ({audio_confidence:.0%})\n"
            fusion_text += f"最终决策: {final_emotion}"

        # TODO: 调用LLM生成回复
        # from modules.llm import generate_empathetic_response
        # response_text = ...

        # 模拟回复（实际使用时删除）
        response_text = "我听到你今天感觉不太好，能和我说说发生了什么吗？我在这里陪着你。"

        # TODO: 调用TTS生成语音
        # from modules.tts import text_to_speech
        # audio_output_path = ...

        # 模拟音频路径（实际使用时删除）
        audio_output_path = None

        # 更新对话历史
        self.chat_history.append([user_text, response_text])

        # 获取 GPU 显存
        gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

        # 更新日志
        log_entries = [
            f"[{time.strftime('%H:%M:%S')}] 录音完成: {audio_path}",
            f"[{time.strftime('%H:%M:%S')}] ASR识别: {user_text}",
            f"[{time.strftime('%H:%M:%S')}] 语音情感: {audio_emotion} ({audio_confidence:.0%})",
            f"[{time.strftime('%H:%M:%S')}] 融合决策: {final_emotion}",
            f"[{time.strftime('%H:%M:%S')}] LLM生成完成"
        ]
        self.system_logs.extend(log_entries)
        if len(self.system_logs) > 50:
            self.system_logs = self.system_logs[-50:]

        return (
            self.chat_history,
            user_text,
            audio_output_path,
            fusion_text,
            visual_weight,
            audio_weight,
            "\n".join(self.system_logs[-10:]),
            "处理完成",
            gpu_mem
        )

    def clear_chat(self):
        """清空对话历史"""
        self.chat_history = []
        self.system_logs = ["对话已清空"]
        return (
            self.chat_history,
            "",
            None,
            "对话已清空"
        )

    def show_example(self):
        """显示示例对话"""
        example_history = [
            ["我最近工作压力很大", "听起来你最近承受了很大的工作压力，能具体说说是什么让你感到压力吗？"],
            ["老板总是给我加任务", "老板不断给你增加任务确实会让人感到疲惫和焦虑。你有没有尝试过和老板沟通这个问题？"],
        ]
        self.chat_history = example_history
        self.chat_display.value = example_history

    def _get_empty_emotion_bars(self) -> str:
        """获取空的情绪条HTML"""
        # 使用当前已知情绪，如果没有则使用默认列表
        emotions = list(self.last_emotion_probs.keys()) if hasattr(self, 'last_emotion_probs') and self.last_emotion_probs else ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]
        colors = ["#4CAF50", "#2196F3", "#F44336", "#9E9E9E", "#FF9800", "#9C27B0"]
        html = "<div style='margin: 10px 0;'>"
        for i, emotion in enumerate(emotions):
            color = colors[i % len(colors)]
            html += f"""
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <span style='width: 80px; font-size: 12px;'>{emotion}</span>
                <div style='flex: 1; height: 20px; background: #e0e0e0; border-radius: 10px; margin-left: 10px;'>
                    <div style='width: 0%; height: 100%; background: {color}; border-radius: 10px;'></div>
                </div>
                <span style='width: 40px; text-align: right; font-size: 12px;'>0%</span>
            </div>
            """
        html += "</div>"
        return html

    def _get_emotion_bars(self, emotion_probs: dict) -> str:
        """
        获取情绪概率分布的HTML条形图

        Args:
            emotion_probs: 情绪概率字典 {emotion: probability}

        Returns:
            HTML格式的条形图
        """
        # 如果为空，返回空条形图
        if not emotion_probs:
            return self._get_empty_emotion_bars()

        # 保存最后一次的 emotion_probs 用于后续显示
        self.last_emotion_probs = emotion_probs

        # 按概率排序，取前6个情绪
        sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)[:6]

        colors = ["#4CAF50", "#2196F3", "#F44336", "#9E9E9E", "#FF9800", "#9C27B0"]
        html = "<div style='margin: 10px 0;'>"
        for i, (emotion, prob) in enumerate(sorted_emotions):
            color = colors[i % len(colors)]
            prob_percent = prob * 100
            html += f"""
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <span style='width: 80px; font-size: 12px;'>{emotion}</span>
                <div style='flex: 1; height: 20px; background: #e0e0e0; border-radius: 10px; margin-left: 10px;'>
                    <div style='width: {prob_percent}%; height: 100%; background: {color}; border-radius: 10px; transition: width 0.3s;'></div>
                </div>
                <span style='width: 40px; text-align: right; font-size: 12px;'>{prob_percent:.0f}%</span>
            </div>
            """
        html += "</div>"
        return html

    def _get_custom_css(self) -> str:
        """获取自定义CSS样式"""
        return """
        #emotion-display {
            font-size: 24px !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 10px !important;
        }
        /* 固定视频流容器大小 */
        #video-stream {
            border-radius: 10px !important;
        }
        #video-stream .preview-container {
            width: 640px !important;
            height: 400px !important;
            min-width: 640px !important;
            min-height: 400px !important;
            max-width: 640px !important;
            max-height: 400px !important;
        }
        #video-stream video,
        #video-stream img,
        #video-stream canvas {
            width: 640px !important;
            height: 400px !important;
            object-fit: cover !important;
        }
        #video-stream button {
            min-width: auto !important;
        }
        """

    def launch(self, **kwargs):
        """启动界面"""
        # 设置 Gradio 6.0 的主题和CSS
        theme = gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
        )
        self.interface.launch(
            theme=theme,
            css=self._get_custom_css(),
            **kwargs
        )


def main():
    """主函数 - 启动界面"""
    app = EmotionChatInterface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    main()
 