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
import asyncio
import collections

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入视觉模块
try:
    from modules.vision import predict_emotion
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"视觉模块导入失败: {e}")
    VISION_AVAILABLE = False

# 导入音频模块
try:
    from modules.audio import transcribe_audio
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"音频模块导入失败: {e}")
    AUDIO_AVAILABLE = False

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

        # === 新增状态管理 ===
        self.is_recording = False          # 正在录音标记
        self.is_asr_processing = False     # 正在 ASR 推理标记（算力避让锁）
        self.recording_emotion_buffer = [] # 录音期间的情感缓冲区
        self.last_proc_time = 0          # 【频率控制】记录上次处理时间戳

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

                    # 摄像头视频流 - 强制低分辨率采集 (320x240 减少带宽)
                    # 注意: height/width 参数会影响 getUserMedia 约束
                    self.video_input = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="实时视频流",
                        elem_id="video-stream",
                        height=240,   # 强制采集高度 240
                        width=320     # 强制采集宽度 320
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
                            type="filepath",  # 返回文件路径
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
        """绑定界面事件 - 开启并发与异步通道"""

        # 视频流：高频预测 + 录音期间数据累积
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
            show_progress="hidden",
            concurrency_limit=1,  # 单线程循环，防止队列积压
            queue=False
        )

        # 新增：录音开始事件
        self.audio_input.start_recording(
            fn=self.handle_recording_start,
            outputs=[self.recording_status]
        )

        # 录音结束：触发 ASR + 时序融合
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
            ],
            concurrency_limit=1,  # ASR 独占一个高优先级通道
            queue=True
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

    def handle_recording_start(self):
        """录音开始：初始化缓冲区"""
        self.is_recording = True
        self.recording_emotion_buffer = []  # 清空之前的记录
        return "正在录音：系统正在同步分析您的全段表情..."

    async def process_video_stream(
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

        # 【频率控制】强制每 150ms 最多处理一帧（约 6-7 FPS），减少 CPU 负担
        curr_time = time.time()
        if curr_time - self.last_proc_time < 0.15:
            return (gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip())
        self.last_proc_time = curr_time

        # 【优先级调度】ASR 运行期间，视觉模块停止向 GPU 发送新任务，防止抢占 ASR 的 WebSocket 管道
        if self.is_asr_processing:
            return (
                self.current_emotion,
                f"{self.visual_confidence:.2%}",
                self._get_emotion_bars(self.last_emotion_probs),
                self._frame_count_internal,
                time.strftime("%H:%M:%S"),
                "GPU 算力优先分配给 ASR 推理...",
                "--"
            )

        # 调用视觉模块进行情绪识别
        try:
            if VISION_AVAILABLE:
                # 【优化】直接传 numpy 数组，torch.from_numpy 在 vision.py 中直接接管显存，避免 CPU 拷贝
                emotion, confidence, probs = await asyncio.to_thread(predict_emotion, video_frame, skip_frames=1)

                # 更新实时状态
                self.current_emotion, self.visual_confidence, self.last_emotion_probs = emotion, confidence, probs
                emotion_probs = probs

                # 【核心功能】如果在录音，则把每一帧结果存入缓冲区
                if self.is_recording:
                    self.recording_emotion_buffer.append(emotion)
        except Exception as e:
            print(f"Vision Error: {e}")
            emotion = self.current_emotion
            confidence = self.visual_confidence
            emotion_probs = self.last_emotion_probs

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

    async def process_dialogue(
        self,
        audio_path: Optional[str],
        current_text: str
    ) -> Tuple:
        """
        对话处理：执行 ASR 并融合录音全段的视觉特征

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
                "请先录音",
                gpu_mem
            )

        try:
            self.is_asr_processing = True  # 开启算力保护锁
            self.is_recording = False      # 标记录音结束

            # 1. 执行真实的 ASR 推理 (派发到独立线程)
            if AUDIO_AVAILABLE:
                user_text = await asyncio.to_thread(transcribe_audio, audio_path)
            else:
                user_text = "[音频模块未加载]"

            # 2. 【全程视觉融合】计算录音全过程出现频率最高的情绪（众数过滤噪声）
            if self.recording_emotion_buffer:
                # 统计出现次数最多的表情作为最终视觉决策
                visual_decision = collections.Counter(self.recording_emotion_buffer).most_common(1)[0][0]
                fusion_info = f"融合成功：分析了录音期间 {len(self.recording_emotion_buffer)} 帧表情，主要情绪：{visual_decision}"
            else:
                visual_decision = self.current_emotion
                fusion_info = "未捕获到录音期间的视觉数据，采用末帧。"

            # 3. 模拟融合逻辑（暂时保持占位数据，之后接入 SER）
            audio_emotion = "Sad"  # 占位
            audio_confidence = 0.85  # 占位

            # 融合决策改用录音期间的综合结果 visual_decision
            final_emotion = audio_emotion if audio_confidence > 0.8 else visual_decision
            visual_weight = f"{(1 - audio_confidence) * 100:.0f}%"
            audio_weight = f"{audio_confidence * 100:.0f}%"

            # 更新融合显示
            if MONITOR_AVAILABLE:
                fusion_text = monitor.format_fusion_report(
                    v_emo=visual_decision,
                    v_conf=self.visual_confidence,
                    a_emo=audio_emotion,
                    a_conf=audio_confidence,
                    final=final_emotion
                )
            else:
                fusion_text = f"时序融合决策: {final_emotion}\n"
                fusion_text += f"录音期间视觉: {visual_decision}\n"
                fusion_text += f"音频情感: {audio_emotion} ({audio_confidence:.0%})"

            # TODO: 调用LLM生成回复
            # from modules.llm import generate_empathetic_response
            # response_text = ...
            response_text = f"我感受到了你的情绪变化。你说：{user_text}"

            # 更新对话历史 (Gradio 6.0+ 格式: 使用 role 和 content)
            self.chat_history.append({"role": "user", "content": user_text})
            self.chat_history.append({"role": "assistant", "content": response_text})

            # 获取 GPU 显存
            gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

            # 更新日志
            log_entries = [
                f"[{time.strftime('%H:%M:%S')}] 录音完成: {audio_path}",
                f"[{time.strftime('%H:%M:%S')}] ASR识别: {user_text}",
                f"[{time.strftime('%H:%M:%S')}] 录音期间视觉众数: {visual_decision}",
                f"[{time.strftime('%H:%M:%S')}] 时序融合决策: {final_emotion}",
                fusion_info
            ]
            self.system_logs.extend(log_entries)
            if len(self.system_logs) > 50:
                self.system_logs = self.system_logs[-50:]

            return (
                self.chat_history,
                user_text,
                None,
                fusion_text,
                visual_weight,
                audio_weight,
                "\n".join(self.system_logs[-10:]),
                "处理完成",
                gpu_mem
            )
        finally:
            self.is_asr_processing = False  # 释放算力锁定，视觉恢复

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
            {"role": "user", "content": "我最近工作压力很大"},
            {"role": "assistant", "content": "听起来你最近承受了很大的工作压力，能具体说说是什么让你感到压力吗？"},
            {"role": "user", "content": "老板总是给我加任务"},
            {"role": "assistant", "content": "老板不断给你增加任务确实会让人感到疲惫和焦虑。你有没有尝试过和老板沟通这个问题？"},
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
    """主函数 - 在启动界面前完成模型加载"""

    print("🚀 [系统启动] 正在预加载多模态模型到 GPU 显存...")

    # 1. 预加载视觉模型
    if VISION_AVAILABLE:
        from modules.vision import get_detector
        get_detector()  # 此调用会触发 ViT 模型的加载
        print("✅ 视觉模块预加载完成")

    # 2. 预加载音频模型
    if AUDIO_AVAILABLE:
        from modules.audio import _get_model
        _get_model()    # 此调用会触发 Whisper 模型加载到 GPU
        print("✅ 音频转录模型预加载完成")

    print("🎉 所有模型预加载完成，启动 Gradio 界面...")

    # 3. 实例化界面
    app = EmotionChatInterface()

    # 【必须开启】这是异步和并发的前提开关
    app.interface.queue()

    # 启动服务
    app.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        show_error=True,
        max_threads=40  # 增加线程池大小适配 L40S
    )


if __name__ == "__main__":
    main()
 