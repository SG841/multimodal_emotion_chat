"""
Gradio 交互界面模块 - 角色绝对隔离架构版
支持普通用户态（多模态感知交互）与管理员态（全局监控），两侧页面物理隔离
"""
import gradio as gr
import numpy as np
import time
import sys
import os
import asyncio
import collections

# 引入数据库模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import db_manager

# 动态加载算法模块
try:
    from modules.vision import predict_emotion
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"视觉模块未加载: {e}")
    VISION_AVAILABLE = False

try:
    from modules.audio import transcribe_audio, predict_audio_emotion
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"音频模块未加载: {e}")
    AUDIO_AVAILABLE = False

try:
    from modules.llm import generate_empathetic_response, _get_llm_model
    LLM_AVAILABLE = True
except ImportError as e:
    print(f"LLM模块未加载: {e}")
    LLM_AVAILABLE = False

try:
    from modules.tts import generate_audio_reply
    TTS_AVAILABLE = True
except ImportError as e:
    print(f"TTS模块未加载: {e}")
    TTS_AVAILABLE = False

try:
    from utils.monitor import monitor
    MONITOR_AVAILABLE = True
except ImportError as e:
    print(f"监控模块未加载: {e}")
    MONITOR_AVAILABLE = False


class SystemInterface:
    def __init__(self):
        # 系统运行时用户状态
        self.current_user_id = None
        self.current_username = None
        self.current_session_id = None
        
        # 多模态感知状态
        self.current_emotion = "Neutral"
        self.visual_confidence = 0.0
        self.chat_history = []
        self.system_logs = []
        self._frame_count_internal = 0
        self.last_emotion_probs = {}
        
        # 录音与并发控制状态
        self.is_recording = False
        self.is_asr_processing = False  
        self.recording_emotion_buffer = [] 
        self.last_proc_time = 0

        self.create_interface()

    def _get_custom_css(self) -> str:
        """完全保留原始的 CSS 样式"""
        return """
        #emotion-display {
            font-size: 24px !important;
            font-weight: bold !important;
            text-align: center !important;
            padding: 10px !important;
        }
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

    def _get_empty_emotion_bars(self) -> str:
        """获取空的情绪条HTML"""
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
        """获取情绪概率分布的HTML条形图"""
        if not emotion_probs: return self._get_empty_emotion_bars()
        self.last_emotion_probs = emotion_probs
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

    def create_interface(self):
        with gr.Blocks(title="多模态情感感知共情对话系统", theme=gr.themes.Soft(), css=self._get_custom_css()) as self.app:
            
            # ================== 模块 1：通用登录入口 ==================
            with gr.Column(visible=True) as self.login_page:
                gr.Markdown("# 🔑 系统登录")
                gr.Markdown("请登录或注册以使用系统。不同的账号类型将跳转至对应的专属控制台。")
                with gr.Row():
                    with gr.Column(scale=1): pass
                    with gr.Column(scale=2):
                        self.login_user = gr.Textbox(label="用户名", placeholder="请输入用户名", max_lines=1)
                        self.login_pwd = gr.Textbox(label="密码", placeholder="请输入密码", type="password", max_lines=1)
                        with gr.Row():
                            self.btn_login = gr.Button("安全登录", variant="primary")
                            self.btn_register = gr.Button("注册用户", variant="secondary")
                        self.login_msg = gr.Markdown("")
                    with gr.Column(scale=1): pass


            # ================== 模块 2：普通用户专属界面 ==================
            # 【架构修改】脱离 Tabs，独立为一个绝对隔离的 Column 容器
            with gr.Column(visible=False) as self.user_page:
                with gr.Row():
                    self.user_header_msg = gr.Markdown("### 欢迎使用系统")
                    self.btn_user_logout = gr.Button("🚪 退出账号", size="sm", variant="stop")

                with gr.Row():
                    # 1. 历史会话区
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 📂 历史记录")
                        self.session_dropdown = gr.Dropdown(label="选择历史会话", choices=[], interactive=True)
                        self.btn_new_session = gr.Button("➕ 新建对话", variant="primary")
                        gr.Markdown("---")
                        self.btn_export = gr.Button("📥 导出聊天记录")
                        self.export_file = gr.File(label="下载文件", interactive=False)

                    # 2. 视觉感知区
                    with gr.Column(scale=3):
                        gr.Markdown("### 👁️ 视觉感知区")
                        self.video_input = gr.Image(sources=["webcam"], label="实时视频流", elem_id="video-stream")
                        with gr.Row():
                            with gr.Column():
                                self.emotion_display = gr.Textbox(label="当前识别情绪", value="等待中...", interactive=False, elem_id="emotion-display")
                            with gr.Column():
                                self.confidence_display = gr.Textbox(label="视觉置信度", value="--", interactive=False)
                        self.emotion_bars = gr.HTML(value=self._get_empty_emotion_bars(), label="情绪概率分布")
                        with gr.Row():
                            self.frame_count = gr.Number(label="已处理帧数", value=0, interactive=False)
                            self.last_update = gr.Textbox(label="最后更新时间", value="--", interactive=False)

                    # 3. 对话交互区
                    with gr.Column(scale=2):
                        gr.Markdown("### 💬 对话交互区")
                        self.chat_display = gr.Chatbot(label="对话历史", height=300, avatar_images=(None, "🤖"))
                        with gr.Row():
                            self.audio_input = gr.Audio(sources=["microphone"], type="filepath", label="语音输入")
                            self.recording_status = gr.Textbox(value="点击录音开始说话...", interactive=False, scale=2)
                        self.recognized_text = gr.Textbox(label="语音识别结果", placeholder="您的语音将在这里显示...", lines=2, interactive=False)
                        self.sys_audio_output = gr.Audio(label="系统回复语音", autoplay=True)
                        with gr.Row():
                            self.clear_btn = gr.Button("清空对话", variant="secondary")
                            self.example_btn = gr.Button("示例对话", variant="secondary")

                # 4. 系统监控区
                gr.Markdown("### 📊 本地任务监控日志")
                with gr.Row():
                    with gr.Column(scale=1):
                        self.fusion_display = gr.Textbox(label="多模态参考信息", value="等待数据...", interactive=False, lines=4)
                    with gr.Column(scale=2):
                        self.log_display = gr.Textbox(label="系统运行日志", value="系统初始化完成...", interactive=False, lines=8, max_lines=10, autoscroll=True)
                        with gr.Row():
                            self.gpu_memory = gr.Textbox(label="GPU 显存占用", value="-- MB", interactive=False)


            # ================== 模块 3：超级管理员专属界面 ==================
            # 【架构修改】完全独立，普通用户无法通过审查元素或 Tab 切换看到此页面
            with gr.Column(visible=False) as self.admin_page:
                with gr.Row():
                    self.admin_header_msg = gr.Markdown("### 🛡️ 超级管理员全局控制台")
                    self.btn_admin_logout = gr.Button("🚪 退出管理后台", size="sm", variant="stop")
                    
                gr.Markdown("---")
                gr.Markdown("### 👥 系统用户大盘管理")
                with gr.Row():
                    with gr.Column(scale=2):
                        self.admin_user_table = gr.Dataframe(headers=["用户ID", "用户名", "注册时间"], interactive=False)
                        self.admin_refresh_btn = gr.Button("🔄 手动同步系统用户数据", variant="secondary")
                    with gr.Column(scale=1):
                        gr.Markdown("#### ⚠️ 危险操作授权区")
                        gr.Markdown("执行删除将触发级联销毁，永久抹除该用户的所有历史会话及多模态实验数据，不可逆转。")
                        self.admin_target_id = gr.Number(label="请输入锁定目标的用户ID", precision=0)
                        self.admin_delete_btn = gr.Button("🗑️ 强行封禁并销毁该用户", variant="stop")
                        self.admin_msg = gr.Markdown("")

            self.bind_events()

    def bind_events(self):
        # ================== 核心鉴权与独立路由分发 ==================
        def handle_register(u, p):
            success, msg = db_manager.register_user(u, p)
            return msg

        def handle_login(u, p):
            success, uid, role = db_manager.login_user(u, p)
            if not success:
                # 登录失败：不发生页面跳转
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "❌ 账号或密码错误，请检查", gr.update(), gr.update(), gr.update(), gr.update()
                )
            
            self.current_username = u
            self.current_user_id = uid

            # 【严格路由分发】
            if role == "user":
                # 跳转至普通用户页面
                self.current_session_id = db_manager.create_session(uid)
                self.chat_history = []
                sessions_map = db_manager.get_user_sessions(uid)
                session_choices = list(sessions_map.keys())
                
                return (
                    gr.update(visible=False),  # 隐藏登录页
                    gr.update(visible=True),   # 开启用户专属页
                    gr.update(visible=False),  # 确保管理页关闭
                    "", f"### 👤 欢迎回来，{u} 👋", gr.update(), 
                    gr.update(choices=session_choices, value=session_choices[0] if session_choices else None),
                    gr.update()
                )
            else:
                # 跳转至管理员专属页面
                admin_data = db_manager.get_all_users_for_admin()
                return (
                    gr.update(visible=False),  # 隐藏登录页
                    gr.update(visible=False),  # 确保用户页关闭
                    gr.update(visible=True),   # 开启管理专属页
                    "", gr.update(), "### 🛡️ 超级管理员后台授权成功", 
                    gr.update(), gr.update(value=admin_data)
                )

        self.btn_register.click(handle_register, [self.login_user, self.login_pwd], [self.login_msg])
        
        # 登录动作映射到了 8 个核心组件的路由控制
        self.btn_login.click(
            handle_login, [self.login_user, self.login_pwd], 
            [self.login_page, self.user_page, self.admin_page, 
             self.login_msg, self.user_header_msg, self.admin_header_msg, 
             self.session_dropdown, self.admin_user_table]
        )

        # 两边独立的登出按钮绑定
        self.btn_user_logout.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[self.login_page, self.user_page])
        self.btn_admin_logout.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[self.login_page, self.admin_page])


        # ================== 用户模块：对话与历史操作 ==================
        def handle_new_session():
            self.current_session_id = db_manager.create_session(self.current_user_id)
            self.chat_history = []
            self.system_logs = ["对话已清空并创建新会话"]
            sessions_map = db_manager.get_user_sessions(self.current_user_id)
            session_choices = list(sessions_map.keys())
            return (gr.update(choices=session_choices, value=session_choices[0]), self.chat_history, "", None, "对话已清空")

        def show_example():
            example_history = [
                {"role": "user", "content": "我最近工作压力很大"},
                {"role": "assistant", "content": "听起来你最近承受了很大的工作压力，能具体说说是什么让你感到压力吗？"},
            ]
            self.chat_history = example_history
            return example_history

        def handle_load_session(session_name):
            if not session_name: return []
            sessions_map = db_manager.get_user_sessions(self.current_user_id)
            self.current_session_id = sessions_map.get(session_name)
            self.chat_history = db_manager.get_session_messages(self.current_session_id)
            return self.chat_history

        def handle_export():
            if self.current_session_id:
                path = db_manager.export_session(self.current_session_id)
                return gr.update(value=path, visible=True)
            return gr.update()

        self.btn_new_session.click(handle_new_session, outputs=[self.session_dropdown, self.chat_display, self.recognized_text, self.sys_audio_output, self.log_display])
        self.clear_btn.click(handle_new_session, outputs=[self.session_dropdown, self.chat_display, self.recognized_text, self.sys_audio_output, self.log_display])
        self.example_btn.click(show_example, outputs=[self.chat_display])
        self.session_dropdown.change(handle_load_session, inputs=[self.session_dropdown], outputs=[self.chat_display])
        self.btn_export.click(handle_export, outputs=[self.export_file])

        # ================== 管理员模块：监控与数据干预 ==================
        self.admin_refresh_btn.click(lambda: db_manager.get_all_users_for_admin(), outputs=[self.admin_user_table])
        self.admin_delete_btn.click(
            lambda uid: (db_manager.get_all_users_for_admin(), db_manager.admin_delete_user(uid)), 
            inputs=[self.admin_target_id], outputs=[self.admin_user_table, self.admin_msg]
        )

        # ================== 核心调度流 (仅作用于用户页面) ==================
        self.video_input.stream(
            fn=self.process_video_stream,
            inputs=[self.video_input],
            outputs=[self.emotion_display, self.confidence_display, self.emotion_bars, 
                     self.frame_count, self.last_update, self.log_display, self.gpu_memory],
            show_progress="hidden", queue=False
        )

        self.audio_input.start_recording(
            fn=lambda: "正在录音：系统正在同步提取您的全段面部表情...", outputs=[self.recording_status]
        ).then(fn=lambda: setattr(self, 'is_recording', True))

        self.audio_input.stop_recording(
            fn=self.process_dialogue,
            inputs=[self.audio_input, self.recognized_text],
            outputs=[self.chat_display, self.recognized_text, self.sys_audio_output, 
                     self.fusion_display, self.log_display, self.recording_status, self.gpu_memory],
            queue=True
        )

    async def process_video_stream(self, video_frame):
        """处理视觉流，与原版毫无差异"""
        if video_frame is None or time.time() - self.last_proc_time < 0.15:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        self.last_proc_time = time.time()

        if self.is_asr_processing:
            return (self.current_emotion, f"{self.visual_confidence:.2%}", self._get_emotion_bars(self.last_emotion_probs),
                    self._frame_count_internal, time.strftime("%H:%M:%S"), "系统算力目前优先分配给语言大模型处理...", "--")

        try:
            if VISION_AVAILABLE:
                emotion, confidence, probs = await asyncio.to_thread(predict_emotion, video_frame, skip_frames=1)
                self.current_emotion, self.visual_confidence, self.last_emotion_probs = emotion, confidence, probs
                if self.is_recording:
                    self.recording_emotion_buffer.append(emotion)
        except Exception as e:
            emotion, confidence, probs = self.current_emotion, self.visual_confidence, self.last_emotion_probs

        emotion_bars_html = self._get_emotion_bars(probs)
        log_entry = f"[{time.strftime('%H:%M:%S')}] 视觉识别: {emotion} (置信度: {confidence:.2%})"
        self.system_logs.append(log_entry)
        if len(self.system_logs) > 50: self.system_logs.pop(0)
        log_text = "\n".join(self.system_logs[-10:])
        
        self._frame_count_internal += 1
        gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

        return (emotion, f"{confidence:.2%}", emotion_bars_html, self._frame_count_internal, 
                time.strftime("%H:%M:%S"), log_text, gpu_mem)

    async def process_dialogue(self, audio_path, current_text):
        """全链路推断 + 数据库拦截写入，完全保留原生输出结构"""
        if not audio_path:
            gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"
            return self.chat_history, current_text, None, "等待语音输入...", "\n".join(self.system_logs[-10:]), "请先录音", gpu_mem

        try:
            self.is_asr_processing = True
            self.is_recording = False

            # ASR 与 音频情感识别
            user_text = await asyncio.to_thread(transcribe_audio, audio_path) if AUDIO_AVAILABLE else "[音频模块未加载]"
            audio_emo, audio_conf, _ = await asyncio.to_thread(predict_audio_emotion, audio_path) if AUDIO_AVAILABLE else ("Neutral", 0.0, {})

            # 视觉众数融合
            if self.recording_emotion_buffer:
                visual_decision = collections.Counter(self.recording_emotion_buffer).most_common(1)[0][0]
                fusion_info = f"融合成功：提取录音期间 {len(self.recording_emotion_buffer)} 帧表情进行综合过滤，主要情绪：{visual_decision}"
            else:
                visual_decision = self.current_emotion
                fusion_info = "由于未捕获录音期间的视频特征，采取最终时刻情绪基线。"
            self.recording_emotion_buffer = []

            # LLM 推理
            if LLM_AVAILABLE:
                llm_emotion, response_text = await asyncio.to_thread(
                    generate_empathetic_response, user_text, visual_decision, audio_emo, visual_decision, self.chat_history
                )
            else:
                llm_emotion, response_text = "Neutral", f"你说：{user_text}"

            # TTS
            audio_output_path = await generate_audio_reply(response_text) if TTS_AVAILABLE else None

            # [静默执行] 数据库写入
            if self.current_session_id:
                db_manager.add_dialogue_turn(self.current_session_id, user_text, response_text, visual_decision, self.visual_confidence, audio_emo, audio_conf, llm_emotion)

            fusion_text = f"多模态情感分析结果：\n📷 视觉判定基调: {visual_decision}\n🎤 音频情感检测: {audio_emo}\n🧠 大语言模型最终评估: {llm_emotion}"
            
            self.system_logs.extend([
                f"[{time.strftime('%H:%M:%S')}] ASR文字转写: {user_text}",
                f"[{time.strftime('%H:%M:%S')}] 录音视觉众数: {visual_decision}",
                f"[{time.strftime('%H:%M:%S')}] 音频情感提取: {audio_emo}",
                fusion_info,
                f"[{time.strftime('%H:%M:%S')}] 模型最终判定: {llm_emotion}"
            ])
            if len(self.system_logs) > 50: self.system_logs = self.system_logs[-50:]

            # 页面刷新：用数据库查询替代单纯的 append，确保数据绝对一致性
            self.chat_history = db_manager.get_session_messages(self.current_session_id)
            gpu_mem = monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

            return (self.chat_history, user_text, audio_output_path, fusion_text, 
                    "\n".join(self.system_logs[-10:]), "多模态情感计算完成", gpu_mem)
        finally:
            self.is_asr_processing = False

if __name__ == "__main__":
    print("🚀 正在加载 AI 系统内核与数据库架构...")
    if VISION_AVAILABLE: from modules.vision import get_detector; get_detector()
    if AUDIO_AVAILABLE: from modules.audio import _get_model, _get_emotion_model; _get_model(); _get_emotion_model()
    if LLM_AVAILABLE: from modules.llm import _get_llm_model; _get_llm_model()
    
    app = SystemInterface()
    app.app.queue().launch(server_name="0.0.0.0", server_port=7863, share=True, max_threads=40)