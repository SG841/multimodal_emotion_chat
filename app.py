"""Gradio 交互界面模块。"""

import asyncio
import collections
import os
import sys
import time

import gradio as gr

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import ADMIN_INVITE_CODE
from utils import db_manager

USER_ACTIONS = ["个性化设置", "修改密码", "退出账号"]
TTS_VOICE_OPTIONS = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunyangNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-liaoning-XiaobeiNeural",
    "zh-CN-shaanxi-XiaoniNeural",
]

try:
    from modules.vision import predict_emotion
    VISION_AVAILABLE = True
except ImportError as e:
    print(f"视觉模块未加载: {e}")
    VISION_AVAILABLE = False

try:
    from modules.audio import predict_audio_emotion, transcribe_audio
    AUDIO_AVAILABLE = True
except ImportError as e:
    print(f"音频模块未加载: {e}")
    AUDIO_AVAILABLE = False

try:
    from modules.llm import _get_llm_model, generate_empathetic_response
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
        self.current_user_id = None
        self.current_username = None
        self.current_session_id = None
        self.current_user_profile = None

        self.current_emotion = "Neutral"
        self.visual_confidence = 0.0
        self.chat_history = []
        self.system_logs = []
        self._frame_count_internal = 0
        self.last_emotion_probs = {}

        self.is_recording = False
        self.is_asr_processing = False
        self.recording_emotion_buffer = []
        self.last_proc_time = 0

        self.create_interface()

    def _get_gpu_mem(self):
        return monitor.get_resource_status().get("gpu_mem", "--") if MONITOR_AVAILABLE else "--"

    def _admin_user_choices(self):
        return db_manager.get_user_choices(include_admin=True)

    def _load_current_profile(self):
        if self.current_user_id is None:
            self.current_user_profile = None
            return None
        self.current_user_profile = db_manager.get_user_profile(self.current_user_id)
        return self.current_user_profile

    def _current_voice(self):
        profile = self.current_user_profile or self._load_current_profile() or {}
        return profile.get("preferred_tts_voice", TTS_VOICE_OPTIONS[0])

    def _log_user_event(self, module_name, info, session_id=None):
        if self.current_user_id is None:
            return
        db_manager.log_system_metric(module_name, self._get_gpu_mem(), info, user_id=self.current_user_id, session_id=session_id)

    def _get_custom_css(self) -> str:
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
        emotions = list(self.last_emotion_probs.keys()) if self.last_emotion_probs else ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]
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
        if not emotion_probs:
            return self._get_empty_emotion_bars()
        self.last_emotion_probs = emotion_probs
        sorted_emotions = sorted(emotion_probs.items(), key=lambda item: item[1], reverse=True)[:6]
        colors = ["#4CAF50", "#2196F3", "#F44336", "#9E9E9E", "#FF9800", "#9C27B0"]
        html = "<div style='margin: 10px 0;'>"
        for i, (emotion, prob) in enumerate(sorted_emotions):
            color = colors[i % len(colors)]
            percent = prob * 100
            html += f"""
            <div style='display: flex; align-items: center; margin: 5px 0;'>
                <span style='width: 80px; font-size: 12px;'>{emotion}</span>
                <div style='flex: 1; height: 20px; background: #e0e0e0; border-radius: 10px; margin-left: 10px;'>
                    <div style='width: {percent}%; height: 100%; background: {color}; border-radius: 10px; transition: width 0.3s;'></div>
                </div>
                <span style='width: 40px; text-align: right; font-size: 12px;'>{percent:.0f}%</span>
            </div>
            """
        html += "</div>"
        return html

    def create_interface(self):
        with gr.Blocks(title="多模态情感感知共情对话系统", theme=gr.themes.Soft(), css=self._get_custom_css()) as self.app:
            with gr.Column(visible=True) as self.login_page:
                gr.Markdown("# 系统登录")
                gr.Markdown("请选择登录或注册方式。管理员注册需要提供邀请码。")
                with gr.Row():
                    with gr.Column(scale=1):
                        pass
                    with gr.Column(scale=2):
                        self.auth_mode = gr.Radio(["用户登录", "管理员登录", "用户注册", "管理员注册"], value="用户登录", label="操作类型")
                        self.login_user = gr.Textbox(label="用户名", placeholder="请输入用户名", max_lines=1)
                        self.login_pwd = gr.Textbox(label="密码", placeholder="请输入密码", type="password", max_lines=1)
                        self.invite_code = gr.Textbox(label="管理员邀请码", placeholder="仅管理员注册需要填写", type="password", visible=False)
                        self.btn_auth = gr.Button("安全登录", variant="primary")
                        self.login_msg = gr.Markdown("")
                    with gr.Column(scale=1):
                        pass

            with gr.Column(visible=False) as self.user_page:
                with gr.Row():
                    self.user_header_msg = gr.Markdown("### 欢迎使用系统")

                with gr.Row():
                    with gr.Column(scale=1, variant="panel"):
                        gr.Markdown("### 历史记录")
                        self.session_dropdown = gr.Dropdown(label="选择历史会话", choices=[], interactive=True)
                        self.btn_new_session = gr.Button("新建对话", variant="primary")
                        gr.Markdown("---")
                        self.btn_export = gr.Button("导出聊天记录")
                        self.export_file = gr.File(label="下载文件", interactive=False)

                        with gr.Accordion("账户操作", open=False):
                            self.user_action_menu = gr.Radio(USER_ACTIONS, value="个性化设置", label="请选择操作")

                        with gr.Column(visible=True) as self.user_profile_panel:
                            gr.Markdown("### 个性化设置")
                            self.user_profile_username = gr.Textbox(label="当前用户名", interactive=False)
                            self.user_nickname = gr.Textbox(label="昵称")
                            self.user_bio = gr.Textbox(label="个人资料", lines=4, placeholder="介绍一下自己或写下你的偏好")
                            self.user_voice = gr.Dropdown(TTS_VOICE_OPTIONS, label="偏好音色", value=TTS_VOICE_OPTIONS[0], interactive=True)
                            self.user_profile_save_btn = gr.Button("保存个性化设置", variant="secondary")
                            self.user_profile_msg = gr.Markdown("")

                        with gr.Column(visible=False) as self.user_password_panel:
                            gr.Markdown("### 修改密码")
                            self.user_old_password = gr.Textbox(label="原密码", type="password")
                            self.user_new_password = gr.Textbox(label="新密码", type="password")
                            self.user_confirm_password = gr.Textbox(label="确认新密码", type="password")
                            self.user_change_pwd_btn = gr.Button("修改我的密码", variant="secondary")
                            self.user_password_msg = gr.Markdown("")

                        with gr.Column(visible=False) as self.user_logout_panel:
                            gr.Markdown("### 退出账号")
                            gr.Markdown("点击下方按钮退出当前账号并返回登录界面。")
                            self.btn_user_logout = gr.Button("退出账号", variant="stop")
                            self.user_logout_msg = gr.Markdown("")

                    with gr.Column(scale=3):
                        gr.Markdown("### 视觉感知区")
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

                    with gr.Column(scale=2):
                        gr.Markdown("### 对话交互区")
                        self.chat_display = gr.Chatbot(label="对话历史", height=300)
                        with gr.Row():
                            self.audio_input = gr.Audio(sources=["microphone"], type="filepath", label="语音输入")
                            self.recording_status = gr.Textbox(value="点击录音开始说话...", interactive=False, scale=2)
                        self.recognized_text = gr.Textbox(label="语音识别结果", placeholder="您的语音将显示在这里...", lines=2, interactive=False)
                        self.sys_audio_output = gr.Audio(label="系统回复语音", autoplay=True)
                        with gr.Row():
                            self.clear_btn = gr.Button("清空对话", variant="secondary")
                            self.example_btn = gr.Button("示例对话", variant="secondary")

                gr.Markdown("### 本地任务监控日志")
                with gr.Row():
                    with gr.Column(scale=1):
                        self.fusion_display = gr.Textbox(label="多模态参考信息", value="等待数据...", interactive=False, lines=4)
                    with gr.Column(scale=2):
                        self.log_display = gr.Textbox(label="系统运行日志", value="系统初始化完成...", interactive=False, lines=8, max_lines=10, autoscroll=True)
                        with gr.Row():
                            self.gpu_memory = gr.Textbox(label="GPU 显存占用", value="-- MB", interactive=False)

            with gr.Column(visible=False) as self.admin_page:
                with gr.Row():
                    self.admin_header_msg = gr.Markdown("### 超级管理员全局控制台")
                    self.btn_admin_logout = gr.Button("退出管理后台", size="sm", variant="stop")

                gr.Markdown("---")
                gr.Markdown("### 系统用户管理")
                with gr.Row():
                    with gr.Column(scale=2):
                        self.admin_user_table = gr.Dataframe(headers=["用户ID", "用户名", "角色", "注册时间"], interactive=False)
                        self.admin_refresh_btn = gr.Button("同步系统用户数据", variant="secondary")
                    with gr.Column(scale=1):
                        gr.Markdown("#### 删除用户")
                        gr.Markdown("删除会级联移除该普通用户的历史会话和多模态数据，管理员账号不允许删除。")
                        self.admin_target_id = gr.Number(label="请输入目标用户ID", precision=0)
                        self.admin_delete_btn = gr.Button("删除该用户", variant="stop")
                        self.admin_msg = gr.Markdown("")

                gr.Markdown("### 用户监控与日志")
                with gr.Row():
                    with gr.Column(scale=1):
                        self.admin_user_selector = gr.Dropdown(label="选择要监控的用户", choices=self._admin_user_choices(), interactive=True)
                        self.admin_refresh_monitor_btn = gr.Button("刷新该用户监控数据", variant="secondary")
                        self.admin_user_summary = gr.Textbox(label="用户使用概览", lines=8, interactive=False)
                        self.admin_user_gpu = gr.Textbox(label="当前服务器 GPU 显存占用", value="-- MB", interactive=False)
                    with gr.Column(scale=2):
                        self.admin_user_logs = gr.Textbox(label="该用户的监控日志", lines=16, max_lines=20, interactive=False, autoscroll=True)

                gr.Markdown("### 管理员修改用户密码")
                with gr.Row():
                    with gr.Column(scale=1):
                        self.admin_password_user = gr.Dropdown(label="选择要改密的用户", choices=self._admin_user_choices(), interactive=True)
                    with gr.Column(scale=1):
                        self.admin_new_password = gr.Textbox(label="新密码", type="password")
                    with gr.Column(scale=1):
                        self.admin_confirm_password = gr.Textbox(label="确认新密码", type="password")
                self.admin_change_pwd_btn = gr.Button("修改用户密码", variant="secondary")
                self.admin_password_msg = gr.Markdown("")

            self.bind_events()

    def bind_events(self):
        def update_auth_form(mode):
            need_invite = mode == "管理员注册"
            if mode == "管理员注册":
                button_text = "注册管理员"
                helper_text = f"管理员注册需要邀请码。当前配置邀请码为：`{ADMIN_INVITE_CODE}`"
            elif mode == "用户注册":
                button_text = "注册用户"
                helper_text = ""
            elif mode == "管理员登录":
                button_text = "管理员登录"
                helper_text = ""
            else:
                button_text = "安全登录"
                helper_text = ""
            return gr.update(visible=need_invite, value=""), gr.update(value=button_text), helper_text

        def refresh_admin_panels(message=""):
            user_choices = self._admin_user_choices()
            return gr.update(value=db_manager.get_all_users_for_admin()), gr.update(choices=user_choices), gr.update(choices=user_choices), message

        def get_user_panel_updates(action):
            return (
                gr.update(visible=action == "个性化设置"),
                gr.update(visible=action == "修改密码"),
                gr.update(visible=action == "退出账号"),
            )

        def load_user_preferences():
            profile = self._load_current_profile() or {}
            return (
                profile.get("username", self.current_username or ""),
                profile.get("nickname", ""),
                profile.get("bio", ""),
                profile.get("preferred_tts_voice", TTS_VOICE_OPTIONS[0]),
            )

        def handle_auth(mode, username, password, invite_code):
            role_map = {"用户登录": "user", "管理员登录": "admin", "用户注册": "user", "管理员注册": "admin"}
            target_role = role_map[mode]
            is_register = "注册" in mode

            if is_register:
                _, msg = db_manager.register_user(username, password, role=target_role, invite_code=invite_code)
                admin_table, admin_selector, admin_password_user, admin_message = refresh_admin_panels(msg)
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    msg, gr.update(), gr.update(), gr.update(), admin_table,
                    admin_selector, admin_password_user, admin_message,
                    gr.update(), gr.update(), gr.update(), gr.update(),
                )

            success, uid, role = db_manager.login_user(username, password, expected_role=target_role)
            if not success:
                return (
                    gr.update(visible=True), gr.update(visible=False), gr.update(visible=False),
                    "账号、密码或登录类型不正确，请检查后重试", gr.update(), gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                )

            self.current_username = username
            self.current_user_id = uid
            self._load_current_profile()

            if role == "user":
                self.current_session_id = db_manager.create_session(uid)
                self.chat_history = []
                sessions_map = db_manager.get_user_sessions(uid)
                session_choices = list(sessions_map.keys())
                self._log_user_event("auth", f"用户 {username} 登录成功并创建默认会话", self.current_session_id)
                profile_username, nickname, bio, voice = load_user_preferences()
                return (
                    gr.update(visible=False), gr.update(visible=True), gr.update(visible=False),
                    "", f"### 欢迎回来，{username}", gr.update(),
                    gr.update(choices=session_choices, value=session_choices[0] if session_choices else None), gr.update(),
                    gr.update(), gr.update(), gr.update(),
                    profile_username, nickname, bio, voice,
                )

            admin_table, admin_selector, admin_password_user, admin_message = refresh_admin_panels("管理员登录成功")
            return (
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=True),
                "", gr.update(), "### 超级管理员后台授权成功",
                gr.update(), admin_table, admin_selector, admin_password_user, admin_message,
                gr.update(), gr.update(), gr.update(), gr.update(),
            )

        def handle_new_session():
            self.current_session_id = db_manager.create_session(self.current_user_id)
            self.chat_history = []
            self.system_logs = ["对话已清空并创建新会话"]
            sessions_map = db_manager.get_user_sessions(self.current_user_id)
            session_choices = list(sessions_map.keys())
            self._log_user_event("session", "用户创建了新会话", self.current_session_id)
            return gr.update(choices=session_choices, value=session_choices[0]), self.chat_history, "", None, "对话已清空"

        def show_example():
            example_history = [("我最近工作压力很大", "听起来你最近承受了不少压力。如果你愿意，可以和我说说最困扰你的是什么。")]
            self.chat_history = example_history
            return example_history

        def handle_load_session(session_name):
            if not session_name:
                return []
            sessions_map = db_manager.get_user_sessions(self.current_user_id)
            self.current_session_id = sessions_map.get(session_name)
            self.chat_history = db_manager.get_session_messages(self.current_session_id)
            return self.chat_history

        def handle_export():
            if self.current_session_id:
                path = db_manager.export_session(self.current_session_id)
                self._log_user_event("export", f"导出了会话记录: {self.current_session_id}", self.current_session_id)
                return gr.update(value=path, visible=True)
            return gr.update()

        def handle_user_change_password(old_pwd, new_pwd, confirm_pwd):
            if new_pwd != confirm_pwd:
                return "两次输入的新密码不一致"
            success, msg = db_manager.change_password(self.current_user_id, old_pwd, new_pwd)
            if success:
                self._log_user_event("password", "用户修改了自己的密码", self.current_session_id)
            return msg

        def handle_save_profile(nickname, bio, voice):
            success, msg = db_manager.update_user_profile(self.current_user_id, nickname, bio, voice)
            if success:
                self._load_current_profile()
                self._log_user_event("profile", f"用户更新了个性化设置，音色={voice}", self.current_session_id)
            return msg

        def handle_user_logout():
            self.current_user_id = None
            self.current_username = None
            self.current_session_id = None
            self.current_user_profile = None
            self.chat_history = []
            return gr.update(visible=True), gr.update(visible=False), "已退出当前账号"

        def handle_admin_delete(uid):
            result = db_manager.admin_delete_user(uid)
            admin_table, admin_selector, admin_password_user, _ = refresh_admin_panels()
            return admin_table, result, admin_selector, admin_password_user

        def handle_admin_change_password(user_choice, new_pwd, confirm_pwd):
            if new_pwd != confirm_pwd:
                admin_table, admin_selector, admin_password_user, _ = refresh_admin_panels()
                return admin_table, admin_selector, admin_password_user, "两次输入的新密码不一致"
            target_user_id = db_manager.parse_user_choice(user_choice)
            if target_user_id is None:
                admin_table, admin_selector, admin_password_user, _ = refresh_admin_panels()
                return admin_table, admin_selector, admin_password_user, "请选择目标用户"
            _, msg = db_manager.admin_reset_password(target_user_id, new_pwd)
            db_manager.log_system_metric("admin_password", self._get_gpu_mem(), f"管理员修改了用户 {target_user_id} 的密码", user_id=target_user_id)
            admin_table, admin_selector, admin_password_user, _ = refresh_admin_panels()
            return admin_table, admin_selector, admin_password_user, msg

        def handle_admin_monitor(user_choice):
            user_id = db_manager.parse_user_choice(user_choice)
            if user_id is None:
                return "请选择一个用户", "暂无监控日志", self._get_gpu_mem()
            return db_manager.get_user_activity_summary(user_id), db_manager.get_user_metrics(user_id), self._get_gpu_mem()

        self.auth_mode.change(update_auth_form, inputs=[self.auth_mode], outputs=[self.invite_code, self.btn_auth, self.login_msg])
        self.btn_auth.click(
            handle_auth,
            [self.auth_mode, self.login_user, self.login_pwd, self.invite_code],
            [
                self.login_page, self.user_page, self.admin_page,
                self.login_msg, self.user_header_msg, self.admin_header_msg,
                self.session_dropdown, self.admin_user_table,
                self.admin_user_selector, self.admin_password_user, self.admin_password_msg,
                self.user_profile_username, self.user_nickname, self.user_bio, self.user_voice,
            ],
        )

        self.user_action_menu.change(get_user_panel_updates, inputs=[self.user_action_menu], outputs=[self.user_profile_panel, self.user_password_panel, self.user_logout_panel])
        self.btn_user_logout.click(handle_user_logout, outputs=[self.login_page, self.user_page, self.user_logout_msg])
        self.btn_admin_logout.click(lambda: (gr.update(visible=True), gr.update(visible=False)), outputs=[self.login_page, self.admin_page])

        self.btn_new_session.click(handle_new_session, outputs=[self.session_dropdown, self.chat_display, self.recognized_text, self.sys_audio_output, self.log_display])
        self.clear_btn.click(handle_new_session, outputs=[self.session_dropdown, self.chat_display, self.recognized_text, self.sys_audio_output, self.log_display])
        self.example_btn.click(show_example, outputs=[self.chat_display])
        self.session_dropdown.change(handle_load_session, inputs=[self.session_dropdown], outputs=[self.chat_display])
        self.btn_export.click(handle_export, outputs=[self.export_file])
        self.user_change_pwd_btn.click(handle_user_change_password, inputs=[self.user_old_password, self.user_new_password, self.user_confirm_password], outputs=[self.user_password_msg])
        self.user_profile_save_btn.click(handle_save_profile, inputs=[self.user_nickname, self.user_bio, self.user_voice], outputs=[self.user_profile_msg])

        self.admin_refresh_btn.click(lambda: db_manager.get_all_users_for_admin(), outputs=[self.admin_user_table])
        self.admin_delete_btn.click(handle_admin_delete, inputs=[self.admin_target_id], outputs=[self.admin_user_table, self.admin_msg, self.admin_user_selector, self.admin_password_user])
        self.admin_refresh_monitor_btn.click(handle_admin_monitor, inputs=[self.admin_user_selector], outputs=[self.admin_user_summary, self.admin_user_logs, self.admin_user_gpu])
        self.admin_change_pwd_btn.click(handle_admin_change_password, inputs=[self.admin_password_user, self.admin_new_password, self.admin_confirm_password], outputs=[self.admin_user_table, self.admin_user_selector, self.admin_password_user, self.admin_password_msg])

        self.video_input.stream(
            fn=self.process_video_stream,
            inputs=[self.video_input],
            outputs=[self.emotion_display, self.confidence_display, self.emotion_bars, self.frame_count, self.last_update, self.log_display, self.gpu_memory],
            show_progress="hidden",
            queue=False,
        )
        self.audio_input.start_recording(fn=lambda: "正在录音：系统正在同步提取您的面部表情...", outputs=[self.recording_status]).then(fn=lambda: setattr(self, "is_recording", True))
        self.audio_input.stop_recording(
            fn=self.process_dialogue,
            inputs=[self.audio_input, self.recognized_text],
            outputs=[self.chat_display, self.recognized_text, self.sys_audio_output, self.fusion_display, self.log_display, self.recording_status, self.gpu_memory],
            queue=True,
        )

    async def process_video_stream(self, video_frame):
        if video_frame is None or time.time() - self.last_proc_time < 0.15:
            return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
        self.last_proc_time = time.time()

        if self.is_asr_processing:
            return self.current_emotion, f"{self.visual_confidence:.2%}", self._get_emotion_bars(self.last_emotion_probs), self._frame_count_internal, time.strftime("%H:%M:%S"), "系统算力当前优先分配给语音与大模型处理...", "--"

        emotion, confidence, probs = self.current_emotion, self.visual_confidence, self.last_emotion_probs
        try:
            if VISION_AVAILABLE:
                emotion, confidence, probs = await asyncio.to_thread(predict_emotion, video_frame, skip_frames=1)
                self.current_emotion = emotion
                self.visual_confidence = confidence
                self.last_emotion_probs = probs
                if self.is_recording:
                    self.recording_emotion_buffer.append(emotion)
        except Exception:
            pass

        emotion_bars_html = self._get_emotion_bars(probs)
        self.system_logs.append(f"[{time.strftime('%H:%M:%S')}] 视觉识别: {emotion} (置信度: {confidence:.2%})")
        if len(self.system_logs) > 50:
            self.system_logs.pop(0)
        log_text = "\n".join(self.system_logs[-10:])
        self._frame_count_internal += 1
        return emotion, f"{confidence:.2%}", emotion_bars_html, self._frame_count_internal, time.strftime("%H:%M:%S"), log_text, self._get_gpu_mem()

    async def process_dialogue(self, audio_path, current_text):
        if not audio_path:
            return self.chat_history, current_text, None, "等待语音输入...", "\n".join(self.system_logs[-10:]), "请先录音", self._get_gpu_mem()

        try:
            self.is_asr_processing = True
            self.is_recording = False

            user_text = await asyncio.to_thread(transcribe_audio, audio_path) if AUDIO_AVAILABLE else "[音频模块未加载]"
            audio_emo, audio_conf, _ = await asyncio.to_thread(predict_audio_emotion, audio_path) if AUDIO_AVAILABLE else ("Neutral", 0.0, {})

            if self.recording_emotion_buffer:
                visual_decision = collections.Counter(self.recording_emotion_buffer).most_common(1)[0][0]
                fusion_info = f"融合成功：提取录音期间 {len(self.recording_emotion_buffer)} 帧表情进行综合过滤，主要情绪：{visual_decision}"
            else:
                visual_decision = self.current_emotion
                fusion_info = "录音期间未捕获到有效视频特征，采用当前视觉情绪作为参考。"
            self.recording_emotion_buffer = []

            if LLM_AVAILABLE:
                llm_emotion, response_text = await asyncio.to_thread(generate_empathetic_response, user_text, visual_decision, audio_emo, visual_decision, self.chat_history)
            else:
                llm_emotion, response_text = "Neutral", f"你说：{user_text}"

            audio_output_path = await generate_audio_reply(response_text, voice=self._current_voice()) if TTS_AVAILABLE else None

            if self.current_session_id:
                db_manager.add_dialogue_turn(self.current_session_id, user_text, response_text, visual_decision, self.visual_confidence, audio_emo, audio_conf, llm_emotion)
                self.chat_history = db_manager.get_session_messages(self.current_session_id)
            else:
                self.chat_history = self.chat_history + [(user_text, response_text)]

            fusion_text = f"多模态情感分析结果：\n视觉判定基调: {visual_decision}\n音频情感检测: {audio_emo}\n大语言模型最终评估: {llm_emotion}"
            self.system_logs.extend([
                f"[{time.strftime('%H:%M:%S')}] ASR文字转写: {user_text}",
                f"[{time.strftime('%H:%M:%S')}] 录音视觉众数: {visual_decision}",
                f"[{time.strftime('%H:%M:%S')}] 音频情感提取: {audio_emo}",
                fusion_info,
                f"[{time.strftime('%H:%M:%S')}] 模型最终判定: {llm_emotion}",
            ])
            if len(self.system_logs) > 50:
                self.system_logs = self.system_logs[-50:]

            self._log_user_event("dialogue", f"完成一次多模态对话。文本='{user_text[:30]}'，视觉={visual_decision}，音频={audio_emo}，最终={llm_emotion}，音色={self._current_voice()}", self.current_session_id)
            return self.chat_history, user_text, audio_output_path, fusion_text, "\n".join(self.system_logs[-10:]), "多模态情感计算完成", self._get_gpu_mem()
        finally:
            self.is_asr_processing = False


if __name__ == "__main__":
    print("正在加载 AI 系统内核与数据库结构...")
    if VISION_AVAILABLE:
        from modules.vision import get_detector
        get_detector()
    if AUDIO_AVAILABLE:
        from modules.audio import _get_emotion_model, _get_model
        _get_model()
        _get_emotion_model()
    if LLM_AVAILABLE:
        _get_llm_model()

    app = SystemInterface()
    app.app.queue().launch(server_name="0.0.0.0", server_port=7863, share=True, max_threads=40)
