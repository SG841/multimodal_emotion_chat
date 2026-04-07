"""
utils/db_manager.py
系统级关系型数据库管理模块 (SQLite3)
对应论文核心架构：持久化存储与 CRUD 业务逻辑
"""

import sqlite3
import os
import hashlib
from datetime import datetime

# 数据库文件存放路径 (存放在 assets 目录下)
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "system_data.db")

def get_connection():
    """获取数据库连接（每次请求新建连接，保证多线程安全）"""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """初始化 6 张核心数据表 (对应论文 ER 图)"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    # 1. 账号表
    cursor.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # 2. 个人画像表
    cursor.execute('''CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY, nickname TEXT, preferred_tts_voice TEXT DEFAULT 'zh-CN-XiaoxiaoNeural',
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')

    # 3. 会话表 (管理每次独立的聊天)
    cursor.execute('''CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
        session_name TEXT, start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)''')

    # 4. 消息记录表
    cursor.execute('''CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
        role TEXT NOT NULL, content TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE)''')

    # 5. 多模态特征记录表 (科研核心数据)
    cursor.execute('''CREATE TABLE IF NOT EXISTS multimodal_features (
        message_id INTEGER PRIMARY KEY, vision_emotion TEXT, vision_confidence REAL,
        audio_emotion TEXT, audio_confidence REAL, llm_decision TEXT,
        FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE)''')

    # 6. 系统监控日志表
    cursor.execute('''CREATE TABLE IF NOT EXISTS system_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        module_name TEXT, gpu_memory_used TEXT, info TEXT)''')

    # 强制开启外键约束支持
    cursor.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()

# 初始化数据库
init_db()

# ==================== 论文用例 1-4：账号与权限 ====================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, password):
    if not username or len(password) < 4: return False, "格式错误"
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO users (username, password_hash) VALUES (?, ?)", (username, hash_password(password)))
        user_id = cur.lastrowid
        # 同步创建用户画像
        cur.execute("INSERT INTO user_profiles (user_id, nickname) VALUES (?, ?)", (user_id, f"用户_{username}"))
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    finally: conn.close()

def login_user(username, password):
    if username == "admin" and password == "admin123": return True, 0, "admin"  # 硬编码超级管理员
    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE username=? AND password_hash=?", (username, hash_password(password))).fetchone()
    conn.close()
    if user: return True, user['id'], "user"
    return False, None, None

# ==================== 论文用例 6, 11-14：会话与消息管理 ====================
def create_session(user_id):
    """创建新会话"""
    conn = get_connection()
    cur = conn.cursor()
    session_name = f"对话 {datetime.now().strftime('%m-%d %H:%M')}"
    cur.execute("INSERT INTO sessions (user_id, session_name) VALUES (?, ?)", (user_id, session_name))
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id

def get_user_sessions(user_id):
    """获取用户的历史会话列表"""
    conn = get_connection()
    sessions = conn.execute("SELECT id, session_name FROM sessions WHERE user_id=? ORDER BY id DESC", (user_id,)).fetchall()
    conn.close()
    return {s['session_name']: s['id'] for s in sessions}

def add_dialogue_turn(session_id, user_text, sys_text, v_emo, v_conf, a_emo, a_conf, llm_emo):
    """写入一轮对话，并绑定多模态特征"""
    conn = get_connection()
    cur = conn.cursor()
    # 写入用户消息
    cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)", (session_id, user_text))
    # 写入系统消息
    cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)", (session_id, sys_text))
    sys_msg_id = cur.lastrowid
    # 绑定多模态科研特征到系统回复上
    cur.execute("""INSERT INTO multimodal_features 
        (message_id, vision_emotion, vision_confidence, audio_emotion, audio_confidence, llm_decision) 
        VALUES (?, ?, ?, ?, ?, ?)""", 
        (sys_msg_id, v_emo, v_conf, a_emo, a_conf, llm_emo))
    conn.commit()
    conn.close()

def get_session_messages(session_id):
    """拉取某次会话的所有聊天记录，用于页面回显"""
    conn = get_connection()
    msgs = conn.execute("SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC", (session_id,)).fetchall()
    conn.close()
    return [{"role": m["role"], "content": m["content"]} for m in msgs]

def export_session(session_id):
    """导出聊天记录为本地 TXT 文件"""
    msgs = get_session_messages(session_id)
    export_path = os.path.join(os.path.dirname(DB_PATH), f"export_session_{session_id}.txt")
    with open(export_path, "w", encoding="utf-8") as f:
        f.write("=== 多模态共情对话系统 导出记录 ===\n\n")
        for m in msgs:
            role = "用户" if m["role"] == "user" else "AI助手"
            f.write(f"[{role}]: {m['content']}\n\n")
    return export_path

# ==================== 论文用例 15-17：管理员后台功能 ====================
def get_all_users_for_admin():
    conn = get_connection()
    res = conn.execute("SELECT id, username, created_at FROM users").fetchall()
    conn.close()
    return [[r['id'], r['username'], r['created_at']] for r in res]

def admin_delete_user(user_id):
    conn = get_connection()
    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return f"已成功删除用户ID: {user_id}"

def log_system_metric(module, gpu_mem, info):
    conn = get_connection()
    conn.execute("INSERT INTO system_metrics (module_name, gpu_memory_used, info) VALUES (?, ?, ?)", (module, gpu_mem, info))
    conn.commit()
    conn.close()