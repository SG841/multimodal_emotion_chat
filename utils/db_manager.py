"""
utils/db_manager.py
系统级关系型数据库管理模块 (SQLite3)
"""

import hashlib
import os
import sqlite3
from datetime import datetime

from config import ADMIN_INVITE_CODE

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets", "system_data.db")


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        role TEXT NOT NULL DEFAULT 'user',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS user_profiles (
        user_id INTEGER PRIMARY KEY,
        nickname TEXT,
        preferred_tts_voice TEXT DEFAULT 'zh-CN-XiaoxiaoNeural',
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        session_name TEXT,
        start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
    )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL,
        role TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS multimodal_features (
        message_id INTEGER PRIMARY KEY,
        vision_emotion TEXT,
        vision_confidence REAL,
        audio_emotion TEXT,
        audio_confidence REAL,
        llm_decision TEXT,
        FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
    )"""
    )

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS system_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        module_name TEXT,
        gpu_memory_used TEXT,
        info TEXT
    )"""
    )

    user_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")

    conn.commit()
    conn.close()


init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, role="user", invite_code=None):
    if role not in ("user", "admin"):
        return False, "账户类型无效"
    if not username or len(password) < 4:
        return False, "格式错误"
    if role == "admin" and invite_code != ADMIN_INVITE_CODE:
        return False, "管理员邀请码错误"

    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role),
        )
        user_id = cur.lastrowid
        cur.execute(
            "INSERT INTO user_profiles (user_id, nickname) VALUES (?, ?)",
            (user_id, f"用户_{username}"),
        )
        conn.commit()
        return True, "注册成功"
    except sqlite3.IntegrityError:
        return False, "用户名已存在"
    finally:
        conn.close()


def login_user(username, password, expected_role=None):
    conn = get_connection()
    user = conn.execute(
        "SELECT id, role FROM users WHERE username=? AND password_hash=?",
        (username, hash_password(password)),
    ).fetchone()
    conn.close()

    if not user:
        return False, None, None

    role = user["role"]
    if expected_role and role != expected_role:
        return False, None, None
    return True, user["id"], role


def create_session(user_id):
    conn = get_connection()
    cur = conn.cursor()
    session_name = f"对话 {datetime.now().strftime('%m-%d %H:%M')}"
    cur.execute("INSERT INTO sessions (user_id, session_name) VALUES (?, ?)", (user_id, session_name))
    session_id = cur.lastrowid
    conn.commit()
    conn.close()
    return session_id


def get_user_sessions(user_id):
    conn = get_connection()
    sessions = conn.execute(
        "SELECT id, session_name FROM sessions WHERE user_id=? ORDER BY id DESC",
        (user_id,),
    ).fetchall()
    conn.close()
    return {s["session_name"]: s["id"] for s in sessions}


def add_dialogue_turn(session_id, user_text, sys_text, v_emo, v_conf, a_emo, a_conf, llm_emo):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'user', ?)", (session_id, user_text))
    cur.execute("INSERT INTO messages (session_id, role, content) VALUES (?, 'assistant', ?)", (session_id, sys_text))
    sys_msg_id = cur.lastrowid
    cur.execute(
        """INSERT INTO multimodal_features
        (message_id, vision_emotion, vision_confidence, audio_emotion, audio_confidence, llm_decision)
        VALUES (?, ?, ?, ?, ?, ?)""",
        (sys_msg_id, v_emo, v_conf, a_emo, a_conf, llm_emo),
    )
    conn.commit()
    conn.close()


def get_session_messages(session_id):
    conn = get_connection()
    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY id ASC",
        (session_id,),
    ).fetchall()
    conn.close()
    return [{"role": m["role"], "content": m["content"]} for m in msgs]


def export_session(session_id):
    msgs = get_session_messages(session_id)
    export_path = os.path.join(os.path.dirname(DB_PATH), f"export_session_{session_id}.txt")
    with open(export_path, "w", encoding="utf-8") as file:
        file.write("=== 多模态共情对话系统导出记录 ===\n\n")
        for msg in msgs:
            role = "用户" if msg["role"] == "user" else "AI助手"
            file.write(f"[{role}]: {msg['content']}\n\n")
    return export_path


def get_all_users_for_admin():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [[row["id"], row["username"], row["role"], row["created_at"]] for row in rows]


def admin_delete_user(user_id):
    conn = get_connection()
    target = conn.execute("SELECT role FROM users WHERE id=?", (user_id,)).fetchone()
    if not target:
        conn.close()
        return f"用户ID不存在: {user_id}"
    if target["role"] == "admin":
        conn.close()
        return "不能删除管理员账号"

    conn.execute("DELETE FROM users WHERE id=?", (user_id,))
    conn.commit()
    conn.close()
    return f"已成功删除用户ID: {user_id}"


def log_system_metric(module, gpu_mem, info):
    conn = get_connection()
    conn.execute(
        "INSERT INTO system_metrics (module_name, gpu_memory_used, info) VALUES (?, ?, ?)",
        (module, gpu_mem, info),
    )
    conn.commit()
    conn.close()
