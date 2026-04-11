"""系统级关系型数据库管理模块 (SQLite3)。"""

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
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL UNIQUE,
        nickname TEXT,
        bio TEXT DEFAULT '',
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
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id INTEGER NOT NULL UNIQUE,
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
        user_id INTEGER,
        session_id INTEGER,
        module_name TEXT,
        gpu_memory_used TEXT,
        info TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE,
        FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
    )"""
    )

    user_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(users)").fetchall()]
    if "role" not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")

    profile_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(user_profiles)").fetchall()]
    if "bio" not in profile_columns:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN bio TEXT DEFAULT ''")
    if "preferred_tts_voice" not in profile_columns:
        cursor.execute("ALTER TABLE user_profiles ADD COLUMN preferred_tts_voice TEXT DEFAULT 'zh-CN-XiaoxiaoNeural'")

    profile_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(user_profiles)").fetchall()]
    if "id" not in profile_columns:
        cursor.execute("ALTER TABLE user_profiles RENAME TO user_profiles_old")
        cursor.execute(
            """CREATE TABLE user_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            nickname TEXT,
            bio TEXT DEFAULT '',
            preferred_tts_voice TEXT DEFAULT 'zh-CN-XiaoxiaoNeural',
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
        )"""
        )
        cursor.execute(
            """INSERT INTO user_profiles (user_id, nickname, bio, preferred_tts_voice)
            SELECT user_id, nickname, COALESCE(bio, ''), COALESCE(preferred_tts_voice, 'zh-CN-XiaoxiaoNeural')
            FROM user_profiles_old"""
        )
        cursor.execute("DROP TABLE user_profiles_old")

    feature_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(multimodal_features)").fetchall()]
    if "id" not in feature_columns:
        cursor.execute("ALTER TABLE multimodal_features RENAME TO multimodal_features_old")
        cursor.execute(
            """CREATE TABLE multimodal_features (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id INTEGER NOT NULL UNIQUE,
            vision_emotion TEXT,
            vision_confidence REAL,
            audio_emotion TEXT,
            audio_confidence REAL,
            llm_decision TEXT,
            FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
        )"""
        )
        cursor.execute(
            """INSERT INTO multimodal_features
            (message_id, vision_emotion, vision_confidence, audio_emotion, audio_confidence, llm_decision)
            SELECT message_id, vision_emotion, vision_confidence, audio_emotion, audio_confidence, llm_decision
            FROM multimodal_features_old"""
        )
        cursor.execute("DROP TABLE multimodal_features_old")

    metric_columns = [row["name"] for row in cursor.execute("PRAGMA table_info(system_metrics)").fetchall()]
    if "user_id" not in metric_columns:
        cursor.execute("ALTER TABLE system_metrics ADD COLUMN user_id INTEGER")
    if "session_id" not in metric_columns:
        cursor.execute("ALTER TABLE system_metrics ADD COLUMN session_id INTEGER")

    conn.commit()
    conn.close()


init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def register_user(username, password, role="user", invite_code=None):
    if role not in ("user", "admin"):
        return False, "账户类型无效"
    if not username or len(password) < 4:
        return False, "用户名不能为空，且密码长度不能小于 4 位"
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
            "INSERT INTO user_profiles (user_id, nickname, bio, preferred_tts_voice) VALUES (?, ?, '', 'zh-CN-XiaoxiaoNeural')",
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


def get_user_profile(user_id):
    conn = get_connection()
    profile = conn.execute(
        """SELECT u.username, u.role, up.nickname, up.bio, up.preferred_tts_voice
        FROM users u
        JOIN user_profiles up ON u.id = up.user_id
        WHERE u.id=?""",
        (user_id,),
    ).fetchone()
    conn.close()
    if not profile:
        return None
    return {
        "username": profile["username"],
        "role": profile["role"],
        "nickname": profile["nickname"] or "",
        "bio": profile["bio"] or "",
        "preferred_tts_voice": profile["preferred_tts_voice"] or "zh-CN-XiaoxiaoNeural",
    }


def update_user_profile(user_id, nickname, bio, preferred_tts_voice):
    if not nickname:
        return False, "昵称不能为空"
    conn = get_connection()
    exists = conn.execute("SELECT id FROM users WHERE id=?", (user_id,)).fetchone()
    if not exists:
        conn.close()
        return False, "用户不存在"
    conn.execute(
        "UPDATE user_profiles SET nickname=?, bio=?, preferred_tts_voice=? WHERE user_id=?",
        (nickname.strip(), (bio or "").strip(), preferred_tts_voice, user_id),
    )
    conn.commit()
    conn.close()
    return True, "个性化设置已保存"


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
    return [{"role": msg["role"], "content": msg["content"]} for msg in msgs]


def export_session(session_id):
    msgs = get_session_messages(session_id)
    export_path = os.path.join(os.path.dirname(DB_PATH), f"export_session_{session_id}.txt")
    with open(export_path, "w", encoding="utf-8") as file:
        file.write("=== 多模态共情对话系统导出记录 ===\n\n")
        for msg in msgs:
            if msg["role"] == "user":
                file.write(f"[用户]: {msg['content']}\n\n")
            elif msg["role"] == "assistant":
                file.write(f"[AI助手]: {msg['content']}\n\n")
    return export_path


def get_all_users_for_admin():
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, username, role, created_at FROM users ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [[row["id"], row["username"], row["role"], row["created_at"]] for row in rows]


def get_user_choices(include_admin=True):
    conn = get_connection()
    if include_admin:
        rows = conn.execute("SELECT id, username, role FROM users ORDER BY id ASC").fetchall()
    else:
        rows = conn.execute("SELECT id, username, role FROM users WHERE role='user' ORDER BY id ASC").fetchall()
    conn.close()
    return [f"{row['id']} - {row['username']} ({row['role']})" for row in rows]


def parse_user_choice(choice):
    if not choice:
        return None
    try:
        return int(str(choice).split(" - ", 1)[0])
    except (TypeError, ValueError):
        return None


def change_password(user_id, old_password, new_password):
    if not new_password or len(new_password) < 4:
        return False, "新密码长度不能小于 4 位"

    conn = get_connection()
    user = conn.execute("SELECT password_hash FROM users WHERE id=?", (user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "用户不存在"
    if user["password_hash"] != hash_password(old_password):
        conn.close()
        return False, "原密码错误"

    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new_password), user_id))
    conn.commit()
    conn.close()
    return True, "密码修改成功"


def admin_reset_password(target_user_id, new_password):
    if not new_password or len(new_password) < 4:
        return False, "新密码长度不能小于 4 位"

    conn = get_connection()
    user = conn.execute("SELECT id FROM users WHERE id=?", (target_user_id,)).fetchone()
    if not user:
        conn.close()
        return False, "目标用户不存在"

    conn.execute("UPDATE users SET password_hash=? WHERE id=?", (hash_password(new_password), target_user_id))
    conn.commit()
    conn.close()
    return True, f"已成功修改用户 {target_user_id} 的密码"


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


def log_system_metric(module, gpu_mem, info, user_id=None, session_id=None):
    conn = get_connection()
    conn.execute(
        "INSERT INTO system_metrics (module_name, gpu_memory_used, info, user_id, session_id) VALUES (?, ?, ?, ?, ?)",
        (module, gpu_mem, info, user_id, session_id),
    )
    conn.commit()
    conn.close()


def get_user_activity_summary(user_id):
    conn = get_connection()
    user = conn.execute(
        "SELECT id, username, role, created_at FROM users WHERE id=?",
        (user_id,),
    ).fetchone()
    if not user:
        conn.close()
        return "未找到该用户"

    session_count = conn.execute("SELECT COUNT(*) AS count FROM sessions WHERE user_id=?", (user_id,)).fetchone()["count"]
    message_count = conn.execute(
        "SELECT COUNT(*) AS count FROM messages m JOIN sessions s ON m.session_id=s.id WHERE s.user_id=?",
        (user_id,),
    ).fetchone()["count"]
    latest_session = conn.execute("SELECT start_time FROM sessions WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    latest_metric = conn.execute("SELECT timestamp FROM system_metrics WHERE user_id=? ORDER BY id DESC LIMIT 1", (user_id,)).fetchone()
    conn.close()

    latest_session_text = latest_session["start_time"] if latest_session else "无"
    latest_metric_text = latest_metric["timestamp"] if latest_metric else "无"
    return (
        f"用户ID: {user['id']}\n"
        f"用户名: {user['username']}\n"
        f"角色: {user['role']}\n"
        f"注册时间: {user['created_at']}\n"
        f"会话数量: {session_count}\n"
        f"消息数量: {message_count}\n"
        f"最近会话时间: {latest_session_text}\n"
        f"最近监控记录时间: {latest_metric_text}"
    )


def get_user_metrics(user_id, limit=20):
    conn = get_connection()
    rows = conn.execute(
        """SELECT timestamp, module_name, gpu_memory_used, info, session_id
        FROM system_metrics
        WHERE user_id=?
        ORDER BY id DESC
        LIMIT ?""",
        (user_id, limit),
    ).fetchall()
    conn.close()
    if not rows:
        return "暂无该用户的监控记录"

    lines = []
    for row in rows:
        session_text = f"会话ID={row['session_id']}" if row["session_id"] else "无会话"
        lines.append(
            f"[{row['timestamp']}] 模块={row['module_name']} | GPU={row['gpu_memory_used']} | {session_text} | {row['info']}"
        )
    return "\n".join(lines)
