"""
用户认证模块（基于 JSON 存储）
用于毕设 Demo 的简单用户管理系统
"""

import json
import os
import hashlib

USER_DB_PATH = "assets/users.json"


def hash_password(password):
    """使用 SHA-256 对密码进行哈希加密"""
    return hashlib.sha256(password.encode()).hexdigest()


def init_db():
    """初始化用户数据库（如果不存在）"""
    if not os.path.exists(os.path.dirname(USER_DB_PATH)):
        os.makedirs(os.path.dirname(USER_DB_PATH), exist_ok=True)
    if not os.path.exists(USER_DB_PATH):
        with open(USER_DB_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)


def register_user(username, password):
    """
    注册新用户

    Args:
        username: 用户名
        password: 密码（明文）

    Returns:
        str: 注册结果消息
    """
    init_db()

    if not username or not password:
        return "❌ 用户名或密码不能为空"

    if len(username) < 2:
        return "❌ 用户名至少需要2个字符"

    if len(password) < 4:
        return "❌ 密码至少需要4个字符"

    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)

    if username in users:
        return "❌ 该用户名已被注册"

    users[username] = {
        "password": hash_password(password),
        "created_at": __import__('time').strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(USER_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

    return "✅ 注册成功，请登录"


def login_user(username, password):
    """
    验证用户登录

    Args:
        username: 用户名
        password: 密码（明文）

    Returns:
        bool: 登录是否成功
    """
    init_db()

    if not username or not password:
        return False

    with open(USER_DB_PATH, "r", encoding="utf-8") as f:
        users = json.load(f)

    if username in users and users[username]["password"] == hash_password(password):
        return True

    return False
