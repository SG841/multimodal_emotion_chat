"""
👄 语音合成模块
调用 Edge-TTS 将大语言模型的文本回复转换为音频文件
"""

import edge_tts
import asyncio
import os
import time
import sys

# 导入配置
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TTS_VOICE

# 确保输出目录存在
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "output_tts")
os.makedirs(OUTPUT_DIR, exist_ok=True)

async def generate_audio_reply(text: str) -> str:
    """
    异步生成语音文件

    Args:
        text: 需要合成的中文文本

    Returns:
        生成好的 .mp3 文件本地绝对路径，如果失败则返回 None
    """
    if not text:
        return None

    # 生成带时间戳的唯一文件名，防止高频对话时文件被覆盖或读写冲突
    filename = f"reply_{int(time.time() * 1000)}.mp3"
    output_path = os.path.join(OUTPUT_DIR, filename)

    try:
        print(f"⏳ [TTS模块] 正在合成语音: {text[:15]}...")
        communicate = edge_tts.Communicate(text, TTS_VOICE)
        await communicate.save(output_path)
        print(f"✅ [TTS模块] 语音合成完成: {output_path}")
        return output_path
    except Exception as e:
        print(f"❌ [TTS模块] 语音合成失败: {e}")
        return None
