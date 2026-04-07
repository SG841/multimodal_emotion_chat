"""语音合成模块。"""

import os
import sys
import time

import edge_tts

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TTS_VOICE

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "output_tts")
os.makedirs(OUTPUT_DIR, exist_ok=True)


async def generate_audio_reply(text: str, voice: str | None = None) -> str | None:
    if not text:
        return None

    filename = f"reply_{int(time.time() * 1000)}.mp3"
    output_path = os.path.join(OUTPUT_DIR, filename)
    selected_voice = voice or TTS_VOICE

    try:
        print(f"[TTS] 正在合成语音，音色={selected_voice}: {text[:15]}...")
        communicate = edge_tts.Communicate(text, selected_voice)
        await communicate.save(output_path)
        print(f"[TTS] 语音合成完成: {output_path}")
        return output_path
    except Exception as exc:
        print(f"[TTS] 语音合成失败: {exc}")
        return None
