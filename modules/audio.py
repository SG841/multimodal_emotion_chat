import os
import time
from faster_whisper import WhisperModel

# 1. 设置镜像环境变量（进程级临时生效）
os.environ["HF_ENDPOINT"] = "http://hfmirror.mas.zetyun.cn:8082"

# 2. 修改后的存放路径：下载到 models 目录下的 faster-whisper 文件夹中
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../models/faster-whisper")


print(f"🚀 准备从镜像下载/加载模型至: {os.path.abspath(model_path)}")

# 3. 初始化 WhisperModel
# 使用 large-v3，针对 L40S 开启 float16 半精度加速
model = WhisperModel(
    "large-v3", 
    device="cuda", 
    compute_type="float16", 
    download_root=model_path
)

def run_test():
    # 使用基于脚本目录的路径
    test_media_dir = os.path.join(script_dir, "../assets/test_media")

    # 支持的音频格式
    audio_extensions = ['.wav', '.mp3', '.ogg', '.flac', '.m4a', '.aac']

    # 获取所有音频文件
    audio_files = []
    if os.path.exists(test_media_dir):
        for file in os.listdir(test_media_dir):
            if os.path.splitext(file)[1].lower() in audio_extensions:
                audio_files.append(os.path.join(test_media_dir, file))

    if not audio_files:
        print(f"❌ 错误：在 {test_media_dir} 未找到音频文件")
        return

    print(f"🎙️ 找到 {len(audio_files)} 个音频文件，开始转录...\n")

    # 4. 执行转录，针对中文优化
    # initial_prompt 强制引导输出简体中文
    for i, test_audio in enumerate(audio_files, 1):
        print(f"\n{'='*50}")
        print(f"文件 {i}/{len(audio_files)}: {os.path.basename(test_audio)}")
        print(f"{'='*50}")

        start_time = time.time()

        try:
            segments, info = model.transcribe(
                test_audio,
                beam_size=5,
                language="zh",
                initial_prompt="以下是普通话，请转录为简体中文。"
            )

            for segment in segments:
                print(f"[{segment.start:.2f}s -> {segment.end:.2f}s]: {segment.text}")

            duration = time.time() - start_time
            print(f"✅ 转录完成！耗时: {duration:.2f}s")
            print(f"检测语言: {info.language} (置信度: {info.language_probability:.2%})")

        except Exception as e:
            print(f"❌ 转录失败: {e}")

    print(f"\n{'='*50}")
    print(f"🎉 所有音频转录完成！")

if __name__ == "__main__":
    run_test()