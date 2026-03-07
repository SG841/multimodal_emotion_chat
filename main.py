from modelscope.hub.snapshot_download import snapshot_download
import os

# 定义你的项目模型存放路径
model_dir = '/root/multimodal_emotion_chat/models'

print(f"🚀 正在开始下载 Emotion2Vec+ 升级版...")
print(f"📂 目标路径: {model_dir}")

# 下载模型
# iic/emotion2vec_plus_large 是目前最推荐的升级版，支持中英双语和 9 分类
snapshot_download(
    'iic/emotion2vec_plus_large',
    local_dir=model_dir
)

print("✅ 下载完成！你可以直接在 audio.py 中引用此路径。")