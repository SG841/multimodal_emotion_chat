"""
utils/monitor.py
硬件级监控模块：专注显存状态
"""

import time
import torch
import pynvml
from typing import Dict

class SystemMonitor:
    def __init__(self):
        self.has_cuda = torch.cuda.is_available()
        self.nvml_initialized = False

        if self.has_cuda:
            try:
                pynvml.nvmlInit()
                # 关联 L40S 显卡句柄
                self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                self.nvml_initialized = True
            except Exception as e:
                print(f"⚠️ [监控系统] 硬件接口调用失败: {e}")

    def get_resource_status(self) -> Dict[str, str]:
        """
        直接通过 NVML 获取显卡真实的已用显存。
        这能准确反映 Whisper (C++/Ctranslate2) 和 ViT 的总占用。
        """
        status = {"gpu_mem": "-- MB"}

        if self.nvml_initialized:
            try:
                # 获取显存原始数据 (Bytes)
                info = pynvml.nvmlDeviceGetMemoryInfo(self.handle)
                used_mb = info.used / (1024**2)
                total_mb = info.total / (1024**2)

                # 格式化输出：已用 / 总共
                status["gpu_mem"] = f"{used_mb:.0f} / {total_mb:.0f} MB"
            except:
                status["gpu_mem"] = "读取异常"

        return status

    def format_fusion_report(self, v_emo: str, v_conf: float, a_emo: str, a_conf: float, final: str) -> str:
        """保持原有方法签名，确保 app.py 调用不报错"""
        timestamp = time.strftime("%H:%M:%S")
        return (
            f"[{timestamp}] 融合决策：{final}\n"
            f"视觉判定: {v_emo} ({v_conf:.2%})\n"
            f"听觉判定: {a_emo} ({a_conf:.2%})"
        )

# 导出单例
monitor = SystemMonitor()
