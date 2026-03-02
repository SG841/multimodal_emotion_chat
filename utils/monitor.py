"""
utils/monitor.py
状态监控模块：专注于硬件资源监测及多模态融合决策日志的格式化
"""

import time
import psutil
import torch
from typing import Dict

class SystemMonitor:
    def __init__(self):
        # 初始化 GPU 监测
        self.has_cuda = torch.cuda.is_available()

    def get_resource_status(self) -> Dict[str, str]:
        """获取系统资源占用状况，支撑 app.py 中的显存显示"""
        status = {"cpu": f"{psutil.cpu_percent()}"}

        if self.has_cuda:
            # 获取显存已用空间
            mem_allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            status["gpu_mem"] = f"{mem_allocated:.0f} MB"
        else:
            status["gpu_mem"] = "无 GPU"

        return status

    def format_fusion_report(self, v_emo: str, v_conf: float, a_emo: str, a_conf: float, final: str) -> str:
        """
        将后端的融合逻辑转化为前端易读的文本报告。
        用于解释为什么系统最终选择了某种情绪。
        """
        timestamp = time.strftime("%H:%M:%S")

        # 这里的逻辑可以帮助你在答辩时解释"置信度对齐"的过程
        report = (
            f"[{timestamp}] 融合决策路径：\n"
            f"  - 视觉序列聚合结果: {v_emo} (置信度: {v_conf:.2f})\n"
            f"  - 听觉片段识别结果: {a_emo} (置信度: {a_conf:.2f})\n"
            f"  - 综合决策判定 >> {final} <<"
        )
        return report


# 导出单例，方便 app.py 直接调用
monitor = SystemMonitor()
