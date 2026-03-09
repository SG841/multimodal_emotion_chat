import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import re

# 导入配置文件
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import LLM_MODEL, LLM_MAX_LENGTH, LLM_TEMPERATURE

_model = None
_tokenizer = None

def _get_llm_model():
    """懒加载获取大语言模型实例"""
    global _model, _tokenizer
    if _model is None:
        print(f"🚀 [LLM模块] 正在加载 Qwen 模型: {LLM_MODEL}")
        # 加载分词器
        _tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)

        # 针对 L40S 优化：使用 bfloat16 和 device_map="auto"
        _model = AutoModelForCausalLM.from_pretrained(
            LLM_MODEL,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        _model.eval()
        print(f"✅ [LLM模块] 模型加载完成，运行在 {_model.device}")
    return _model, _tokenizer

def generate_empathetic_response(user_text: str, visual_emotion: str, audio_emotion: str, final_emotion: str, chat_history: list) -> tuple:
    """
    结合多模态情感生成共情回复，并要求大模型输出最终的情感标签
    返回: (llm_emotion, response_text)
    """
    model, tokenizer = _get_llm_model()

    # 1. 强格式化的系统 Prompt
# 1. 强格式化的系统 Prompt (全中文强制版)
    system_prompt = (
        "你是一个富有同理心且专业的心理咨询师。你需要根据用户的文本输入以及系统提供的多模态情感分析结果，给出回答。\n\n"
        "【最高级指令 - 严格遵守】：\n"
        "1. 拒绝任何分析过程、思考步骤或多余解释。直接输出最终结果！\n"
        "2. 【语言强制规定】：除了情绪标签必须是指定的纯英文单词外，其余所有内容（包括你的回复）**必须全部使用纯中文**！绝对不允许在回复中夹杂英文。\n"
        "3. 你的输出必须只有两行，严格按照以下确切格式：\n"
        "情绪：[只能从 Happy, Sad, Angry, Neutral, Surprise, Fear 中选一个]\n"
        "回复：[你的纯中文共情回复，像真人一样自然流畅]\n\n"
        "【正确输出示例】：\n"
        "情绪：Happy\n"
        "回复：很高兴认识你！听到你分享自己喜欢的唱跳和篮球，感觉你非常有活力呢，今天是有什么开心的事情想跟我分享吗？"
    )

    # 2. 转换对话历史
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history:
        messages.append(msg)

    # 3. 注入多模态信息的当前输入
    current_input = (
        f"（后台参考数据：视觉检测为 {visual_emotion}，语音情感为 {audio_emotion}，系统初步判定为 {final_emotion}。）\n\n"
        f"用户说：{user_text}"
    )
    messages.append({"role": "user", "content": current_input})

    # 4. 生成推理
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=LLM_MAX_LENGTH,
            temperature=LLM_TEMPERATURE,
            top_p=0.8,
            repetition_penalty=1.05
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 5. 解析模型的格式化输出
    llm_emotion = "Neutral"  # 默认值
    response_text = raw_output  # 如果解析失败，返回全部原始文本

    try:
        # 使用正则提取"情绪："和"回复："后面的内容
        emotion_match = re.search(r"情绪：\s*([A-Za-z]+)", raw_output)
        reply_match = re.search(r"回复：\s*(.*)", raw_output, re.DOTALL)

        if emotion_match:
            parsed_emotion = emotion_match.group(1).capitalize()
            # 确保提取出的情绪在我们的六个目标情绪内
            if parsed_emotion in ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]:
                llm_emotion = parsed_emotion

        if reply_match:
            response_text = reply_match.group(1).strip()

    except Exception as e:
        print(f"⚠️ [LLM模块] 解析模型输出失败: {e}\n原始输出: {raw_output}")

    return llm_emotion, response_text
