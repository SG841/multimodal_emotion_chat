import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os
import re
import json  # 新增 json 导入

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
    结合多模态情感生成共情回复，使用前缀引导强制 JSON 输出
    """
    model, tokenizer = _get_llm_model()

    # 1. 极致精简的 System Prompt
    system_prompt = """你是一个专业的心理咨询师,需要给照顾用户的同时提供一些有建设性的建议。请直接输出JSON格式的回复，不需要任何分析或解释。
格式：{"emotion": "Happy/Sad/Angry/Neutral/Surprise/Fear", "response": "你的纯中文共情回复"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        # 极简打样
        {"role": "user", "content": "（后台参考数据：视觉检测为 Neutral，语音情感为 Sad，系统初步判定为 Sad。）\n\n用户说：我最近压力真的很大，天天失眠。"},
        {"role": "assistant", "content": '{"emotion": "Sad", "response": "听到你最近因为压力大而失眠，我感到很心疼。长期休息不好身体肯定吃不消，能跟我具体说说是什么让你这么有压力吗？"}'}
    ]

    for msg in chat_history:
        messages.append(msg)

    current_input = (
        f"（后台参考数据：视觉检测为 {visual_emotion}，语音情感为 {audio_emotion}，系统初步判定为 {final_emotion}。）\n\n"
        f"用户说：{user_text}"
    )
    messages.append({"role": "user", "content": current_input})

    # 2. 转换 Prompt
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # 【终极必杀技：物理外挂前缀引导】
    # 我们直接帮模型打出左大括号和第一个键名，它就绝对不可能再输出 Thinking Process 了！
    force_prefix = '{\n    "emotion": "'
    text += force_prefix

    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # 3. 推理生成
    with torch.inference_mode():
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            top_p=0.8,
            repetition_penalty=1.05,
            pad_token_id=tokenizer.eos_token_id # 顺手把终端里的警告消除掉
        )

    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    raw_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    # 把我们帮它写的前缀拼回去，构成完整的 JSON 字符串
    full_json_string = force_prefix + raw_output
    print(f"DEBUG [LLM完整输出]:\n{full_json_string}")

    # 4. JSON 提取与解析
    llm_emotion = "Neutral"
    response_text = ""

    try:
        # 【修复正则 bug】：找最后一个像 JSON 的块，防止抓到废话
        json_matches = re.findall(r'\{.*?\}', full_json_string, re.DOTALL)
        if json_matches:
            json_str = json_matches[-1] # 取最后一个匹配项
            data = json.loads(json_str)

            parsed_emotion = data.get("emotion", "Neutral").capitalize()
            if parsed_emotion in ["Happy", "Sad", "Angry", "Neutral", "Surprise", "Fear"]:
                llm_emotion = parsed_emotion

            response_text = data.get("response", "").strip()

            if not response_text:
                raise ValueError("JSON中response字段为空")
        else:
            # 极限兜底：如果 JSON 还是坏了，尝试硬切字符串
            if '"response":' in full_json_string:
                response_text = full_json_string.split('"response":')[1].strip(' "}\n')
            else:
                raise ValueError("未找到有效内容")

    except json.JSONDecodeError as e:
        print(f"⚠️ [LLM模块] JSON 解析失败: {e}")
        # 如果少了右括号，尝试硬切
        if '"response":' in full_json_string:
            response_text = full_json_string.split('"response":')[1].strip(' "}\n')
        else:
            response_text = "我感受到了你的情绪，可以多和我说说吗？"
    except Exception as e:
        print(f"⚠️ [LLM模块] 提取严重异常: {e}")
        response_text = "我正在努力理解你的感受，请稍等。"

    return llm_emotion, response_text
