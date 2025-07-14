import re
# from math_verify import parse, verify
import json
from typing import Dict, List, Optional
from mathruler.grader import extract_boxed_content, grade_answer

def compute_score(predicts: List[str], ground_truths: List[str], tool_rewards: List[List[float]], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for i, (predict, ground_truth) in enumerate(zip(predicts, ground_truths)):
        reward_value,format_reward_value,acc_reward_value,tool_reward_value = 0.0, 0.0, 0.0, 0.0
        if isinstance(predict, list):
            if len(predict) == 1:
                predict = predict[0]
                answer, format_reward_value = format_reward(predict, ground_truth)
            else:
                for pred in predict:
                    # answer会不断被更新，直到赋值为最后一项的答案
                    # format reward value要么为0，要么为1，并且不考虑有多少轮，因为都为1的话求平均还是1
                    answer, round_format_reward_value = format_reward(pred, ground_truth)
                    format_reward_value = round_format_reward_value
                    # 只要有一轮的reward为0，那就设置为0
                    if round_format_reward_value == 0:
                        format_reward_value = 0
                        break
        # 如果是字符串
        elif isinstance(predict, str):
            answer, format_reward_value = format_reward(predict, ground_truth)
        
        else:
            # predit不符合预期，报错
            raise TypeError(f"predict 类型不符合预期，应为 list 或 str，实际类型为: {type(predict)}")

        # 只有有了format_reward_value才会去考虑acc和tool reward
        if format_reward_value == 1.0:
            try:
                # grade_answer的输出事True或者False，可以当1或者0使用
                # answr来自于format_reward的返回
                acc_reward_value = grade_answer(answer, ground_truth)
                tool_reward = tool_rewards[i]
                if isinstance(tool_reward, list):
                    if len(tool_reward) > 1:
                        # 如果最后一轮的工具得分是0，则不计算最后一轮的工具得分,说明它回答了
                        # 但是如果最后一轮的工具得分不为0，则计算最后一轮的工具得分
                        if tool_reward[-1] == 0.0:
                            tool_reward_value = sum(tool_reward[:-1]) / (len(tool_reward) - 1)
                        else:
                            tool_reward_value = sum(tool_reward) / len(tool_reward)
                    else:
                        tool_reward_value = tool_reward[0]
                elif isinstance(tool_reward, float):
                    tool_reward_value = tool_reward
                else:
                    raise TypeError(f"tool_reward 类型不符合预期，应为 list 或 float: {type(tool_reward)}")
            except: 
                acc_reward_value = 0.0
            # reward_value = (1 - format_weight) * acc_reward_value + format_weight * 
            # 分数范围为0-6分
            reward_value = format_reward_value + acc_reward_value + tool_reward_value
        # print(f"当模型回答为{predict}时,传进来的tool_reward_value为{tool_rewards[i]}, overall_reward_value为{reward_value}, format_reward_value为{format_reward_value}, acc_reward_value为{acc_reward_value}")
        scores.append(
            {
                "overall": reward_value,
                "format": format_reward_value,
                "accuracy": acc_reward_value,
                "tool": tool_reward_value
            }
        )

    return scores

# 目前实现了<think>, <tool_call> or <response>标签的格式检查，以及<tool_call>和<response>的互斥检查
# 实现了三者的顺序检查
def format_reward(predict_str: str, ground_truth: str):
    """Reward function that checks if the completion has the correct format with <think>, <tool_call> or <response> tags."""
    reward = 0.0
    answer = None
    
    # 清理输入文本，去除前后空白
    cleaned_text = predict_str.strip()
    
        # 检查标签是否成对出现
    tags_to_check = [
        ('<think>', '</think>'),
        ('<tool_call>', '</tool_call>'),
        ('<response>', '</response>')
    ]
    
    # 判断配对关系
    for open_tag, close_tag in tags_to_check:
        if cleaned_text.count(open_tag) != cleaned_text.count(close_tag):
            return None, 0.0

    # 检查是否包含<think>标签
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)
    
    # 检查是否包含<response>标签
    response_pattern = r'<response>(.*?)</response>'
    response_match = re.search(response_pattern, cleaned_text, re.DOTALL)
    
    # 检查是否包含<tool_call>标签
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_call_match = re.search(tool_call_pattern, cleaned_text, re.DOTALL)
    
    # 检查</tool_call>或</response>后面是否有内容
    if tool_call_match:
        tool_call_end = cleaned_text.rfind('</tool_call>') + len('</tool_call>')
        if tool_call_end < len(cleaned_text):
            # 检查后面是否有非空白字符
            if cleaned_text[tool_call_end:].strip():
                return None, 0.0
    
    if response_match:
        response_end = cleaned_text.rfind('</response>') + len('</response>')
        if response_end < len(cleaned_text):
            # 检查后面是否有非空白字符
            if cleaned_text[response_end:].strip():
                return None, 0.0
    
    # 检查标签顺序和互斥性
    if think_match and ((tool_call_match and not response_match) or (response_match and not tool_call_match)):
        # 提取所有标签的位置
        think_start = cleaned_text.find('<think>')
        
        # 检查<think>是否在开头
        if think_start != 0:
            return None, 0.0
                
        if response_match:
            # 提取答案
            answer = response_match.group(1).strip()
            if "\\boxed{" in answer:
                answer = answer.split("\\boxed{")[1].split("}")[0].strip()
            reward = 1.0
        
        # 如果只有tool_call，之前忘记了这个
        elif tool_call_match:
            reward = 1.0
    else:
        reward = 0.0
        answer = None

    # 检查标签之间不能有包含关系
    if think_match:
        think_content = think_match.group(1)
        if '<tool_call>' in think_content or '</tool_call>' in think_content or '<response>' in think_content or '</response>' in think_content:
            return None, 0.0
    
    if tool_call_match:
        tool_call_content = tool_call_match.group(1)
        if '<think>' in tool_call_content or '</think>' in tool_call_content or '<response>' in tool_call_content or '</response>' in tool_call_content:
            return None, 0.0
    
    if response_match:
        response_content = response_match.group(1)
        if '<think>' in response_content or '</think>' in response_content or '<tool_call>' in response_content or '</tool_call>' in response_content:
            return None, 0.0
        
    return answer, reward
    
def extract_json_objects(text, max_objects=20, max_length=4096):
    """
    Find all JSON objects in a string and parse them into dictionaries.
    Uses pruning strategies to avoid excessive computation on complex inputs.
    
    Args:
        text: String that may contain JSON objects
        max_objects: Maximum number of objects to extract
        max_length: Maximum length to consider for a JSON object
        
    Returns:
        List of parsed JSON dictionaries
    """
    json_objects = []
    i = 0
    
    while i < len(text) and len(json_objects) < max_objects:
        # Skip non-opening braces
        if text[i] != '{':
            i += 1
            continue
            
        # Found opening brace, track nesting level
        start = i
        brace_level = 0
        j = i
        
        # Find matching closing brace with length limit
        while j < len(text) and j - start <= max_length:
            if text[j] == '{':
                brace_level += 1
            elif text[j] == '}':
                brace_level -= 1
                if brace_level == 0:
                    # Found a balanced JSON candidate
                    try:
                        json_str = text[start:j+1]
                        json_obj = json.loads(json_str)
                        if isinstance(json_obj, dict):
                            json_objects.append(json_obj)
                            i = j  # Skip to end of this JSON object
                    except json.JSONDecodeError:
                        pass
                    break
            j += 1
        
        # Move to next character if no valid JSON found
        i += 1
    
    return json_objects