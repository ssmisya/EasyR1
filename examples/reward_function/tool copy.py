import re
# from math_verify import parse, verify
import json
from typing import Dict, List
from mathruler.grader import extract_boxed_content, grade_answer

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    print("==============开始计算得分===============\n")
    print("predicts",predicts)
    print("=============================\n")
    print("ground_truths",ground_truths)
    breakpoint()
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        format_reward_value,reward_value,acc_reward_value = 0.0, 0.0, 0.0
        # print(f"predict: {predict},predict_type: {type(predict)}, ground_truth: {ground_truth}, ground_truth_type: {type(ground_truth)}")
        print("predict",predict)
        print("=============================\n")
        print("ground_truth",ground_truth)
        # 确保predict是字符串
        # 这一步有很大问题，因为predict是list，所以需要修改
        if isinstance(predict, list):
            predict = predict[0] if predict else ""
        answer, format_reward_value = format_reward(predict, ground_truth)
        if format_reward_value == 1.0:
            try:
                acc_reward_value = grade_answer(answer, ground_truth)
            except: 
                acc_reward_value = 0.0
            reward_value = (1 - format_weight) * acc_reward_value + format_weight * format_reward_value
        scores.append(
            {
                "overall": reward_value,
                "format": format_reward_value,
                "accuracy": acc_reward_value,
            }
        )
        # print("==============得到的scores===============\n",scores)

    return scores

def tool_compute_score(predict_str: str, ground_truth: str) -> float:
    reward = 0.0
    answer, format_reward_value = format_reward(predict_str, ground_truth)
    if format_reward_value == 1.0:
        reward = 0.1
        try:
            acc_reward = grade_answer(answer, ground_truth)
            reward = 0.1 * format_reward_value + 0.9 * acc_reward
        except: 
            reward = 0.1
    
    return reward

# 目前实现了<think>, <tool_call> or <response>标签的格式检查，以及<tool_call>和<response>的互斥检查
# 实现了三者的顺序检查
# tool_call的内部格式，工具调用参数这些后续或许可以加
def format_reward(predict_str: str, ground_truth: str):
    """Reward function that checks if the completion has the correct format with <think>, <tool_call> or <response> tags."""
    reward = 0.0
    answer = None
    
    # 清理输入文本，去除前后空白
    cleaned_text = predict_str.strip()
    
    # 检查是否包含<think>标签
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, cleaned_text, re.DOTALL)
    
    # 检查是否包含<response>标签
    response_pattern = r'<response>(.*?)</response>'
    response_match = re.search(response_pattern, cleaned_text, re.DOTALL)
    
    # 检查是否包含<tool_call>标签
    tool_call_pattern = r'<tool_call>(.*?)</tool_call>'
    tool_call_match = re.search(tool_call_pattern, cleaned_text, re.DOTALL)
    
    # 检查标签顺序和互斥性
    if think_match and ((tool_call_match and not response_match) or (response_match and not tool_call_match)):
        # 提取所有标签的位置
        think_start = cleaned_text.find('<think>')
        
        # 检查<think>是否在开头
        if think_start != 0:
            return None, 0.0
            
        # 处理<tool_call>或<response>
        if tool_call_match:
            # 验证<tool_call>内容是否为有效的JSON格式
            try:
                tool_call_content = tool_call_match.group(1).strip()
                json_obj = json.loads(tool_call_content)
                if "name" in json_obj and "parameters" in json_obj:
                    reward = 1.0
                else:
                    return None, 0.0
            except:
                return None, 0.0
                
        elif response_match:
            # 提取答案
            answer = response_match.group(1).strip()
            reward = 1.0
    else:
        reward = 0.0
        answer = None
        
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