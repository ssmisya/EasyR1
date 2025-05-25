import re
# from math_verify import parse, verify
import json
from typing import Dict, List
from mathruler.grader import extract_boxed_content, grade_answer

def compute_score(predicts: List[str], ground_truths: List[str], format_weight: float = 0.1) -> List[Dict[str, float]]:
    scores = []
    for predict, ground_truth in zip(predicts, ground_truths):
        format_reward_value,reward_value,acc_reward_value = 0.0, 0.0, 0.0
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


def format_reward(predict_str: str, ground_truth: str):
    """Reward function that checks if the completion has a specific format, in this case, if it contains the word 'Terminate'."""
    reward = 0.0
    answer = None
    dict_list = extract_json_objects(predict_str)
    try:
        if len(dict_list) > 0:
            last_dict = dict_list[-1]
            actions = last_dict["actions"]
            last_action = actions[-1]
            if last_action["name"] == "Terminate":
                answer = last_action["arguments"]["ans"]
                reward = 1.0
    except: 
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